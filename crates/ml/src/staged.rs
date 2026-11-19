//! Coarse-to-fine subject matting: one model decides the subject on the whole image, a second
//! reruns on a crop around it, and the coarse result is kept wherever it is confident.
//!
//! Private build: the box padding and the blend come from the reconstruction of Photoshop's
//! generic refinement in `photoshop_pipelines`, so this file is not clean-room.

/// Opacity above which the coarse mask counts as subject when placing the crop (50 of 255).
const ACTIVE: f32 = 50.0 / 255.0;
/// Crop padding, as a share of the box's short side.
const BOX_PAD: f32 = 0.1;
const ERODE_PASSES: usize = 5;
/// Below this share of the coarse subject, the whole-image result has lost the subject.
const LOST_SHARE: f32 = 0.35;

/// The padded box around the subject as `(x, y, width, height)`, or `None` with no subject.
pub fn active_box(mask: &[f32], width: usize, height: usize) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 || mask.len() != width.checked_mul(height)? {
        return None;
    }
    let (mut left, mut top, mut right, mut bottom) = (width, height, 0, 0);
    for (y, row) in mask.chunks_exact(width).enumerate() {
        for (x, value) in row.iter().enumerate() {
            if *value > ACTIVE {
                left = left.min(x);
                right = right.max(x + 1);
                top = top.min(y);
                bottom = bottom.max(y + 1);
            }
        }
    }
    if left >= right || top >= bottom {
        return None;
    }
    let pad = ((right - left).min(bottom - top) as f32 * BOX_PAD) as usize;
    let (left, top) = (left.saturating_sub(pad), top.saturating_sub(pad));
    let (right, bottom) = ((right + pad).min(width), (bottom + pad).min(height));
    Some((left, top, right - left, bottom - top))
}

/// The minimum of each pixel and its eight neighbours, as a row pass then a column pass.
fn erode(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut rows = Vec::with_capacity(values.len());
    for row in values.chunks_exact(width) {
        for x in 0..width {
            let around = row.get(x.saturating_sub(1)..(x + 2).min(width)).unwrap_or_default();
            rows.push(around.iter().copied().fold(f32::INFINITY, f32::min));
        }
    }
    let at = |x: usize, y: usize| rows.get(y * width + x).copied().unwrap_or(f32::INFINITY);
    let mut out = Vec::with_capacity(values.len());
    for y in 0..height {
        for x in 0..width {
            let above = if y > 0 { at(x, y - 1) } else { f32::INFINITY };
            out.push(above.min(at(x, y)).min(at(x, y + 1)));
        }
    }
    out
}

/// `c * hard(coarse) + (1 - c) * fine`, where `c` is the coarse mask's confidence, eroded so the
/// fine result takes over a little way in from every uncertain edge. All masks are 0..=1.
/// Whether `mask` kept so little of what `coarse` found that the subject must have been missed.
/// A small subject is the usual cause, and a crop around it recovers it.
pub fn lost_subject(mask: &[f32], coarse: &[f32]) -> bool {
    let active = |values: &[f32]| values.iter().filter(|v| **v > ACTIVE).count() as f32;
    active(mask) < active(coarse) * LOST_SHARE
}

pub fn confidence_blend(coarse: &[f32], fine: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
    let pixels = width.checked_mul(height).filter(|n| *n > 0)?;
    if coarse.len() != pixels || fine.len() != pixels {
        return None;
    }
    let mut confidence: Vec<f32> = coarse.iter().map(|v| (v - 0.5).abs() * 2.0).collect();
    for _ in 0..ERODE_PASSES {
        confidence = erode(&confidence, width, height);
    }
    let hard = coarse.iter().map(|v| if *v > 0.5 { 1.0 } else { 0.0 });
    Some(confidence.iter().zip(hard).zip(fine).map(|((c, hard), fine)| c * hard + (1.0 - c) * fine).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_subject_counts_as_lost_only_when_little_of_it_remains() {
        let coarse = vec![1.0; 10];
        let kept = |n: usize| (0..10).map(|i| if i < n { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
        assert!(lost_subject(&kept(0), &coarse));
        assert!(lost_subject(&kept(3), &coarse));
        assert!(!lost_subject(&kept(4), &coarse));
        assert!(!lost_subject(&coarse, &kept(0)), "nothing to lose when the coarse pass found nothing");
    }

    #[test]
    fn the_box_is_padded_by_its_short_side_and_stays_on_the_image() {
        let mut mask = vec![0.0; 100 * 100];
        for y in 20..40 {
            for x in 30..70 {
                mask[y * 100 + x] = 1.0;
            }
        }
        assert_eq!(active_box(&mask, 100, 100), Some((28, 18, 44, 24)));
        mask.fill(1.0);
        assert_eq!(active_box(&mask, 100, 100), Some((0, 0, 100, 100)));
    }

    #[test]
    fn faint_masks_and_bad_sizes_have_no_box() {
        assert_eq!(active_box(&[0.1; 4], 2, 2), None);
        assert_eq!(active_box(&[1.0; 3], 2, 2), None);
        assert_eq!(active_box(&[], 0, 0), None);
        assert_eq!(active_box(&[1.0; 2], usize::MAX, 2), None);
    }

    #[test]
    fn confident_coarse_wins_and_uncertain_coarse_defers_to_fine() {
        let (width, height) = (32, 32);
        let solid = vec![1.0; width * height];
        let fine = vec![0.25; width * height];
        assert_eq!(confidence_blend(&solid, &fine, width, height).unwrap(), solid);
        let unsure = vec![0.5; width * height];
        assert_eq!(confidence_blend(&unsure, &fine, width, height).unwrap(), fine);
    }

    #[test]
    fn the_fine_result_reaches_past_the_uncertain_edge() {
        let (width, height) = (32, 8);
        let coarse: Vec<f32> = (0..width * height).map(|i| if i % width == 16 { 0.5 } else { 1.0 }).collect();
        let fine = vec![0.0; width * height];
        let blended = confidence_blend(&coarse, &fine, width, height).unwrap();
        // Five passes carry the doubt five pixels either side of the seam, and no further.
        assert_eq!(blended[3 * width + 11], 0.0);
        assert_eq!(blended[3 * width + 21], 0.0);
        assert_eq!(blended[3 * width + 10], 1.0);
        assert_eq!(blended[3 * width + 22], 1.0);
    }

    #[test]
    fn mismatched_masks_are_refused() {
        assert_eq!(confidence_blend(&[1.0], &[1.0, 1.0], 1, 1), None);
        assert_eq!(confidence_blend(&[], &[], 0, 0), None);
        assert_eq!(confidence_blend(&[1.0], &[1.0], usize::MAX, 2), None);
    }
}
