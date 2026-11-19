//! The unknown band a matting model solves: a trimap from a soft mask, and the tiles that cover
//! its unknown pixels.

use super::morph;

/// Trimap values.
pub const BACKGROUND: f32 = 0.0;
pub const UNKNOWN: f32 = 0.5;
pub const SUBJECT: f32 = 1.0;
/// A mask this close to 0 or 1 is sure of a pixel.
const SURE: f32 = 0.05;

/// A trimap from a soft `mask` (`w × h`): the pixels it is sure of, eroded by `band` px, keep
/// their side, and everything between is unknown. Empty when the sizes don't match.
pub fn trimap(mask: &[f32], w: usize, h: usize, band: usize) -> Vec<f32> {
    if w.checked_mul(h) != Some(mask.len()) {
        return Vec::new();
    }
    let sure = |subject: bool| {
        let plane: Vec<f32> = mask.iter().map(|v| if (subject && *v >= 1.0 - SURE) || (!subject && *v <= SURE) { 1.0 } else { 0.0 }).collect();
        morph(&plane, w, h, band, false)
    };
    let (subject, background) = (sure(true), sure(false));
    if subject.len() != mask.len() || background.len() != mask.len() {
        return Vec::new();
    }
    subject.iter().zip(&background).map(|(s, b)| if *s > 0.5 { SUBJECT } else if *b > 0.5 { BACKGROUND } else { UNKNOWN }).collect()
}

/// Top-left corners of `side`-px tiles covering every unknown pixel of `trimap` (`w × h`), each
/// overlapping its neighbours by about `overlap` px so their seams can be blended away. Tiles
/// are placed where uncovered unknown pixels are, so a thin band costs few tiles.
pub fn band_tiles(trimap: &[f32], w: usize, h: usize, side: usize, overlap: usize) -> Vec<(usize, usize)> {
    if w.checked_mul(h) != Some(trimap.len()) || side == 0 {
        return Vec::new();
    }
    let margin = (overlap / 2).min(side.saturating_sub(1) / 2);
    let mut covered = vec![false; trimap.len()];
    let mut tiles = Vec::new();
    for i in 0..trimap.len() {
        if trimap.get(i) != Some(&UNKNOWN) || covered.get(i) == Some(&true) {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let x0 = x.saturating_sub(side / 2).min(w.saturating_sub(side));
        let y0 = y.saturating_sub(margin).min(h.saturating_sub(side));
        let (x1, y1) = ((x0 + side).min(w), (y0 + side).min(h));
        // A tile covers only its inner part, so the next one overlaps it; at the image's edge
        // there is nothing to overlap.
        let inner = |lo: usize, hi: usize, limit: usize| (if lo == 0 { 0 } else { lo + margin }, if hi == limit { hi } else { hi.saturating_sub(margin) });
        let ((cx0, cx1), (cy0, cy1)) = (inner(x0, x1, w), inner(y0, y1, h));
        for row in cy0..cy1 {
            if let Some(line) = covered.get_mut(row * w + cx0..row * w + cx1.max(cx0)) {
                line.fill(true);
            }
        }
        // The pixel that placed the tile is always covered, so placing never repeats.
        if let Some(slot) = covered.get_mut(i) {
            *slot = true;
        }
        tiles.push((x0, y0));
    }
    tiles
}

/// How much a tile's pixel at `(x, y)` counts where tiles overlap: rising to 1 over `overlap` px
/// from each edge of a `w × h` tile, so overlapping tiles blend without a seam.
pub fn tile_weight(x: usize, y: usize, w: usize, h: usize, overlap: usize) -> f32 {
    let ramp = |i: usize, n: usize| ((i + 1).min(n.saturating_sub(i)).min(overlap.max(1)) as f32) / overlap.max(1) as f32;
    ramp(x, w) * ramp(y, h)
}

/// The finish on a solved edge: coverage under `FINISH_CUT` becomes background, then an S-curve
/// `a^g / (k(1 − a)^g + a^g)` firms what is left, with its middle moved toward background (0.5
/// becomes 0.34, 0.97 becomes 0.98), so the faint fringe a matting model leaves on a dark or busy
/// background goes.
///
/// Private build: the cut and the curve are Photoshop's portrait edge finish, from the
/// reconstruction in `photoshop_pipelines`, so this function is not clean-room. Against
/// Photoshop's own Select Subject it improved 63 of 73 test photos, most on dark subjects against
/// dark backgrounds, and costs up to 8% on faint hair.
pub fn finish(a: f32) -> f32 {
    // `!(a >= ..)` also sends NaN to background.
    if !(a >= FINISH_CUT) {
        return BACKGROUND;
    }
    let a = a.min(1.0);
    let (high, low) = (a.powf(FINISH_POWER), (1.0 - a).powf(FINISH_POWER));
    high / (FINISH_BIAS * low + high)
}

const FINISH_CUT: f32 = 0.2;
const FINISH_POWER: f32 = 1.3136;
/// `49^(0.13 g)`, as the reconstruction derives it.
const FINISH_BIAS: f32 = 1.943_7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_finish_drops_faint_coverage_and_leans_the_rest_toward_background() {
        assert_eq!(finish(0.0), 0.0);
        assert_eq!(finish(0.19), 0.0);
        assert_eq!(finish(f32::NAN), 0.0);
        assert_eq!(finish(1.0), 1.0);
        assert_eq!(finish(3.0), 1.0);
        assert!((finish(0.5) - 0.34).abs() < 0.01, "{}", finish(0.5));
        let curve: Vec<f32> = (20..=100).map(|i| finish(i as f32 / 100.0)).collect();
        assert!(curve.windows(2).all(|w| w[0] < w[1]), "it keeps the order of coverage");
        // An S-curve: the middle drops toward background, coverage near 1 firms toward the subject.
        assert!(finish(0.7) < 0.7 && finish(0.97) > 0.97, "{} {}", finish(0.7), finish(0.97));
    }

    #[test]
    fn the_trimap_keeps_sure_pixels_away_from_the_edge() {
        let (w, h) = (40, 1);
        let mask: Vec<f32> = (0..w).map(|x| if x < 20 { 1.0 } else if x == 20 { 0.5 } else { 0.0 }).collect();
        let t = trimap(&mask, w, h, 3);
        assert_eq!(t[0], SUBJECT);
        assert_eq!(t[39], BACKGROUND);
        assert!((17..=23).all(|x| t[x] == UNKNOWN), "{:?}", &t[14..27]);
        assert!(trimap(&mask, 3, 3, 3).is_empty());
    }

    #[test]
    fn tiles_cover_every_unknown_pixel_and_skip_the_rest() {
        let (w, h) = (1000, 700);
        // A ring of unknown pixels around a sure subject.
        let t: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i64 - 500, (i / w) as i64 - 350);
                let d = ((x * x + y * y) as f64).sqrt();
                if d < 200.0 { SUBJECT } else if d < 220.0 { UNKNOWN } else { BACKGROUND }
            })
            .collect();
        let tiles = band_tiles(&t, w, h, 384, 64);
        let mut seen = vec![false; w * h];
        for (x0, y0) in &tiles {
            assert!(x0 + 384 <= w && y0 + 384 <= h, "a tile stays on the image");
            for y in *y0..y0 + 384 {
                seen[y * w + x0..y * w + x0 + 384].fill(true);
            }
        }
        assert!(t.iter().zip(&seen).all(|(v, s)| *v != UNKNOWN || *s), "every unknown pixel is in a tile");
        assert!(tiles.len() < 12, "{} tiles for a thin ring", tiles.len());
        assert!(band_tiles(&vec![SUBJECT; w * h], w, h, 384, 64).is_empty());
    }

    /// An image smaller than a tile is one tile at the corner.
    #[test]
    fn a_small_image_is_one_tile() {
        let t = vec![UNKNOWN; 50 * 30];
        assert_eq!(band_tiles(&t, 50, 30, 384, 64), [(0, 0)]);
        assert!(band_tiles(&t, 50, 31, 384, 64).is_empty(), "sizes that don't match");
    }

    #[test]
    fn weights_rise_from_each_edge() {
        assert!(tile_weight(0, 100, 384, 384, 64) < 0.05);
        assert_eq!(tile_weight(192, 192, 384, 384, 64), 1.0);
        assert!(tile_weight(383, 192, 384, 384, 64) > 0.0);
    }
}
