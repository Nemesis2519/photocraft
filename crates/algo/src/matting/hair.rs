//! Refine Hair: re-solves a selection's edge from the image, so strands and fur that the selection
//! only roughly follows get their own coverage.
//!
//! The selection stands in for the trimap network of the local reconstruction of Photoshop's
//! portrait path, whose band and blend this follows: threshold, close and dilate the selection,
//! leave out its solid core, and treat what remains as unknown ([`band`]). There is no learned hair
//! detector and no matting network here. [`shared_sampling`] solves the whole band, so any soft or
//! detailed edge is refined, hair or not, and the solver's output is the matte.
//!
//! It works on a crop around the selection, read reduced when large, and adds what changed back at
//! full size a tile at a time, so detail the selection already has survives and a huge canvas costs
//! no canvas-sized buffer.

use photocraft_geom::Rect;
use photocraft_raster::Surface;

use super::shared_sampling::{self, KNOWN_BG, KNOWN_FG, UNKNOWN};
use super::{gaussian_blur, guided_filter_gray, morph};
use crate::segment::{Sampler, SurfaceSampler};
use crate::selection::mask_from_surface;

/// Longest side of the crop the solver reads (px).
const WORK_SIDE: usize = 1280;
/// The side the band's kernels were tuned at (px); larger crops scale them up.
const BASE_SIDE: usize = 640;
/// Covered from this level on, as Select Subject's box is.
const ACTIVE: f32 = 50.0 / 255.0;
/// Background kept around the selection, as a share of its short side.
const BOX_PAD: f32 = 0.1;
/// The band reaches this far past the edge at [`BASE_SIDE`] (px), so the crop must reach further
/// for there to be known background to sample.
const MIN_PAD: usize = 24;
/// Below this mean coverage in the box there is no subject to speak of.
const MIN_MEAN: f32 = 1.785 / 255.0;
/// A smaller box has no room for known pixels around a band.
const MIN_SIDE: usize = 16;

const SEED_LEVEL: f32 = 0.12;
const CORE_LEVEL: f32 = 0.30;
const SOLID_LEVEL: f32 = 0.95;
const EDGE_LOW: f32 = 0.60;
const EDGE_HIGH: f32 = 0.70;
const UNKNOWN_LEVEL: f32 = 0.80;
/// A matte rarely reaches exactly 0 or 1, so these count as known.
const KNOWN_LEVEL: f32 = 0.05;
/// Most unknown pixels one solve takes; a wider soft edge is read more reduced.
const MAX_UNKNOWN: usize = 400_000;
/// The guided filter that cleans the result, as the paper's local smoothing does.
const CLEAN_RADIUS: usize = 2;
const CLEAN_EPSILON: f32 = 1e-3;
/// Sigma of the 5- and 7-tap Gaussians.
const SMALL_BLUR: f32 = 1.1;
const LARGE_BLUR: f32 = 1.4;

/// Where the matte applies (0 to 1) and the trimap, from the selection `mask` (`w × h`). Kernels
/// are multiplied by `scale`.
fn band(mask: &[f32], w: usize, h: usize, scale: usize) -> (Vec<f32>, Vec<u8>) {
    let above = |level: f32| -> Vec<f32> { mask.iter().map(|v| if *v > level { 1.0 } else { 0.0 }).collect() };
    let close = |src: &[f32], r: usize| morph(&morph(src, w, h, r, true), w, h, r, false);
    let seed = close(&above(SEED_LEVEL), 2 * scale);
    let seed = morph(&seed, w, h, 4 * scale, true);
    let core = morph(&morph(&above(CORE_LEVEL), w, h, 2 * scale, false), w, h, 2 * scale, true);
    let smooth = gaussian_blur(&core, w, h, SMALL_BLUR * scale as f32);
    let mut weight = close(&seed, 6 * scale);
    for ((value, solid), m) in weight.iter_mut().zip(&smooth).zip(mask) {
        if *solid > SOLID_LEVEL {
            *value = 0.0;
        }
        // A selection, unlike a network's output, can be soft well inside its core.
        if (*m > EDGE_LOW && *solid < EDGE_HIGH) || (*m > KNOWN_LEVEL && *m < 1.0 - KNOWN_LEVEL) {
            *value = 1.0;
        }
    }
    let blur = LARGE_BLUR * scale as f32;
    let weight = gaussian_blur(&morph(&gaussian_blur(&weight, w, h, blur), w, h, 2 * scale, true), w, h, blur);
    let trimap = mask
        .iter()
        .zip(&weight)
        .map(|(m, e)| {
            if *e > UNKNOWN_LEVEL {
                UNKNOWN
            } else if *m >= 1.0 - KNOWN_LEVEL {
                KNOWN_FG
            } else if *m <= KNOWN_LEVEL {
                KNOWN_BG
            } else {
                // Neither: too soft to sample, and outside the band.
                (m * 255.0) as u8
            }
        })
        .collect();
    (weight, trimap)
}

/// Coverage over `area` reduced by `step`, without holding `area` at full size.
fn coverage(selection: &Surface, area: Rect, step: usize) -> (Vec<f32>, usize, usize) {
    let image = SurfaceSampler(selection).rgb_scaled(area, step);
    let values = image.px.iter().map(|p| p[0]).collect();
    (values, image.w, image.h)
}

/// The box around what `values` (`w × h`) covers, with its mean coverage.
fn covered_box(values: &[f32], w: usize, h: usize) -> Option<(usize, usize, usize, usize, f32)> {
    let (mut left, mut top, mut right, mut bottom) = (w, h, 0, 0);
    for (y, row) in values.chunks_exact(w.max(1)).enumerate() {
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
    let total: f32 = (top..bottom).filter_map(|y| values.get(y * w + left..y * w + right)).flatten().sum();
    Some((left, top, right, bottom, total / ((right - left) * (bottom - top)) as f32))
}

/// `delta` (`w × h`, each cell `step` px wide, starting at `area`'s corner) over `r`, bilinear.
fn enlarged(delta: &[f32], w: usize, h: usize, step: usize, area: Rect, r: Rect) -> Vec<f32> {
    let at = |x: usize, y: usize| delta.get(y.min(h - 1) * w + x.min(w - 1)).copied().unwrap_or(0.0);
    // Where a pixel falls on the grid: the two cells either side and how far between them.
    let place = |offset: i32| {
        let v = ((offset as f32 + 0.5) / step as f32 - 0.5).max(0.0);
        (v as usize, v.fract())
    };
    let mut out = Vec::with_capacity(r.width() as usize * r.height() as usize);
    for y in r.y0..r.y1 {
        let (gy, ty) = place(y - area.y0);
        for x in r.x0..r.x1 {
            let (gx, tx) = place(x - area.x0);
            let upper = at(gx, gy) + (at(gx + 1, gy) - at(gx, gy)) * tx;
            let lower = at(gx, gy + 1) + (at(gx + 1, gy + 1) - at(gx, gy + 1)) * tx;
            out.push(upper + (lower - upper) * ty);
        }
    }
    out
}

/// `selection` with its edge re-solved from `sampler`'s image; `None` when there is no edge to
/// work on, which leaves the selection as it was.
pub fn refine_hair(sampler: &dyn Sampler, selection: &Surface, canvas: Rect) -> Option<Surface> {
    let filled = selection.default_pixel().first().is_some_and(|v| *v > 0.0);
    let content = if filled { canvas } else { selection.content_bounds().intersect(&canvas) };
    if content.is_empty() {
        return None;
    }
    // A first look, reduced, finds what is really selected.
    let longest = content.width().max(content.height()) as usize;
    let look = longest.div_ceil(WORK_SIDE).max(1);
    let (values, look_w, look_h) = coverage(selection, content, look);
    let (left, top, right, bottom, mean) = covered_box(&values, look_w, look_h)?;
    if mean < MIN_MEAN {
        return None;
    }
    let reach = MIN_PAD * look_w.max(look_h).div_ceil(BASE_SIDE).max(1);
    let pad = (((right - left).min(bottom - top) as f32 * BOX_PAD).ceil() as usize).max(reach);
    let side = |grid: usize, origin: i32| origin.saturating_add(i32::try_from(grid.saturating_mul(look)).unwrap_or(i32::MAX));
    let area = Rect::new(
        side(left.saturating_sub(pad), content.x0),
        side(top.saturating_sub(pad), content.y0),
        side(right + pad, content.x0),
        side(bottom + pad, content.y0),
    )
    .intersect(&content.inflate(i32::try_from(pad.saturating_mul(look)).unwrap_or(0)))
    .intersect(&canvas);
    let longest = area.width().max(area.height()) as usize;
    if (area.width().min(area.height()) as usize) < MIN_SIDE {
        return None;
    }
    let mut step = longest.div_ceil(WORK_SIDE).max(1);
    let (mask, w, h, weight, trimap) = loop {
        let (mask, w, h) = coverage(selection, area, step);
        let scale = (w.max(h) as f32 / BASE_SIDE as f32).round().max(1.0) as usize;
        let (weight, trimap) = band(&mask, w, h, scale);
        let unknown = trimap.iter().filter(|label| **label == UNKNOWN).count();
        if unknown == 0 {
            return None;
        }
        if unknown <= MAX_UNKNOWN || w.min(h) < MIN_SIDE * 2 {
            break (mask, w, h, weight, trimap);
        }
        step = step.saturating_mul(2);
    };
    let image = sampler.rgb_scaled(area, step);
    if (image.w, image.h) != (w, h) {
        return None;
    }
    let solved = shared_sampling::solve_alpha(&image, &trimap, &mask);
    let trusted: Vec<f32> = (0..mask.len())
        .map(|i| {
            let [m, s, c] = [&mask, &solved.alpha, &solved.confidence].map(|v| v.get(i).copied().unwrap_or(0.0));
            m + (s - m) * c
        })
        .collect();
    let grey: Vec<f32> = image.px.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
    let clean = guided_filter_gray(&grey, &trusted, w, h, CLEAN_RADIUS, CLEAN_EPSILON);
    let delta: Vec<f32> = (0..mask.len())
        .map(|i| {
            let (m, c, e) = (mask.get(i).copied().unwrap_or(0.0), clean.get(i).copied().unwrap_or(0.0), weight.get(i).copied().unwrap_or(0.0));
            if trimap.get(i) == Some(&UNKNOWN) { (c.clamp(0.0, 1.0) - m) * e } else { 0.0 }
        })
        .collect();
    let mut out = selection.clone();
    for tile in area.tiles() {
        let r = tile.rect().intersect(&area);
        let change = enlarged(&delta, w, h, step, area, r);
        if change.iter().all(|v| *v == 0.0) {
            continue;
        }
        let mut values = mask_from_surface(Some(selection), r);
        for (value, change) in values.iter_mut().zip(&change) {
            *value = (*value + change).clamp(0.0, 1.0);
        }
        out.write_region(r, &values);
    }
    out.prune();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{ImageSampler, RgbImage};
    use crate::selection::mask_to_surface;

    const FG: [f32; 3] = [0.9, 0.7, 0.2];
    const BG: [f32; 3] = [0.1, 0.3, 0.8];

    fn mask_of(canvas: Rect, f: impl Fn(i32, i32) -> f32) -> Surface {
        let mut values = Vec::new();
        for y in canvas.y0..canvas.y1 {
            for x in canvas.x0..canvas.x1 {
                values.push(f(x, y));
            }
        }
        mask_to_surface(&values, canvas)
    }

    /// Strands reaching past a selection that stops short of them: the band takes in the nearest
    /// ones and clears the background between them.
    #[test]
    fn strands_next_to_the_edge_come_into_the_selection() {
        let (w, h) = (200usize, 120usize);
        let strand = |x: usize, y: usize| (x < 100 || ((100..108).contains(&x) && x.is_multiple_of(3))) && (20..100).contains(&y);
        let image = RgbImage::from_fn(w, h, |x, y| if strand(x, y) { FG } else { BG });
        let sampler = ImageSampler { img: &image, origin: (0, 0) };
        let canvas = Rect::new(0, 0, w as i32, h as i32);
        let selection = mask_of(canvas, |x, y| if x < 100 && (20..100).contains(&y) { 1.0 } else { 0.0 });
        let refined = refine_hair(&sampler, &selection, canvas).expect("an edge to refine");
        let at = |x: i32, y: i32| refined.sample_channel(x, y, 0);
        // Not all the way: the cleanup that stops speckle takes something off a 1 px strand.
        assert!(at(102, 60) > 0.6, "a strand comes in: {}", at(102, 60));
        assert!(at(103, 60) < 0.1 && at(104, 60) < 0.1, "the gaps stay out: {} {}", at(103, 60), at(104, 60));
        assert!(at(50, 60) > 0.99 && at(170, 60) < 0.01, "away from the edge nothing moves");
    }

    /// A soft edge on an image with no soft edge snaps to the image.
    #[test]
    fn a_vague_edge_tightens_onto_the_image() {
        let (w, h) = (160usize, 160usize);
        let image = RgbImage::from_fn(w, h, |x, y| if (40..120).contains(&x) && (40..120).contains(&y) { FG } else { BG });
        let sampler = ImageSampler { img: &image, origin: (0, 0) };
        let canvas = Rect::new(0, 0, w as i32, h as i32);
        let ramp = |d: i32| ((d as f32 + 3.0) / 6.0).clamp(0.0, 1.0);
        let selection = mask_of(canvas, |x, y| ramp((x - 40).min(119 - x).min(y - 40).min(119 - y)));
        let refined = refine_hair(&sampler, &selection, canvas).expect("an edge to refine");
        let at = |x: i32, y: i32| refined.sample_channel(x, y, 0);
        assert!(at(41, 80) > 0.9 && at(38, 80) < 0.1, "{} {}", at(41, 80), at(38, 80));
    }

    /// A selection soft all over stays bounded: it is read more reduced, not solved whole.
    #[test]
    fn a_selection_soft_all_over_finishes() {
        let side = 1400usize;
        let image = RgbImage::from_fn(side, side, |x, y| if (x / 9 + y / 9) % 2 == 0 { FG } else { BG });
        let sampler = ImageSampler { img: &image, origin: (0, 0) };
        let canvas = Rect::new(0, 0, side as i32, side as i32);
        let mut selection = Surface::new(photocraft_color::PixelFormat::GRAY8);
        selection.fill_rect(canvas, &[0.5]);
        let t = std::time::Instant::now();
        let refined = refine_hair(&sampler, &selection, canvas);
        assert!(refined.is_none_or(|s| s.sample_channel(700, 700, 0).is_finite()));
        assert!(t.elapsed() < std::time::Duration::from_secs(30), "{:?}", t.elapsed());
    }

    /// A large canvas is read reduced, and only the band's tiles change.
    #[test]
    fn a_large_selection_keeps_its_detail_away_from_the_band() {
        let (w, h) = (3000usize, 2000usize);
        let edge = 1500;
        let image = RgbImage::from_fn(w, h, |x, _| if x < edge { FG } else { BG });
        let sampler = ImageSampler { img: &image, origin: (0, 0) };
        let canvas = Rect::new(0, 0, w as i32, h as i32);
        let mut selection = Surface::new(photocraft_color::PixelFormat::GRAY8);
        selection.fill_rect(Rect::new(100, 100, edge as i32 + 6, 1900), &[1.0]);
        selection.fill_rect(Rect::new(300, 300, 301, 1700), &[0.25]);
        let refined = refine_hair(&sampler, &selection, canvas).expect("an edge to refine");
        assert!((refined.sample_channel(300, 900, 0) - 0.25).abs() < 0.01, "detail off the band stays");
        assert!(refined.sample_channel(edge as i32 + 4, 900, 0) < 0.5, "{}", refined.sample_channel(edge as i32 + 4, 900, 0));
        assert!(refined.sample_channel(edge as i32 - 8, 900, 0) > 0.9);
    }

    /// Nothing to work on: no change, and no panic.
    #[test]
    fn empty_tiny_and_hostile_selections_are_left_alone() {
        let image = RgbImage::from_fn(64, 64, |_, _| FG);
        let sampler = ImageSampler { img: &image, origin: (0, 0) };
        let canvas = Rect::new(0, 0, 64, 64);
        let empty = Surface::new(photocraft_color::PixelFormat::GRAY8);
        assert!(refine_hair(&sampler, &empty, canvas).is_none());
        assert!(refine_hair(&sampler, &empty, Rect::EMPTY).is_none());
        let mut faint = Surface::new(photocraft_color::PixelFormat::GRAY8);
        faint.fill_rect(canvas, &[0.1]);
        assert!(refine_hair(&sampler, &faint, canvas).is_none(), "below the level that counts as selected");
        let mut sliver = Surface::new(photocraft_color::PixelFormat::GRAY8);
        sliver.fill_rect(Rect::new(30, 30, 31, 31), &[1.0]);
        let mut small = Surface::new(photocraft_color::PixelFormat::GRAY8);
        small.fill_rect(Rect::new(0, 0, 4, 4), &[1.0]);
        assert!(refine_hair(&sampler, &small, Rect::new(0, 0, 4, 4)).is_none(), "too small for a band");
        let mut outside = Surface::new(photocraft_color::PixelFormat::GRAY8);
        outside.fill_rect(Rect::new(500, 500, 600, 600), &[1.0]);
        assert!(refine_hair(&sampler, &outside, canvas).is_none(), "off the canvas");
        let _ = refine_hair(&sampler, &sliver, canvas);
        // Everything selected, the default a revealing layer mask has: no background to sample.
        let all = Surface::with_default(photocraft_color::PixelFormat::GRAY8, &[1.0]);
        let kept = refine_hair(&sampler, &all, canvas);
        assert!(kept.is_none_or(|s| s.sample_channel(32, 32, 0) > 0.99));
    }
}
