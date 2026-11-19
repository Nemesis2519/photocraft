//! Shared Sampling matting: E. Gastal and M. Oliveira, "Shared Sampling for Real-Time Alpha
//! Matting" (Eurographics 2010).
//!
//! For each unknown pixel of a trimap, walk rays out to the first known foreground and background
//! pixels, keep the pair that best explains the pixel's neighbourhood, then let neighbours share
//! their best pairs and solve the coverage from the average of the three best. The settings below
//! are the ones the local reconstruction of Photoshop's matting kernels recorded; on this private
//! branch only (`docs/gaps.md`).
//!
//! It departs from that reconstruction in two ways. A pixel no ray finds a pair for keeps its
//! starting coverage, where the reconstruction borrows the image's first known pixel; that keeps
//! the result local, so a crop gives the same answer as the whole image. And every pixel gets a
//! confidence: where the foreground and background colours nearly match, or the pixel fits neither,
//! the solved coverage is close to arbitrary and shows as speckle, so callers can keep what they
//! had there. The paper's own local smoothing stage serves the same end.

use crate::segment::RgbImage;

/// Trimap labels.
pub const KNOWN_FG: u8 = 255;
pub const KNOWN_BG: u8 = 0;
pub const UNKNOWN: u8 = 128;
/// Never used as a sample, by neighbours either.
pub const PROTECTED: u8 = 64;

const DIRECTIONS: usize = 4;
const MAX_SAMPLE_DISTANCE: usize = 100;
const SAMPLE_STEP: usize = 7;
const SHARED: usize = 3;
/// Weighs a neighbour's error up the closer its coverage is to one half.
const SHARING_ALPHA_BIAS: f32 = 1.2;
/// Sharing radius as a share of width + height.
const RADIUS_FACTOR: f32 = 9.0 / 2048.0;
const FLAT_COLOUR: f32 = 1e-4;
/// Squared distance between the samples from which the solve is trusted not at all, and fully.
const SEPARATION: (f32, f32) = (0.015, 0.08);
/// Squared distance from the line between the samples over which that trust falls away.
const MISFIT: (f32, f32) = (0.002, 0.02);

type Colour = [f32; 3];

/// A solve's coverage, with how far to trust each pixel of it (0 to 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Solved {
    pub alpha: Vec<f32>,
    pub confidence: Vec<f32>,
}

/// 0 up to `low`, 1 from `high`, and smooth between.
fn ramp(value: f32, (low, high): (f32, f32)) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn separation(a: Colour, b: Colour) -> f32 {
    a.iter().zip(&b).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// Where `colour` falls on the line from `bg` to `fg`; one half when they match.
fn estimate_alpha(colour: Colour, fg: Colour, bg: Colour) -> f32 {
    let (mut length, mut along) = (0.0, 0.0);
    for ((c, f), b) in colour.iter().zip(&fg).zip(&bg) {
        let delta = b - f;
        length += delta * delta;
        along += (b - c) * delta;
    }
    if length < FLAT_COLOUR { 0.5 } else { (along / length).clamp(0.0, 1.0) }
}

fn matting_error(colour: Colour, fg: Colour, bg: Colour, alpha: f32) -> f32 {
    colour.iter().zip(&fg).zip(&bg).map(|((c, f), b)| (c - (alpha * f + (1.0 - alpha) * b)).powi(2)).sum()
}

/// The image and trimap, with lookups that clamp to the edge.
struct Scene<'a> {
    width: usize,
    height: usize,
    colours: &'a [Colour],
    trimap: &'a [u8],
}

impl Scene<'_> {
    fn clamped(&self, x: isize, y: isize) -> usize {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        y * self.width + x
    }

    fn colour(&self, index: usize) -> Colour {
        self.colours.get(index).copied().unwrap_or_default()
    }

    fn label(&self, index: usize) -> u8 {
        self.trimap.get(index).copied().unwrap_or(PROTECTED)
    }

    /// The first known foreground and background along each ray from `(x, y)`. The rays are 90
    /// degrees apart and turn 10 degrees per position in a 3×3 tile.
    fn first_hits(&self, x: usize, y: usize) -> ([Option<usize>; DIRECTIONS], [Option<usize>; DIRECTIONS]) {
        let (mut fg, mut bg) = ([None; DIRECTIONS], [None; DIRECTIONS]);
        let tile = (x % 3 + (y % 3) * 3) as f32 * 10.0;
        for k in 0..DIRECTIONS {
            let (sine, cosine) = (tile + 90.0 * k as f32).to_radians().sin_cos();
            let distances = std::iter::once(1).chain((SAMPLE_STEP..MAX_SAMPLE_DISTANCE - SAMPLE_STEP).step_by(SAMPLE_STEP));
            for distance in distances {
                let nx = (x as f32 + distance as f32 * cosine + 0.5).floor();
                let ny = (y as f32 + distance as f32 * sine + 0.5).floor();
                if nx < 0.0 || ny < 0.0 || nx >= self.width as f32 || ny >= self.height as f32 {
                    break;
                }
                let index = ny as usize * self.width + nx as usize;
                match (self.label(index), &mut fg[k], &mut bg[k]) {
                    (KNOWN_FG, slot @ None, _) | (KNOWN_BG, _, slot @ None) => *slot = Some(index),
                    _ => {}
                }
                if fg[k].is_some() && bg[k].is_some() {
                    break;
                }
            }
        }
        (fg, bg)
    }

    /// How badly the pair explains the 3×3 around `(x, y)`.
    fn neighbourhood_error(&self, x: usize, y: usize, fg: Colour, bg: Colour) -> f32 {
        let mut total = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let colour = self.colour(self.clamped(x as isize + dx, y as isize + dy));
                total += matting_error(colour, fg, bg, estimate_alpha(colour, fg, bg));
            }
        }
        total
    }

    /// The best foreground and background sample for the unknown pixel `(x, y)`, as colours.
    fn best_pair(&self, x: usize, y: usize) -> Option<(Colour, Colour)> {
        let (fg, bg) = self.first_hits(x, y);
        let distance = |index: usize| {
            let (dx, dy) = ((index % self.width) as f64 - x as f64, (index / self.width) as f64 - y as f64);
            dx * dx + dy * dy
        };
        let mut best = None;
        let mut best_energy = f64::INFINITY;
        for f in fg.into_iter().flatten() {
            for b in bg.into_iter().flatten() {
                let (fg, bg) = (self.colour(f), self.colour(b));
                let near = f64::from(self.neighbourhood_error(x, y, fg, bg));
                let energy = near * near * distance(f) * distance(b) * distance(b);
                if energy < best_energy {
                    best_energy = energy;
                    best = Some((fg, bg));
                }
            }
        }
        best
    }

    /// The coverage and confidence of the unknown pixel `(x, y)` from the three neighbours that
    /// explain it best, `None` when none has a sample to share. `pairs` holds each unknown pixel's
    /// own pair.
    fn shared_alpha(&self, x: usize, y: usize, radius: isize, pairs: &[Option<(Colour, Colour)>]) -> Option<(f32, f32)> {
        let colour = self.colour(y * self.width + x);
        let mut top = [(f32::INFINITY, None::<(u8, Colour, Colour)>); SHARED];
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let index = self.clamped(x as isize + dx, y as isize + dy);
                let label = self.label(index);
                let (fg, bg, alpha) = match label {
                    PROTECTED => continue,
                    KNOWN_FG => (self.colour(index), self.colour(index), 1.0),
                    KNOWN_BG => (self.colour(index), self.colour(index), 0.0),
                    UNKNOWN => {
                        let Some((fg, bg)) = pairs.get(index).copied().flatten() else { continue };
                        (fg, bg, estimate_alpha(colour, fg, bg))
                    }
                    _ => (self.colour(index), self.colour(index), 0.5),
                };
                let error = matting_error(colour, fg, bg, alpha) * (1.0 + SHARING_ALPHA_BIAS * (0.5 - (alpha - 0.5).abs()));
                if let Some(slot) = top.iter().position(|(best, _)| error < *best) {
                    top[slot..].rotate_right(1);
                    top[slot] = (error, Some((label, fg, bg)));
                }
            }
        }
        let (mut fg_sum, mut bg_sum) = ([0.0; 3], [0.0; 3]);
        let (mut fg_count, mut bg_count, mut any) = (0.0f32, 0.0f32, false);
        for (label, fg, bg) in top.iter().filter_map(|(_, sample)| *sample) {
            any = true;
            if label != KNOWN_BG {
                fg_count += 1.0;
                fg_sum.iter_mut().zip(&fg).for_each(|(sum, v)| *sum += v);
            }
            if label != KNOWN_FG {
                bg_count += 1.0;
                bg_sum.iter_mut().zip(&bg).for_each(|(sum, v)| *sum += v);
            }
        }
        if !any {
            return None;
        }
        // Samples of one side only say nothing about the other, so the pixel's own pair judges it.
        let own = pairs.get(y * self.width + x).copied().flatten();
        let trust = |fg: Colour, bg: Colour, alpha: f32| ramp(separation(fg, bg), SEPARATION) * (1.0 - ramp(matting_error(colour, fg, bg, alpha), MISFIT));
        if fg_count == 0.0 {
            Some((0.0, own.map_or(0.0, |(fg, bg)| trust(fg, bg, 0.0))))
        } else if bg_count == 0.0 {
            Some((1.0, own.map_or(0.0, |(fg, bg)| trust(fg, bg, 1.0))))
        } else {
            let (fg, bg) = (fg_sum.map(|v| v / fg_count), bg_sum.map(|v| v / bg_count));
            let alpha = estimate_alpha(colour, fg, bg);
            Some((alpha, trust(fg, bg, alpha)))
        }
    }
}

fn each<T: Send>(items: &[usize], f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        items.par_iter().map(|i| f(*i)).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter().map(|i| f(*i)).collect()
    }
}

/// `start` with every [`UNKNOWN`] pixel of `trimap` solved from `image`. Everything else, and an
/// unknown pixel with no sample in reach, keeps its value at no confidence. Sizes that don't match
/// give `start`.
pub fn solve_alpha(image: &RgbImage, trimap: &[u8], start: &[f32]) -> Solved {
    let (width, height) = (image.w, image.h);
    let size = width.checked_mul(height);
    if size.is_none_or(|n| n == 0 || n != image.px.len() || n != trimap.len() || n != start.len()) {
        return Solved { alpha: start.to_vec(), confidence: vec![0.0; start.len()] };
    }
    let scene = Scene { width, height, colours: &image.px, trimap };
    let unknown: Vec<usize> = trimap.iter().enumerate().filter(|(_, label)| **label == UNKNOWN).map(|(i, _)| i).collect();
    let found = each(&unknown, |i| scene.best_pair(i % width, i / width));
    let mut pairs = vec![None; trimap.len()];
    for (i, pair) in unknown.iter().zip(found) {
        if let Some(slot) = pairs.get_mut(*i) {
            *slot = pair;
        }
    }
    let radius = (((width + height) as f32 * RADIUS_FACTOR) as isize).max(1);
    let solved = each(&unknown, |i| scene.shared_alpha(i % width, i / width, radius, &pairs));
    let mut out = Solved { alpha: start.to_vec(), confidence: vec![0.0; start.len()] };
    for (i, value) in unknown.iter().zip(solved) {
        if let (Some(alpha), Some(confidence), Some((value, trust))) = (out.alpha.get_mut(*i), out.confidence.get_mut(*i), value) {
            *alpha = value;
            *confidence = trust;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FG: Colour = [0.9, 0.7, 0.2];
    const BG: Colour = [0.1, 0.3, 0.8];

    fn mix(alpha: f32) -> Colour {
        [0, 1, 2].map(|c| FG[c] * alpha + BG[c] * (1.0 - alpha))
    }

    /// A ramp from foreground to background, known on either side of it.
    #[test]
    fn it_recovers_a_soft_edge_from_the_colours() {
        let (w, h) = (80, 40);
        let truth = |x: usize| 1.0 - ((x as f32 - 30.0) / 20.0).clamp(0.0, 1.0);
        let image = RgbImage::from_fn(w, h, |x, _| mix(truth(x)));
        let trimap: Vec<u8> = (0..w * h).map(|i| if i % w < 25 { KNOWN_FG } else if i % w >= 55 { KNOWN_BG } else { UNKNOWN }).collect();
        let solved = solve_alpha(&image, &trimap, &vec![0.5; w * h]).alpha;
        for x in 25..55 {
            let got = solved[20 * w + x];
            assert!((got - truth(x)).abs() < 0.03, "at {x}: {got} for {}", truth(x));
        }
    }

    /// One-pixel strands past the edge come out whole, with the gaps between them clear.
    #[test]
    fn it_separates_thin_strands_from_the_background() {
        let (w, h) = (90, 30);
        let strand = |x: usize| x < 30 || ((36..60).contains(&x) && x.is_multiple_of(4));
        let image = RgbImage::from_fn(w, h, |x, _| if strand(x) { FG } else { BG });
        let trimap: Vec<u8> = (0..w * h).map(|i| if i % w < 30 { KNOWN_FG } else if i % w >= 64 { KNOWN_BG } else { UNKNOWN }).collect();
        let solved = solve_alpha(&image, &trimap, &vec![0.5; w * h]).alpha;
        assert!(solved[15 * w + 44] > 0.95 && solved[15 * w + 46] < 0.05, "{} {}", solved[15 * w + 44], solved[15 * w + 46]);
    }

    /// With nothing known in reach there are no samples, so the start stands.
    #[test]
    fn a_pixel_with_no_samples_keeps_its_value() {
        let (w, h) = (300, 8);
        let image = RgbImage::from_fn(w, h, |x, _| mix(x as f32 / 300.0));
        let trimap: Vec<u8> = (0..w * h).map(|i| if i % w == 0 { KNOWN_FG } else if i % w == w - 1 { KNOWN_BG } else { UNKNOWN }).collect();
        assert_eq!(solve_alpha(&image, &trimap, &vec![0.25; w * h]).alpha[4 * w + 150], 0.25);
    }

    #[test]
    fn protected_pixels_are_left_alone_and_never_shared() {
        let (w, h) = (60, 20);
        let image = RgbImage::from_fn(w, h, |x, _| mix(1.0 - x as f32 / 59.0));
        let trimap: Vec<u8> = (0..w * h)
            .map(|i| match i % w {
                x if x < 10 => KNOWN_FG,
                x if x >= 50 => KNOWN_BG,
                x if x < 30 => UNKNOWN,
                _ => PROTECTED,
            })
            .collect();
        assert_eq!(solve_alpha(&image, &trimap, &vec![0.25; w * h]).alpha[10 * w + 40], 0.25);
    }

    #[test]
    fn sizes_that_do_not_match_change_nothing() {
        let image = RgbImage::from_fn(4, 4, |_, _| FG);
        assert_eq!(solve_alpha(&image, &[UNKNOWN; 3], &[0.5; 16]).alpha, vec![0.5; 16]);
        assert_eq!(solve_alpha(&image, &[UNKNOWN; 16], &[0.5; 2]).alpha, vec![0.5; 2]);
        assert!(solve_alpha(&RgbImage::new(0, 0), &[], &[]).alpha.is_empty());
    }

    /// A trimap of nothing but unknown pixels has no samples at all.
    #[test]
    fn an_all_unknown_trimap_finishes_with_the_start() {
        let image = RgbImage::from_fn(32, 32, |x, y| mix(((x ^ y) % 7) as f32 / 7.0));
        assert_eq!(solve_alpha(&image, &[UNKNOWN; 32 * 32], &[0.3; 32 * 32]).alpha, vec![0.3; 32 * 32]);
    }

    /// Samples that nearly match can't say how much of each a pixel holds; samples far apart can.
    #[test]
    fn confidence_falls_where_the_colours_nearly_match() {
        let (w, h) = (60, 20);
        let solve = |fg: Colour, bg: Colour| {
            let image = RgbImage::from_fn(w, h, |x, _| if x < 30 { fg } else { bg });
            let trimap: Vec<u8> = (0..w * h).map(|i| if i % w < 20 { KNOWN_FG } else if i % w >= 40 { KNOWN_BG } else { UNKNOWN }).collect();
            solve_alpha(&image, &trimap, &vec![0.5; w * h]).confidence[10 * w + 30]
        };
        assert!(solve(FG, BG) > 0.95, "{}", solve(FG, BG));
        assert!(solve([0.50, 0.50, 0.50], [0.53, 0.53, 0.53]) < 0.05);
    }
}
