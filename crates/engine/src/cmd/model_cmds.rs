//! Optional model lifecycle commands and the colour/depth-agnostic selection adapter.
use std::sync::Arc;

use photocraft_algo::matting::{band, gaussian_blur, guided_filter_gray, morph};
use photocraft_algo::segment::{RgbImage, Sampler, antialias_u8, trim_region};
use photocraft_algo::selection::Region;
use photocraft_cms::{Builtin, Transform, transform::TransformOptions};
use photocraft_doc::Document;
use photocraft_geom::Rect;
use photocraft_ml::{AlphaMask, InferenceBackend, MaskKind, ModelId, Prompt, staged};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

fn error(e: photocraft_ml::Error) -> EngineError {
    match e {
        photocraft_ml::Error::Cancelled => EngineError::Cancelled,
        e => EngineError::Other(e.to_string()),
    }
}

pub(crate) fn backend(s: &Session) -> Result<Arc<dyn InferenceBackend>> {
    s.model_backend
        .clone()
        .ok_or_else(|| EngineError::Other("local models are not available in this build or session; choose Classical in Preferences › Integrations".into()))
}

/// An explicit command choice overrides the saved preference, allowing an action or script to
/// request the same method every time. Inference never silently falls back or starts a download.
pub(crate) fn subject_model(s: &Session, p: &Value) -> Result<Option<ModelId>> {
    model_choice(p, saved_subject(s), &[ModelId::BiRefNetLite, ModelId::IsNet, ModelId::BiRefNet])
}

pub(crate) fn object_model(s: &Session, p: &Value) -> Result<Option<ModelId>> {
    model_choice(p, saved(s, s.prefs().integrations.object_model.name()), &[ModelId::Sam2])
}

/// The saved choice. Default and web builds can't run a model and hide the choice, so one saved
/// by a build with local models (the preferences file is shared) means Classical there instead of
/// failing every Select Subject. Builds with local models keep it strict; explicit params too.
fn saved(s: &Session, name: &'static str) -> &'static str {
    if crate::prefs::NATIVE_MODELS || s.model_backend.is_some() { name } else { "classical" }
}

/// The saved subject model, with `auto` resolved: the most accurate installed model that suits
/// ordinary machines, else Classical, so an install with no models behaves as it always did.
fn saved_subject(s: &Session) -> &'static str {
    let choice = saved(s, s.prefs().integrations.subject_model.name());
    if choice != "auto" {
        return choice;
    }
    let status = s.model_backend.as_ref().map(|b| b.status()).unwrap_or_default();
    let installed = |id| status.iter().any(|m| m.id == id && m.installed);
    [ModelId::BiRefNetLite, ModelId::IsNet].into_iter().find(|id| installed(*id)).map_or("classical", ModelId::name)
}

/// Resolve the preference before dispatch so recorded actions keep the method used, even if
/// the preference changes before playback. Explicit choices (including invalid ones) stay intact.
pub(crate) fn inject_model(s: &Session, command: &str, mut params: Value) -> Value {
    let model = match command {
        "select.subject" | "layer.removeBackground" => saved_subject(s),
        // Point clicks commit the built-in Object Finder's hover result. SAM supports boxes,
        // so record Classical for this route even when the saved box preference is SAM.
        "select.object" if params.get("rect").is_none() && params.get("point").is_some() => "classical",
        "select.object" => saved(s, s.prefs().integrations.object_model.name()),
        _ => return params,
    };
    if params.is_null() {
        params = json!({});
    }
    if let Some(fields) = params.as_object_mut() {
        fields.entry("model").or_insert_with(|| json!(model));
    }
    params
}

/// The models a Select Subject click runs: the chosen one, and U2-Net and ViTMatte beside the two
/// that use them.
fn warm_ids(s: &Session) -> Vec<ModelId> {
    let Ok(Some(main)) = subject_model(s, &Value::Null) else { return Vec::new() };
    let status = s.model_backend.as_ref().map(|b| b.status()).unwrap_or_default();
    let mut ids = vec![main];
    if matches!(main, ModelId::BiRefNetLite | ModelId::IsNet) {
        ids.extend([ModelId::U2Net, ModelId::ViTMatte].into_iter().filter(|id| status.iter().any(|m| m.id == *id && m.installed)));
    }
    ids
}

fn model_choice(p: &Value, default: &str, allowed: &[ModelId]) -> Result<Option<ModelId>> {
    let name = match p.get("model") {
        Some(v) => v.as_str().ok_or_else(|| EngineError::Other("model must be a string".into()))?,
        None => default,
    };
    let name = if name == "auto" { default } else { name };
    if name == "classical" {
        return Ok(None);
    }
    match ModelId::parse(name) {
        Some(id) if allowed.contains(&id) => Ok(Some(id)),
        _ => {
            let names: Vec<&str> = allowed.iter().map(|m| m.name()).collect();
            Err(EngineError::Other(format!("model must be classical or {} for this command", names.join(" or "))))
        }
    }
}

struct ModelSampler<'a> {
    source: &'a dyn Sampler,
    cmyk: Option<Arc<photocraft_color::convert::CmykSpace>>,
    transform: Arc<Transform>,
}

impl Sampler for ModelSampler<'_> {
    fn rgba(&self, rect: Rect) -> Vec<[f32; 4]> {
        // rgb_scaled reads strips on rayon workers. Enter the document's CMYK context on each
        // worker, and convert its RGB/gray profile before downsampling or compositing transparency.
        let mut px = photocraft_color::convert::with_cmyk_space(self.cmyk.as_ref(), || self.source.rgba(rect));
        for p in &mut px {
            let mut rgb = [0.0; 3];
            self.transform.eval(&p[..3], &mut rgb);
            for (dst, v) in p.iter_mut().take(3).zip(rgb) {
                *dst = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
            }
            p[3] = if p[3].is_finite() { p[3].clamp(0.0, 1.0) } else { 0.0 };
        }
        px
    }
}

/// Largest side of the image the final edge snap reads, so edges come from the picture rather
/// than from the model's resolution.
const SNAP_SIDE: usize = 2048;
/// Radius and regularisation of the final edge snap, on opacities and colours in 0..=1.
const SNAP_MIN_RADIUS: usize = 5;
const SNAP_RADIUS_DIVISOR: usize = 320;
/// Approximates the 3x3 blur on the snap's grey guide.
const SNAP_GUIDE_SIGMA: f32 = 0.8;
const SNAP_EPSILON: f32 = 1e-4;
/// The model is sure of a pixel this close to 0 or 1, and the snap leaves it there.
const SNAP_SURE: f32 = 0.02;
/// How far past the model's unsure pixels the snap may reach (px at the snap's size).
const SNAP_REACH: usize = 1;
/// ViTMatte's unknown band: how far sure pixels are eroded (px at the snap's size), and how much
/// its tiles overlap so their seams blend away.
const MATTE_BAND: usize = 12;
const MATTE_OVERLAP: usize = 64;

/// Most pixels a model's input is read at, in units of its square input (`side²`).
const MAX_READ: usize = 16;

/// The reduction to read a `w × h` area at for a model taking `side × side`: as little as keeps
/// both sides at least `side`, since the model stretches its input square and a short side read
/// below it loses detail across it. An extreme aspect is capped at [`MAX_READ`] inputs' worth.
fn read_step(w: usize, h: usize, side: usize) -> usize {
    let side = side.max(1);
    let keep_detail = (w.min(h) / side).max(1);
    let cap = (w.saturating_mul(h) as f64 / (MAX_READ * side * side) as f64).sqrt().ceil() as usize;
    keep_detail.max(cap)
}

/// `mask` resampled to `width` by `height`.
fn resampled(mask: &AlphaMask, width: usize, height: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            out.push(mask.sample((x as f32 + 0.5) / width as f32, (y as f32 + 0.5) / height as f32));
        }
    }
    out
}

/// U2-Net's view of the subject on the whole area, at the size of `image`.
fn coarse_subject(backend: &dyn InferenceBackend, image: &RgbImage, from: f32, to: f32, ctx: &JobCtx) -> Result<Vec<f32>> {
    let mask = ctx.stage(from, to, "Finding subject", |ctl| backend.infer(ModelId::U2Net, image, Prompt::Subject, ctl)).map_err(error)?;
    mask.validate().map_err(error)?;
    Ok(resampled(&mask, image.w, image.h))
}

/// The coarse-to-fine pipeline: `fine` reruns on a padded crop read back from the document, and
/// the result is blended with `coarse`.
#[allow(clippy::too_many_arguments)]
fn refined(backend: &dyn InferenceBackend, fine: ModelId, sampler: &dyn Sampler, bounds: Rect, image: &RgbImage, coarse: Vec<f32>, from: f32, ctx: &JobCtx) -> Result<Vec<f32>> {
    let (w, h) = (image.w, image.h);
    let Some((x, y, box_w, box_h)) = staged::active_box(&coarse, w, h) else {
        return Ok(coarse);
    };
    let (scale_x, scale_y) = (bounds.width() as f32 / w as f32, bounds.height() as f32 / h as f32);
    let area = Rect::new(
        bounds.x0 + (x as f32 * scale_x).floor() as i32,
        bounds.y0 + (y as f32 * scale_y).floor() as i32,
        bounds.x0 + ((x + box_w) as f32 * scale_x).ceil() as i32,
        bounds.y0 + ((y + box_h) as f32 * scale_y).ceil() as i32,
    )
    .intersect(&bounds);
    let step = read_step(area.width() as usize, area.height() as usize, fine.info().input_side);
    let crop = sampler.rgb_scaled(area, step);
    ctx.check()?;
    let second = ctx.stage(from, 0.75, "Refining subject", |ctl| backend.infer(fine, &crop, Prompt::Subject, ctl)).map_err(error)?;
    second.validate().map_err(error)?;
    let inner = resampled(&second, box_w, box_h);
    let mut detail = coarse.clone();
    for (row, values) in inner.chunks_exact(box_w).enumerate() {
        let start = (y + row) * w + x;
        if let Some(dst) = detail.get_mut(start..start + box_w) {
            dst.copy_from_slice(values);
        }
    }
    staged::confidence_blend(&coarse, &detail, w, h).ok_or_else(|| EngineError::Other("staged masks differ in size".into()))
}

/// BiRefNet Lite on the whole area, which decides most images alone. When that loses a small
/// subject that U2-Net can still see, it reruns on a crop around what U2-Net found.
fn subject_matte(backend: &dyn InferenceBackend, sampler: &dyn Sampler, bounds: Rect, image: &RgbImage, coarse_too: bool, ctx: &JobCtx) -> Result<Vec<f32>> {
    let whole = ctx.stage(0.1, 0.6, "Running local model", |ctl| backend.infer(ModelId::BiRefNetLite, image, Prompt::Subject, ctl)).map_err(error)?;
    whole.validate().map_err(error)?;
    let values = resampled(&whole, image.w, image.h);
    if !coarse_too {
        return Ok(values);
    }
    let coarse = coarse_subject(backend, image, 0.6, 0.65, ctx)?;
    if !staged::lost_subject(&values, &coarse) {
        return Ok(values);
    }
    refined(backend, ModelId::BiRefNetLite, sampler, bounds, image, coarse, 0.65, ctx)
}

/// The picture around the subject, read at up to `SNAP_SIDE`, and the model's mask over the whole
/// area at that size.
struct EdgeArea {
    mask: Vec<f32>,
    width: usize,
    height: usize,
    /// Where `guide` sits in `mask`.
    x0: usize,
    y0: usize,
    guide: RgbImage,
}

impl EdgeArea {
    /// `values`, sized to `image`, at the edge steps' size with the picture around the subject;
    /// `Err` holds the mask to return as it is when there is no subject or nothing to read.
    fn read(values: Vec<f32>, image: &RgbImage, sampler: &dyn Sampler, bounds: Rect, ctx: &JobCtx) -> Result<std::result::Result<EdgeArea, AlphaMask>> {
        let subject = staged::active_box(&values, image.w, image.h);
        let mask = AlphaMask { width: image.w, height: image.h, values, kind: MaskKind::Alpha };
        let Some((x, y, box_w, box_h)) = subject else {
            return Ok(Err(mask));
        };
        let (w, h) = (bounds.width() as usize, bounds.height() as usize);
        let step = w.max(h).div_ceil(SNAP_SIDE).max(1);
        let (width, height) = (w.div_ceil(step), h.div_ceil(step));
        let (x0, x1) = (x * width / image.w, ((x + box_w) * width).div_ceil(image.w).min(width));
        let (y0, y1) = (y * height / image.h, ((y + box_h) * height).div_ceil(image.h).min(height));
        let area = Rect::new(bounds.x0 + (x0 * step) as i32, bounds.y0 + (y0 * step) as i32, bounds.x0 + (x1 * step) as i32, bounds.y0 + (y1 * step) as i32).intersect(&bounds);
        let guide = sampler.rgb_scaled(area, step);
        ctx.check()?;
        let mask = resampled(&mask, width, height);
        if (guide.w, guide.h) != (x1 - x0, y1 - y0) {
            return Ok(Err(AlphaMask { width, height, values: mask, kind: MaskKind::Alpha }));
        }
        Ok(Ok(EdgeArea { mask, width, height, x0, y0, guide }))
    }

    /// The mask under the picture.
    fn crop(&self) -> Vec<f32> {
        (self.y0..self.y0 + self.guide.h).filter_map(|row| self.mask.get(row * self.width + self.x0..row * self.width + self.x0 + self.guide.w)).flatten().copied().collect()
    }

    /// `values`, sized to the picture, written over the mask.
    fn put(&mut self, values: &[f32]) {
        for (row, part) in values.chunks_exact(self.guide.w.max(1)).enumerate() {
            let start = (self.y0 + row) * self.width + self.x0;
            if let Some(dst) = self.mask.get_mut(start..start + self.guide.w) {
                for (d, v) in dst.iter_mut().zip(part) {
                    *d = v.clamp(0.0, 1.0);
                }
            }
        }
    }

    fn finish(self) -> AlphaMask {
        AlphaMask { width: self.width, height: self.height, values: self.mask, kind: MaskKind::Alpha }
    }
}

/// The final edge snap: `values`, sized to `image`, follow the picture read at up to `SNAP_SIDE`.
/// It reads and filters only around the subject, on a grey guide, since the rest stays flat.
fn snapped(values: Vec<f32>, image: &RgbImage, sampler: &dyn Sampler, bounds: Rect, ctx: &JobCtx) -> Result<AlphaMask> {
    ctx.progress(0.75, "Refining edges");
    let mut area = match EdgeArea::read(values, image, sampler, bounds, ctx)? {
        Ok(area) => area,
        Err(mask) => return Ok(mask),
    };
    let (w, h) = (area.guide.w, area.guide.h);
    let grey: Vec<f32> = area.guide.px.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
    let grey = gaussian_blur(&grey, w, h, SNAP_GUIDE_SIGMA);
    let crop = area.crop();
    let radius = (area.width.min(area.height) / SNAP_RADIUS_DIVISOR).max(SNAP_MIN_RADIUS);
    let edges = within_unsure(&crop, guided_filter_gray(&grey, &crop, w, h, radius, SNAP_EPSILON), w, h);
    area.put(&edges);
    Ok(area.finish())
}

/// The edge solved by ViTMatte: a band around the model's edge becomes unknown, and ViTMatte
/// solves it from the picture at up to `SNAP_SIDE`, a tile at a time. Only the band changes, so
/// what the model was sure of stays.
fn matted(backend: &dyn InferenceBackend, values: Vec<f32>, image: &RgbImage, sampler: &dyn Sampler, bounds: Rect, ctx: &JobCtx) -> Result<AlphaMask> {
    ctx.progress(0.75, "Refining edges");
    let mut area = match EdgeArea::read(values, image, sampler, bounds, ctx)? {
        Ok(area) => area,
        Err(mask) => return Ok(mask),
    };
    let (w, h) = (area.guide.w, area.guide.h);
    let crop = area.crop();
    let trimap = band::trimap(&crop, w, h, MATTE_BAND);
    if trimap.len() != crop.len() {
        return Ok(area.finish());
    }
    let side = ModelId::ViTMatte.info().input_side;
    let tiles = band::band_tiles(&trimap, w, h, side, MATTE_OVERLAP);
    let (mut sum, mut weight) = (vec![0.0f32; crop.len()], vec![0.0f32; crop.len()]);
    for (n, (x0, y0)) in tiles.iter().enumerate() {
        let (tw, th) = (side.min(w.saturating_sub(*x0)), side.min(h.saturating_sub(*y0)));
        let at = |x: usize, y: usize| (y0 + y) * w + x0 + x;
        let tile = RgbImage::from_fn(tw, th, |x, y| area.guide.px.get(at(x, y)).copied().unwrap_or_default());
        let tile_trimap: Vec<f32> = (0..th).flat_map(|y| (0..tw).map(move |x| (x, y))).map(|(x, y)| trimap.get(at(x, y)).copied().unwrap_or(band::UNKNOWN)).collect();
        let from = 0.75 + 0.2 * n as f32 / tiles.len() as f32;
        let to = from + 0.2 / tiles.len() as f32;
        let alpha = ctx.stage(from, to, "Refining edges", |ctl| backend.matte_tile(ModelId::ViTMatte, &tile, &tile_trimap, ctl)).map_err(error)?;
        if (alpha.width, alpha.height) != (tw, th) {
            return Err(EngineError::Other("ViTMatte returned a tile of the wrong size".into()));
        }
        for y in 0..th {
            for x in 0..tw {
                let k = band::tile_weight(x, y, tw, th, MATTE_OVERLAP);
                if let (Some(s), Some(t), Some(a)) = (sum.get_mut(at(x, y)), weight.get_mut(at(x, y)), alpha.values.get(y * tw + x)) {
                    *s += a * k;
                    *t += k;
                }
            }
        }
    }
    let solved: Vec<f32> = crop.iter().zip(&trimap).zip(sum.iter().zip(&weight)).map(|((m, t), (s, k))| if *t == band::UNKNOWN && *k > 0.0 { band::finish(s / k) } else { *m }).collect();
    area.put(&solved);
    Ok(area.finish())
}

/// The edge step: ViTMatte when it is installed, else the guided-filter snap.
fn edged(backend: &dyn InferenceBackend, matting: bool, values: Vec<f32>, image: &RgbImage, sampler: &dyn Sampler, bounds: Rect, ctx: &JobCtx) -> Result<AlphaMask> {
    if matting { matted(backend, values, image, sampler, bounds, ctx) } else { snapped(values, image, sampler, bounds, ctx) }
}

/// `snapped` (`w × h`) where `model` was unsure or next to that, and `model` elsewhere. Unlimited,
/// the filter smears faint coverage across what the model was sure of, which reads as a halo once
/// the subject or the background is deleted.
fn within_unsure(model: &[f32], snapped: Vec<f32>, w: usize, h: usize) -> Vec<f32> {
    let unsure: Vec<f32> = model.iter().map(|v| if *v > SNAP_SURE && *v < 1.0 - SNAP_SURE { 1.0 } else { 0.0 }).collect();
    let reach = morph(&unsure, w, h, SNAP_REACH, true);
    if reach.len() != model.len() || snapped.len() != model.len() {
        return model.to_vec();
    }
    model.iter().zip(snapped).zip(reach).map(|((m, s), r)| if r > 0.0 { s.clamp(0.0, 1.0) } else { *m }).collect()
}

pub(crate) fn infer_region(
    backend: &dyn InferenceBackend,
    id: ModelId,
    source: &dyn Sampler,
    doc: &Document,
    bounds: Rect,
    prompt: Prompt,
    ctx: &JobCtx,
) -> Result<Option<Region>> {
    let (w, h) = (bounds.width() as usize, bounds.height() as usize);
    let n = w
        .checked_mul(h)
        .filter(|n| *n > 0 && *n <= 64_000_000)
        .ok_or_else(|| EngineError::Other("local models currently support canvases up to 64 megapixels".into()))?;
    ctx.check()?;
    ctx.progress(0.02, "Preparing image");
    let transform = photocraft_cms::transform::cached(&crate::color_cmds::composite_profile(doc), Builtin::Srgb.profile(), TransformOptions::default())
        .map_err(|e| EngineError::Other(format!("model colour conversion: {e}")))?;
    let sampler = ModelSampler { source, cmyk: photocraft_compose::cmyk_space(doc), transform };
    let step = read_step(w, h, id.info().input_side);
    let image = sampler.rgb_scaled(bounds, step);
    ctx.check()?;
    ctx.progress(0.1, "Running local model");
    let status = backend.status();
    let installed = |id| status.iter().any(|m| m.id == id && m.installed);
    let coarse_too = prompt == Prompt::Subject && installed(ModelId::U2Net);
    let matting = prompt == Prompt::Subject && installed(ModelId::ViTMatte);
    let mask = if id == ModelId::BiRefNetLite && prompt == Prompt::Subject {
        let values = subject_matte(backend, &sampler, bounds, &image, coarse_too, ctx)?;
        edged(backend, matting, values, &image, &sampler, bounds, ctx)?
    } else if id == ModelId::IsNet && coarse_too {
        // ISNet alone leaves solid subjects see-through.
        let coarse = coarse_subject(backend, &image, 0.1, 0.3, ctx)?;
        let values = refined(backend, ModelId::IsNet, &sampler, bounds, &image, coarse, 0.3, ctx)?;
        edged(backend, matting, values, &image, &sampler, bounds, ctx)?
    } else {
        ctx.stage(0.1, 0.85, "Running local model", |ctl| backend.infer(id, &image, prompt, ctl)).map_err(error)?
    };
    mask.validate().map_err(error)?;
    let mut values = vec![0u8; n];
    for (y, row) in values.chunks_exact_mut(w).enumerate() {
        ctx.check()?;
        for (x, dst) in row.iter_mut().enumerate() {
            *dst = mask.coverage((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
        }
        if y % 64 == 0 {
            ctx.progress(0.85 + 0.14 * y as f32 / h as f32, "Building mask");
        }
    }
    if mask.kind == MaskKind::Logits {
        antialias_u8(&mut values, w, h);
    }
    ctx.check()?;
    Ok(trim_region(Region { bbox: bounds, mask: values }))
}

impl Session {
    /// Inject a native model cache. Host code supplies the location; no arbitrary model path or
    /// URL is accepted by an engine command. Default builds and wasm have no native backend.
    #[cfg(all(feature = "local-ml", not(target_arch = "wasm32")))]
    pub fn configure_local_models(&mut self, directory: std::path::PathBuf) {
        self.model_backend = Some(Arc::new(photocraft_ml::NativeBackend::new(directory)));
    }

    /// Loads the models Select Subject and Remove Background will run, on a background thread, so
    /// the first click is as quick as the rest. Nothing happens with Classical or no models.
    pub fn warm_local_models(&self) {
        let (Some(backend), ids) = (self.model_backend.clone(), warm_ids(self)) else { return };
        if ids.is_empty() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = std::thread::Builder::new().name("model-warm-up".into()).spawn(move || {
            for id in ids {
                // A model that can't load fails again, with its message, on the first click.
                let _ = backend.warm(id);
            }
        });
        #[cfg(target_arch = "wasm32")]
        let _ = (backend, ids);
    }

    pub fn local_model_status(&self) -> Value {
        let states = self.model_backend.as_ref().map(|b| b.status()).unwrap_or_default();
        let models: Vec<Value> = photocraft_ml::MODELS
            .iter()
            .map(|m| {
                let status = states.iter().find(|s| s.id == m.id);
                json!({"id": m.id, "label": m.label, "purpose": m.purpose, "license": m.license,
                "downloadBytes": m.download_bytes(), "upstream": m.upstream, "exportSource": m.export_source,
                "revision": m.revision, "installed": status.is_some_and(|s| s.installed), "busy": status.is_some_and(|s| s.busy)})
            })
            .collect();
        json!({"available": self.model_backend.is_some(), "models": models})
    }
}

fn model_id(p: &Value, cmd: &str) -> Result<ModelId> {
    p.get("id")
        .and_then(Value::as_str)
        .and_then(ModelId::parse)
        .ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: "id must be birefnet-lite, isnet-general-use, birefnet-hr-matting or sam2.1-large".into() })
}

fn available(s: &Session) -> std::result::Result<(), String> {
    if s.model_backend.is_some() { Ok(()) } else { Err("local models are not available in this build or session".into()) }
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    let id = model_id(p, "models.download")?;
    let backend = backend(s)?;
    crate::jobs::run(
        s,
        &format!("Downloading {}", id.info().label),
        false,
        move |ctx| ctx.stage(0.0, 1.0, "Downloading model", |ctl| backend.download(id, ctl)).map_err(error),
        move |s, ()| Ok(s.local_model_status()),
    )
}

fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    let id = model_id(p, "models.remove")?;
    let backend = backend(s)?;
    crate::jobs::run(
        s,
        &format!("Removing {}", id.info().label),
        false,
        move |ctx| ctx.stage(0.0, 1.0, "Removing model", |ctl| backend.remove(id, ctl)).map_err(error),
        move |s, ()| {
            s.prefs.edit(|p| match id {
                ModelId::Sam2 => p.integrations.object_model = crate::prefs::ObjectModel::Classical,
                _ if p.integrations.subject_model.name() == id.name() => p.integrations.subject_model = crate::prefs::SubjectModel::Auto,
                _ => {}
            });
            Ok(s.local_model_status())
        },
    )
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "models.list",
            label: "Local Selection Models",
            menu: &[],
            shortcut: None,
            params: "{} → {available,models:[{id,label,installed,busy,downloadBytes,license,revision,upstream,exportSource}]}",
            enabled: |_| Ok(()),
            run: |s, _| Ok(s.local_model_status()),
            journal: false,
        },
        CommandSpec {
            id: "models.download",
            label: "Download Local Selection Model",
            menu: &[],
            shortcut: None,
            params: r#"{"id":"birefnet-lite|isnet-general-use|birefnet-hr-matting|sam2.1-large"} (explicit download; size/SHA-256 verified; no document edits)"#,
            enabled: available,
            run: download,
            journal: false,
        },
        CommandSpec {
            id: "models.remove",
            label: "Remove Local Selection Model",
            menu: &[],
            shortcut: None,
            params: r#"{"id":"birefnet-lite|isnet-general-use|birefnet-hr-matting|sam2.1-large"} (remove only this model; restore its selection preference to classical)"#,
            enabled: available,
            run: remove,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_algo::segment::RgbImage;
    use photocraft_ml::{AlphaMask, ModelStatus};
    use photocraft_raster::Interrupt;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Default)]
    struct Fake {
        installed: AtomicBool,
        downloads: AtomicUsize,
        block: AtomicBool,
        entered: AtomicBool,
        calls: Mutex<Vec<(ModelId, RgbImage, Prompt)>>,
        misses_subject: AtomicBool,
        /// ViTMatte counts as installed only with this set, and how many tiles it solved.
        matting: AtomicBool,
        tiles: AtomicUsize,
    }
    impl InferenceBackend for Fake {
        fn status(&self) -> Vec<ModelStatus> {
            let installed = |id| self.installed.load(Ordering::Relaxed) && (id != ModelId::ViTMatte || self.matting.load(Ordering::Relaxed));
            photocraft_ml::MODELS.iter().map(|m| ModelStatus { id: m.id, installed: installed(m.id), busy: false }).collect()
        }
        /// Every unknown pixel comes back half covered; known ones keep their side.
        fn matte_tile(&self, _: ModelId, image: &RgbImage, trimap: &[f32], _: &Interrupt<'_>) -> photocraft_ml::Result<AlphaMask> {
            self.tiles.fetch_add(1, Ordering::Relaxed);
            let values = trimap.iter().map(|t| if *t == band::UNKNOWN { 0.5 } else { *t }).collect();
            Ok(AlphaMask { width: image.w, height: image.h, values, kind: MaskKind::Alpha })
        }
        fn download(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
            self.downloads.fetch_add(1, Ordering::Relaxed);
            self.installed.store(true, Ordering::Relaxed);
            Ok(())
        }
        fn remove(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
            self.installed.store(false, Ordering::Relaxed);
            Ok(())
        }
        fn infer(&self, id: ModelId, image: &RgbImage, prompt: Prompt, ctl: &Interrupt<'_>) -> photocraft_ml::Result<AlphaMask> {
            self.entered.store(true, Ordering::Relaxed);
            if !self.installed.load(Ordering::Relaxed) {
                return Err(photocraft_ml::Error::Unavailable("download first".into()));
            }
            self.calls.lock().unwrap().push((id, RgbImage { w: image.w, h: image.h, px: image.px.clone() }, prompt));
            if self.block.load(Ordering::Relaxed) {
                for _ in 0..400 {
                    if ctl.cancelled() {
                        return Err(photocraft_ml::Error::Cancelled);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                return Err(photocraft_ml::Error::Inference("test backend timed out".into()));
            }
            // The first BiRefNet Lite pass sees the whole image, so a miss is one call in.
            if id == ModelId::BiRefNetLite && self.misses_subject.swap(false, Ordering::Relaxed) {
                return Ok(AlphaMask { width: 2, height: 1, values: vec![0.0, 0.0], kind: MaskKind::Alpha });
            }
            Ok(AlphaMask { width: 2, height: 1, values: vec![0.0, 1.0], kind: MaskKind::Alpha })
        }
    }
    fn session(mode: &str, depth: u32) -> (Session, Arc<Fake>) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":8,"height":4,"depth":depth,"mode":mode})).unwrap();
        let fake = Arc::new(Fake::default());
        fake.installed.store(true, Ordering::Relaxed);
        s.model_backend = Some(fake.clone());
        (s, fake)
    }
    fn cov(s: &Session, x: i32) -> f32 {
        s.active().unwrap().doc.selection.as_ref().map_or(0.0, |m| m.sample_channel(x, 2, 0))
    }
    fn pixels(s: &Session) -> Vec<f32> {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(d.doc.bounds())
    }

    #[test]
    fn auto_picks_the_installed_model_and_is_classical_without_one() {
        let (mut s, fake) = session("rgb", 8);
        assert_eq!(s.execute("select.subject", json!({})).unwrap()["model"], "birefnet-lite");
        assert_eq!(fake.calls.lock().unwrap()[0].0, ModelId::BiRefNetLite);

        let calls = fake.calls.lock().unwrap().len();
        fake.installed.store(false, Ordering::Relaxed);
        s.execute("select.subject", json!({})).unwrap();
        assert_eq!(fake.calls.lock().unwrap().len(), calls, "no model installed, so nothing reaches the backend");
    }

    #[test]
    fn isnet_runs_staged_on_a_crop_of_the_subject_when_u2net_is_installed() {
        let (mut s, fake) = session("rgb", 8);
        let result = s.execute("select.subject", json!({"model":"isnet-general-use"})).unwrap();
        assert_eq!(result["model"], "isnet-general-use");
        let calls = fake.calls.lock().unwrap();
        assert_eq!(calls.iter().map(|c| c.0).collect::<Vec<_>>(), [ModelId::U2Net, ModelId::IsNet]);
        assert_eq!((calls[0].1.w, calls[0].1.h), (8, 4));
        // The fake's coarse mask is the right half, so the crop is that half and its padding.
        assert!(calls[1].1.w < 8, "crop width {}", calls[1].1.w);
    }

    /// A phone portrait is read whole, not shrunk below the model's input across its width; a
    /// large photo is reduced, and an extreme panorama stays bounded.
    #[test]
    fn the_model_input_is_read_without_losing_detail() {
        assert_eq!(read_step(736, 1308, 640), 1);
        assert_eq!(read_step(6000, 4000, 640), 6);
        assert_eq!(read_step(640, 640, 640), 1);
        assert_eq!(read_step(10, 10, 640), 1);
        let step = read_step(200_000, 700, 640);
        assert!((200_000 / step) * (700 / step) <= MAX_READ * 640 * 640, "{step}");
        assert_eq!(read_step(0, 0, 0), 1);
    }

    /// The snap reshapes only the model's unsure edge: a blur of a sure edge stays at the edge.
    #[test]
    fn the_snap_leaves_what_the_model_was_sure_of() {
        let (w, h) = (40, 3);
        let model: Vec<f32> = (0..w * h).map(|i| if i % w < 20 { 1.0 } else if i % w == 20 { 0.5 } else { 0.0 }).collect();
        let blurred: Vec<f32> = (0..w * h).map(|i| (1.0 - (i % w) as f32 / 39.0).clamp(0.0, 1.0)).collect();
        let kept = within_unsure(&model, blurred.clone(), w, h);
        for x in 0..w {
            let expected = if (19..=21).contains(&x) { blurred[w + x] } else { model[w + x] };
            assert_eq!(kept[w + x], expected, "at {x}");
        }
        assert_eq!(within_unsure(&model, vec![0.5; 3], w, h), model, "sizes that don't match keep the model");
    }

    /// With ViTMatte installed, it solves the band around the model's edge; without, nothing asks it.
    #[test]
    fn vitmatte_solves_the_edge_band_when_installed() {
        let (mut s, fake) = session("rgb", 8);
        s.execute("select.subject", json!({"model":"birefnet-lite"})).unwrap();
        assert_eq!(fake.tiles.load(Ordering::Relaxed), 0, "not installed, not asked");
        assert!(!warm_ids(&s).contains(&ModelId::ViTMatte));

        fake.matting.store(true, Ordering::Relaxed);
        s.execute("select.subject", json!({"model":"birefnet-lite"})).unwrap();
        assert_eq!(fake.tiles.load(Ordering::Relaxed), 1, "an 8 x 4 canvas is one tile");
        // The canvas is narrower than the band, so all of it is unknown and ViTMatte decides it,
        // then the finish leans its half coverage toward background.
        assert!((cov(&s, 6) - band::finish(0.5)).abs() < 0.02, "{}", cov(&s, 6));
        assert_eq!(warm_ids(&s), [ModelId::BiRefNetLite, ModelId::U2Net, ModelId::ViTMatte]);
    }

    /// Warm-up loads what a click would run, and nothing when that is the classical method.
    #[test]
    fn warm_up_loads_the_models_a_click_runs() {
        let (mut s, fake) = session("rgb", 8);
        assert_eq!(warm_ids(&s), [ModelId::BiRefNetLite, ModelId::U2Net]);
        s.edit_prefs(|p| p.integrations.subject_model = crate::prefs::SubjectModel::BiRefNet);
        assert_eq!(warm_ids(&s), [ModelId::BiRefNet], "HR Matting runs alone");
        s.edit_prefs(|p| p.integrations.subject_model = crate::prefs::SubjectModel::Classical);
        assert!(warm_ids(&s).is_empty());
        s.edit_prefs(|p| p.integrations.subject_model = crate::prefs::SubjectModel::Auto);
        fake.installed.store(false, Ordering::Relaxed);
        assert!(warm_ids(&s).is_empty(), "auto with nothing installed is classical");
        s.warm_local_models();
    }

    /// Select and Mask's Select Subject runs the saved model, and a model that fails says so.
    #[test]
    fn the_workspace_select_subject_runs_the_saved_model() {
        let (mut s, fake) = session("rgb", 8);
        s.execute("select.all", json!({})).unwrap();
        let params = json!({"edits": [{"tool": "subject"}]});
        s.execute("select.refineEdge", params.clone()).unwrap();
        assert_eq!(fake.calls.lock().unwrap().first().map(|c| c.0), Some(ModelId::BiRefNetLite));
        assert!(cov(&s, 6) > 0.5 && cov(&s, 1) < 0.5, "the model's mask is the selection");

        let calls = fake.calls.lock().unwrap().len();
        s.execute("select.refineEdge", json!({"edits": [{"tool": "invert"}]})).unwrap();
        assert_eq!(fake.calls.lock().unwrap().len(), calls, "other edits leave the models alone");

        fake.installed.store(false, Ordering::Relaxed);
        s.execute("select.refineEdge", params.clone()).expect("auto with nothing installed is the classical method");
        s.edit_prefs(|p| p.integrations.subject_model = crate::prefs::SubjectModel::BiRefNetLite);
        assert!(s.execute("select.refineEdge", params).is_err(), "a chosen model that can't run is an error");
    }

    #[test]
    fn birefnet_lite_decides_alone_unless_it_loses_the_subject() {
        let (mut s, fake) = session("rgb", 8);
        s.execute("select.subject", json!({"model":"birefnet-lite"})).unwrap();
        let ids = |fake: &Fake| fake.calls.lock().unwrap().iter().map(|c| c.0).collect::<Vec<_>>();
        assert_eq!(ids(&fake), [ModelId::BiRefNetLite, ModelId::U2Net]);
        assert!(cov(&s, 6) > 0.5, "the subject is selected");
        assert!(cov(&s, 1) < 0.5, "the background is not");

        fake.calls.lock().unwrap().clear();
        fake.misses_subject.store(true, Ordering::Relaxed);
        s.execute("select.subject", json!({"model":"birefnet-lite"})).unwrap();
        assert_eq!(ids(&fake), [ModelId::BiRefNetLite, ModelId::U2Net, ModelId::BiRefNetLite]);
        // U2-Net's mask is the right half, so the rerun sees that half and its padding.
        assert!(fake.calls.lock().unwrap()[2].1.w < 8);
        assert!(cov(&s, 6) > 0.5, "the crop recovered the subject");
    }

    #[test]
    fn an_explicit_model_must_suit_the_command() {
        let (mut s, _) = session("rgb", 8);
        assert!(s.execute("select.subject", json!({"model":"sam2.1-large"})).is_err());
        assert!(s.execute("select.object", json!({"rect":[0,0,4,4],"model":"isnet-general-use"})).is_err());
        assert!(s.execute("select.subject", json!({"model":"not-a-model"})).is_err());
    }

    #[test]
    fn remove_background_sends_the_model_only_the_layers_own_pixels() {
        // A photo smaller than the canvas must not show the model its empty surround, which would
        // make the whole rectangle the subject (#8235).
        let (mut s, fake) = session("rgb", 8);
        s.execute("layer.new.layer", json!({})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        s.edit("paint", |doc, _| {
            let layer = doc.layer_mut(id).unwrap().surface_mut().unwrap();
            let channels = layer.channels();
            layer.write_region(Rect::new(2, 1, 6, 3), &vec![1.0; 4 * 2 * channels]);
            Ok(())
        })
        .unwrap();
        s.execute("layer.removeBackground", json!({"model":"isnet-general-use"})).unwrap();
        let calls = fake.calls.lock().unwrap();
        assert_eq!((calls[0].1.w, calls[0].1.h), (4, 2));
        assert!(calls.iter().all(|c| c.1.w <= 4 && c.1.h <= 2));
    }

    #[test]
    fn learned_background_mask_preserves_pixels_soft_alpha_and_one_undo_at_all_depths_and_modes() {
        for mode in ["rgb", "gray", "cmyk", "lab"] {
            for depth in [8, 16, 32] {
                let (mut s, fake) = session(mode, depth);
                s.execute("select.rect", json!({"x":0,"y":0,"width":2,"height":4})).unwrap();
                let old_selection = s.active().unwrap().doc.selection.clone();
                let before = pixels(&s);
                let steps = s.active().unwrap().history.past_len();
                s.execute("layer.removeBackground", json!({"model":"birefnet-hr-matting"})).unwrap();
                let d = s.active().unwrap();
                let mask = d.doc.layer(d.active_layer.unwrap()).unwrap().mask.as_ref().unwrap();
                assert_eq!(mask.value(0, 2), 0.0);
                assert_eq!(mask.value(7, 2), 1.0);
                assert!((mask.value(3, 2) - 0.376).abs() < 0.01);
                assert_eq!(pixels(&s), before, "{mode} {depth}");
                assert!(d.doc.selection.is_none());
                assert_eq!(d.history.past_len(), steps + 1);
                let calls = fake.calls.lock().unwrap();
                assert_eq!((calls[0].1.w, calls[0].1.h), (8, 4));
                assert!(calls[0].1.px.iter().flatten().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
                drop(calls);
                s.execute("edit.undo", json!({})).unwrap();
                let d = s.active().unwrap();
                assert!(d.doc.layer(d.active_layer.unwrap()).unwrap().mask.is_none());
                assert_eq!(d.doc.selection.as_ref().unwrap().read_region(d.doc.bounds()), old_selection.as_ref().unwrap().read_region(d.doc.bounds()));
                s.execute("edit.redo", json!({})).unwrap();
                assert!(s.active().unwrap().doc.layers[0].mask.is_some());
            }
        }
    }

    #[test]
    fn learned_selection_combines_modes_and_maps_boxes_on_non_square_images() {
        for (mode, left, right) in [("replace", false, true), ("add", true, true), ("subtract", true, false), ("intersect", false, false)] {
            let (mut s, fake) = session("rgb", 16);
            s.execute("select.rect", json!({"x":0,"y":0,"width":2,"height":4})).unwrap();
            let steps = s.active().unwrap().history.past_len();
            s.execute("select.object", json!({"rect":[6,3,-4,-2],"model":"sam2.1-large","mode":mode})).unwrap();
            assert_eq!(cov(&s, 0) > 0.5, left, "{mode}");
            assert_eq!(cov(&s, 7) > 0.5, right, "{mode}");
            let calls = fake.calls.lock().unwrap();
            assert_eq!(calls[0].2, Prompt::Box([0.25, 0.25, 0.75, 0.75]));
            drop(calls);
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(cov(&s, 0), 1.0);
            assert_eq!(cov(&s, 7), 0.0);
        }
    }

    #[test]
    fn missing_models_invalid_choices_and_bad_boxes_never_edit_or_download() {
        let (mut s, fake) = session("rgb", 8);
        fake.installed.store(false, Ordering::Relaxed);
        let revision = s.active().unwrap().revision;
        for (cmd, p) in [
            ("select.subject", json!({"model":"birefnet-hr-matting"})),
            ("layer.removeBackground", json!({"model":"birefnet-hr-matting"})),
            ("select.object", json!({"model":"sam2.1-large","rect":[0,0,8,4]})),
            ("select.subject", json!({"model":"sam2.1-large"})),
            ("select.subject", json!({"model":42})),
            ("models.download", json!({"id":"../unsafe"})),
            ("models.remove", json!({"id":42})),
        ] {
            assert!(s.execute(cmd, p).is_err(), "{cmd}");
        }
        for rect in [json!([0, 0, "bad", 8, 4]), json!([0, 0, 8]), json!([1e100, 0, 8, 4]), json!([0, 0, 0, 0]), json!([100, 100, 8, 4])] {
            assert!(s.execute("select.object", json!({"model":"sam2.1-large","rect":rect})).is_err());
        }
        assert_eq!(s.active().unwrap().revision, revision);
        assert_eq!(fake.downloads.load(Ordering::Relaxed), 0);
        s.model_backend = None;
        assert_eq!(s.execute("models.list", json!({})).unwrap()["available"], false);
        assert!(s.execute("select.subject", json!({"model":"birefnet-hr-matting"})).is_err());
    }

    #[test]
    fn a_model_saved_by_a_local_ml_build_means_classical_in_a_build_without_models() {
        if crate::prefs::NATIVE_MODELS {
            return;
        }
        let mut s = Session::new();
        s.execute("file.new", json!({"width":8,"height":4,"depth":8,"mode":"rgb"})).unwrap();
        s.execute("prefs.set", json!({"path":"integrations.subjectModel","value":"birefnet-hr-matting"})).unwrap();
        s.execute("prefs.set", json!({"path":"integrations.objectModel","value":"sam2.1-large"})).unwrap();
        s.execute("actions.record", json!({"name":"Saved"})).unwrap();
        for (cmd, p) in [("select.subject", json!({})), ("select.object", json!({"rect":[0,0,8,4]})), ("layer.removeBackground", json!({}))] {
            let r = s.execute(cmd, p);
            assert!(!matches!(&r, Err(EngineError::Other(m)) if m.contains("local models")), "{cmd}: {r:?}");
        }
        s.execute("actions.stop", json!({})).unwrap();
        let steps = &s.actions.list[0].steps;
        assert!(!steps.is_empty());
        assert!(steps.iter().all(|(_, p)| p["model"] == "classical"), "{steps:?}");
        // Asking for a model by name still fails: only the saved choice falls back.
        assert!(s.execute("select.subject", json!({"model":"birefnet-hr-matting"})).is_err());
    }

    #[test]
    fn downloads_need_no_document_never_replay_and_removal_resets_only_its_preference() {
        let mut s = Session::new();
        let fake = Arc::new(Fake::default());
        s.model_backend = Some(fake.clone());
        s.execute("models.download", json!({"id":"birefnet-hr-matting"})).unwrap();
        assert!(s.journal.is_empty());
        assert_eq!(fake.downloads.load(Ordering::Relaxed), 1);
        s.execute("prefs.set", json!({"values":{"integrations.subjectModel":"birefnet-hr-matting","integrations.objectModel":"sam2.1-large"}})).unwrap();
        s.execute("models.remove", json!({"id":"birefnet-hr-matting"})).unwrap();
        assert_eq!(s.prefs().integrations.subject_model, crate::prefs::SubjectModel::Auto);
        assert_eq!(s.prefs().integrations.object_model, crate::prefs::ObjectModel::Sam2);
        assert!(s.execute("prefs.set", json!({"path":"integrations.objectModel","value":"birefnet-hr-matting"})).is_err());
        let mut old = s.prefs_value();
        old["integrations"].as_object_mut().unwrap().remove("objectModel");
        old["integrations"].as_object_mut().unwrap().remove("subjectModel");
        let loaded: crate::prefs::Preferences = serde_json::from_value(old).unwrap();
        assert_eq!(loaded.integrations.subject_model, crate::prefs::SubjectModel::Auto);
        assert_eq!(loaded.integrations.object_model, crate::prefs::ObjectModel::Classical);
    }

    #[test]
    fn managing_models_does_not_drop_a_floating_selection_or_edit_the_document() {
        let (mut s, _) = session("rgb", 8);
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, layer| {
            let bounds = doc.bounds();
            doc.layer_mut(layer.unwrap()).unwrap().surface_mut().unwrap().fill_rect(bounds, &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s.execute("select.rect", json!({"x":0,"y":0,"width":2,"height":2})).unwrap();
        s.execute("select.float", json!({"dx":2,"dy":1})).unwrap();
        let doc = s.active().unwrap().doc.clone();
        let history = s.active().unwrap().history.past_len();
        let journal = s.journal.len();
        for (command, params) in
            [("models.list", json!({})), ("models.download", json!({"id":"birefnet-hr-matting"})), ("models.remove", json!({"id":"birefnet-hr-matting"}))]
        {
            s.execute(command, params).unwrap();
            let state = s.active().unwrap();
            assert!(Arc::ptr_eq(&state.doc, &doc));
            assert_eq!(state.history.past_len(), history);
            assert_eq!(state.floating.as_ref().unwrap().offset, (2, 1));
            assert_eq!(s.journal.len(), journal);
        }
    }

    #[test]
    fn preferences_select_a_model_and_explicit_classical_overrides_it() {
        let (mut s, fake) = session("rgb", 32);
        s.execute("prefs.set", json!({"path":"integrations.subjectModel","value":"birefnet-hr-matting"})).unwrap();
        s.execute("select.subject", json!({})).unwrap();
        assert_eq!(fake.calls.lock().unwrap().len(), 1);
        s.execute("select.subject", json!({"model":"classical"})).unwrap();
        assert_eq!(fake.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn learned_subject_runs_independently_of_the_classical_result_cache() {
        let (mut s, fake) = session("rgb", 16);
        s.execute("select.subject", json!({"model":"classical"})).unwrap();
        s.edit_prefs(|p| p.integrations.subject_model = crate::prefs::SubjectModel::BiRefNet);
        for expected_calls in 1..=2 {
            let result = s.execute("select.subject", json!({})).unwrap();
            assert_eq!(result["model"], "birefnet-hr-matting");
            assert_eq!(fake.calls.lock().unwrap().len(), expected_calls);
            assert_eq!(cov(&s, 0), 0.0);
            assert_eq!(cov(&s, 7), 1.0);
        }
        s.execute("select.subject", json!({"model":"classical"})).unwrap();
        assert_eq!(fake.calls.lock().unwrap().len(), 2);
    }

    #[test]
    fn recorded_actions_keep_the_selected_method_when_preferences_change() {
        let (mut s, fake) = session("rgb", 8);
        s.edit_prefs(|p| {
            p.integrations.subject_model = crate::prefs::SubjectModel::BiRefNet;
            p.integrations.object_model = crate::prefs::ObjectModel::Sam2;
        });
        s.execute("actions.record", json!({"name":"Models"})).unwrap();
        s.execute("select.subject", Value::Null).unwrap();
        s.execute("select.object", json!({"rect":[0,0,8,4]})).unwrap();
        s.execute("layer.removeBackground", json!({})).unwrap();
        s.execute("actions.stop", json!({})).unwrap();
        let action = &s.actions.list[0];
        assert_eq!(action.steps.len(), 3);
        assert_eq!(action.steps[0].1["model"], "birefnet-hr-matting");
        assert_eq!(action.steps[1].1["model"], "sam2.1-large");
        assert_eq!(action.steps[2].1["model"], "birefnet-hr-matting");
        s.edit_prefs(|p| p.integrations = Default::default());
        assert!(s.execute("actions.play", json!({"action":"Models"})).unwrap().get("failed").is_none());
        assert_eq!(fake.calls.lock().unwrap().len(), 6);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn batch_and_droplet_share_only_the_explicitly_configured_model_capability() {
        let dir = std::env::temp_dir().join(format!("photocraft-model-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("input.png");
        let output = dir.join("output");
        std::fs::create_dir(&output).unwrap();
        let (mut s, fake) = session("rgb", 8);
        let exported = photocraft_io::export(&s.active().unwrap().doc, "input.png", &Default::default()).unwrap();
        std::fs::write(&input, exported.bytes).unwrap();
        s.edit_prefs(|p| p.integrations.subject_model = crate::prefs::SubjectModel::BiRefNet);
        let droplet = dir.join("models.pcdroplet");
        std::fs::write(
            &droplet,
            serde_json::to_vec(&json!({"photocraftDroplet":1,"action":{"steps":[["layer.removeBackground",{}]]},"options":{"format":"png"}})).unwrap(),
        )
        .unwrap();
        let result = s.execute("file.automate.runDroplet", json!({"droplet":droplet,"input":[input],"output":output})).unwrap();
        assert_eq!(result["errors"], json!([]));
        assert_eq!(fake.calls.lock().unwrap().len(), 1);
        // Current upstream supports called actions in batch. Keep that action list alongside
        // the host-granted backend and method preference; losing either breaks this call.
        s.actions.list.push(crate::actions_cmds::Action::new("Models", vec![("select.subject".into(), json!({}))]));
        let called_action = json!({"steps":[["actions.play",{"action":"Models"}]],"input":[input],"output":output});
        let result = s.execute("file.automate.batch", called_action.clone()).unwrap();
        assert_eq!(result["errors"], json!([]), "{result}");
        assert_eq!(result["files"].as_array().unwrap().len(), 1);
        assert_eq!(fake.calls.lock().unwrap().len(), 2);
        s.model_backend = None;
        let result = s.execute("file.automate.batch", called_action).unwrap();
        if crate::prefs::NATIVE_MODELS {
            assert_eq!(result["files"], json!([]), "a called action cannot grant a model backend");
            assert_eq!(result["errors"].as_array().unwrap().len(), 1);
        } else {
            // A build without local models reads the saved model as Classical.
            assert_eq!(result["errors"], json!([]), "{result}");
        }
        assert_eq!(fake.calls.lock().unwrap().len(), 2);
        let result =
            s.execute("file.automate.batch", json!({"steps":[["select.subject",{"model":"birefnet-hr-matting"}]],"input":[input],"output":output})).unwrap();
        assert_eq!(result["errors"].as_array().unwrap().len(), 1);
        assert_eq!(fake.calls.lock().unwrap().len(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn model_input_uses_document_profile_and_existing_mask_replacement_undo_is_preserved() {
        let (mut s, fake) = session("rgb", 32);
        s.edit("linear pixels and old mask", |doc, _| {
            doc.icc_profile = Some(Builtin::LinearSrgb.profile().to_bytes());
            let bounds = doc.bounds();
            doc.layers[0].surface_mut().unwrap().fill_rect(bounds, &[0.25, 0.25, 0.25, 1.0]);
            doc.layers[0].mask = Some(photocraft_doc::LayerMask::hide_all());
            Ok(())
        })
        .unwrap();
        s.execute("select.subject", json!({"model":"birefnet-hr-matting","sampleAllLayers":false})).unwrap();
        let calls = fake.calls.lock().unwrap();
        assert!((calls[0].1.px[0][0] - 0.5371).abs() < 0.005, "linear RGB must be transformed to sRGB");
        drop(calls);
        s.execute("layer.removeBackground", json!({"model":"birefnet-hr-matting"})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers[0].mask.as_ref().unwrap().value(7, 2), 1.0);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers[0].mask.as_ref().unwrap().value(7, 2), 0.0);
    }

    #[test]
    fn out_of_range_and_non_finite_alpha_only_affect_the_model_input() {
        // The existing segmentation sampler composites transparent pixels over neutral gray.
        for (alpha, expected) in [(2.0, 0.25), (f32::NAN, 0.5)] {
            let (mut s, fake) = session("rgb", 32);
            s.edit("HDR alpha", |doc, _| {
                doc.icc_profile = Some(Builtin::Srgb.profile().to_bytes());
                let bounds = doc.bounds();
                doc.layers[0].surface_mut().unwrap().fill_rect(bounds, &[0.25, 0.25, 0.25, alpha]);
                Ok(())
            })
            .unwrap();
            let before: Vec<u32> = pixels(&s).iter().map(|v| v.to_bits()).collect();
            s.execute("select.subject", json!({"model":"birefnet-hr-matting","sampleAllLayers":false})).unwrap();
            assert_eq!(pixels(&s).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
            assert!(fake.calls.lock().unwrap()[0].1.px.iter().flatten().all(|v| v.is_finite() && (*v - expected).abs() < 0.005));
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn cancelling_model_inference_discards_result_and_unlocks_document() {
        let (mut s, fake) = session("rgb", 8);
        fake.block.store(true, Ordering::Relaxed);
        let before = s.active().unwrap().revision;
        let crate::jobs::Started::Job(job) = s.start("layer.removeBackground", json!({"model":"birefnet-hr-matting"})).unwrap() else { panic!("expected job") };
        for _ in 0..200 {
            if fake.entered.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(fake.entered.load(Ordering::Relaxed));
        assert!(s.cancel_job(job));
        assert!(matches!(s.wait_job(job), Err(EngineError::Cancelled)));
        assert_eq!(s.active().unwrap().revision, before);
        assert!(s.active().unwrap().doc.layers[0].mask.is_none());
        s.execute("select.all", json!({})).unwrap();
        assert!(s.active().unwrap().doc.selection.is_some());
    }
}
