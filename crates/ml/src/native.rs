//! Native backend: the GPU routes built in, then the CPU. Model installation is explicit;
//! inference never downloads anything.
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::{RunOptions, Session as OnnxSession};
use ort::value::Tensor;
use photocraft_algo::segment::RgbImage;
use photocraft_raster::Interrupt;
use sha2::{Digest, Sha256};
use ureq::unversioned::transport::{Buffers, ConnectionDetails, Connector, DefaultConnector, NextTimeout, Transport};

use crate::catalog::Artifact;
use crate::{AlphaMask, Error, InferenceBackend, MaskKind, ModelId, ModelStatus, Prompt, Result, check, normalize};

impl<T> From<ort::Error<T>> for Error {
    fn from(e: ort::Error<T>) -> Self {
        Self::Inference(e.to_string())
    }
}

/// Platform frontends choose the directory. Constructing a backend does not create it, contact
/// a server, load ONNX Runtime or load any model. One model stays resident at a time, except
/// the light models the staged pipeline runs together on every selection.
pub struct NativeBackend {
    root: PathBuf,
    operations: [Mutex<()>; 6],
    sessions: Mutex<Vec<Loaded>>,
    /// Models that failed on the GPU this run, so an out-of-memory model does not fail again on
    /// every click before reaching the CPU.
    cpu_only: [AtomicBool; 6],
    last_device: Mutex<Option<&'static str>>,
    /// Held while [`InferenceBackend::warm`] runs; a real use waits on it.
    warming: Mutex<()>,
}

/// How often a use waiting on a warm-up checks whether it was cancelled.
const WARM_WAIT: Duration = Duration::from_millis(20);

enum Sessions {
    Matte(OnnxSession),
    Sam2 { encoder: OnnxSession, decoder: OnnxSession },
}

struct Loaded {
    id: ModelId,
    sessions: Sessions,
    device: &'static str,
}

/// ureq's body timeout covers the whole download, rather than each read. Keep large downloads
/// possible while bounding stalled reads (and cancellation cleanup) to 30 seconds. Delegate TLS,
/// redirects and proxy handling to its default connector; never weaken certificate validation.
#[derive(Debug)]
struct IdleConnector;
#[derive(Debug)]
struct IdleTransport(Box<dyn Transport>);
impl Connector for IdleConnector {
    type Out = IdleTransport;
    fn connect(&self, details: &ConnectionDetails<'_>, chained: Option<()>) -> std::result::Result<Option<Self::Out>, ureq::Error> {
        Ok(DefaultConnector::default().connect(details, chained)?.map(IdleTransport))
    }
}
fn idle_timeout(timeout: NextTimeout) -> NextTimeout {
    let after = ureq::unversioned::transport::time::Duration::from_secs(30);
    if *timeout.after > *after { NextTimeout { after, reason: ureq::Timeout::RecvBody } } else { timeout }
}
impl Transport for IdleTransport {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.0.buffers()
    }
    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> std::result::Result<(), ureq::Error> {
        self.0.transmit_output(amount, timeout)
    }
    fn await_input(&mut self, timeout: NextTimeout) -> std::result::Result<bool, ureq::Error> {
        self.0.await_input(idle_timeout(timeout))
    }
    fn is_open(&mut self) -> bool {
        self.0.is_open()
    }
    fn is_tls(&self) -> bool {
        self.0.is_tls()
    }
}

impl NativeBackend {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            operations: Default::default(),
            sessions: Mutex::new(Vec::new()),
            cpu_only: Default::default(),
            last_device: Mutex::new(None),
            warming: Mutex::new(()),
        }
    }

    fn load(&self, id: ModelId, cpu_only: bool) -> Result<Loaded> {
        let dir = self.directory(id);
        Ok(match id {
            ModelId::BiRefNet | ModelId::IsNet | ModelId::BiRefNetLite | ModelId::U2Net | ModelId::ViTMatte => {
                let (model, device) = session(&dir.join("model.onnx"), cpu_only, pinned_dims(id))?;
                Loaded { id, sessions: Sessions::Matte(model), device }
            }
            ModelId::Sam2 => {
                let (encoder, device) = session(&dir.join("vision_encoder.onnx"), cpu_only, &[])?;
                let (decoder, _) = session(&dir.join("prompt_encoder_mask_decoder.onnx"), cpu_only, &[])?;
                Loaded { id, sessions: Sessions::Sam2 { encoder, decoder }, device }
            }
        })
    }

    fn directory(&self, id: ModelId) -> PathBuf {
        self.root.join(format!("{}-{}", id.name(), id.info().revision))
    }

    fn operation(&self, id: ModelId) -> Result<MutexGuard<'_, ()>> {
        let lock = self.operations.get(id.index()).ok_or_else(|| Error::Input("unknown model".into()))?;
        match lock.try_lock() {
            Ok(guard) => Ok(guard),
            Err(std::sync::TryLockError::Poisoned(e)) => {
                lock.clear_poison();
                Ok(e.into_inner())
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                Err(Error::Unavailable(format!("{} is in use or downloading; try again when it finishes", id.info().label)))
            }
        }
    }

    fn installed(&self, id: ModelId) -> bool {
        let dir = self.directory(id);
        fs::symlink_metadata(&dir).is_ok_and(|m| m.is_dir())
            && id.info().artifacts.iter().all(|a| fs::symlink_metadata(dir.join(a.file)).is_ok_and(|m| m.is_file() && m.len() == a.bytes))
    }

    fn verify(&self, id: ModelId, ctl: &Interrupt<'_>) -> Result<()> {
        if !self.installed(id) {
            return Err(Error::Unavailable(format!("download {} in Preferences › Integrations first", id.info().label)));
        }
        for a in id.info().artifacts {
            check(ctl)?;
            // Hash before parsing, including external weights. No ONNX from a mutable URL or an
            // unverified local file is executed, and external data paths are pinned with it.
            copy_verified(&mut File::open(self.directory(id).join(a.file))?, &mut std::io::sink(), a, ctl, |_| {})?;
        }
        Ok(())
    }
}

impl InferenceBackend for NativeBackend {
    fn status(&self) -> Vec<ModelStatus> {
        crate::MODELS
            .iter()
            .map(|m| ModelStatus {
                id: m.id,
                installed: self.installed(m.id),
                busy: self.operations.get(m.id.index()).is_some_and(|l| matches!(l.try_lock(), Err(std::sync::TryLockError::WouldBlock))),
            })
            .collect()
    }

    fn download(&self, id: ModelId, ctl: &Interrupt<'_>) -> Result<()> {
        let _guard = self.operation(id)?;
        check(ctl)?;
        if self.installed(id) {
            self.verify(id, ctl)?;
            ctl.progress(1.0);
            return Ok(());
        }
        let dest = self.directory(id);
        if dest.try_exists()? {
            return Err(Error::Download("incomplete or corrupt installation; remove this model in Preferences and download it again".into()));
        }
        fs::create_dir_all(&self.root)?;
        // Every error, timeout or cancellation drops the staging directory. Windows rename is
        // atomic too because the destination does not exist; installed files are never overwritten.
        let staging = tempfile::Builder::new().prefix("photocraft-model-").tempdir_in(&self.root)?;
        let config = ureq::Agent::config_builder()
            .https_only(true)
            .max_redirects(5)
            .timeout_resolve(Some(Duration::from_secs(15)))
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_response(Some(Duration::from_secs(30)))
            .timeout_recv_body(Some(Duration::from_secs(3600)))
            .build();
        let agent = ureq::Agent::with_parts(config, IdleConnector, ureq::unversioned::resolver::DefaultResolver::default());
        let total = id.info().download_bytes() as f32;
        let mut completed = 0u64;
        for a in id.info().artifacts {
            check(ctl)?;
            let mut response = agent
                .get(a.url)
                .header("User-Agent", "PhotoCraft-model-download")
                .call()
                .map_err(|e| Error::Download(format!("{}: {e}; try again", a.file)))?;
            let mut file = File::create(staging.path().join(a.file))?;
            copy_verified(&mut response.body_mut().as_reader(), &mut file, a, ctl, |bytes| ctl.progress((completed + bytes) as f32 / total * 0.98)).map_err(
                |e| match e {
                    Error::Cancelled => e,
                    e => Error::Download(format!("{}: {e}; partial download discarded; try again", a.file)),
                },
            )?;
            file.flush()?;
            file.sync_all()?;
            completed += a.bytes;
        }
        let license = match id {
            ModelId::BiRefNet | ModelId::BiRefNetLite => include_str!("../licenses/BiRefNet-MIT.txt"),
            ModelId::Sam2 => include_str!("../licenses/SAM2-Apache-2.0.txt"),
            ModelId::IsNet => include_str!("../licenses/ISNet-Apache-2.0.txt"),
            ModelId::U2Net => include_str!("../licenses/U2Net-Apache-2.0.txt"),
            ModelId::ViTMatte => include_str!("../licenses/ViTMatte-Apache-2.0.txt"),
        };
        fs::write(staging.path().join("LICENSE.txt"), license)?;
        fs::write(
            staging.path().join("SOURCE.txt"),
            format!("{}\nUpstream: {}\nONNX export: {}\nRevision: {}\n", id.info().label, id.info().upstream, id.info().export_source, id.info().revision),
        )?;
        check(ctl)?;
        fs::rename(staging.path(), dest)?;
        ctl.progress(1.0);
        Ok(())
    }

    fn remove(&self, id: ModelId, ctl: &Interrupt<'_>) -> Result<()> {
        let _guard = self.operation(id)?;
        check(ctl)?;
        let mut loaded = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        check(ctl)?;
        // Close external weight files before removing on Windows.
        loaded.retain(|l| l.id != id);
        let path = self.directory(id);
        if path.try_exists()? {
            if !fs::symlink_metadata(&path)?.is_dir() {
                return Err(Error::Unavailable("model installation is not a regular directory".into()));
            }
            fs::remove_dir_all(path)?;
        }
        ctl.progress(1.0);
        Ok(())
    }

    fn infer(&self, id: ModelId, image: &RgbImage, prompt: Prompt, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
        self.after_warm_up(ctl)?;
        self.run_now(id, Input::Picture(image), prompt, ctl)
    }

    fn matte_tile(&self, id: ModelId, image: &RgbImage, trimap: &[f32], ctl: &Interrupt<'_>) -> Result<AlphaMask> {
        self.after_warm_up(ctl)?;
        let full = self.run_now(id, Input::Tile(image, trimap), Prompt::Trimap, ctl)?;
        let values = full.values.chunks_exact(full.width).take(image.h).flat_map(|row| row.iter().take(image.w)).copied().collect();
        Ok(AlphaMask { width: image.w, height: image.h, values, kind: MaskKind::Alpha })
    }

    fn warm(&self, id: ModelId) -> Result<()> {
        let _warming = self.warming.lock().unwrap_or_else(PoisonError::into_inner);
        match id {
            ModelId::Sam2 => Ok(()),
            ModelId::ViTMatte => {
                let side = id.info().input_side;
                self.run_now(id, Input::Tile(&RgbImage::new(side, side), &vec![0.5; side * side]), Prompt::Trimap, &Interrupt::NONE).map(|_| ())
            }
            _ => self.run_now(id, Input::Picture(&RgbImage::new(64, 64)), Prompt::Subject, &Interrupt::NONE).map(|_| ()),
        }
    }

    fn last_device(&self) -> Option<&'static str> {
        *self.last_device.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What a run feeds a model: a picture it resizes to its input, or a matting tile with its trimap.
enum Input<'a> {
    Picture(&'a RgbImage),
    Tile(&'a RgbImage, &'a [f32]),
}

impl NativeBackend {
    /// A warm-up would make a use fail as busy, so it waits, and can still be cancelled.
    fn after_warm_up(&self, ctl: &Interrupt<'_>) -> Result<()> {
        loop {
            match self.warming.try_lock() {
                Ok(_) | Err(std::sync::TryLockError::Poisoned(_)) => return Ok(()),
                Err(std::sync::TryLockError::WouldBlock) => {
                    check(ctl)?;
                    std::thread::sleep(WARM_WAIT);
                }
            }
        }
    }

    fn run_now(&self, id: ModelId, input: Input<'_>, prompt: Prompt, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
        let _guard = self.operation(id)?;
        validate_prompt(id, prompt)?;
        let input = match input {
            Input::Picture(image) => normalize(image, id.info().input_side, id.info().scaling, ctl)?,
            Input::Tile(image, trimap) => tile_input(image, trimap, id.info().input_side)?,
        };
        let mut loaded = match self.sessions.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(e)) => {
                let mut guard = e.into_inner();
                guard.clear();
                self.sessions.clear_poison();
                guard
            }
            Err(std::sync::TryLockError::WouldBlock) => return Err(Error::Unavailable("another local model is running; try again when it finishes".into())),
        };
        let flag = self.cpu_only.get(id.index()).ok_or_else(|| Error::Input("unknown model".into()))?;
        let index = match loaded.iter().position(|l| l.id == id) {
            Some(index) => index,
            None => {
                loaded.retain(|l| light(l.id) && light(id));
                // Hash once per load: re-reading hundreds of MB would cost every selection a delay.
                self.verify(id, ctl)?;
                check(ctl)?;
                loaded.push(self.load(id, flag.load(Ordering::Relaxed))?);
                loaded.len() - 1
            }
        };
        check(ctl)?;
        ctl.progress(0.2);
        let current = loaded.get_mut(index).ok_or_else(|| Error::Inference("model did not load".into()))?;
        let mut device = current.device;
        let result = if device != CPU {
            match run(id, current, input.clone(), prompt, ctl) {
                Err(Error::Inference(_)) => {
                    // Usually the GPU running out of memory, which only shows while running.
                    flag.store(true, Ordering::Relaxed);
                    loaded.remove(index);
                    check(ctl)?;
                    let mut cpu = self.load(id, true)?;
                    device = cpu.device;
                    let result = run(id, &mut cpu, input, prompt, ctl);
                    loaded.push(cpu);
                    result
                }
                other => other,
            }
        } else {
            run(id, current, input, prompt, ctl)
        }?;
        result.validate()?;
        // U2-Net only assists the others, so the device reported is the main model's.
        if id != ModelId::U2Net {
            *self.last_device.lock().unwrap_or_else(PoisonError::into_inner) = Some(device);
        }
        check(ctl)?;
        ctl.progress(1.0);
        Ok(result)
    }
}

/// Whether a model is light enough to stay resident beside the other light ones.
fn light(id: ModelId) -> bool {
    matches!(id, ModelId::IsNet | ModelId::U2Net | ModelId::BiRefNetLite | ModelId::ViTMatte)
}

/// The free dimensions to fix a size-free export at. Fixing them folds away the shape arithmetic,
/// taking BiRefNet Lite from 18,300 graph nodes to 1,363 and its load from about 11 s to 2.4 s.
fn pinned_dims(id: ModelId) -> &'static [(&'static str, i64)] {
    match id {
        ModelId::BiRefNetLite => &[("h", 640), ("w", 640)],
        ModelId::ViTMatte => &[("batch_size", 1), ("num_channels", 4), ("height", 384), ("width", 384)],
        _ => &[],
    }
}

/// A matting tile as ViTMatte takes it: the picture scaled to -1..=1 and the trimap, as four
/// channels of `side × side`, the picture at the top left and zeros past it.
fn tile_input(image: &RgbImage, trimap: &[f32], side: usize) -> Result<Vec<f32>> {
    let (w, h) = (image.w, image.h);
    if w == 0 || h == 0 || w > side || h > side || image.px.len() != w * h || trimap.len() != w * h {
        return Err(Error::Input(format!("a matting tile must be at most {side} px each way, with one trimap value per pixel")));
    }
    if image.px.iter().flatten().chain(trimap).any(|v| !v.is_finite()) {
        return Err(Error::Input("a matting tile holds a value that is not a number".into()));
    }
    let plane = side * side;
    let mut input = vec![0.0f32; 4 * plane];
    for y in 0..h {
        for x in 0..w {
            let (i, o) = (y * w + x, y * side + x);
            if let (Some(p), Some(t)) = (image.px.get(i), trimap.get(i)) {
                for (c, v) in p.iter().enumerate() {
                    if let Some(slot) = input.get_mut(c * plane + o) {
                        *slot = (v.clamp(0.0, 1.0) - 0.5) / 0.5;
                    }
                }
                if let Some(slot) = input.get_mut(3 * plane + o) {
                    *slot = t.clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(input)
}

fn run(id: ModelId, loaded: &mut Loaded, input: Vec<f32>, prompt: Prompt, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
    match &mut loaded.sessions {
        Sessions::Matte(model) if id == ModelId::IsNet => matte(model, input, MatteExport { side: 1024, output: "output_image", ending: Ending::Sigmoid }, ctl),
        Sessions::Matte(model) if id == ModelId::BiRefNetLite => matte(model, input, MatteExport { side: 640, output: "output_image", ending: Ending::RawScores }, ctl),
        Sessions::Matte(model) if id == ModelId::U2Net => matte(model, input, MatteExport { side: 320, output: "1959", ending: Ending::Stretch }, ctl),
        Sessions::Matte(model) if id == ModelId::ViTMatte => vitmatte(model, input, id.info().input_side, ctl),
        Sessions::Matte(model) => birefnet(model, input, ctl),
        Sessions::Sam2 { encoder, decoder } => sam2(encoder, decoder, input, prompt, ctl),
    }
}

fn validate_prompt(id: ModelId, p: Prompt) -> Result<()> {
    match (id, p) {
        (ModelId::BiRefNet | ModelId::IsNet | ModelId::BiRefNetLite | ModelId::U2Net, Prompt::Subject) => Ok(()),
        (ModelId::ViTMatte, Prompt::Trimap) => Ok(()),
        (ModelId::Sam2, Prompt::Box([x0, y0, x1, y1])) if [x0, y0, x1, y1].iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) && x0 < x1 && y0 < y1 => {
            Ok(())
        }
        _ => Err(Error::Input("this model does not support the requested prompt".into())),
    }
}

const CPU: &str = "CPU";

/// GPU routes compiled into this build, best first. Each build enables what its platform has.
// Which pushes exist depends on the build, so this cannot be a `vec!` literal.
#[allow(clippy::vec_init_then_push)]
fn accelerators() -> Vec<(&'static str, ort::ep::ExecutionProviderDispatch)> {
    #[allow(unused_mut)]
    let mut routes = Vec::new();
    #[cfg(feature = "cuda")]
    routes.push(("CUDA", ort::ep::CUDA::default().build().error_on_failure()));
    #[cfg(all(feature = "directml", target_os = "windows"))]
    routes.push(("DirectML", ort::ep::DirectML::default().build().error_on_failure()));
    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    routes.push(("CoreML", ort::ep::CoreML::default().build().error_on_failure()));
    routes
}

fn pinned(mut builder: SessionBuilder, dims: &[(&str, i64)]) -> Result<SessionBuilder> {
    for (name, size) in dims {
        builder = builder.with_dimension_override(*name, *size)?;
    }
    Ok(builder)
}

fn accelerated(path: &Path, route: ort::ep::ExecutionProviderDispatch, side: &[(&str, i64)]) -> Result<OnnxSession> {
    // DirectML needs the memory pattern off, and it does the others no harm.
    let builder = OnnxSession::builder()?.with_execution_providers([route])?.with_memory_pattern(false)?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    Ok(pinned(builder, side)?.commit_from_file(path)?)
}

/// Returns the session and the device it runs on.
fn session(path: &Path, cpu_only: bool, side: &[(&str, i64)]) -> Result<(OnnxSession, &'static str)> {
    let threads = std::thread::available_parallelism().map_or(2, |v| v.get()).clamp(1, 8);
    if !cpu_only {
        for (device, route) in accelerators() {
            if let Ok(model) = accelerated(path, route, side) {
                return Ok((model, device));
            }
        }
    }
    // Keeping weights resident is useful, but retaining peak 2048px intermediates in an arena
    // (then allocating a contiguous memory-pattern block on the second run) can exhaust RAM.
    // Prefer releasing temporary buffers over latency: these models are explicitly optional.
    let builder = OnnxSession::builder()?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])?
        .with_memory_pattern(false)?
        .with_intra_threads(threads)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    Ok((pinned(builder, side)?.commit_from_file(path)?, CPU))
}

/// ORT supports terminating a run from another thread. The watcher exits at completion or
/// cancellation; it owns no document and never outlives the job. No fictitious per-node progress.
fn cancellable<T>(ctl: &Interrupt<'_>, f: impl FnOnce(&RunOptions) -> Result<T>) -> Result<T> {
    check(ctl)?;
    let options = Arc::new(RunOptions::new()?);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let monitor = std::thread::Builder::new().name("photocraft-model-cancel".into()).spawn_scoped(scope, || {
            while !done.load(Ordering::Relaxed) {
                if ctl.cancelled() {
                    let _ = options.terminate();
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        })?;
        // Release the scoped watcher even if a dependency unexpectedly unwinds: the engine's
        // last-resort panic guard must be able to report it instead of hanging while joining.
        struct Complete<'a>(&'a AtomicBool);
        impl Drop for Complete<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let complete = Complete(&done);
        let result = f(&options);
        drop(complete);
        monitor.join().map_err(|_| Error::Inference("cancellation watcher failed".into()))?;
        check(ctl)?;
        result
    })
}

fn birefnet(model: &mut OnnxSession, input: Vec<f32>, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
    let tensor = Tensor::from_array(([1usize, 3, 2048, 2048], input.into_boxed_slice()))?;
    cancellable(ctl, |options| {
        let outputs = model.run_with_options(ort::inputs!["image" => tensor], options)?;
        let output = outputs.get("alpha").ok_or_else(|| Error::Inference("BiRefNet export has no alpha output".into()))?;
        let (shape, values) = output.try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, 1, 2048, 2048] {
            return Err(Error::Inference("unexpected BiRefNet output shape".into()));
        }
        // This reviewed export already includes sigmoid: applying it twice turns backgrounds gray.
        Ok(AlphaMask { width: 2048, height: 2048, values: values.to_vec(), kind: MaskKind::Alpha })
    })
}

/// How an export's single-channel output becomes opacity.
#[derive(Clone, Copy)]
enum Ending {
    Sigmoid,
    RawScores,
    /// Ends in a sigmoid, but rarely spans 0..=1, so it is stretched to fill it.
    Stretch,
}

#[derive(Clone, Copy)]
struct MatteExport {
    side: usize,
    output: &'static str,
    ending: Ending,
}

fn opacities(values: &[f32], ending: Ending) -> Vec<f32> {
    match ending {
        Ending::Sigmoid => values.to_vec(),
        Ending::RawScores => values.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        Ending::Stretch => {
            let (low, high) = values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
            let spread = (high - low).max(1e-6);
            values.iter().map(|v| (v - low) / spread).collect()
        }
    }
}

fn matte(model: &mut OnnxSession, input: Vec<f32>, export: MatteExport, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
    let MatteExport { side, output, ending } = export;
    let tensor = Tensor::from_array(([1usize, 3, side, side], input.into_boxed_slice()))?;
    let name = model.inputs().first().ok_or_else(|| Error::Inference("model export has no input".into()))?.name().to_string();
    cancellable(ctl, |options| {
        let outputs = model.run_with_options(ort::inputs![name.as_str() => tensor], options)?;
        let output = outputs.get(output).ok_or_else(|| Error::Inference(format!("model export has no {output} output")))?;
        let (shape, values) = output.try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, 1, side as i64, side as i64] {
            return Err(Error::Inference("unexpected model output shape".into()));
        }
        Ok(AlphaMask { width: side, height: side, values: opacities(values, ending), kind: MaskKind::Alpha })
    })
}

/// ViTMatte on one tile: four channels in, opacity out, both `side × side`.
fn vitmatte(model: &mut OnnxSession, input: Vec<f32>, side: usize, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
    let tensor = Tensor::from_array(([1usize, 4, side, side], input.into_boxed_slice()))?;
    let name = model.inputs().first().ok_or_else(|| Error::Inference("model export has no input".into()))?.name().to_string();
    let output = model.outputs().first().ok_or_else(|| Error::Inference("model export has no output".into()))?.name().to_string();
    cancellable(ctl, |options| {
        let outputs = model.run_with_options(ort::inputs![name.as_str() => tensor], options)?;
        let alpha = outputs.get(output.as_str()).ok_or_else(|| Error::Inference("ViTMatte export has no output".into()))?;
        let (shape, values) = alpha.try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, 1, side as i64, side as i64] {
            return Err(Error::Inference("unexpected ViTMatte output shape".into()));
        }
        Ok(AlphaMask { width: side, height: side, values: values.iter().map(|v| v.clamp(0.0, 1.0)).collect(), kind: MaskKind::Alpha })
    })
}

fn sam2(encoder: &mut OnnxSession, decoder: &mut OnnxSession, input: Vec<f32>, prompt: Prompt, ctl: &Interrupt<'_>) -> Result<AlphaMask> {
    let Prompt::Box(bbox) = prompt else { return Err(Error::Input("SAM needs a box".into())) };
    let image = Tensor::from_array(([1usize, 3, 1024, 1024], input.into_boxed_slice()))?;
    let (e0, e1, e2) = cancellable(ctl, |options| {
        let outputs = encoder.run_with_options(ort::inputs!["pixel_values" => image], options)?;
        let embedding = |name| -> Result<Tensor<f32>> {
            let v = outputs.get(name).ok_or_else(|| Error::Inference(format!("SAM export has no {name} output")))?;
            let (shape, values) = v.try_extract_tensor::<f32>()?;
            Ok(Tensor::from_array((shape.to_vec(), values.to_vec().into_boxed_slice()))?)
        };
        Ok((embedding("image_embeddings.0")?, embedding("image_embeddings.1")?, embedding("image_embeddings.2")?))
    })?;
    ctl.progress(0.85);
    // A padding point is ignored by SAM; the box supplies corner labels 2 and 3 inside the
    // decoder. The reviewed export uses int64 labels and coordinates on its 1024-pixel square.
    let points = Tensor::from_array(([1usize, 1, 1, 2], vec![0.0f32; 2].into_boxed_slice()))?;
    let labels = Tensor::from_array(([1usize, 1, 1], vec![-1i64].into_boxed_slice()))?;
    let boxes = Tensor::from_array(([1usize, 1, 4], bbox.map(|v| v * 1024.0).to_vec().into_boxed_slice()))?;
    cancellable(ctl, |options| {
        let outputs = decoder.run_with_options(
            ort::inputs![
            "input_points" => points,
            "input_labels" => labels,
            "input_boxes" => boxes,
            "image_embeddings.0" => e0,
            "image_embeddings.1" => e1,
            "image_embeddings.2" => e2,
            ],
            options,
        )?;
        let masks = outputs.get("pred_masks").ok_or_else(|| Error::Inference("SAM has no mask output".into()))?;
        let scores = outputs.get("iou_scores").ok_or_else(|| Error::Inference("SAM has no score output".into()))?;
        let (shape, values) = masks.try_extract_tensor::<f32>()?;
        let (_, scores) = scores.try_extract_tensor::<f32>()?;
        sam_mask(shape.as_ref(), values, scores)
    })
}

fn sam_mask(shape: &[i64], values: &[f32], scores: &[f32]) -> Result<AlphaMask> {
    let [1, 1, count, height, width] = shape else { return Err(Error::Inference("unexpected SAM output shape".into())) };
    if ![count, height, width].iter().all(|v| **v > 0)
        || *count > 3
        || *height > 2048
        || *width > 2048
        || scores.len() != *count as usize
        || scores.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Inference("invalid SAM mask dimensions or scores".into()));
    }
    let plane = (*width as usize).checked_mul(*height as usize).ok_or_else(|| Error::Inference("mask too large".into()))?;
    if values.len() != plane * *count as usize {
        return Err(Error::Inference("SAM mask length mismatch".into()));
    }
    let best = scores.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map_or(0, |(i, _)| i);
    let chosen = values.get(best * plane..(best + 1) * plane).ok_or_else(|| Error::Inference("SAM mask missing".into()))?;
    Ok(AlphaMask { width: *width as usize, height: *height as usize, values: chosen.to_vec(), kind: MaskKind::Logits })
}

fn copy_verified(reader: &mut dyn Read, writer: &mut dyn Write, artifact: &Artifact, ctl: &Interrupt<'_>, progress: impl Fn(u64)) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut count = 0u64;
    loop {
        check(ctl)?;
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        count = count.checked_add(read as u64).ok_or_else(|| Error::Download("download too large".into()))?;
        if count > artifact.bytes {
            return Err(Error::Download(format!("{} exceeds its pinned size", artifact.file)));
        }
        let chunk = buffer.get(..read).ok_or_else(|| Error::Download("invalid read length".into()))?;
        writer.write_all(chunk)?;
        hasher.update(chunk);
        progress(count);
    }
    check(ctl)?;
    if count != artifact.bytes || format!("{:x}", hasher.finalize()) != artifact.sha256 {
        return Err(Error::Download(format!("{} failed size/SHA-256 verification; no model was installed", artifact.file)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABC: Artifact = Artifact {
        file: "test.onnx",
        url: "https://example.invalid/test.onnx",
        bytes: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    };

    #[test]
    fn stalled_reads_are_bounded_without_shortening_a_stricter_deadline() {
        use ureq::unversioned::transport::time::Duration;
        let t = NextTimeout { after: Duration::NotHappening, reason: ureq::Timeout::RecvBody };
        assert_eq!(idle_timeout(t).after, Duration::from_secs(30));
        let t = NextTimeout { after: Duration::from_secs(3), reason: ureq::Timeout::Global };
        assert_eq!(idle_timeout(t), t);
    }

    #[test]
    fn cancellation_watcher_exits_when_a_dependency_unwinds() {
        let started = std::time::Instant::now();
        let r = std::panic::catch_unwind(|| cancellable::<()>(&Interrupt::NONE, |_| panic!("synthetic dependency panic")));
        assert!(r.is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn a_dependency_panic_does_not_leave_the_model_permanently_busy() {
        let temp = tempfile::tempdir().unwrap();
        let backend = NativeBackend::new(temp.path().into());
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = backend.operation(ModelId::Sam2).unwrap();
            panic!("synthetic dependency panic");
        }));
        assert!(backend.status().iter().all(|s| !s.busy));
        assert!(backend.operation(ModelId::Sam2).is_ok());
    }

    #[test]
    fn cpu_runtime_executes_both_export_contracts_without_downloads() {
        let dir = tempfile::tempdir().unwrap();
        for (name, bytes) in
            [("matte.onnx", crate::test_models::MATTE), ("encoder.onnx", crate::test_models::ENCODER), ("decoder.onnx", crate::test_models::DECODER)]
        {
            fs::write(dir.path().join(name), bytes).unwrap();
        }
        let (mut matte, _) = session(&dir.path().join("matte.onnx"), true, &[]).unwrap();
        let alpha = birefnet(&mut matte, vec![0.0; 3 * 2048 * 2048], &Interrupt::NONE).unwrap();
        assert_eq!(alpha.kind, MaskKind::Alpha);
        assert!(alpha.values.iter().all(|v| (*v - 0.5).abs() < 1e-6));
        let (mut encoder, _) = session(&dir.path().join("encoder.onnx"), true, &[]).unwrap();
        let (mut decoder, _) = session(&dir.path().join("decoder.onnx"), true, &[]).unwrap();
        let mask = sam2(&mut encoder, &mut decoder, vec![0.0; 3 * 1024 * 1024], Prompt::Box([0.1, 0.2, 0.9, 0.8]), &Interrupt::NONE).unwrap();
        assert_eq!(mask.kind, MaskKind::Logits);
        assert_eq!(mask.values, vec![1.0, 2.0]);
        assert!(matches!(birefnet(&mut matte, vec![0.0; 3 * 2048 * 2048], &Interrupt::cancel_only(&|| true)), Err(Error::Cancelled)));
    }

    #[test]
    fn verified_stream_rejects_truncation_corruption_overflow_and_cancel() {
        let mut out = Vec::new();
        copy_verified(&mut &b"abc"[..], &mut out, &ABC, &Interrupt::NONE, |_| {}).unwrap();
        assert_eq!(out, b"abc");
        for input in [b"ab".as_slice(), b"abd".as_slice(), b"abcd".as_slice()] {
            assert!(copy_verified(&mut &input[..], &mut Vec::new(), &ABC, &Interrupt::NONE, |_| {}).is_err());
        }
        assert!(matches!(copy_verified(&mut &b"abc"[..], &mut Vec::new(), &ABC, &Interrupt::cancel_only(&|| true), |_| {}), Err(Error::Cancelled)));
    }

    #[test]
    fn constructing_and_querying_a_backend_does_not_create_a_directory_or_download() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let b = NativeBackend::new(root.clone());
        assert!(b.status().iter().all(|m| !m.installed));
        assert!(!root.exists());
        assert!(b.infer(ModelId::BiRefNet, &RgbImage::new(1, 1), Prompt::Subject, &Interrupt::NONE).is_err());
        assert!(!root.exists());
    }

    #[test]
    fn remove_is_scoped_to_the_selected_model_and_busy_models_are_protected() {
        let temp = tempfile::tempdir().unwrap();
        let b = NativeBackend::new(temp.path().into());
        let dir = b.directory(ModelId::BiRefNet);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("partial"), b"bad").unwrap();
        let unrelated = temp.path().join("preferences.json");
        fs::write(&unrelated, b"keep").unwrap();
        let guard = b.operation(ModelId::BiRefNet).unwrap();
        assert!(b.remove(ModelId::BiRefNet, &Interrupt::NONE).is_err());
        drop(guard);
        b.remove(ModelId::BiRefNet, &Interrupt::NONE).unwrap();
        assert!(!dir.exists());
        assert_eq!(fs::read(unrelated).unwrap(), b"keep");
    }

    /// A tile goes in as four planes, picture first, padded with zeros; bad tiles are refused.
    #[test]
    fn matting_tiles_are_laid_out_as_vitmatte_takes_them() {
        let image = RgbImage::from_fn(2, 1, |x, _| if x == 0 { [1.0, 0.5, 0.0] } else { [0.0, 0.0, 0.0] });
        let input = tile_input(&image, &[0.5, 1.0], 3).unwrap();
        assert_eq!(input.len(), 4 * 9);
        assert_eq!((input[0], input[9], input[18], input[27]), (1.0, 0.0, -1.0, 0.5), "the first pixel in each plane");
        assert_eq!((input[1], input[28]), (-1.0, 1.0));
        assert!(input[2..9].iter().all(|v| *v == 0.0) && input[29..36].iter().all(|v| *v == 0.0), "padding stays zero");
        assert!(tile_input(&image, &[0.5], 3).is_err(), "one trimap value per pixel");
        assert!(tile_input(&RgbImage::from_fn(4, 1, |_, _| [0.0; 3]), &[0.5; 4], 3).is_err(), "wider than a tile");
        assert!(tile_input(&RgbImage::new(0, 0), &[], 3).is_err());
        assert!(tile_input(&image, &[f32::NAN, 0.0], 3).is_err());
        assert!(validate_prompt(ModelId::ViTMatte, Prompt::Subject).is_err() && validate_prompt(ModelId::BiRefNetLite, Prompt::Trimap).is_err());
    }

    #[test]
    fn sam_selects_the_highest_scoring_mask_and_rejects_bad_shapes() {
        let m = sam_mask(&[1, 1, 3, 1, 2], &[-1.0, -1.0, 1.0, 2.0, 9.0, 9.0], &[0.2, 0.9, 0.4]).unwrap();
        assert_eq!(m.values, vec![1.0, 2.0]);
        assert!(sam_mask(&[1, 1, i64::MAX, 1, 2], &[], &[]).is_err());
        assert!(sam_mask(&[1, 1, 1, 1, 2], &[0.0], &[1.0]).is_err());
        assert!(validate_prompt(ModelId::Sam2, Prompt::Box([0.0, 0.0, f32::NAN, 1.0])).is_err());
    }

    /// Explicit developer opt-in only. No normal test or first launch downloads weights.
    #[test]
    #[ignore = "requires PHOTOCRAFT_MODEL_TEST_DIR; PHOTOCRAFT_MODEL_TEST_DOWNLOAD=1 explicitly downloads ~1.8 GB"]
    fn real_models_cpu_smoke() {
        let root = std::env::var_os("PHOTOCRAFT_MODEL_TEST_DIR").expect("set a dedicated model test directory");
        let backend = NativeBackend::new(root.into());
        let mut image = RgbImage::new(192, 128);
        for (i, p) in image.px.iter_mut().enumerate() {
            let (x, y) = ((i % 192) as i32 - 96, (i / 192) as i32 - 64);
            *p = if x * x + y * y < 45 * 45 { [0.9, 0.2, 0.1] } else { [0.1, 0.4, 0.7] };
        }
        if std::env::var("PHOTOCRAFT_MODEL_TEST_DOWNLOAD").as_deref() == Ok("1") {
            for id in [ModelId::BiRefNet, ModelId::Sam2] {
                eprintln!("Downloading / verifying {}", id.name());
                backend.download(id, &Interrupt::NONE).unwrap();
            }
        }
        for id in [ModelId::BiRefNet, ModelId::Sam2] {
            let prompt = if id == ModelId::Sam2 { Prompt::Box([0.2, 0.1, 0.8, 0.9]) } else { Prompt::Subject };
            let start = std::time::Instant::now();
            let mask = backend.infer(id, &image, prompt, &Interrupt::NONE).unwrap();
            mask.validate().unwrap();
            assert!(mask.values.iter().any(|v| v.abs() > 0.01));
            eprintln!("{} CPU: {:.2}s; mask {}x{}", id.name(), start.elapsed().as_secs_f64(), mask.width, mask.height);
        }
    }
}
