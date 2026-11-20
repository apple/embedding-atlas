use crate::layout_modern::{
    ModernLayout, ModernLayoutParams, ModernOptimizerKind, ModernOptions, RateMode,
};
use ndarray::Array2;
use nndescent::rng::TauRng;
use nndescent::Logger;
use rayon::prelude::*;

/// Which optimizer to use for the layout.
///
/// `Sgd` is the reference umap-learn 0.5 incremental SGD. `Momentum` and `Adam`
/// are the umap-learn 0.6 layouts ([`crate::layout_modern`]); with a GPU
/// optimizer they run in `gpu_layout_modern`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Optimizer {
    /// Reference-faithful incremental SGD. Default.
    Sgd,
    /// Heavy-ball momentum with blocked Jacobi updates (umap-learn 0.6).
    Momentum,
    /// Lazy Adam with deterministic hashed negative sampling and Jacobi updates.
    Adam,
}

impl std::str::FromStr for Optimizer {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "sgd" => Ok(Self::Sgd),
            "momentum" => Ok(Self::Momentum),
            "adam" => Ok(Self::Adam),
            other => Err(format!("unknown optimizer \"{other}\"")),
        }
    }
}

impl Optimizer {
    fn modern_kind(self) -> Option<ModernOptimizerKind> {
        match self {
            Optimizer::Sgd => None,
            Optimizer::Momentum => Some(ModernOptimizerKind::Momentum),
            Optimizer::Adam => Some(ModernOptimizerKind::Adam),
        }
    }
}

/// Sub-batches of a GPU SGD run to a horizon, summed over all epochs: short
/// runs use more sub-batches per epoch, which keeps their high learning rates
/// from overshooting.
#[cfg(feature = "gpu")]
const GPU_SUB_BATCH_BUDGET: usize = 1280;

/// Live-tunable optimizer parameters. These can be changed between steps without
/// rebuilding the optimizer (the caller controls learning-rate decay).
#[derive(Clone, Copy, Debug)]
pub struct LiveParams {
    /// Learning rate (no decay applied here — see [`UmapOptimizer::run_to`]).
    pub alpha: f32,
    /// Repulsion strength (γ).
    pub gamma: f32,
    /// Number of negative samples per positive sample.
    pub negative_sample_rate: f32,
    /// Output-distance curve parameter `a` (from `find_ab_params`).
    pub a: f32,
    /// Output-distance curve parameter `b` (from `find_ab_params`).
    pub b: f32,
}

/// Clamp gradient to [-4.0, 4.0] to prevent instability.
#[inline]
fn clip(val: f32) -> f32 {
    val.clamp(-4.0, 4.0)
}

/// Squared Euclidean distance in embedding space.
#[inline]
pub(crate) fn rdist(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

/// Collect the edges that are due to be sampled at `current_epoch`, advancing the
/// per-edge sampling schedule counters in place.
///
/// Returns `(edge_index, n_negative_samples)` for each active edge.
///
/// `epochs_per_negative_sample` is computed **inline** as
/// `epochs_per_sample[i] / negative_sample_rate`, so changing the rate live needs
/// no rebuild of any precomputed table.
pub(crate) fn collect_active_edges(
    epochs_per_sample: &[f64],
    epoch_of_next_sample: &mut [f64],
    epoch_of_next_negative_sample: &mut [f64],
    current_epoch: u64,
    negative_sample_rate: f32,
) -> Vec<(usize, usize)> {
    let n_edges = epochs_per_sample.len();
    let cur = current_epoch as f64;
    let rate = negative_sample_rate as f64;

    let mut active_edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_edges {
        if epoch_of_next_sample[i] > cur {
            continue;
        }

        // Negative-sample count for this edge this epoch. When the rate is 0 we
        // simply do no repulsion (and don't advance the negative counter, which
        // would otherwise divide by an infinite interval).
        let n_neg = if rate > 0.0 {
            let epn = epochs_per_sample[i] / rate;
            let n = ((cur - epoch_of_next_negative_sample[i]) / epn) as usize;
            epoch_of_next_negative_sample[i] += n as f64 * epn;
            n
        } else {
            0
        };

        active_edges.push((i, n_neg));
        epoch_of_next_sample[i] += epochs_per_sample[i];
    }

    active_edges
}

/// Run one epoch of Hogwild SGD in place.
///
/// The gradient math (attractive `-2ab·d^{2b}/d² / (a·d^{2b}+1)`, repulsive
/// `2γb/((0.001+d²)(a·d^{2b}+1))`, per-edge clip to ±4) is identical to the
/// reference implementation; `g` is the position delta (`x += α·g`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_one_epoch_hogwild(
    emb: &mut [f32],
    dim: usize,
    n_vertices: usize,
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f64],
    epoch_of_next_sample: &mut [f64],
    epoch_of_next_negative_sample: &mut [f64],
    current_epoch: u64,
    params: LiveParams,
    rng_state: [i64; 3],
) {
    // Phase 1: sequentially collect active edges and advance scheduling counters.
    let active_edges = collect_active_edges(
        epochs_per_sample,
        epoch_of_next_sample,
        epoch_of_next_negative_sample,
        current_epoch,
        params.negative_sample_rate,
    );

    let alpha = params.alpha;
    let gamma = params.gamma;
    let a = params.a;
    let b = params.b;
    let epoch = current_epoch;

    // SAFETY: Hogwild-style parallel SGD (Recht et al., 2011). The embedding array
    // is shared mutably across threads WITHOUT synchronization. This is intentional:
    // - Each gradient update touches only `dim` floats (typically 2-4) per vertex.
    // - Concurrent writes to the same vertex produce slightly stale/torn reads,
    //   which act as additional stochastic noise and do not affect convergence in
    //   practice.
    // - This matches reference Python UMAP (numba prange) and is standard practice.
    //
    // Formally, multiple `&mut` slices to overlapping memory is UB under Rust's
    // aliasing model; in practice the generated code is safe because all accesses
    // are simple f32 loads/stores, and the primary target is single-threaded WASM.
    let emb_ptr = emb.as_mut_ptr() as usize;

    // Phase 2: apply gradients in parallel (Hogwild). Each edge gets its own RNG
    // seeded deterministically from (rng_state, edge_index, epoch).
    active_edges.par_iter().for_each_init(
        || (vec![0.0f32; dim], vec![0.0f32; dim], vec![0.0f32; dim]),
        |(current, other, neg_other), &(edge_idx, n_neg)| {
            // SAFETY: see Hogwild comment above on emb_ptr.
            let emb =
                unsafe { std::slice::from_raw_parts_mut(emb_ptr as *mut f32, n_vertices * dim) };

            let j = head[edge_idx];
            let k = tail[edge_idx];
            let j_off = j * dim;
            let k_off = k * dim;

            current.copy_from_slice(&emb[j_off..j_off + dim]);
            other.copy_from_slice(&emb[k_off..k_off + dim]);
            let dist_squared = rdist(current, other);

            // Positive sample: attractive force.
            let grad_coeff = if dist_squared > 0.0 {
                let pow_b = dist_squared.powf(b);
                -2.0 * a * b * (pow_b / dist_squared) / (a * pow_b + 1.0)
            } else {
                0.0
            };

            for d in 0..dim {
                let g = clip(grad_coeff * (current[d] - other[d]));
                emb[j_off + d] += alpha * g;
                emb[k_off + d] -= alpha * g;
            }

            // Negative samples: repulsive force (j only).
            let mut rng = TauRng::from_state([
                rng_state[0]
                    .wrapping_add(edge_idx as i64)
                    .wrapping_mul(epoch as i64 + 1),
                rng_state[1].wrapping_add(j as i64),
                rng_state[2].wrapping_add(edge_idx as i64),
            ]);

            for _ in 0..n_neg {
                let neg_k = (rng.tau_rand_int().unsigned_abs() as usize) % n_vertices;
                let neg_off = neg_k * dim;
                neg_other.copy_from_slice(&emb[neg_off..neg_off + dim]);
                // Re-read current since it may have been updated.
                current.copy_from_slice(&emb[j_off..j_off + dim]);
                let dist_squared = rdist(current, neg_other);

                let grad_coeff = if dist_squared > 0.0 {
                    2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_squared.powf(b) + 1.0))
                } else if j == neg_k {
                    continue;
                } else {
                    0.0
                };

                for d in 0..dim {
                    let g = if grad_coeff > 0.0 {
                        clip(grad_coeff * (current[d] - neg_other[d]))
                    } else {
                        0.0
                    };
                    emb[j_off + d] += alpha * g;
                }
            }
        },
    );
}

/// A resumable UMAP layout optimizer.
///
/// Advances the layout `k` epochs at a time ([`step`](Self::step)), or runs to a
/// fixed horizon ([`run_to`](Self::run_to), which reproduces standard
/// run-to-completion UMAP). Parameters and the optimizer can be changed live
/// between steps, and the embedding can be re-initialized
/// ([`reinit_embedding`](Self::reinit_embedding)) to restart from a new layout.
///
/// On GPU, the resident buffers live in the optimizer; `step`/`run_to` download
/// the embedding into the CPU `embedding` mirror once at the end, and all getters
/// read that mirror.
pub struct UmapOptimizer {
    embedding: Array2<f32>,
    n_vertices: usize,
    dim: usize,
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: Vec<f64>,
    epoch_of_next_sample: Vec<f64>,
    epoch_of_next_negative_sample: Vec<f64>,
    current_epoch: u64,
    params: LiveParams,
    optimizer: Optimizer,
    rng_state: [i64; 3],
    /// Momentum/Adam state on the CPU.
    modern: Option<ModernLayout>,
    /// Schedule horizon of the Momentum/Adam schedules for [`step`](Self::step).
    n_epochs: usize,
    good_initialization: bool,
    modern_options: ModernOptions,
    #[cfg(feature = "gpu")]
    gpu_sgd: Option<crate::gpu_optimize::OptimizeGpuContext>,
    /// f32-counter coordinate for the GPU SGD schedule; reset to 0 on rebase.
    #[cfg(feature = "gpu")]
    gpu_epoch: u32,
    /// Device shared by the GPU optimizers; `Some` for a GPU-backed optimizer.
    #[cfg(feature = "gpu")]
    gpu_device: Option<(wgpu::Device, wgpu::Queue)>,
    /// Momentum/Adam state on the GPU; replaces `modern` when present.
    #[cfg(feature = "gpu")]
    gpu_modern: Option<crate::gpu_layout_modern::GpuModernLayout>,
    /// Why the GPU Momentum/Adam layout was abandoned for the CPU one.
    #[cfg(feature = "gpu")]
    gpu_error: Option<String>,
    #[cfg(feature = "gpu")]
    n_neighbors: usize,
}

impl UmapOptimizer {
    /// Create a CPU optimizer over the given edge list and initial embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        head: Vec<usize>,
        tail: Vec<usize>,
        epochs_per_sample: &[f32],
        embedding: Array2<f32>,
        params: LiveParams,
        optimizer: Optimizer,
        rng_state: [i64; 3],
    ) -> Self {
        let mut opt = Self::without_layout_state(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
        );
        opt.install_layout_state();
        opt
    }

    fn without_layout_state(
        head: Vec<usize>,
        tail: Vec<usize>,
        epochs_per_sample: &[f32],
        embedding: Array2<f32>,
        params: LiveParams,
        optimizer: Optimizer,
        rng_state: [i64; 3],
    ) -> Self {
        let n_vertices = embedding.nrows();
        let dim = embedding.ncols();

        let eps: Vec<f64> = epochs_per_sample.iter().map(|&e| e as f64).collect();
        let epoch_of_next_sample = eps.clone();
        let rate = params.negative_sample_rate as f64;
        let epoch_of_next_negative_sample: Vec<f64> = eps
            .iter()
            .map(|&e| if rate > 0.0 { e / rate } else { f64::INFINITY })
            .collect();

        UmapOptimizer {
            embedding,
            n_vertices,
            dim,
            head,
            tail,
            epochs_per_sample: eps,
            epoch_of_next_sample,
            epoch_of_next_negative_sample,
            current_epoch: 0,
            params,
            optimizer,
            rng_state,
            modern: None,
            n_epochs: if n_vertices <= 10_000 { 500 } else { 200 },
            good_initialization: false,
            modern_options: ModernOptions::default(),
            #[cfg(feature = "gpu")]
            gpu_sgd: None,
            #[cfg(feature = "gpu")]
            gpu_epoch: 0,
            #[cfg(feature = "gpu")]
            gpu_device: None,
            #[cfg(feature = "gpu")]
            gpu_modern: None,
            #[cfg(feature = "gpu")]
            gpu_error: None,
            #[cfg(feature = "gpu")]
            n_neighbors: 0,
        }
    }

    fn modern_layout_params(&self, kind: ModernOptimizerKind) -> ModernLayoutParams {
        let mut params = ModernLayoutParams {
            kind,
            n_epochs: self.n_epochs,
            a: self.params.a,
            b: self.params.b,
            gamma: self.params.gamma,
            initial_alpha: self.params.alpha,
            negative_sample_rate: self.params.negative_sample_rate,
            good_initialization: self.good_initialization,
            seed: self
                .rng_state
                .iter()
                .fold(0u64, |seed, &s| seed.rotate_left(21) ^ s as u64),
            ..ModernLayoutParams::default()
        };
        params.set_options(self.modern_options);
        params
    }

    /// Install the state of the current optimizer: Momentum/Adam on the GPU when
    /// a device is available, else on the CPU; the SGD GPU context for Sgd.
    fn install_layout_state(&mut self) {
        self.modern = None;
        #[cfg(feature = "gpu")]
        {
            self.gpu_modern = None;
        }
        let Some(kind) = self.optimizer.modern_kind() else {
            #[cfg(feature = "gpu")]
            self.install_gpu_sgd_context();
            return;
        };
        #[cfg(feature = "gpu")]
        {
            self.gpu_sgd = None;
        }
        let (indptr, indices, epochs_per_sample) = edges_to_csr(
            self.n_vertices,
            &self.head,
            &self.tail,
            &self.epochs_per_sample,
        );
        let params = self.modern_layout_params(kind);
        let start_epoch = self.current_epoch as usize;
        #[cfg(feature = "gpu")]
        if let (Some((device, queue)), None) = (&self.gpu_device, &self.gpu_error) {
            self.gpu_modern = crate::gpu_layout_modern::GpuModernLayout::new(
                device.clone(),
                queue.clone(),
                indptr.clone(),
                indices.clone(),
                epochs_per_sample.clone(),
                self.embedding.as_slice().expect("contiguous"),
                self.dim,
                params.clone(),
                start_epoch,
            );
            if self.gpu_modern.is_some() {
                return;
            }
        }
        self.modern = Some(ModernLayout::from_csr_parts(
            indptr,
            indices,
            epochs_per_sample,
            self.dim,
            params,
            start_epoch,
        ));
    }

    /// Create the resident SGD GPU context if a device is available, or upload
    /// the current embedding into the existing one.
    #[cfg(feature = "gpu")]
    fn install_gpu_sgd_context(&mut self) {
        let Some((device, queue)) = &self.gpu_device else {
            return;
        };
        let embedding = self.embedding.as_slice().expect("contiguous");
        if let Some(gpu) = &self.gpu_sgd {
            gpu.reupload_embedding(embedding);
            return;
        }
        let eps_f32: Vec<f32> = self.epochs_per_sample.iter().map(|&e| e as f32).collect();
        let head_u32: Vec<u32> = self.head.iter().map(|&h| h as u32).collect();
        let tail_u32: Vec<u32> = self.tail.iter().map(|&t| t as u32).collect();
        self.gpu_sgd = crate::gpu_optimize::OptimizeGpuContext::new(
            device.clone(),
            queue.clone(),
            embedding,
            &head_u32,
            &tail_u32,
            &eps_f32,
            self.n_vertices as u32,
            self.dim as u32,
            self.n_neighbors,
            self.params.negative_sample_rate,
            self.rng_state,
        );
        self.gpu_epoch = 0;
    }

    /// Set the schedule horizon that [`step`](Self::step) uses for Momentum and
    /// Adam. [`run_to`](Self::run_to) sets it to its target.
    pub fn set_n_epochs(&mut self, n_epochs: usize) {
        self.n_epochs = n_epochs;
        if let Some(modern) = &mut self.modern {
            modern.set_n_epochs(n_epochs);
        }
        #[cfg(feature = "gpu")]
        if let Some(layout) = &mut self.gpu_modern {
            layout.set_n_epochs(n_epochs);
        }
    }

    /// Select the shorter Momentum/Adam warm-up for an already good initial layout.
    pub fn set_good_initialization(&mut self, good_initialization: bool) {
        self.good_initialization = good_initialization;
        if let Some(modern) = &mut self.modern {
            modern.set_good_initialization(good_initialization);
        }
        #[cfg(feature = "gpu")]
        if let Some(layout) = &mut self.gpu_modern {
            layout.set_good_initialization(good_initialization);
        }
    }

    /// Set the negative-selection options of Momentum and Adam. This resets
    /// their orderings and repulsion scale.
    pub fn set_modern_options(&mut self, options: ModernOptions) {
        self.modern_options = options;
        if let Some(modern) = &mut self.modern {
            modern.set_options(options);
        }
        #[cfg(feature = "gpu")]
        if let Some(layout) = &mut self.gpu_modern {
            layout.set_options(options);
        }
    }

    /// Create a GPU-backed optimizer. Falls back to the CPU path (no resident GPU
    /// context) if no suitable adapter is available.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    pub async fn new_gpu(
        head: Vec<usize>,
        tail: Vec<usize>,
        epochs_per_sample: &[f32],
        embedding: Array2<f32>,
        params: LiveParams,
        optimizer: Optimizer,
        rng_state: [i64; 3],
        n_neighbors: usize,
    ) -> Self {
        let device = crate::gpu_optimize::request_device().await;
        Self::new_with_device(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
            n_neighbors,
            device,
        )
    }

    /// Like [`new_gpu`](Self::new_gpu), on an already requested device (`None`
    /// selects the CPU path).
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_device(
        head: Vec<usize>,
        tail: Vec<usize>,
        epochs_per_sample: &[f32],
        embedding: Array2<f32>,
        params: LiveParams,
        optimizer: Optimizer,
        rng_state: [i64; 3],
        n_neighbors: usize,
        device: Option<(wgpu::Device, wgpu::Queue)>,
    ) -> Self {
        let mut opt = Self::without_layout_state(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
        );
        opt.n_neighbors = n_neighbors;
        opt.gpu_device = device;
        opt.install_layout_state();
        opt
    }

    /// The GPU error that moved Momentum/Adam to the CPU, if any.
    #[cfg(feature = "gpu")]
    pub fn gpu_error(&self) -> Option<&str> {
        self.gpu_error.as_deref()
    }

    #[cfg(feature = "gpu")]
    fn gpu_sgd_active(&self) -> bool {
        self.gpu_sgd.is_some() && self.optimizer == Optimizer::Sgd
    }

    #[cfg(feature = "gpu")]
    async fn sync_gpu_modern(&mut self) {
        let Some(layout) = &mut self.gpu_modern else {
            return;
        };
        layout.sync_embedding().await;
        self.embedding
            .as_slice_memory_order_mut()
            .expect("embedding not contiguous")
            .copy_from_slice(layout.embedding());
    }

    #[cfg(feature = "gpu")]
    fn run_gpu_sgd_epoch(&mut self, params: LiveParams, min_sub_batches: usize) {
        let gpu = self.gpu_sgd.as_ref().unwrap();
        gpu.run_epoch(self.gpu_epoch, params, min_sub_batches);
        self.gpu_epoch += 1;
        if self.gpu_epoch >= (1u32 << 18) {
            gpu.rebase(self.gpu_epoch);
            self.gpu_epoch = 0;
        }
    }

    /// Run one CPU SGD epoch in place at the given parameters.
    fn run_cpu_epoch(&mut self, params: LiveParams) {
        let emb = self
            .embedding
            .as_slice_memory_order_mut()
            .expect("embedding not contiguous");
        run_one_epoch_hogwild(
            emb,
            self.dim,
            self.n_vertices,
            &self.head,
            &self.tail,
            &self.epochs_per_sample,
            &mut self.epoch_of_next_sample,
            &mut self.epoch_of_next_negative_sample,
            self.current_epoch,
            params,
            self.rng_state,
        );
    }

    /// Run Momentum/Adam epochs until `until` epochs are done, logging progress
    /// against `n_epochs` when a logger is given.
    async fn run_modern(
        &mut self,
        until: u64,
        mode: RateMode,
        mut logger: Option<(&mut Logger, usize)>,
    ) {
        #[cfg(feature = "gpu")]
        if let Some(layout) = &mut self.gpu_modern {
            let mut failure = None;
            while self.current_epoch < until {
                let epoch = self.current_epoch;
                if let Err(error) = layout.step_with(mode).await {
                    failure = Some(error);
                    break;
                }
                self.current_epoch += 1;
                if let Some((logger, n_epochs)) = &mut logger {
                    Self::log_progress(logger, epoch, *n_epochs);
                }
            }
            match failure {
                None => {
                    self.sync_gpu_modern().await;
                    return;
                }
                Some(error) => {
                    let message = format!("GPU layout failed ({error}); continuing on the CPU");
                    crate::gpu_optimize::report_gpu_error(&message);
                    if let Some((logger, _)) = &mut logger {
                        logger.log(&message);
                    }
                    self.gpu_error = Some(error);
                    self.install_layout_state();
                }
            }
        }
        let modern = self.modern.as_mut().expect("Momentum/Adam state");
        while self.current_epoch < until {
            let epoch = self.current_epoch;
            modern.step_with(&mut self.embedding, mode);
            self.current_epoch += 1;
            if let Some((logger, n_epochs)) = &mut logger {
                Self::log_progress(logger, epoch, *n_epochs);
            }
        }
    }

    /// Advance the layout by `n_epochs` at the live learning rate, held flat (no
    /// decay). SGD and Adam use α = `alpha`; Momentum uses 0.25·`alpha` with
    /// momentum 0.5. Intended for interactive stepping.
    pub async fn step(&mut self, n_epochs: usize) {
        let until = self.current_epoch + n_epochs as u64;
        if self.optimizer.modern_kind().is_some() {
            self.run_modern(until, RateMode::Flat, None).await;
            return;
        }

        #[cfg(feature = "gpu")]
        if self.gpu_sgd_active() {
            while self.current_epoch < until {
                self.run_gpu_sgd_epoch(self.params, 0);
                self.current_epoch += 1;
            }
            self.download_from_gpu().await;
            return;
        }

        while self.current_epoch < until {
            self.run_cpu_epoch(self.params);
            self.current_epoch += 1;
        }
    }

    /// Run to a fixed horizon with linear learning-rate decay, mirroring standard
    /// run-to-completion UMAP. A fresh optimizer started at epoch 0 reproduces the
    /// reference layout (modulo the f64 schedule counters).
    ///
    /// The decay peak is the live learning rate (`self.params.alpha` — the
    /// configured rate, or whatever [`set_params`](Self::set_params) last set).
    /// Decay is applied to a local copy of the parameters; `self.params` is left
    /// untouched so a subsequent [`step`](Self::step) continues at that same
    /// learning rate rather than at the decayed-to-zero value.
    ///
    /// Momentum and Adam instead follow the umap-learn 0.6 schedules with horizon
    /// `n_epochs` (which becomes the horizon of later [`step`](Self::step)s).
    pub async fn run_to(&mut self, n_epochs: usize, logger: &mut Logger) {
        if self.optimizer.modern_kind().is_some() {
            self.set_n_epochs(n_epochs);
            self.run_modern(
                n_epochs as u64,
                RateMode::Schedule,
                Some((logger, n_epochs)),
            )
            .await;
            return;
        }

        let initial_alpha = self.params.alpha;
        let decayed = |params: LiveParams, epoch: u64| LiveParams {
            alpha: (initial_alpha * (1.0 - epoch as f32 / n_epochs as f32)).max(0.0),
            ..params
        };

        #[cfg(feature = "gpu")]
        if self.gpu_sgd_active() {
            let min_sub_batches = GPU_SUB_BATCH_BUDGET.div_ceil(n_epochs.max(1));
            while self.current_epoch < n_epochs as u64 {
                let epoch = self.current_epoch;
                self.run_gpu_sgd_epoch(decayed(self.params, epoch), min_sub_batches);
                self.current_epoch += 1;
                Self::log_progress(logger, epoch, n_epochs);
            }
            self.download_from_gpu().await;
            return;
        }

        while self.current_epoch < n_epochs as u64 {
            let epoch = self.current_epoch;
            self.run_cpu_epoch(decayed(self.params, epoch));
            self.current_epoch += 1;
            Self::log_progress(logger, epoch, n_epochs);
        }
    }

    fn log_progress(logger: &mut Logger, epoch: u64, n_epochs: usize) {
        if n_epochs >= 10 && (epoch as usize).is_multiple_of(n_epochs / 10) {
            logger.log(&format!("completed {} / {} epochs", epoch, n_epochs));
        }
        logger.stage_progress((epoch + 1) as f64 / n_epochs as f64, None);
    }

    #[cfg(feature = "gpu")]
    async fn download_from_gpu(&mut self) {
        let result = self.gpu_sgd.as_ref().unwrap().download_embedding().await;
        self.embedding
            .as_slice_memory_order_mut()
            .expect("embedding not contiguous")
            .copy_from_slice(&result);
    }

    /// (Re)seed the per-edge negative-sample counters for the current rate so the
    /// next negative sample for each edge falls ~`epn` epochs after `from_epoch`.
    ///
    /// At `from_epoch == 0` this reproduces the construction baseline (`e / rate`).
    /// Keeping the rate fully live requires calling this on any rate change: the
    /// counter is rate-dependent, so a stale baseline (or the `+inf` park used when
    /// the rate is 0) would otherwise leave repulsion mis-scheduled — or, for a
    /// 0 → positive change, permanently disabled.
    fn reseed_negative_schedule(&mut self, from_epoch: f64) {
        let rate = self.params.negative_sample_rate as f64;
        for (i, &e) in self.epochs_per_sample.iter().enumerate() {
            self.epoch_of_next_negative_sample[i] = if rate > 0.0 {
                from_epoch + e / rate
            } else {
                f64::INFINITY
            };
        }
    }

    /// Update the live parameters (learning rate, repulsion, negative-sample rate,
    /// and the a/b curve parameters).
    ///
    /// Changing `negative_sample_rate` re-seeds the negative-sample schedule
    /// relative to the current epoch, so the new rate takes effect cleanly from
    /// here on (no stale baseline, and a 0 → positive change re-arms repulsion).
    /// Changing `alpha` rescales the Momentum velocity to the new rate (so 0
    /// stops the layout). Changing `a`, `b`, or `gamma` resets the Adam
    /// moments, whose stale second moments would otherwise damp the response
    /// to the new forces.
    pub fn set_params(&mut self, params: LiveParams) {
        let rate_changed = params.negative_sample_rate != self.params.negative_sample_rate;
        let forces_changed =
            (params.a, params.b, params.gamma) != (self.params.a, self.params.b, self.params.gamma);
        let velocity_scale = if self.params.alpha > 0.0 {
            params.alpha / self.params.alpha
        } else {
            0.0
        };
        self.params = params;
        if velocity_scale != 1.0 {
            if let Some(modern) = &mut self.modern {
                modern.scale_velocity(velocity_scale);
            }
            #[cfg(feature = "gpu")]
            if let Some(layout) = &mut self.gpu_modern {
                layout.scale_velocity(velocity_scale);
            }
        }
        if forces_changed {
            if let Some(modern) = &mut self.modern {
                modern.reset_moments();
            }
            #[cfg(feature = "gpu")]
            if let Some(layout) = &mut self.gpu_modern {
                layout.reset_moments();
            }
        }
        if let Some(modern) = &mut self.modern {
            modern.set_live_params(params);
        }
        #[cfg(feature = "gpu")]
        if let Some(layout) = &mut self.gpu_modern {
            layout.set_live_params(params);
        }
        if rate_changed {
            let from = self.current_epoch as f64;
            self.reseed_negative_schedule(from);
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &self.gpu_sgd {
                gpu.reseed_negative_schedule(params.negative_sample_rate, self.gpu_epoch);
            }
        }
    }

    /// Switch optimizer. Momentum and Adam start with fresh state (moments,
    /// sampling schedules, orderings), continuing their schedules from the
    /// current epoch.
    pub fn set_optimizer(&mut self, optimizer: Optimizer) {
        if optimizer == Optimizer::Sgd && self.optimizer != Optimizer::Sgd {
            let from = self.current_epoch as f64;
            for (next, &interval) in self
                .epoch_of_next_sample
                .iter_mut()
                .zip(&self.epochs_per_sample)
            {
                *next = from + interval;
            }
            self.reseed_negative_schedule(from);
        }
        self.optimizer = optimizer;
        self.install_layout_state();
    }

    /// Replace the embedding and reset the schedule, epoch, and optimizer state —
    /// i.e. a full restart of the layout optimization from a new initial position.
    pub fn reinit_embedding(&mut self, embedding: Array2<f32>) {
        assert_eq!(embedding.nrows(), self.n_vertices);
        assert_eq!(embedding.ncols(), self.dim);
        self.embedding = embedding;

        self.epoch_of_next_sample
            .copy_from_slice(&self.epochs_per_sample);
        self.reseed_negative_schedule(0.0);
        self.current_epoch = 0;
        if let Some(modern) = &mut self.modern {
            modern.restart(0);
        }
        #[cfg(feature = "gpu")]
        if let Some(layout) = &mut self.gpu_modern {
            layout.upload_embedding(self.embedding.as_slice().expect("contiguous"));
            layout.restart(0);
        }

        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu_sgd {
            gpu.reupload_embedding(self.embedding.as_slice().expect("contiguous"));
            gpu.reset_schedule(self.params.negative_sample_rate);
            self.gpu_epoch = 0;
        }
    }

    /// The current live parameters.
    pub fn params(&self) -> LiveParams {
        self.params
    }

    /// The current embedding (CPU mirror; valid after `step`/`run_to`).
    pub fn embedding(&self) -> &Array2<f32> {
        &self.embedding
    }

    /// Consume the optimizer and return the embedding.
    pub fn into_embedding(self) -> Array2<f32> {
        self.embedding
    }

    /// Copy the current embedding into `out` (synchronous memcpy from the mirror).
    pub fn copy_embedding_into(&self, out: &mut [f32]) {
        let src = self.embedding.as_slice().expect("embedding not contiguous");
        let n = out.len().min(src.len());
        out[..n].copy_from_slice(&src[..n]);
    }

    /// The number of epochs run so far.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }
}

/// Edge list as CSR (rows by `head`, columns sorted) with the matching
/// per-edge sampling intervals.
fn edges_to_csr(
    n_vertices: usize,
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut order: Vec<usize> = (0..head.len()).collect();
    if !order.is_sorted_by_key(|&e| (head[e], tail[e])) {
        order.sort_by_key(|&e| (head[e], tail[e]));
    }
    let mut indptr = vec![0usize; n_vertices + 1];
    for &h in head {
        indptr[h + 1] += 1;
    }
    for i in 0..n_vertices {
        indptr[i + 1] += indptr[i];
    }
    let indices = order.iter().map(|&e| tail[e]).collect();
    let epochs = order.iter().map(|&e| epochs_per_sample[e]).collect();
    (indptr, indices, epochs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nndescent::rng::Xoshiro256StarStar;

    /// Symmetric ring-of-cliques edge list in CSR order.
    fn clique_ring(n_cliques: usize, clique_size: usize) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let n = n_cliques * clique_size;
        let mut triples = Vec::new();
        for i in 0..n {
            let clique = i / clique_size;
            for j in 0..n {
                let bridge = j == (i + clique_size) % n || i == (j + clique_size) % n;
                if i != j && (j / clique_size == clique || bridge) {
                    triples.push((i, j, if bridge { 0.3 } else { 1.0 }));
                }
            }
        }
        let head = triples.iter().map(|t| t.0).collect();
        let tail = triples.iter().map(|t| t.1).collect();
        let weights: Vec<f32> = triples.iter().map(|t| t.2).collect();
        (
            head,
            tail,
            crate::graph::make_epochs_per_sample(&weights, 200),
        )
    }

    fn adam_optimizer(head: Vec<usize>, tail: Vec<usize>, eps: &[f32]) -> UmapOptimizer {
        let n = head.iter().max().unwrap() + 1;
        let mut rng = Xoshiro256StarStar::seed_from_u64(3);
        let embedding = Array2::from_shape_fn((n, 2), |_| rng.random_f32() * 20.0 - 10.0);
        let params = LiveParams {
            alpha: 1.0,
            gamma: 1.0,
            negative_sample_rate: 5.0,
            a: 1.577,
            b: 0.895,
        };
        UmapOptimizer::new(
            head,
            tail,
            eps,
            embedding,
            params,
            Optimizer::Adam,
            [1, 2, 3],
        )
    }

    #[test]
    fn adam_run_to_is_deterministic_and_restartable() {
        let (head, tail, eps) = clique_ring(12, 8);
        let mut first = adam_optimizer(head.clone(), tail.clone(), &eps);
        let mut second = adam_optimizer(head, tail, &eps);
        let initial = first.embedding().clone();
        let mut logger = Logger::new(false, None, &[]);
        pollster::block_on(first.run_to(100, &mut logger));
        pollster::block_on(second.run_to(100, &mut logger));
        assert_eq!(first.current_epoch(), 100);
        assert_eq!(first.embedding(), second.embedding());
        assert!(first.embedding().iter().all(|x| x.is_finite()));

        second.reinit_embedding(initial);
        pollster::block_on(second.run_to(100, &mut logger));
        assert_eq!(first.embedding(), second.embedding());
    }

    #[test]
    fn adam_accepts_unsorted_edges() {
        let (head, tail, eps) = clique_ring(6, 6);
        let mut sorted = adam_optimizer(head.clone(), tail.clone(), &eps);
        let mut permutation: Vec<usize> = (0..head.len()).collect();
        permutation.reverse();
        let mut unsorted = adam_optimizer(
            permutation.iter().map(|&e| head[e]).collect(),
            permutation.iter().map(|&e| tail[e]).collect(),
            &permutation.iter().map(|&e| eps[e]).collect::<Vec<_>>(),
        );
        pollster::block_on(sorted.step(30));
        pollster::block_on(unsorted.step(30));
        assert_eq!(sorted.embedding(), unsorted.embedding());
    }

    #[cfg(feature = "gpu")]
    fn adam_gpu_optimizer(
        head: Vec<usize>,
        tail: Vec<usize>,
        eps: &[f32],
    ) -> Option<UmapOptimizer> {
        let cpu = adam_optimizer(head.clone(), tail.clone(), eps);
        let optimizer = pollster::block_on(UmapOptimizer::new_gpu(
            head,
            tail,
            eps,
            cpu.embedding().clone(),
            cpu.params(),
            Optimizer::Adam,
            [1, 2, 3],
            10,
        ));
        if optimizer.gpu_modern.is_none() {
            eprintln!("skipping: no GPU adapter");
            return None;
        }
        Some(optimizer)
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn adam_gpu_follows_cpu() {
        let (head, tail, eps) = clique_ring(40, 30);
        let Some(mut gpu) = adam_gpu_optimizer(head.clone(), tail.clone(), &eps) else {
            return;
        };
        let mut cpu = adam_optimizer(head, tail, &eps);
        pollster::block_on(cpu.step(1));
        pollster::block_on(gpu.step(1));
        let difference = cpu
            .embedding()
            .iter()
            .zip(gpu.embedding())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(difference < 1e-4, "first epoch differs by {difference}");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn adam_gpu_step_api() {
        let (head, tail, eps) = clique_ring(10, 6);
        let Some(mut opt) = adam_gpu_optimizer(head, tail, &eps) else {
            return;
        };
        let initial = opt.embedding().clone();
        opt.set_n_epochs(80);
        opt.set_good_initialization(true);
        pollster::block_on(opt.step(20));
        let mut params = opt.params();
        params.negative_sample_rate = 0.0;
        opt.set_params(params);
        pollster::block_on(opt.step(10));
        params.negative_sample_rate = 3.0;
        opt.set_params(params);
        pollster::block_on(opt.step(10));
        opt.set_optimizer(Optimizer::Sgd);
        assert!(opt.gpu_modern.is_none() && opt.gpu_sgd.is_some());
        pollster::block_on(opt.step(10));
        opt.set_optimizer(Optimizer::Adam);
        assert!(opt.gpu_modern.is_some() && opt.modern.is_none());
        pollster::block_on(opt.step(30));
        assert_eq!(opt.current_epoch(), 80);
        assert!(opt.embedding().iter().all(|x| x.is_finite()));
        let mut copy = vec![0.0; opt.embedding().len()];
        opt.copy_embedding_into(&mut copy);
        assert_eq!(copy.as_slice(), opt.embedding().as_slice().unwrap());

        opt.reinit_embedding(initial.clone());
        let mut logger = Logger::new(false, None, &[]);
        pollster::block_on(opt.run_to(50, &mut logger));
        let first = opt.embedding().clone();
        opt.reinit_embedding(initial);
        pollster::block_on(opt.run_to(50, &mut logger));
        assert_eq!(opt.current_epoch(), 50);
        assert_eq!(&first, opt.embedding());
    }

    /// (active edges, negative samples) of the next CPU SGD epoch.
    fn next_sgd_epoch_load(opt: &UmapOptimizer) -> (usize, usize) {
        let active = collect_active_edges(
            &opt.epochs_per_sample,
            &mut opt.epoch_of_next_sample.clone(),
            &mut opt.epoch_of_next_negative_sample.clone(),
            opt.current_epoch,
            opt.params.negative_sample_rate,
        );
        (active.len(), active.iter().map(|&(_, n)| n).sum())
    }

    #[test]
    fn switching_to_sgd_resumes_the_sgd_schedule() {
        let (head, tail, eps) = clique_ring(12, 8);
        let mut opt = adam_optimizer(head, tail, &eps);
        let (first_active, first_negatives) = next_sgd_epoch_load(&opt);
        pollster::block_on(opt.step(100));
        opt.set_optimizer(Optimizer::Sgd);
        let (active, negatives) = next_sgd_epoch_load(&opt);
        assert!(active <= first_active, "{active} active edges");
        assert!(
            negatives <= 2 * first_negatives.max(1),
            "{negatives} negatives"
        );

        let max_abs =
            |opt: &UmapOptimizer| opt.embedding().iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let before = max_abs(&opt);
        pollster::block_on(opt.step(1));
        assert!(max_abs(&opt) < 1.5 * before + 1.0);
    }

    fn realtime_optimizer(optimizer: Optimizer, gpu: bool) -> UmapOptimizer {
        let data = crate::layout_modern::tests::gaussian_blobs(700, 8, 6.0, 5);
        let builder = crate::Umap::builder(&data)
            .n_neighbors(15)
            .optimizer(optimizer)
            .init_method(crate::Init::Random)
            .random_state(5)
            .gpu(gpu);
        pollster::block_on(builder.setup_async()).unwrap().optimizer
    }

    /// RMS coordinate change relative to the RMS distance to the centroid.
    fn relative_movement(before: &Array2<f32>, after: &Array2<f32>) -> f64 {
        let n = after.nrows() as f64;
        let centroid = after.mean_axis(ndarray::Axis(0)).unwrap();
        let spread: f64 = after
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .zip(&centroid)
                    .map(|(x, c)| ((x - c) as f64).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n;
        let moved: f64 = before
            .iter()
            .zip(after)
            .map(|(b, a)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / n;
        (moved / spread).sqrt()
    }

    const OPTIMIZERS: [Optimizer; 3] = [Optimizer::Sgd, Optimizer::Momentum, Optimizer::Adam];

    /// Movement after a live minDist 0.1 → 0.8 and gamma 1 → 4 change, past the
    /// 500-epoch default horizon.
    fn force_change_response(optimizer: Optimizer, gpu: bool) -> f64 {
        let mut opt = realtime_optimizer(optimizer, gpu);
        pollster::block_on(opt.step(600));
        let before = opt.embedding().clone();
        let (a, b) = crate::find_ab_params(1.0, 0.8);
        opt.set_params(LiveParams {
            a: a as f32,
            b: b as f32,
            gamma: 4.0,
            ..opt.params()
        });
        pollster::block_on(opt.step(50));
        assert!(opt.embedding().iter().all(|x| x.is_finite()));
        relative_movement(&before, opt.embedding())
    }

    fn assert_step_contract(gpu: bool) {
        let movements: Vec<f64> = OPTIMIZERS
            .iter()
            .map(|&optimizer| force_change_response(optimizer, gpu))
            .collect();
        eprintln!("gpu {gpu}: relative movement sgd/momentum/adam {movements:?}");
        assert!(movements[1] >= 0.5 * movements[0], "{movements:?}");
        assert!(movements[2] >= 0.5 * movements[0], "{movements:?}");

        for optimizer in OPTIMIZERS {
            let mut opt = realtime_optimizer(optimizer, gpu);
            pollster::block_on(opt.step(30));
            opt.set_params(LiveParams {
                alpha: 0.0,
                ..opt.params()
            });
            let before = opt.embedding().clone();
            pollster::block_on(opt.step(20));
            let moved = before
                .iter()
                .zip(opt.embedding())
                .map(|(b, a)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                moved < 1e-4,
                "{optimizer:?} moved {moved} at learning rate 0"
            );

            let response = |alpha: f32| {
                let mut opt = realtime_optimizer(optimizer, gpu);
                pollster::block_on(opt.step(30));
                opt.set_params(LiveParams {
                    alpha,
                    ..opt.params()
                });
                let before = opt.embedding().clone();
                pollster::block_on(opt.step(10));
                relative_movement(&before, opt.embedding())
            };
            let (slow, fast) = (response(0.2), response(2.0));
            assert!(
                fast > slow,
                "{optimizer:?}: rate 2 moved {fast}, rate 0.2 moved {slow}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_failure_falls_back_to_the_cpu_layout() {
        for optimizer in [Optimizer::Adam, Optimizer::Momentum] {
            let mut opt = realtime_optimizer(optimizer, true);
            if opt.gpu_modern.is_none() {
                eprintln!("skipping: no GPU adapter");
                return;
            }
            pollster::block_on(opt.step(3));
            opt.gpu_modern.as_ref().unwrap().simulate_device_loss();
            let before = opt.embedding().clone();
            pollster::block_on(opt.step(5));
            assert_eq!(opt.current_epoch(), 8);
            assert!(opt.gpu_error().is_some() && opt.gpu_modern.is_none());
            assert!(opt.modern.is_some());
            assert!(relative_movement(&before, opt.embedding()) > 1e-3);
            opt.set_optimizer(optimizer);
            assert!(opt.gpu_modern.is_none() && opt.modern.is_some());
        }
    }

    #[test]
    fn step_holds_the_live_learning_rate() {
        assert_step_contract(false);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_step_holds_the_live_learning_rate() {
        if pollster::block_on(crate::gpu_optimize::request_device()).is_none() {
            eprintln!("skipping: no GPU adapter");
            return;
        }
        assert_step_contract(true);
    }

    #[test]
    fn adam_live_params_and_optimizer_switches_stay_finite() {
        let (head, tail, eps) = clique_ring(10, 6);
        let mut opt = adam_optimizer(head, tail, &eps);
        opt.set_n_epochs(80);
        opt.set_good_initialization(true);
        pollster::block_on(opt.step(20));
        let mut params = opt.params();
        params.negative_sample_rate = 0.0;
        params.gamma = 2.0;
        opt.set_params(params);
        pollster::block_on(opt.step(10));
        params.negative_sample_rate = 3.0;
        opt.set_params(params);
        pollster::block_on(opt.step(10));
        opt.set_optimizer(Optimizer::Sgd);
        pollster::block_on(opt.step(10));
        opt.set_optimizer(Optimizer::Adam);
        pollster::block_on(opt.step(100));
        assert_eq!(opt.current_epoch(), 150);
        assert!(opt.embedding().iter().all(|x| x.is_finite()));
        let mut copy = vec![0.0; opt.embedding().len()];
        opt.copy_embedding_into(&mut copy);
        assert_eq!(copy.as_slice(), opt.embedding().as_slice().unwrap());
    }
}
