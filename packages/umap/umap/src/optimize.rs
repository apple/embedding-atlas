use ndarray::Array2;
use nndescent::rng::TauRng;
use nndescent::Logger;
use rayon::prelude::*;

/// Which optimizer to use for the layout SGD.
///
/// Both share one β-parameterized velocity-SGD kernel: `Sgd` uses β=0 (the
/// reference UMAP update, no velocity buffer), `Momentum` uses β=[`MOMENTUM_BETA`]
/// and a persistent velocity buffer (better global structure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Optimizer {
    /// Reference-faithful incremental SGD (β=0). Default.
    Sgd,
    /// Velocity-SGD with classical momentum (β=[`MOMENTUM_BETA`]).
    Momentum,
}

/// Fixed momentum coefficient (classical, not Nesterov). Not exposed live — it
/// couples with the learning rate and blows up the layout scale as β→1.
pub(crate) const MOMENTUM_BETA: f32 = 0.9;

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
fn rdist(x: &[f32], y: &[f32]) -> f32 {
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

/// Run one epoch of Hogwild velocity-SGD in place.
///
/// `beta == 0.0` ⇒ plain SGD (the reference UMAP update; `velocity` is unused and
/// may be empty). `beta > 0.0` ⇒ classical momentum using `velocity`
/// (`len == n_vertices * dim`).
///
/// The gradient math (attractive `-2ab·d^{2b}/d² / (a·d^{2b}+1)`, repulsive
/// `2γb/((0.001+d²)(a·d^{2b}+1))`, per-edge clip to ±4) is identical to the
/// reference implementation. **Sign convention:** `g` is the SGD *position delta*
/// (`x += α·g`); velocity-SGD is `v = β·v + g; x += α·v`, so `β=0` reduces to
/// `x += α·g` exactly — no negation anywhere.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_one_epoch_hogwild(
    emb: &mut [f32],
    velocity: &mut [f32],
    dim: usize,
    n_vertices: usize,
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f64],
    epoch_of_next_sample: &mut [f64],
    epoch_of_next_negative_sample: &mut [f64],
    current_epoch: u64,
    params: LiveParams,
    beta: f32,
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
    let use_momentum = beta > 0.0;

    // SAFETY: Hogwild-style parallel SGD (Recht et al., 2011). The embedding array
    // (and, for momentum, the velocity array) is shared mutably across threads
    // WITHOUT synchronization. This is intentional:
    // - Each gradient update touches only `dim` floats (typically 2-4) per vertex.
    // - Concurrent writes to the same vertex produce slightly stale/torn reads,
    //   which act as additional stochastic noise and do not affect convergence in
    //   practice. The velocity buffer is touched only at the same (vertex,
    //   component) slots as the embedding, so it races identically and benignly;
    //   `clip` bounds `g` and α decays, so β·v cannot amplify the divergence.
    // - This matches reference Python UMAP (numba prange) and is standard practice.
    //
    // Formally, multiple `&mut` slices to overlapping memory is UB under Rust's
    // aliasing model; in practice the generated code is safe because all accesses
    // are simple f32 loads/stores, and the primary target is single-threaded WASM.
    let emb_ptr = emb.as_mut_ptr() as usize;
    let vel_ptr = if use_momentum {
        velocity.as_mut_ptr() as usize
    } else {
        0
    };

    // Phase 2: apply gradients in parallel (Hogwild). Each edge gets its own RNG
    // seeded deterministically from (rng_state, edge_index, epoch).
    active_edges.par_iter().for_each_init(
        || (vec![0.0f32; dim], vec![0.0f32; dim], vec![0.0f32; dim]),
        |(current, other, neg_other), &(edge_idx, n_neg)| {
            // SAFETY: see Hogwild comment above on emb_ptr / vel_ptr.
            let emb =
                unsafe { std::slice::from_raw_parts_mut(emb_ptr as *mut f32, n_vertices * dim) };
            let vel: &mut [f32] = if use_momentum {
                unsafe { std::slice::from_raw_parts_mut(vel_ptr as *mut f32, n_vertices * dim) }
            } else {
                &mut []
            };

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
                if use_momentum {
                    let vj = beta * vel[j_off + d] + g;
                    vel[j_off + d] = vj;
                    emb[j_off + d] += alpha * vj;
                    let vk = beta * vel[k_off + d] - g;
                    vel[k_off + d] = vk;
                    emb[k_off + d] += alpha * vk;
                } else {
                    emb[j_off + d] += alpha * g;
                    emb[k_off + d] -= alpha * g;
                }
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
                    if use_momentum {
                        let vj = beta * vel[j_off + d] + g;
                        vel[j_off + d] = vj;
                        emb[j_off + d] += alpha * vj;
                    } else {
                        emb[j_off + d] += alpha * g;
                    }
                }
            }
        },
    );
}

/// A resumable UMAP layout optimizer.
///
/// Advances the SGD layout `k` epochs at a time ([`step`](Self::step)), or runs
/// to a fixed horizon with learning-rate decay ([`run_to`](Self::run_to), which
/// reproduces standard run-to-completion UMAP). Parameters and the optimizer can
/// be changed live between steps, and the embedding can be re-initialized
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
    /// Velocity buffer for Momentum; empty for Sgd (no overhead).
    velocity: Vec<f32>,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu_optimize::OptimizeGpuContext>,
    /// f32-counter coordinate for the GPU schedule; reset to 0 on rebase.
    #[cfg(feature = "gpu")]
    gpu_epoch: u32,
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
        let n_vertices = embedding.nrows();
        let dim = embedding.ncols();

        let eps: Vec<f64> = epochs_per_sample.iter().map(|&e| e as f64).collect();
        let epoch_of_next_sample = eps.clone();
        let rate = params.negative_sample_rate as f64;
        let epoch_of_next_negative_sample: Vec<f64> = eps
            .iter()
            .map(|&e| if rate > 0.0 { e / rate } else { f64::INFINITY })
            .collect();

        let velocity = match optimizer {
            Optimizer::Momentum => vec![0.0f32; n_vertices * dim],
            Optimizer::Sgd => Vec::new(),
        };

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
            velocity,
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_epoch: 0,
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
        let mut opt = Self::new(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
        );

        // Owned, GPU-friendly copies so no borrow of `opt` is held across the await.
        let emb_vec: Vec<f32> = opt.embedding.as_slice().expect("contiguous").to_vec();
        let eps_f32: Vec<f32> = opt.epochs_per_sample.iter().map(|&e| e as f32).collect();
        let head_u32: Vec<u32> = opt.head.iter().map(|&h| h as u32).collect();
        let tail_u32: Vec<u32> = opt.tail.iter().map(|&t| t as u32).collect();

        opt.gpu = crate::gpu_optimize::OptimizeGpuContext::new(
            &emb_vec,
            &head_u32,
            &tail_u32,
            &eps_f32,
            opt.n_vertices as u32,
            opt.dim as u32,
            n_neighbors,
            params.negative_sample_rate,
            rng_state,
        )
        .await;

        opt
    }

    #[inline]
    fn beta(&self) -> f32 {
        match self.optimizer {
            Optimizer::Sgd => 0.0,
            Optimizer::Momentum => MOMENTUM_BETA,
        }
    }

    /// Run one CPU epoch in place at the current parameters.
    fn run_cpu_epoch(&mut self, params: LiveParams, beta: f32) {
        let emb = self
            .embedding
            .as_slice_memory_order_mut()
            .expect("embedding not contiguous");
        run_one_epoch_hogwild(
            emb,
            &mut self.velocity,
            self.dim,
            self.n_vertices,
            &self.head,
            &self.tail,
            &self.epochs_per_sample,
            &mut self.epoch_of_next_sample,
            &mut self.epoch_of_next_negative_sample,
            self.current_epoch,
            params,
            beta,
            self.rng_state,
        );
    }

    /// Advance the layout by `n_epochs` at the current learning rate (no decay).
    /// Intended for interactive/real-time stepping.
    pub async fn step(&mut self, n_epochs: usize) {
        let beta = self.beta();

        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            for _ in 0..n_epochs {
                self.gpu.as_ref().unwrap().run_epoch(
                    self.gpu_epoch,
                    self.params,
                    beta,
                    self.optimizer,
                );
                self.current_epoch += 1;
                self.gpu_epoch += 1;
                if self.gpu_epoch >= (1u32 << 18) {
                    self.gpu.as_ref().unwrap().rebase(self.gpu_epoch);
                    self.gpu_epoch = 0;
                }
            }
            self.download_from_gpu().await;
            return;
        }

        for _ in 0..n_epochs {
            self.run_cpu_epoch(self.params, beta);
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
    pub async fn run_to(&mut self, n_epochs: usize, logger: &mut Logger) {
        let beta = self.beta();
        let initial_alpha = self.params.alpha;

        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            while self.current_epoch < n_epochs as u64 {
                let epoch = self.current_epoch;
                let mut p = self.params;
                p.alpha = (initial_alpha * (1.0 - epoch as f32 / n_epochs as f32)).max(0.0);
                self.gpu
                    .as_ref()
                    .unwrap()
                    .run_epoch(self.gpu_epoch, p, beta, self.optimizer);
                self.current_epoch += 1;
                self.gpu_epoch += 1;
                if self.gpu_epoch >= (1u32 << 18) {
                    self.gpu.as_ref().unwrap().rebase(self.gpu_epoch);
                    self.gpu_epoch = 0;
                }
                Self::log_progress(logger, epoch, n_epochs);
            }
            self.download_from_gpu().await;
            return;
        }

        while self.current_epoch < n_epochs as u64 {
            let epoch = self.current_epoch;
            let mut p = self.params;
            p.alpha = (initial_alpha * (1.0 - epoch as f32 / n_epochs as f32)).max(0.0);
            self.run_cpu_epoch(p, beta);
            self.current_epoch += 1;
            Self::log_progress(logger, epoch, n_epochs);
        }
    }

    fn log_progress(logger: &mut Logger, epoch: u64, n_epochs: usize) {
        if n_epochs >= 10 && (epoch as usize) % (n_epochs / 10) == 0 {
            logger.log(&format!("completed {} / {} epochs", epoch, n_epochs));
        }
        logger.stage_progress((epoch + 1) as f64 / n_epochs as f64, None);
    }

    #[cfg(feature = "gpu")]
    async fn download_from_gpu(&mut self) {
        let result = self.gpu.as_ref().unwrap().download_embedding().await;
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
    pub fn set_params(&mut self, params: LiveParams) {
        let rate_changed = params.negative_sample_rate != self.params.negative_sample_rate;
        self.params = params;
        if rate_changed {
            let from = self.current_epoch as f64;
            self.reseed_negative_schedule(from);
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &self.gpu {
                gpu.reseed_negative_schedule(params.negative_sample_rate, self.gpu_epoch);
            }
        }
    }

    /// Switch optimizer. Resets the velocity (zeroed for Momentum, freed for Sgd).
    pub fn set_optimizer(&mut self, optimizer: Optimizer) {
        self.optimizer = optimizer;
        match optimizer {
            Optimizer::Momentum => {
                if self.velocity.len() == self.n_vertices * self.dim {
                    self.velocity.iter_mut().for_each(|v| *v = 0.0);
                } else {
                    self.velocity = vec![0.0f32; self.n_vertices * self.dim];
                }
            }
            Optimizer::Sgd => self.velocity = Vec::new(),
        }
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            gpu.zero_velocity();
        }
    }

    /// Replace the embedding and reset the schedule, epoch, and velocity — i.e. a
    /// full restart of the layout optimization from a new initial position.
    pub fn reinit_embedding(&mut self, embedding: Array2<f32>) {
        assert_eq!(embedding.nrows(), self.n_vertices);
        assert_eq!(embedding.ncols(), self.dim);
        self.embedding = embedding;

        self.epoch_of_next_sample
            .copy_from_slice(&self.epochs_per_sample);
        self.reseed_negative_schedule(0.0);
        self.current_epoch = 0;
        self.velocity.iter_mut().for_each(|v| *v = 0.0);

        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            let emb_vec: Vec<f32> = self.embedding.as_slice().expect("contiguous").to_vec();
            gpu.reupload_embedding(&emb_vec);
            gpu.reset_schedule(self.params.negative_sample_rate);
            gpu.zero_velocity();
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
