use ndarray::Array2;
use wasm_bindgen::prelude::*;

use nndescent::rng::Xoshiro256StarStar;
use nndescent::NNDescent;
use umap::graph::CsrMatrix;
use umap::spectral::{noisy_scale_coords, random_layout, spectral_layout};
use umap::{find_ab_params, Init, Optimizer, Umap as UmapCore, UmapOptimizer};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_data(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Array2<f32>, JsError> {
    let expected_len = n_rows * n_cols;
    if data.len() != expected_len {
        return Err(JsError::new(&format!(
            "Data length {} does not match {} rows x {} cols = {}",
            data.len(),
            n_rows,
            n_cols,
            expected_len
        )));
    }
    Array2::from_shape_vec((n_rows, n_cols), data.to_vec())
        .map_err(|e| JsError::new(&e.to_string()))
}

fn parse_seed(random_state: i64) -> Option<u64> {
    if random_state >= 0 {
        Some(random_state as u64)
    } else {
        None
    }
}

/// Convert a JS callback `(progress: number, stage: string) => void` into a
/// Rust `ProgressCallback`.
fn js_to_progress_callback(f: js_sys::Function) -> nndescent::ProgressCallback {
    Box::new(move |progress: f32, stage: &str| {
        let _ = f.call2(
            &JsValue::NULL,
            &JsValue::from_f64(progress as f64),
            &JsValue::from_str(stage),
        );
    })
}

// ---------------------------------------------------------------------------
// UMAP
// ---------------------------------------------------------------------------

/// Builder for UMAP dimensionality reduction.
///
/// Usage (JS):
/// ```js
/// const result = new UMAPBuilder(data, 1000, 784, 2)
///   .metric("cosine")
///   .minDist(0.1)
///   .nNeighbors(15)
///   .randomState(42)
///   .build();
/// result.embedding   // Float32Array (n_rows * n_components)
/// result.nRows       // number
/// result.nComponents // number
/// ```
#[wasm_bindgen]
pub struct UMAPBuilder {
    data: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
    n_components: usize,
    // Optional parameters with defaults
    metric: String,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    learning_rate: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    local_connectivity: f32,
    set_op_mix_ratio: f32,
    init: Init,
    optimizer: Optimizer,
    random_state: Option<u64>,
    verbose: bool,
    gpu: bool,
    progress: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl UMAPBuilder {
    /// Create a new UMAP builder.
    ///
    /// @param data - Flat Float32Array, row-major (n_rows * n_cols).
    /// @param n_rows - Number of data points.
    /// @param n_cols - Number of input features.
    /// @param n_components - Target embedding dimensions (typically 2).
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], n_rows: usize, n_cols: usize, n_components: usize) -> UMAPBuilder {
        UMAPBuilder {
            data: data.to_vec(),
            n_rows,
            n_cols,
            n_components,
            metric: "euclidean".to_string(),
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            init: Init::Spectral,
            optimizer: Optimizer::Sgd,
            random_state: None,
            verbose: false,
            gpu: false,
            progress: None,
        }
    }

    /// Distance metric ("euclidean", "cosine", etc.). Default: "euclidean".
    pub fn metric(mut self, metric: &str) -> UMAPBuilder {
        self.metric = metric.to_string();
        self
    }

    /// Number of neighbors for graph construction. Default: 15.
    #[wasm_bindgen(js_name = "nNeighbors")]
    pub fn n_neighbors(mut self, n: usize) -> UMAPBuilder {
        self.n_neighbors = n;
        self
    }

    /// Minimum distance between points in embedding. Default: 0.1.
    #[wasm_bindgen(js_name = "minDist")]
    pub fn min_dist(mut self, d: f32) -> UMAPBuilder {
        self.min_dist = d;
        self
    }

    /// Effective scale of embedded points. Default: 1.0.
    pub fn spread(mut self, s: f32) -> UMAPBuilder {
        self.spread = s;
        self
    }

    /// Number of optimization epochs. Default: auto.
    #[wasm_bindgen(js_name = "nEpochs")]
    pub fn n_epochs(mut self, n: usize) -> UMAPBuilder {
        self.n_epochs = Some(n);
        self
    }

    /// Initial learning rate. Default: 1.0.
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, lr: f32) -> UMAPBuilder {
        self.learning_rate = lr;
        self
    }

    /// Negative samples per positive sample. Default: 5.
    #[wasm_bindgen(js_name = "negativeSampleRate")]
    pub fn negative_sample_rate(mut self, r: usize) -> UMAPBuilder {
        self.negative_sample_rate = r;
        self
    }

    /// Weight of repulsive force. Default: 1.0.
    #[wasm_bindgen(js_name = "repulsionStrength")]
    pub fn repulsion_strength(mut self, s: f32) -> UMAPBuilder {
        self.repulsion_strength = s;
        self
    }

    /// Local connectivity constraint. Default: 1.0.
    #[wasm_bindgen(js_name = "localConnectivity")]
    pub fn local_connectivity(mut self, c: f32) -> UMAPBuilder {
        self.local_connectivity = c;
        self
    }

    /// Interpolation between fuzzy union and intersection. Default: 1.0.
    #[wasm_bindgen(js_name = "mixRatio")]
    pub fn mix_ratio(mut self, r: f32) -> UMAPBuilder {
        self.set_op_mix_ratio = r;
        self
    }

    /// Initialization method: "spectral" or "random". Default: "spectral".
    /// (Momentum forces random init regardless; see the core builder.)
    #[wasm_bindgen(js_name = "initMethod")]
    pub fn init_method(mut self, method: &str) -> UMAPBuilder {
        self.init = match method {
            "random" => Init::Random,
            _ => Init::Spectral,
        };
        self
    }

    /// Optimizer: "sgd" (default) or "momentum".
    pub fn optimizer(mut self, optimizer: &str) -> UMAPBuilder {
        self.optimizer = match optimizer {
            "momentum" => Optimizer::Momentum,
            _ => Optimizer::Sgd,
        };
        self
    }

    /// Random seed for reproducibility. Default: None (random).
    #[wasm_bindgen(js_name = "randomState")]
    pub fn random_state(mut self, seed: i64) -> UMAPBuilder {
        self.random_state = parse_seed(seed);
        self
    }

    /// Enable verbose output. Default: false.
    pub fn verbose(mut self, v: bool) -> UMAPBuilder {
        self.verbose = v;
        self
    }

    /// Enable GPU acceleration via WebGPU. Default: false.
    /// Requires the `gpu` feature.
    pub fn gpu(mut self, g: bool) -> UMAPBuilder {
        self.gpu = g;
        self
    }

    /// Set a progress callback: `(progress: number, stage: string) => void`.
    /// `progress` is in [0, 1], `stage` describes the current processing phase.
    pub fn progress(mut self, callback: js_sys::Function) -> UMAPBuilder {
        self.progress = Some(callback);
        self
    }

    /// Build the graph + initial embedding (eager) and return a stateful [`Umap`].
    ///
    /// The embedding and kNN graph are valid immediately at epoch 0; call
    /// [`Umap::run`] to anneal to completion or [`Umap::step`] to advance
    /// interactively.
    pub async fn build(self) -> Result<Umap, JsError> {
        let array = parse_data(&self.data, self.n_rows, self.n_cols)?;

        let mut builder = UmapCore::builder(&array)
            .n_components(self.n_components)
            .n_neighbors(self.n_neighbors)
            .min_dist(self.min_dist)
            .spread(self.spread)
            .metric(&self.metric)
            .learning_rate(self.learning_rate)
            .negative_sample_rate(self.negative_sample_rate)
            .repulsion_strength(self.repulsion_strength)
            .local_connectivity(self.local_connectivity)
            .set_op_mix_ratio(self.set_op_mix_ratio)
            .init_method(self.init)
            .optimizer(self.optimizer)
            .gpu(self.gpu)
            .verbose(self.verbose);
        if let Some(n) = self.n_epochs {
            builder = builder.n_epochs(n);
        }
        if let Some(seed) = self.random_state {
            builder = builder.random_state(seed);
        }
        // Report progress during the (eager) setup using a clone of the callback;
        // the original is retained for run()/step().
        if let Some(cb) = &self.progress {
            builder = builder.progress(js_to_progress_callback(cb.clone()));
        }

        let setup = builder
            .setup_async()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let n_rows = setup.optimizer.embedding().nrows();
        let n_components = setup.optimizer.embedding().ncols();
        let n_neighbors_out = setup.knn_indices.ncols();

        Ok(Umap {
            optimizer: setup.optimizer,
            graph: setup.graph,
            knn_indices: setup.knn_indices.into_raw_vec_and_offset().0,
            knn_distances: setup.knn_distances.into_raw_vec_and_offset().0,
            n_neighbors_out,
            n_rows,
            n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            seed: self.random_state,
            n_epochs: setup.n_epochs,
            progress: self.progress,
            verbose: self.verbose,
        })
    }
}

/// A stateful, resumable UMAP. The embedding/kNN are valid immediately after
/// [`UMAPBuilder::build`] (eager setup); `run`/`step` advance the layout in place.
#[wasm_bindgen]
pub struct Umap {
    optimizer: UmapOptimizer,
    /// Pruned fuzzy graph, retained for spectral re-initialization in `reset`.
    graph: CsrMatrix,
    knn_indices: Vec<i32>,
    knn_distances: Vec<f32>,
    n_neighbors_out: usize,
    n_rows: usize,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    seed: Option<u64>,
    n_epochs: usize,
    progress: Option<js_sys::Function>,
    verbose: bool,
}

#[wasm_bindgen]
impl Umap {
    /// Anneal the layout to completion: linear learning-rate decay from the
    /// current `learningRate` down to 0 over the default horizon, continuing from
    /// the current epoch. Reports progress over the optimization stage.
    pub async fn run(&mut self) -> Result<(), JsError> {
        let progress = self.progress.clone().map(js_to_progress_callback);
        let mut logger = umap::Logger::new(self.verbose, progress, umap::UMAP_STAGES);
        logger.push_stage_with_message(umap::STAGE_OPTIMIZATION, "Optimizing layout...");
        self.optimizer.run_to(self.n_epochs, &mut logger).await;
        logger.pop_stage();
        Ok(())
    }

    /// Advance the layout by `n_epochs` at the current learning rate (no decay) —
    /// intended for interactive/real-time stepping.
    pub async fn step(&mut self, n_epochs: usize) {
        self.optimizer.step(n_epochs).await;
    }

    /// Update live parameters. `min_dist`/`spread` re-fit the a/b curve, but only
    /// when their (clamped) values actually change — callers typically re-send the
    /// merged parameter set on every partial update, and the fit is not cheap.
    /// `spread` is clamped to `>= min_dist` to avoid degenerate curves. Any argument
    /// may be `null`/`undefined` to leave it unchanged.
    #[wasm_bindgen(js_name = "setParameters")]
    pub fn set_parameters(
        &mut self,
        learning_rate: Option<f32>,
        repulsion_strength: Option<f32>,
        negative_sample_rate: Option<f32>,
        min_dist: Option<f32>,
        spread: Option<f32>,
    ) {
        let mut p = self.optimizer.params();
        if let Some(lr) = learning_rate {
            p.alpha = lr;
        }
        if let Some(rs) = repulsion_strength {
            p.gamma = rs;
        }
        if let Some(nsr) = negative_sample_rate {
            p.negative_sample_rate = nsr;
        }
        if min_dist.is_some() || spread.is_some() {
            let (prev_min_dist, prev_spread) = (self.min_dist, self.spread);
            if let Some(md) = min_dist {
                self.min_dist = md;
            }
            if let Some(sp) = spread {
                self.spread = sp;
            }
            // Clamp spread >= min_dist (low spread triggers an a≈0 explosion /
            // a≈830 collapse in find_ab_params).
            if self.spread < self.min_dist {
                self.spread = self.min_dist;
            }
            // Re-fit the a/b curve only if the clamped values changed. `p` already
            // carries the current a/b, so an unchanged curve is left as-is — this
            // skips a redundant find_ab_params on every partial setParameters.
            if self.min_dist != prev_min_dist || self.spread != prev_spread {
                let (a, b) = find_ab_params(self.spread, self.min_dist);
                p.a = a as f32;
                p.b = b as f32;
            }
        }
        self.optimizer.set_params(p);
    }

    /// Switch optimizer: "sgd" or "momentum". Resets the velocity buffer.
    #[wasm_bindgen(js_name = "setOptimizer")]
    pub fn set_optimizer(&mut self, optimizer: &str) {
        let opt = match optimizer {
            "momentum" => Optimizer::Momentum,
            _ => Optimizer::Sgd,
        };
        self.optimizer.set_optimizer(opt);
    }

    /// Re-initialize the embedding and restart the schedule from epoch 0.
    /// `method` is "spectral" (uses the retained graph) or "random".
    pub fn reset(&mut self, method: &str) {
        let mut rng = match self.seed {
            Some(s) => Xoshiro256StarStar::seed_from_u64(s),
            None => Xoshiro256StarStar::seed_from_os(),
        };
        let embedding = match method {
            "spectral" => {
                let mut emb = spectral_layout(&self.graph, self.n_components, &mut rng);
                noisy_scale_coords(&mut emb, &mut rng, 10.0, 0.0001);
                emb
            }
            _ => random_layout(self.n_rows, self.n_components, &mut rng),
        };
        self.optimizer.reinit_embedding(embedding);
    }

    /// Copy the current embedding into `out` (length n_rows * n_components). The
    /// JS-side typed array is updated in place on return.
    #[wasm_bindgen(js_name = "copyEmbeddingInto")]
    pub fn copy_embedding_into(&self, out: &mut [f32]) {
        self.optimizer.copy_embedding_into(out);
    }

    /// Embedding coordinates as a flat Float32Array (row-major).
    #[wasm_bindgen(getter)]
    pub fn embedding(&self) -> Vec<f32> {
        self.optimizer
            .embedding()
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default()
    }

    /// KNN indices as a flat Int32Array (row-major, n_rows x n_neighbors).
    #[wasm_bindgen(getter, js_name = "knnIndices")]
    pub fn knn_indices(&self) -> Vec<i32> {
        self.knn_indices.clone()
    }

    /// KNN distances as a flat Float32Array (row-major, n_rows x n_neighbors).
    #[wasm_bindgen(getter, js_name = "knnDistances")]
    pub fn knn_distances(&self) -> Vec<f32> {
        self.knn_distances.clone()
    }

    /// Number of data points.
    #[wasm_bindgen(getter, js_name = "nRows")]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of embedding dimensions.
    #[wasm_bindgen(getter, js_name = "nComponents")]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Number of neighbors per point in the kNN graph.
    #[wasm_bindgen(getter, js_name = "nNeighbors")]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors_out
    }

    /// The number of epochs run so far.
    #[wasm_bindgen(getter)]
    pub fn epoch(&self) -> f64 {
        self.optimizer.current_epoch() as f64
    }
}

// ---------------------------------------------------------------------------
// UMAP from a precomputed kNN graph
// ---------------------------------------------------------------------------

/// Builder for UMAP from a precomputed kNN graph (skips NNDescent — no
/// high-dimensional data needed).
///
/// Usage (JS):
/// ```js
/// const result = new UMAPFromKnnBuilder(knnIndices, knnDistances, 1000, 2)
///   .minDist(0.1)
///   .randomState(42)
///   .build();
/// ```
///
/// `knnIndices`/`knnDistances` are row-major (n_rows * k); row i lists the
/// neighbors of point i sorted by ascending distance. The point itself may appear
/// in column 0 (distance 0) or be omitted — detected and normalized per row.
///
/// The caller must supply a well-formed graph (not validated beyond an index-range
/// check): every index in `[0, n_rows)`, at most one self-entry per row, finite
/// distances (no `NaN`), and a uniform real-neighbor count across rows. Violations
/// degrade silently — a `NaN` distance yields a `NaN` embedding, and one short row
/// trims every row to its length.
#[wasm_bindgen]
pub struct UMAPFromKnnBuilder {
    knn_indices: Vec<i32>,
    knn_distances: Vec<f32>,
    n_rows: usize,
    n_components: usize,
    // Optional parameters with defaults (no metric / n_neighbors: k comes from the graph)
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    learning_rate: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    local_connectivity: f32,
    set_op_mix_ratio: f32,
    init: Init,
    optimizer: Optimizer,
    random_state: Option<u64>,
    verbose: bool,
    gpu: bool,
    progress: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl UMAPFromKnnBuilder {
    /// Create a new builder from a precomputed kNN graph.
    ///
    /// @param knn_indices - Flat Int32Array, row-major (n_rows * k).
    /// @param knn_distances - Flat Float32Array, row-major (n_rows * k).
    /// @param n_rows - Number of data points.
    /// @param n_components - Target embedding dimensions (typically 2).
    #[wasm_bindgen(constructor)]
    pub fn new(
        knn_indices: &[i32],
        knn_distances: &[f32],
        n_rows: usize,
        n_components: usize,
    ) -> UMAPFromKnnBuilder {
        UMAPFromKnnBuilder {
            knn_indices: knn_indices.to_vec(),
            knn_distances: knn_distances.to_vec(),
            n_rows,
            n_components,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            init: Init::Spectral,
            optimizer: Optimizer::Sgd,
            random_state: None,
            verbose: false,
            gpu: false,
            progress: None,
        }
    }

    /// Minimum distance between points in embedding. Default: 0.1.
    #[wasm_bindgen(js_name = "minDist")]
    pub fn min_dist(mut self, d: f32) -> UMAPFromKnnBuilder {
        self.min_dist = d;
        self
    }

    /// Effective scale of embedded points. Default: 1.0.
    pub fn spread(mut self, s: f32) -> UMAPFromKnnBuilder {
        self.spread = s;
        self
    }

    /// Number of optimization epochs. Default: auto.
    #[wasm_bindgen(js_name = "nEpochs")]
    pub fn n_epochs(mut self, n: usize) -> UMAPFromKnnBuilder {
        self.n_epochs = Some(n);
        self
    }

    /// Initial learning rate. Default: 1.0.
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, lr: f32) -> UMAPFromKnnBuilder {
        self.learning_rate = lr;
        self
    }

    /// Negative samples per positive sample. Default: 5.
    #[wasm_bindgen(js_name = "negativeSampleRate")]
    pub fn negative_sample_rate(mut self, r: usize) -> UMAPFromKnnBuilder {
        self.negative_sample_rate = r;
        self
    }

    /// Weight of repulsive force. Default: 1.0.
    #[wasm_bindgen(js_name = "repulsionStrength")]
    pub fn repulsion_strength(mut self, s: f32) -> UMAPFromKnnBuilder {
        self.repulsion_strength = s;
        self
    }

    /// Local connectivity constraint. Default: 1.0.
    #[wasm_bindgen(js_name = "localConnectivity")]
    pub fn local_connectivity(mut self, c: f32) -> UMAPFromKnnBuilder {
        self.local_connectivity = c;
        self
    }

    /// Interpolation between fuzzy union and intersection. Default: 1.0.
    #[wasm_bindgen(js_name = "mixRatio")]
    pub fn mix_ratio(mut self, r: f32) -> UMAPFromKnnBuilder {
        self.set_op_mix_ratio = r;
        self
    }

    /// Initialization method: "spectral" or "random". Default: "spectral".
    #[wasm_bindgen(js_name = "initMethod")]
    pub fn init_method(mut self, method: &str) -> UMAPFromKnnBuilder {
        self.init = match method {
            "random" => Init::Random,
            _ => Init::Spectral,
        };
        self
    }

    /// Optimizer: "sgd" (default) or "momentum".
    pub fn optimizer(mut self, optimizer: &str) -> UMAPFromKnnBuilder {
        self.optimizer = match optimizer {
            "momentum" => Optimizer::Momentum,
            _ => Optimizer::Sgd,
        };
        self
    }

    /// Random seed for reproducibility. Default: None (random).
    #[wasm_bindgen(js_name = "randomState")]
    pub fn random_state(mut self, seed: i64) -> UMAPFromKnnBuilder {
        self.random_state = parse_seed(seed);
        self
    }

    /// Enable verbose output. Default: false.
    pub fn verbose(mut self, v: bool) -> UMAPFromKnnBuilder {
        self.verbose = v;
        self
    }

    /// Enable GPU acceleration via WebGPU for the layout optimization. Default: false.
    pub fn gpu(mut self, g: bool) -> UMAPFromKnnBuilder {
        self.gpu = g;
        self
    }

    /// Set a progress callback: `(progress: number, stage: string) => void`.
    pub fn progress(mut self, callback: js_sys::Function) -> UMAPFromKnnBuilder {
        self.progress = Some(callback);
        self
    }

    /// Build the graph + initial embedding (eager) and return a stateful [`Umap`].
    pub async fn build(self) -> Result<Umap, JsError> {
        if self.n_rows == 0 {
            return Err(JsError::new("nRows must be > 0"));
        }
        if self.knn_indices.len() != self.knn_distances.len() {
            return Err(JsError::new(
                "knnIndices and knnDistances must have equal length",
            ));
        }
        if self.knn_indices.len() % self.n_rows != 0 {
            return Err(JsError::new(&format!(
                "knn length {} is not divisible by nRows {}",
                self.knn_indices.len(),
                self.n_rows
            )));
        }
        let k = self.knn_indices.len() / self.n_rows;
        if k < 1 {
            return Err(JsError::new("each point must have at least one neighbor"));
        }

        let idx = Array2::from_shape_vec((self.n_rows, k), self.knn_indices)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let dist = Array2::from_shape_vec((self.n_rows, k), self.knn_distances)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let mut builder = UmapCore::builder_from_knn(idx, dist)
            .n_components(self.n_components)
            .min_dist(self.min_dist)
            .spread(self.spread)
            .learning_rate(self.learning_rate)
            .negative_sample_rate(self.negative_sample_rate)
            .repulsion_strength(self.repulsion_strength)
            .local_connectivity(self.local_connectivity)
            .set_op_mix_ratio(self.set_op_mix_ratio)
            .init_method(self.init)
            .optimizer(self.optimizer)
            .gpu(self.gpu)
            .verbose(self.verbose);
        if let Some(n) = self.n_epochs {
            builder = builder.n_epochs(n);
        }
        if let Some(seed) = self.random_state {
            builder = builder.random_state(seed);
        }
        if let Some(cb) = &self.progress {
            builder = builder.progress(js_to_progress_callback(cb.clone()));
        }

        let setup = builder
            .setup_async()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let n_rows = setup.optimizer.embedding().nrows();
        let n_components = setup.optimizer.embedding().ncols();
        let n_neighbors_out = setup.knn_indices.ncols();

        Ok(Umap {
            optimizer: setup.optimizer,
            graph: setup.graph,
            knn_indices: setup.knn_indices.into_raw_vec_and_offset().0,
            knn_distances: setup.knn_distances.into_raw_vec_and_offset().0,
            n_neighbors_out,
            n_rows,
            n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            seed: self.random_state,
            n_epochs: setup.n_epochs,
            progress: self.progress,
            verbose: self.verbose,
        })
    }
}

// ---------------------------------------------------------------------------
// NNDescent
// ---------------------------------------------------------------------------

/// Builder for NNDescent approximate nearest neighbor index.
///
/// Usage (JS):
/// ```js
/// const index = new NNDescentBuilder(data, 1000, 784, "euclidean", 15)
///   .randomState(42)
///   .nTrees(10)
///   .build();
/// const graph = index.neighborGraph();
/// graph.indices     // Int32Array
/// graph.distances   // Float32Array
/// ```
#[wasm_bindgen]
pub struct NNDescentBuilder {
    data: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
    metric: String,
    n_neighbors: usize,
    // Optional parameters
    random_state: Option<u64>,
    n_trees: Option<usize>,
    n_iters: Option<usize>,
    delta: f32,
    tree_init: bool,
    max_rptree_depth: usize,
    diversify_prob: f32,
    pruning_degree_multiplier: f32,
    verbose: bool,
    gpu: bool,
    progress: Option<nndescent::ProgressCallback>,
}

#[wasm_bindgen]
impl NNDescentBuilder {
    /// Create a new NNDescent builder.
    ///
    /// @param data - Flat Float32Array, row-major (n_rows * n_cols).
    /// @param n_rows - Number of data points.
    /// @param n_cols - Number of features per point.
    /// @param metric - Distance metric ("euclidean", "cosine", etc.).
    /// @param n_neighbors - Number of neighbors to find.
    #[wasm_bindgen(constructor)]
    pub fn new(
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        metric: &str,
        n_neighbors: usize,
    ) -> NNDescentBuilder {
        NNDescentBuilder {
            data: data.to_vec(),
            n_rows,
            n_cols,
            metric: metric.to_string(),
            n_neighbors,
            random_state: None,
            n_trees: None,
            n_iters: None,
            delta: 0.001,
            tree_init: true,
            max_rptree_depth: 200,
            diversify_prob: 1.0,
            pruning_degree_multiplier: 1.5,
            verbose: false,
            gpu: false,
            progress: None,
        }
    }

    /// Random seed for reproducibility. Default: None (random).
    #[wasm_bindgen(js_name = "randomState")]
    pub fn random_state(mut self, seed: i64) -> NNDescentBuilder {
        self.random_state = parse_seed(seed);
        self
    }

    /// Number of random projection trees. Default: auto.
    #[wasm_bindgen(js_name = "nTrees")]
    pub fn n_trees(mut self, n: usize) -> NNDescentBuilder {
        self.n_trees = Some(n);
        self
    }

    /// Number of NN-descent iterations. Default: auto.
    #[wasm_bindgen(js_name = "nIters")]
    pub fn n_iters(mut self, n: usize) -> NNDescentBuilder {
        self.n_iters = Some(n);
        self
    }

    /// Early stopping threshold. Default: 0.001.
    pub fn delta(mut self, d: f32) -> NNDescentBuilder {
        self.delta = d;
        self
    }

    /// Whether to use RP tree initialization. Default: true.
    #[wasm_bindgen(js_name = "treeInit")]
    pub fn tree_init(mut self, t: bool) -> NNDescentBuilder {
        self.tree_init = t;
        self
    }

    /// Maximum RP tree depth. Default: 200.
    #[wasm_bindgen(js_name = "maxRptreeDepth")]
    pub fn max_rptree_depth(mut self, d: usize) -> NNDescentBuilder {
        self.max_rptree_depth = d;
        self
    }

    /// Probability of pruning during diversification. Default: 1.0.
    #[wasm_bindgen(js_name = "diversifyProb")]
    pub fn diversify_prob(mut self, p: f32) -> NNDescentBuilder {
        self.diversify_prob = p;
        self
    }

    /// Pruning degree multiplier. Default: 1.5.
    #[wasm_bindgen(js_name = "pruningDegreeMultiplier")]
    pub fn pruning_degree_multiplier(mut self, m: f32) -> NNDescentBuilder {
        self.pruning_degree_multiplier = m;
        self
    }

    /// Enable verbose output. Default: false.
    pub fn verbose(mut self, v: bool) -> NNDescentBuilder {
        self.verbose = v;
        self
    }

    /// Enable GPU acceleration via WebGPU. Default: false.
    /// Requires the `gpu` feature.
    pub fn gpu(mut self, g: bool) -> NNDescentBuilder {
        self.gpu = g;
        self
    }

    /// Set a progress callback: `(progress: number, stage: string) => void`.
    /// `progress` is in [0, 1], `stage` describes the current processing phase.
    pub fn progress(mut self, callback: js_sys::Function) -> NNDescentBuilder {
        self.progress = Some(js_to_progress_callback(callback));
        self
    }

    /// Build the NNDescent index.
    pub async fn build(self) -> Result<NNDescentIndex, JsError> {
        let array = parse_data(&self.data, self.n_rows, self.n_cols)?;

        let mut builder = NNDescent::builder(array, &self.metric, self.n_neighbors)
            .delta(self.delta)
            .tree_init(self.tree_init)
            .max_rptree_depth(self.max_rptree_depth)
            .diversify_prob(self.diversify_prob)
            .pruning_degree_multiplier(self.pruning_degree_multiplier)
            .gpu(self.gpu)
            .verbose(self.verbose);
        if let Some(n) = self.n_trees {
            builder = builder.n_trees(n);
        }
        if let Some(n) = self.n_iters {
            builder = builder.n_iters(n);
        }
        if let Some(seed) = self.random_state {
            builder = builder.random_state(seed);
        }
        if let Some(cb) = self.progress {
            builder = builder.progress(cb);
        }

        Ok(NNDescentIndex {
            inner: builder
                .build_async()
                .await
                .map_err(|e| JsError::new(&e.to_string()))?,
        })
    }
}

/// An NNDescent nearest neighbor index.
#[wasm_bindgen]
pub struct NNDescentIndex {
    inner: NNDescent,
}

#[wasm_bindgen]
impl NNDescentIndex {
    /// Get the neighbor graph (indices + distances).
    #[wasm_bindgen(js_name = "neighborGraph")]
    pub fn neighbor_graph(&self) -> Result<NeighborResult, JsError> {
        let (indices, distances) = self
            .inner
            .neighbor_graph()
            .ok_or_else(|| JsError::new("Neighbor graph not built yet"))?;

        let n_rows = indices.nrows();
        let n_cols = indices.ncols();

        Ok(NeighborResult {
            indices: indices.into_raw_vec_and_offset().0,
            distances: distances.into_raw_vec_and_offset().0,
            n_rows,
            n_cols,
        })
    }

    /// Prepare the search index for querying.
    pub fn prepare(&mut self) {
        self.inner.prepare();
    }

    /// Query the index for k nearest neighbors.
    ///
    /// @param query_data - Flat Float32Array (n_queries * n_features), row-major.
    /// @param n_queries - Number of query points.
    /// @param n_features - Number of features (must match training data).
    /// @param k - Number of neighbors to return.
    /// @param epsilon - Accuracy/speed tradeoff (higher = more accurate).
    pub fn query(
        &mut self,
        query_data: &[f32],
        n_queries: usize,
        n_features: usize,
        k: usize,
        epsilon: f32,
    ) -> Result<NeighborResult, JsError> {
        let array = parse_data(query_data, n_queries, n_features)?;
        let (indices, distances) = self.inner.query(&array, k, epsilon);
        let n_rows = indices.nrows();
        let n_cols = indices.ncols();

        Ok(NeighborResult {
            indices: indices.into_raw_vec_and_offset().0,
            distances: distances.into_raw_vec_and_offset().0,
            n_rows,
            n_cols,
        })
    }
}

/// Result of a neighbor graph or query operation.
#[wasm_bindgen]
pub struct NeighborResult {
    indices: Vec<i32>,
    distances: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
}

#[wasm_bindgen]
impl NeighborResult {
    /// Neighbor indices as a flat Int32Array (row-major, n_rows x n_cols).
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<i32> {
        self.indices.clone()
    }

    /// Neighbor distances as a flat Float32Array (row-major, n_rows x n_cols).
    #[wasm_bindgen(getter)]
    pub fn distances(&self) -> Vec<f32> {
        self.distances.clone()
    }

    /// Number of rows (points).
    #[wasm_bindgen(getter, js_name = "nRows")]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns (neighbors per point).
    #[wasm_bindgen(getter, js_name = "nCols")]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
}
