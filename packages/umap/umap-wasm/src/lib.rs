use ndarray::Array2;
use wasm_bindgen::prelude::*;

use nndescent::NNDescent;
use umap::{Init, Umap};

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
    random_state: Option<u64>,
    verbose: bool,
    gpu: bool,
    progress: Option<nndescent::ProgressCallback>,
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
    #[wasm_bindgen(js_name = "initMethod")]
    pub fn init_method(mut self, method: &str) -> UMAPBuilder {
        self.init = match method {
            "random" => Init::Random,
            _ => Init::Spectral,
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
        self.progress = Some(js_to_progress_callback(callback));
        self
    }

    /// Run UMAP and return the embedding result.
    pub async fn build(self) -> Result<UMAPResult, JsError> {
        let array = parse_data(&self.data, self.n_rows, self.n_cols)?;

        let mut builder = Umap::builder(&array)
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
            .gpu(self.gpu)
            .verbose(self.verbose);
        if let Some(n) = self.n_epochs {
            builder = builder.n_epochs(n);
        }
        if let Some(seed) = self.random_state {
            builder = builder.random_state(seed);
        }
        if let Some(cb) = self.progress {
            builder = builder.progress(cb);
        }

        let result = builder
            .build_async()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n_rows = result.embedding.nrows();
        let n_components = result.embedding.ncols();
        let n_neighbors = result.knn_indices.ncols();
        Ok(UMAPResult {
            embedding: result.embedding.into_raw_vec_and_offset().0,
            knn_indices: result.knn_indices.into_raw_vec_and_offset().0,
            knn_distances: result.knn_distances.into_raw_vec_and_offset().0,
            n_rows,
            n_components,
            n_neighbors,
        })
    }
}

/// Result of a UMAP embedding.
#[wasm_bindgen]
pub struct UMAPResult {
    embedding: Vec<f32>,
    knn_indices: Vec<i32>,
    knn_distances: Vec<f32>,
    n_rows: usize,
    n_components: usize,
    n_neighbors: usize,
}

#[wasm_bindgen]
impl UMAPResult {
    /// Embedding coordinates as a flat Float32Array (row-major, n_rows x n_components).
    #[wasm_bindgen(getter)]
    pub fn embedding(&self) -> Vec<f32> {
        self.embedding.clone()
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

    /// Number of neighbors per point.
    #[wasm_bindgen(getter, js_name = "nNeighbors")]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
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
