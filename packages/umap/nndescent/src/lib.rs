/// Fast approximate nearest neighbor search using NN-descent.
///
/// Based on the Python PyNNDescent library (https://github.com/lmcinnes/pynndescent).
pub mod distance;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod heap;
pub mod logger;
pub mod nn_descent;
#[cfg(feature = "gpu")]
pub mod nn_descent_gpu;
pub mod rng;
pub mod rp_trees;
pub mod search;
pub mod utils;

use std::fmt;

use ndarray::Array2;

use crate::distance::{CorrectionFunc, DistanceFunc, FLOAT32_EPS};
pub use crate::logger::{Logger, ProgressCallback};
use crate::rng::{TauRng, Xoshiro256StarStar};
use crate::rp_trees::FlatTree;
use crate::search::{batch_search, CsrGraph};

const INT32_MIN: i32 = i32::MIN + 1;
const INT32_MAX: i32 = i32::MAX - 1;

/// Stage name constants for NNDescent.
pub const STAGE_RP_FOREST: &str = "Building RP forest";
pub const STAGE_NN_DESCENT: &str = "NN descent";
pub const STAGE_SORTING: &str = "Sorting results";

/// Estimated time fractions for each NNDescent stage.
pub const NN_STAGES: &[(&str, f32)] = &[
    (STAGE_RP_FOREST, 0.15),
    (STAGE_NN_DESCENT, 0.80),
    (STAGE_SORTING, 0.05),
];

/// Error type for NNDescent operations.
#[derive(Debug)]
pub enum NNDescentError {
    /// Invalid parameter value.
    InvalidParameter(String),
}

impl fmt::Display for NNDescentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NNDescentError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

impl std::error::Error for NNDescentError {}

/// Builder for constructing an NNDescent index.
///
/// # Example
/// ```no_run
/// use ndarray::Array2;
/// use nndescent::NNDescent;
///
/// let data: Array2<f32> = Array2::zeros((1000, 50));
/// let index = NNDescent::builder(data, "euclidean", 15)
///     .random_state(42)
///     .build()
///     .unwrap();
/// let (indices, distances) = index.neighbor_graph().unwrap();
/// ```
pub struct NNDescentBuilder {
    data: Array2<f32>,
    metric: String,
    n_neighbors: usize,
    random_state: Option<u64>,
    n_trees: Option<usize>,
    leaf_size: Option<usize>,
    pruning_degree_multiplier: f32,
    diversify_prob: f32,
    n_iters: Option<usize>,
    delta: f32,
    tree_init: bool,
    max_candidates: Option<usize>,
    max_rptree_depth: usize,
    verbose: bool,
    gpu: bool,
    progress: Option<ProgressCallback>,
}

impl NNDescentBuilder {
    /// Number of random projection trees. Default: auto.
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = Some(n);
        self
    }

    /// Leaf size for random projection trees.
    pub fn leaf_size(mut self, n: usize) -> Self {
        self.leaf_size = Some(n);
        self
    }

    /// Pruning degree multiplier. Default: 1.5.
    pub fn pruning_degree_multiplier(mut self, m: f32) -> Self {
        self.pruning_degree_multiplier = m;
        self
    }

    /// Probability of pruning during diversification. Default: 1.0.
    pub fn diversify_prob(mut self, p: f32) -> Self {
        self.diversify_prob = p;
        self
    }

    /// Number of NN-descent iterations. Default: auto.
    pub fn n_iters(mut self, n: usize) -> Self {
        self.n_iters = Some(n);
        self
    }

    /// Early stopping threshold. Default: 0.001.
    pub fn delta(mut self, d: f32) -> Self {
        self.delta = d;
        self
    }

    /// Whether to use RP tree initialization. Default: true.
    pub fn tree_init(mut self, t: bool) -> Self {
        self.tree_init = t;
        self
    }

    /// Maximum number of candidates per iteration.
    pub fn max_candidates(mut self, n: usize) -> Self {
        self.max_candidates = Some(n);
        self
    }

    /// Maximum RP tree depth. Default: 200.
    pub fn max_rptree_depth(mut self, d: usize) -> Self {
        self.max_rptree_depth = d;
        self
    }

    /// Random seed for reproducibility.
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable verbose output. Default: false.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Enable GPU acceleration for distance computation. Default: false.
    ///
    /// Requires the `gpu` crate feature. When enabled, distance computation is
    /// offloaded to a GPU compute shader via wgpu (Metal, Vulkan, DX12, or
    /// WebGPU). Falls back to CPU if no suitable GPU is available or if the
    /// distance metric is not supported on GPU.
    ///
    /// Most effective for high-dimensional data (n_features >= 64).
    pub fn gpu(mut self, g: bool) -> Self {
        self.gpu = g;
        self
    }

    /// Set a progress callback. Default: none.
    pub fn progress(mut self, cb: ProgressCallback) -> Self {
        self.progress = Some(cb);
        self
    }

    /// Set an optional progress callback. Default: none.
    pub fn progress_option(mut self, cb: Option<ProgressCallback>) -> Self {
        self.progress = cb;
        self
    }

    /// Build the NNDescent index.
    ///
    /// This is a synchronous wrapper around [`build_async`]. On native builds
    /// (including native GPU via `pollster::block_on`), use this method.
    /// For WASM with GPU, use [`build_async`] instead.
    pub fn build(self) -> Result<NNDescent, NNDescentError> {
        pollster::block_on(self.build_async())
    }

    /// Build the NNDescent index (async version).
    ///
    /// Use this on WASM with GPU where `pollster::block_on` cannot block.
    /// On native, prefer [`build`] which wraps this automatically.
    pub async fn build_async(self) -> Result<NNDescent, NNDescentError> {
        let n = self.data.nrows();

        if self.n_neighbors >= n {
            return Err(NNDescentError::InvalidParameter(format!(
                "n_neighbors ({}) must be less than n_samples ({})",
                self.n_neighbors, n
            )));
        }

        // Determine number of trees and iterations
        let n_trees = self
            .n_trees
            .unwrap_or_else(|| 3.max(12.min((2.0 * (n as f64).log10()).round() as usize)));
        let n_iters = self
            .n_iters
            .unwrap_or_else(|| 5.max((n as f64).log2().round() as usize));

        // Set up RNG
        let mut rng = match self.random_state {
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
            None => Xoshiro256StarStar::seed_from_os(),
        };

        let rng_state: [i64; 3] = [
            rng.random_range_i64(INT32_MIN as i64, INT32_MAX as i64),
            rng.random_range_i64(INT32_MIN as i64, INT32_MAX as i64),
            rng.random_range_i64(INT32_MIN as i64, INT32_MAX as i64),
        ];
        let search_rng_state: [i64; 3] = [
            rng.random_range_i64(INT32_MIN as i64, INT32_MAX as i64),
            rng.random_range_i64(INT32_MIN as i64, INT32_MAX as i64),
            rng.random_range_i64(INT32_MIN as i64, INT32_MAX as i64),
        ];

        // Get distance function
        let angular_trees = distance::is_angular_metric(&self.metric);
        let (distance_func, distance_correction) =
            if let Some((fast_fn, correction)) = distance::get_fast_alternative(&self.metric) {
                (fast_fn, Some(correction))
            } else if let Some(fn_) = distance::get_distance_func(&self.metric) {
                (fn_, None)
            } else {
                return Err(NNDescentError::InvalidParameter(format!(
                    "unknown metric: {}",
                    self.metric
                )));
            };

        let tree_init = self.tree_init && n_trees > 0;

        // Initialize GPU context if requested (before forest so we can use GPU projections)
        #[cfg(feature = "gpu")]
        let gpu_ctx = if self.gpu {
            match gpu::GpuContext::new(&self.data, &self.metric).await {
                Some(ctx) => Some(ctx),
                None => {
                    if self.verbose {
                        eprintln!("Warning: GPU unavailable or metric not supported on GPU, falling back to CPU");
                    }
                    None
                }
            }
        } else {
            None
        };

        // Create logger
        let mut logger = Logger::new(self.verbose, self.progress, NN_STAGES);

        // Build RP forest
        let leaf_array = if tree_init {
            logger.push_stage_with_message(
                STAGE_RP_FOREST,
                &format!("Building RP forest with {} trees", n_trees),
            );
            #[cfg(feature = "gpu")]
            let forest = if let Some(ref gpu) = gpu_ctx {
                rp_trees::make_forest_gpu(
                    &self.data,
                    self.n_neighbors,
                    n_trees,
                    self.leaf_size,
                    &rng_state,
                    angular_trees,
                    self.max_rptree_depth,
                    gpu,
                )
                .await
            } else {
                rp_trees::make_forest(
                    &self.data,
                    self.n_neighbors,
                    n_trees,
                    self.leaf_size,
                    &rng_state,
                    angular_trees,
                    self.max_rptree_depth,
                )
            };
            #[cfg(not(feature = "gpu"))]
            let forest = rp_trees::make_forest(
                &self.data,
                self.n_neighbors,
                n_trees,
                self.leaf_size,
                &rng_state,
                angular_trees,
                self.max_rptree_depth,
            );
            let leaves = rp_trees::rptree_leaf_array(&forest);
            logger.pop_stage();
            Some(leaves)
        } else {
            None
        };

        let effective_max_candidates = self
            .max_candidates
            .unwrap_or_else(|| 60.min(self.n_neighbors));

        // Run NN-descent
        let mut rng = TauRng::from_state(rng_state);
        let (indices, distances) = nn_descent::nn_descent(
            &self.data,
            self.n_neighbors,
            &mut rng,
            effective_max_candidates,
            distance_func,
            n_iters,
            self.delta,
            tree_init,
            leaf_array.as_ref(),
            &mut logger,
            #[cfg(feature = "gpu")]
            gpu_ctx.as_ref(),
        )
        .await;

        // Check for missing neighbors
        let any_missing = indices.iter().any(|&v| v < 0);
        if any_missing {
            logger.log(
                "Warning: Failed to correctly find n_neighbors for some samples. \
                Results may be less than ideal.",
            );
        }

        Ok(NNDescent {
            n_neighbors: self.n_neighbors,
            leaf_size: self.leaf_size,
            diversify_prob: self.diversify_prob,
            max_rptree_depth: self.max_rptree_depth,
            raw_data: self.data,
            neighbor_graph: Some((indices, distances)),
            search_graph: None,
            search_forest: Vec::new(),
            vertex_order: None,
            rng_state,
            search_rng_state,
            distance_func,
            distance_correction,
            angular_trees,
            min_distance: FLOAT32_EPS,
            is_prepared: false,
        })
    }
}

/// NNDescent index for fast approximate nearest neighbor queries.
///
/// The neighbor list for each point includes the point itself as a self-neighbor
/// at position 0 with distance 0, consistent with pynndescent. When requesting
/// `n_neighbors`, the result contains the self-neighbor plus `n_neighbors - 1`
/// actual neighbors.
///
/// # Example
/// ```no_run
/// use ndarray::Array2;
/// use nndescent::NNDescent;
///
/// let data: Array2<f32> = Array2::zeros((1000, 50));
/// let index = NNDescent::builder(data, "euclidean", 15)
///     .random_state(42)
///     .build()
///     .unwrap();
/// ```
pub struct NNDescent {
    // Configuration (retained for prepare/query)
    n_neighbors: usize,
    leaf_size: Option<usize>,
    diversify_prob: f32,
    max_rptree_depth: usize,

    // Built state
    raw_data: Array2<f32>,
    neighbor_graph: Option<(Array2<i32>, Array2<f32>)>,
    search_graph: Option<CsrGraph>,
    search_forest: Vec<FlatTree>,
    vertex_order: Option<Vec<i32>>,
    rng_state: [i64; 3],
    search_rng_state: [i64; 3],
    distance_func: DistanceFunc,
    distance_correction: Option<CorrectionFunc>,
    angular_trees: bool,
    min_distance: f32,
    is_prepared: bool,
}

impl NNDescent {
    /// Create a builder for a new NNDescent index.
    ///
    /// # Arguments
    /// * `data` - Array of shape (n_samples, n_features)
    /// * `metric` - Distance metric name (e.g., "euclidean", "cosine")
    /// * `n_neighbors` - Number of neighbors to find
    pub fn builder(data: Array2<f32>, metric: &str, n_neighbors: usize) -> NNDescentBuilder {
        NNDescentBuilder {
            data,
            metric: metric.to_string(),
            n_neighbors,
            random_state: None,
            n_trees: None,
            leaf_size: None,
            pruning_degree_multiplier: 1.5,
            diversify_prob: 1.0,
            n_iters: None,
            delta: 0.001,
            tree_init: true,
            max_candidates: None,
            max_rptree_depth: 200,
            verbose: false,
            gpu: false,
            progress: None,
        }
    }

    /// Get the neighbor graph (indices, distances).
    /// Applies distance correction if using fast alternative distances.
    pub fn neighbor_graph(&self) -> Option<(Array2<i32>, Array2<f32>)> {
        self.neighbor_graph.as_ref().map(|(indices, distances)| {
            if let Some(correction) = self.distance_correction {
                let corrected = distances.mapv(correction);
                (indices.clone(), corrected)
            } else {
                (indices.clone(), distances.clone())
            }
        })
    }

    /// Access the raw neighbor graph without correction.
    pub fn raw_neighbor_graph(&self) -> Option<&(Array2<i32>, Array2<f32>)> {
        self.neighbor_graph.as_ref()
    }

    /// Prepare the search graph for querying.
    ///
    /// This must be called before `query()`, or `query()` will call it automatically.
    ///
    /// # Panics
    /// Panics if the neighbor graph is not present. This cannot happen when using
    /// the builder API, since `build()` always populates the neighbor graph.
    pub fn prepare(&mut self) {
        if self.is_prepared {
            return;
        }

        let (indices, distances) = self
            .neighbor_graph
            .as_ref()
            .expect("neighbor graph not built; this is a bug");
        let n = self.raw_data.nrows();

        // Build search graph from neighbor graph
        // The search graph includes both forward and reverse edges
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        // Forward edges (from diversified neighbor graph)
        let diversified = diversify(
            &indices,
            &distances,
            &self.raw_data,
            self.distance_func,
            self.diversify_prob,
        );

        for i in 0..n {
            for j in 0..diversified.0.ncols() {
                let idx = diversified.0[[i, j]];
                if idx >= 0 {
                    let d = diversified.1[[i, j]];
                    if d > 0.0 {
                        rows.push(i as i32);
                        cols.push(idx);
                        data.push(d);
                        // Also add reverse edge
                        rows.push(idx);
                        cols.push(i as i32);
                        data.push(d);
                    }
                }
            }
        }

        let (search_graph, graph_data) = CsrGraph::from_coo(n, &rows, &cols, &data);

        // Find min distance for search bound calculation
        let min_dist = graph_data
            .iter()
            .filter(|&&d| d > 0.0)
            .cloned()
            .fold(f32::INFINITY, f32::min);
        self.min_distance = if min_dist.is_finite() {
            min_dist
        } else {
            FLOAT32_EPS
        };

        // Build search tree
        let mut rng = TauRng::from_state(self.rng_state);
        let search_leaf_size = self.leaf_size.unwrap_or(30);
        let search_tree = rp_trees::make_hub_tree(
            &self.raw_data,
            indices,
            &mut rng,
            search_leaf_size,
            self.angular_trees,
            self.max_rptree_depth,
        );
        // Don't store search_forest yet — we'll remap it below.

        // Reorder data and graph by tree leaf order for cache locality during search.
        // vertex_order[new_i] = old_i (tree DFS leaf order)
        let vertex_order: Vec<i32> = search_tree.indices.clone();

        // Build inverse mapping: inverse[old_i] = new_i
        let mut inverse_order = vec![0i32; n];
        for (new_i, &old_i) in vertex_order.iter().enumerate() {
            if old_i >= 0 && (old_i as usize) < n {
                inverse_order[old_i as usize] = new_i as i32;
            }
        }

        // Reorder raw_data: new row new_i = old row vertex_order[new_i]
        let dim = self.raw_data.ncols();
        let mut new_data = Array2::zeros((n, dim));
        for new_i in 0..n {
            let old_i = vertex_order[new_i] as usize;
            new_data.row_mut(new_i).assign(&self.raw_data.row(old_i));
        }
        self.raw_data = new_data;

        // Remap search graph: rebuild COO with remapped indices
        let mut remap_rows = Vec::new();
        let mut remap_cols = Vec::new();
        let mut remap_data = Vec::new();
        for old_row in 0..n {
            let new_row = inverse_order[old_row];
            let start = search_graph.indptr[old_row] as usize;
            let end = search_graph.indptr[old_row + 1] as usize;
            for pos in start..end {
                let old_col = search_graph.indices[pos];
                if old_col >= 0 && (old_col as usize) < n {
                    remap_rows.push(new_row);
                    remap_cols.push(inverse_order[old_col as usize]);
                    remap_data.push(1.0f32); // distance values not used after prepare
                }
            }
        }
        let (remapped_graph, _) = CsrGraph::from_coo(n, &remap_rows, &remap_cols, &remap_data);

        // Remap tree indices to new numbering
        let mut remapped_tree = search_tree;
        for idx in remapped_tree.indices.iter_mut() {
            if *idx >= 0 && (*idx as usize) < n {
                *idx = inverse_order[*idx as usize];
            }
        }

        self.search_forest = vec![remapped_tree];
        self.search_graph = Some(remapped_graph);
        self.vertex_order = Some(vertex_order);
        self.is_prepared = true;
    }

    /// Query the index for k nearest neighbors of query points.
    ///
    /// # Arguments
    /// * `query_data` - Array of shape (n_queries, n_features)
    /// * `k` - Number of neighbors to return
    /// * `epsilon` - Controls accuracy vs speed tradeoff (higher = more accurate but slower)
    ///
    /// # Returns
    /// (indices, distances) arrays of shape (n_queries, k)
    pub fn query(
        &mut self,
        query_data: &Array2<f32>,
        k: usize,
        epsilon: f32,
    ) -> (Array2<i32>, Array2<f32>) {
        self.prepare();

        let search_graph = self.search_graph.as_ref().unwrap();
        let (indices, distances) = batch_search(
            query_data,
            &self.raw_data,
            search_graph,
            &self.search_forest,
            self.distance_func,
            k,
            epsilon,
            self.n_neighbors,
            self.min_distance,
            &self.search_rng_state,
        );

        // Apply distance correction if needed
        let distances = if let Some(correction) = self.distance_correction {
            distances.mapv(correction)
        } else {
            distances
        };

        // Map indices back through vertex order to original numbering
        if let Some(ref order) = self.vertex_order {
            let remapped = indices.mapv(|i| if i >= 0 { order[i as usize] } else { i });
            (remapped, distances)
        } else {
            (indices, distances)
        }
    }
}

/// Diversify the neighbor graph using relative neighborhood pruning.
/// Removes edges where a shorter path through a retained neighbor exists.
fn diversify(
    indices: &Array2<i32>,
    distances: &Array2<f32>,
    data: &Array2<f32>,
    dist_fn: DistanceFunc,
    prune_probability: f32,
) -> (Array2<i32>, Array2<f32>) {
    let n = indices.nrows();
    let k = indices.ncols();

    let mut new_indices = Array2::from_elem((n, k), -1i32);
    let mut new_distances = Array2::from_elem((n, k), f32::INFINITY);

    for i in 0..n {
        let mut retained_indices: Vec<i32> = Vec::new();
        let mut retained_distances: Vec<f32> = Vec::new();

        // Always keep first neighbor
        if indices[[i, 0]] >= 0 {
            retained_indices.push(indices[[i, 0]]);
            retained_distances.push(distances[[i, 0]]);
        }

        for j in 1..k {
            if indices[[i, j]] < 0 {
                break;
            }

            let candidate = indices[[i, j]];
            let candidate_dist = distances[[i, j]];
            let mut keep = true;

            for m in 0..retained_indices.len() {
                let c = retained_indices[m];
                let c_dist = retained_distances[m];

                if c_dist > FLOAT32_EPS {
                    let d = dist_fn(
                        data.row(candidate as usize).as_slice().unwrap(),
                        data.row(c as usize).as_slice().unwrap(),
                    );
                    if d < candidate_dist {
                        // There's a shorter path through retained neighbor c
                        if prune_probability >= 1.0 {
                            keep = false;
                            break;
                        }
                    }
                }
            }

            if keep {
                retained_indices.push(candidate);
                retained_distances.push(candidate_dist);
            }
        }

        for j in 0..retained_indices.len().min(k) {
            new_indices[[i, j]] = retained_indices[j];
            new_distances[[i, j]] = retained_distances[j];
        }
    }

    (new_indices, new_distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_nn_data(seed: u64) -> Array2<f32> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let mut data = Array2::zeros((1002, 5));
        for i in 0..1000 {
            for j in 0..5 {
                data[[i, j]] = rng.random_f32();
            }
        }
        // Last 2 rows are zeros (corner case)
        data
    }

    #[test]
    fn test_nndescent_construction() {
        let data = make_test_nn_data(189212);
        let nnd = NNDescent::builder(data, "euclidean", 10)
            .random_state(42)
            .build()
            .unwrap();
        let (indices, distances) = nnd.neighbor_graph().unwrap();
        assert_eq!(indices.nrows(), 1002);
        assert_eq!(indices.ncols(), 10);

        // Check distances are sorted ascending
        for i in 0..1002 {
            for j in 1..10 {
                assert!(
                    distances[[i, j]] >= distances[[i, j - 1]],
                    "Row {} not sorted at col {}: {} < {}",
                    i,
                    j,
                    distances[[i, j]],
                    distances[[i, j - 1]]
                );
            }
        }
    }

    #[test]
    fn test_nndescent_cosine() {
        let data = make_test_nn_data(189212);
        let nnd = NNDescent::builder(data, "cosine", 10)
            .random_state(42)
            .build()
            .unwrap();
        let (indices, _distances) = nnd.neighbor_graph().unwrap();
        assert_eq!(indices.nrows(), 1002);
    }
}
