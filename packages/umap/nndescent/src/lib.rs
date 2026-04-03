/// Fast approximate nearest neighbor search using NN-descent.
///
/// Based on the Python PyNNDescent library (https://github.com/lmcinnes/pynndescent).
pub mod distance;
pub mod heap;
pub mod nn_descent;
pub mod rng;
pub mod rp_trees;
pub mod search;
pub mod utils;

use std::fmt;

use ndarray::Array2;

use crate::distance::{CorrectionFunc, DistanceFunc, FLOAT32_EPS};
use crate::rng::{TauRng, Xoshiro256StarStar};
use crate::rp_trees::FlatTree;
use crate::search::{batch_search, CsrGraph};

const INT32_MIN: i32 = i32::MIN + 1;
const INT32_MAX: i32 = i32::MAX - 1;

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

    /// Build the NNDescent index.
    pub fn build(self) -> Result<NNDescent, NNDescentError> {
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

        // Build RP forest
        let leaf_array = if tree_init {
            if self.verbose {
                println!("Building RP forest with {} trees", n_trees);
            }
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
            Some(leaves)
        } else {
            None
        };

        let effective_max_candidates = self
            .max_candidates
            .unwrap_or_else(|| 60.min(self.n_neighbors));

        if self.verbose {
            println!("NN descent for {} iterations", n_iters);
        }

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
            self.verbose,
        );

        // Check for missing neighbors
        let any_missing = indices.iter().any(|&v| v < 0);
        if any_missing && self.verbose {
            eprintln!(
                "Warning: Failed to correctly find n_neighbors for some samples. \
                Results may be less than ideal."
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
        self.search_forest = vec![search_tree];

        self.search_graph = Some(search_graph);
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
        if let Some(correction) = self.distance_correction {
            let corrected = distances.mapv(correction);

            // Map indices back through vertex order if applicable
            if let Some(ref _order) = self.vertex_order {
                // For now, no reordering
                (indices, corrected)
            } else {
                (indices, corrected)
            }
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
