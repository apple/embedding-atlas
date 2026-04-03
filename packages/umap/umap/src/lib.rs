//! UMAP - Uniform Manifold Approximation and Projection for Dimension Reduction.
//!
//! Based on the Python umap-learn library (https://github.com/lmcinnes/umap).
//!
//! # Example
//! ```no_run
//! use ndarray::Array2;
//! use umap::Umap;
//!
//! let data: Array2<f32> = Array2::zeros((100, 50));
//! let embedding = Umap::builder(&data)
//!     .n_neighbors(15)
//!     .min_dist(0.1)
//!     .random_state(42)
//!     .build()
//!     .unwrap();
//! ```

pub mod graph;
pub mod optimize;
pub mod spectral;

use std::fmt;

use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;
use nndescent::NNDescent;

use crate::graph::{fuzzy_simplicial_set, make_epochs_per_sample};
use crate::optimize::optimize_layout_euclidean;
use crate::spectral::{noisy_scale_coords, random_layout, spectral_layout};

/// Error type for UMAP operations.
#[derive(Debug)]
pub enum UmapError {
    /// The nearest neighbor search failed.
    NNDescent(nndescent::NNDescentError),
    /// The neighbor graph was not computed.
    NoNeighborGraph,
    /// Invalid parameter value.
    InvalidParameter(String),
}

impl fmt::Display for UmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UmapError::NNDescent(e) => write!(f, "NNDescent error: {}", e),
            UmapError::NoNeighborGraph => write!(f, "Neighbor graph was not computed"),
            UmapError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

impl std::error::Error for UmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UmapError::NNDescent(e) => Some(e),
            _ => None,
        }
    }
}

impl From<nndescent::NNDescentError> for UmapError {
    fn from(e: nndescent::NNDescentError) -> Self {
        UmapError::NNDescent(e)
    }
}

/// Initialization method for the embedding.
#[derive(Clone, Debug)]
pub enum Init {
    /// Spectral embedding of the fuzzy simplicial set graph.
    Spectral,
    /// Random uniform initialization in [-10, 10].
    Random,
}

/// Builder for UMAP dimensionality reduction.
///
/// # Example
/// ```no_run
/// use ndarray::Array2;
/// use umap::Umap;
///
/// let data: Array2<f32> = Array2::zeros((100, 50));
/// let embedding = Umap::builder(&data)
///     .n_components(2)
///     .n_neighbors(15)
///     .min_dist(0.1)
///     .metric("cosine")
///     .random_state(42)
///     .build()
///     .unwrap();
/// ```
pub struct UmapBuilder<'a> {
    data: &'a Array2<f32>,
    n_neighbors: usize,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    metric: String,
    n_epochs: Option<usize>,
    learning_rate: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    local_connectivity: f32,
    set_op_mix_ratio: f32,
    random_state: Option<u64>,
    verbose: bool,
    init: Init,
}

impl<'a> UmapBuilder<'a> {
    /// Number of nearest neighbors. Default: 15.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Target embedding dimension. Default: 2.
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Minimum distance between points in embedding. Default: 0.1.
    pub fn min_dist(mut self, d: f32) -> Self {
        self.min_dist = d;
        self
    }

    /// Effective scale of embedded points. Default: 1.0.
    pub fn spread(mut self, s: f32) -> Self {
        self.spread = s;
        self
    }

    /// Distance metric name. Default: "euclidean".
    pub fn metric(mut self, m: &str) -> Self {
        self.metric = m.to_string();
        self
    }

    /// Number of training epochs. Default: auto.
    pub fn n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = Some(n);
        self
    }

    /// Initial learning rate. Default: 1.0.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Number of negative samples per positive sample. Default: 5.
    pub fn negative_sample_rate(mut self, r: usize) -> Self {
        self.negative_sample_rate = r;
        self
    }

    /// Weight of repulsive force. Default: 1.0.
    pub fn repulsion_strength(mut self, s: f32) -> Self {
        self.repulsion_strength = s;
        self
    }

    /// Local connectivity constraint. Default: 1.0.
    pub fn local_connectivity(mut self, c: f32) -> Self {
        self.local_connectivity = c;
        self
    }

    /// Interpolation between fuzzy union and intersection. Default: 1.0.
    pub fn set_op_mix_ratio(mut self, r: f32) -> Self {
        self.set_op_mix_ratio = r;
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

    /// Initialization method. Default: Spectral.
    pub fn init_method(mut self, init: Init) -> Self {
        self.init = init;
        self
    }

    /// Run UMAP and return the result (embedding + KNN graph).
    pub fn build(self) -> Result<UmapResult, UmapError> {
        self.build_model()
    }

    /// Run UMAP and return the fitted result.
    fn build_model(self) -> Result<UmapResult, UmapError> {
        let data = self.data;
        let n_samples = data.nrows();

        if self.n_neighbors >= n_samples {
            return Err(UmapError::InvalidParameter(format!(
                "n_neighbors ({}) must be less than n_samples ({})",
                self.n_neighbors, n_samples
            )));
        }

        // Compute curve parameters a, b
        let (a, b) = find_ab_params(self.spread, self.min_dist);

        if self.verbose {
            eprintln!(
                "UMAP(n_neighbors={}, n_components={}, metric={})",
                self.n_neighbors, self.n_components, self.metric
            );
            eprintln!("Fitted a={:.4}, b={:.4}", a, b);
        }

        // Step 1: Compute nearest neighbors using NNDescent
        if self.verbose {
            eprintln!("Finding nearest neighbors...");
        }
        let nnd_builder = NNDescent::builder(data.clone(), &self.metric, self.n_neighbors);
        let nnd = match self.random_state {
            Some(seed) => nnd_builder.random_state(seed).build()?,
            None => nnd_builder.build()?,
        };
        let (knn_indices, knn_dists) = nnd.neighbor_graph().ok_or(UmapError::NoNeighborGraph)?;

        // Step 2: Build fuzzy simplicial set
        if self.verbose {
            eprintln!("Constructing fuzzy simplicial set...");
        }
        let mut graph = fuzzy_simplicial_set(
            &knn_indices,
            &knn_dists,
            self.n_neighbors,
            self.set_op_mix_ratio,
            self.local_connectivity,
        );

        // Step 3: Determine number of epochs
        let n_epochs = self
            .n_epochs
            .unwrap_or(if n_samples <= 10000 { 500 } else { 200 });

        // Step 4: Prune weak edges
        if n_epochs > 10 {
            let max_weight = graph
                .entries
                .values()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let threshold = max_weight / n_epochs as f32;
            graph.prune(threshold);
        }

        // Step 5: Initialize embedding
        if self.verbose {
            eprintln!("Initializing embedding...");
        }
        let mut rng = match self.random_state {
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
            None => Xoshiro256StarStar::seed_from_os(),
        };

        let mut embedding = match self.init {
            Init::Spectral => {
                let csr = graph.to_csr();
                if self.verbose {
                    eprintln!("Computing spectral initialization...");
                }
                let mut emb = spectral_layout(&csr, self.n_components, &mut rng);
                let is_random_fallback = {
                    // Check if spectral layout likely fell back to random
                    // by testing if values span the [-10, 10] range (random_layout range)
                    let min_val = emb.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = emb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    max_val - min_val > 15.0
                };
                if self.verbose && is_random_fallback {
                    eprintln!(
                        "WARNING: Spectral initialization failed, fell back to random layout"
                    );
                }
                noisy_scale_coords(&mut emb, &mut rng, 10.0, 0.0001);
                emb
            }
            Init::Random => random_layout(n_samples, self.n_components, &mut rng),
        };

        // Step 6: Extract edges and compute sampling schedule
        let (heads, tails, weights) = graph.to_edge_list();
        let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);

        // Step 7: Set up RNG state for optimization
        let rng_state: [i64; 3] = [
            rng.random_i64().abs(),
            rng.random_i64().abs(),
            rng.random_i64().abs(),
        ];

        // Step 8: Optimize embedding
        if self.verbose {
            eprintln!("Optimizing layout for {} epochs...", n_epochs);
        }
        optimize_layout_euclidean(
            &mut embedding,
            &heads,
            &tails,
            &epochs_per_sample,
            n_epochs,
            n_samples,
            a,
            b,
            self.repulsion_strength,
            self.learning_rate,
            self.negative_sample_rate as f32,
            rng_state,
            self.verbose,
        );

        Ok(UmapResult {
            embedding,
            knn_indices,
            knn_distances: knn_dists,
        })
    }
}

/// Result of a UMAP computation.
pub struct UmapResult {
    /// The low-dimensional embedding, shape (n_samples, n_components).
    pub embedding: Array2<f32>,
    /// KNN indices from the neighbor graph, shape (n_samples, n_neighbors).
    pub knn_indices: Array2<i32>,
    /// KNN distances from the neighbor graph, shape (n_samples, n_neighbors).
    pub knn_distances: Array2<f32>,
}

/// UMAP entry point.
pub struct Umap;

impl Umap {
    /// Create a builder for UMAP dimensionality reduction.
    ///
    /// # Arguments
    /// * `data` - Array of shape (n_samples, n_features)
    pub fn builder(data: &Array2<f32>) -> UmapBuilder<'_> {
        UmapBuilder {
            data,
            n_neighbors: 15,
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            metric: "euclidean".to_string(),
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            random_state: None,
            verbose: false,
            init: Init::Spectral,
        }
    }
}

/// Fit the curve parameters (a, b) for the output distance function:
///   f(x) = 1 / (1 + a * x^(2b))
///
/// The target curve is:
///   y = 1.0              if x < min_dist
///   y = exp(-(x - min_dist) / spread)  otherwise
///
/// Uses Levenberg-Marquardt optimization (damped Gauss-Newton).
fn find_ab_params(spread: f32, min_dist: f32) -> (f64, f64) {
    let n = 300;
    let spread = spread as f64;
    let min_dist = min_dist as f64;

    // Generate target curve
    let xv: Vec<f64> = (0..n)
        .map(|i| i as f64 * spread * 3.0 / (n - 1) as f64)
        .collect();
    let yv: Vec<f64> = xv
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    // Levenberg-Marquardt for 2-parameter curve fitting
    // f(x, a, b) = 1 / (1 + a * x^(2b))
    // Jacobian:
    //   df/da = -x^(2b) / (1 + a * x^(2b))^2
    //   df/db = -2 * a * x^(2b) * ln(x) / (1 + a * x^(2b))^2

    let mut a = 1.0f64;
    let mut b = 1.0f64;
    let mut lambda = 1e-3f64;

    for _ in 0..200 {
        // Compute residuals and Jacobian
        let mut jt_j = [[0.0f64; 2]; 2]; // J^T J (2x2)
        let mut jt_r = [0.0f64; 2]; // J^T r (2x1)

        for i in 0..n {
            let x = xv[i];
            if x <= 0.0 {
                continue;
            }
            let x2b = x.powf(2.0 * b);
            let denom = 1.0 + a * x2b;
            let pred = 1.0 / denom;
            let r = pred - yv[i];

            let denom_sq = denom * denom;
            let da = -x2b / denom_sq;
            let db = -2.0 * a * x2b * x.ln() / denom_sq;

            jt_j[0][0] += da * da;
            jt_j[0][1] += da * db;
            jt_j[1][0] += db * da;
            jt_j[1][1] += db * db;
            jt_r[0] += da * r;
            jt_r[1] += db * r;
        }

        // Add damping: (J^T J + lambda * I) * delta = -J^T r
        let h00 = jt_j[0][0] + lambda;
        let h01 = jt_j[0][1];
        let h10 = jt_j[1][0];
        let h11 = jt_j[1][1] + lambda;

        // Solve 2x2 system via Cramer's rule
        let det = h00 * h11 - h01 * h10;
        if det.abs() < 1e-30 {
            lambda *= 10.0;
            continue;
        }

        let da = -(h11 * jt_r[0] - h01 * jt_r[1]) / det;
        let db = -(h00 * jt_r[1] - h10 * jt_r[0]) / det;

        let new_a = a + da;
        let new_b = b + db;

        // Check improvement
        if new_a > 0.0 && new_b > 0.0 {
            let old_cost: f64 = xv
                .iter()
                .zip(yv.iter())
                .map(|(&x, &y)| {
                    let p = if x <= 0.0 {
                        1.0
                    } else {
                        1.0 / (1.0 + a * x.powf(2.0 * b))
                    };
                    (p - y).powi(2)
                })
                .sum();
            let new_cost: f64 = xv
                .iter()
                .zip(yv.iter())
                .map(|(&x, &y)| {
                    let p = if x <= 0.0 {
                        1.0
                    } else {
                        1.0 / (1.0 + new_a * x.powf(2.0 * new_b))
                    };
                    (p - y).powi(2)
                })
                .sum();

            if new_cost < old_cost {
                a = new_a;
                b = new_b;
                lambda *= 0.1;
                if new_cost < 1e-12 {
                    break;
                }
            } else {
                lambda *= 10.0;
            }
        } else {
            lambda *= 10.0;
        }
    }

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_data(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |_| rng.random_f32())
    }

    #[test]
    fn test_find_ab_params() {
        let (a, b) = find_ab_params(1.0, 0.1);
        // Values verified against scipy.optimize.curve_fit
        assert!((a - 1.5769).abs() < 0.01, "a = {}", a);
        assert!((b - 0.8951).abs() < 0.01, "b = {}", b);
    }

    #[test]
    fn test_umap_random_init() {
        let data = make_test_data(200, 10, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_epochs(50)
            .init_method(Init::Random)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 200);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(!result.embedding.iter().any(|x| x.is_nan()));

        // KNN graph should match input dimensions
        assert_eq!(result.knn_indices.nrows(), 200);
        assert_eq!(result.knn_indices.ncols(), 10);
        assert_eq!(result.knn_distances.nrows(), 200);
        assert_eq!(result.knn_distances.ncols(), 10);
    }

    #[test]
    fn test_umap_spectral_init() {
        let data = make_test_data(200, 10, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_epochs(50)
            .init_method(Init::Spectral)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 200);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(!result.embedding.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_umap_3d_output() {
        let data = make_test_data(100, 10, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_components(3)
            .n_epochs(30)
            .init_method(Init::Random)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.ncols(), 3);
    }

    #[test]
    fn test_umap_5d_output() {
        let data = make_test_data(100, 10, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_components(5)
            .n_epochs(30)
            .init_method(Init::Random)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.ncols(), 5);
        assert_eq!(result.embedding.nrows(), 100);
        assert!(!result.embedding.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_umap_10d_output() {
        let data = make_test_data(100, 20, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_components(10)
            .n_epochs(30)
            .init_method(Init::Random)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.ncols(), 10);
        assert_eq!(result.embedding.nrows(), 100);
        assert!(!result.embedding.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_umap_cosine_metric() {
        let data = make_test_data(100, 10, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_epochs(30)
            .metric("cosine")
            .init_method(Init::Random)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 100);
        assert!(!result.embedding.iter().any(|x| x.is_nan()));
    }
}
