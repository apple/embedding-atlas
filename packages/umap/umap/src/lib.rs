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

#[cfg(feature = "gpu")]
pub mod gpu_optimize;

use std::fmt;

use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;
use nndescent::NNDescent;

pub use nndescent::{Logger, ProgressCallback};

use crate::graph::{
    fuzzy_simplicial_set, make_epochs_per_sample, normalize_knn_self_column, CsrMatrix,
    SparseMatrix,
};
use crate::spectral::{noisy_scale_coords, random_layout, spectral_layout};

pub use crate::optimize::{LiveParams, Optimizer, UmapOptimizer};

// ---------- UMAP stage definitions ----------

/// Stage name constants for UMAP.
pub const STAGE_NEIGHBORS: &str = "Finding nearest neighbors";
pub const STAGE_GRAPH: &str = "Constructing fuzzy graph";
pub const STAGE_EMBEDDING_INIT: &str = "Initializing embedding";
pub const STAGE_OPTIMIZATION: &str = "Optimizing layout";

/// Estimated time fractions for each UMAP stage.
pub const UMAP_STAGES: &[(&str, f32)] = &[
    (STAGE_NEIGHBORS, 0.30),
    (STAGE_GRAPH, 0.05),
    (STAGE_EMBEDDING_INIT, 0.10),
    (STAGE_OPTIMIZATION, 0.55),
];

/// Error type for UMAP operations.
#[derive(Debug)]
pub enum UmapError {
    /// The nearest neighbor search failed.
    NNDescent(nndescent::NNDescentError),
    /// The neighbor graph was not computed.
    NoNeighborGraph,
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Supplied kNN indices and distances have mismatched shapes.
    KnnShapeMismatch {
        indices: (usize, usize),
        distances: (usize, usize),
    },
    /// A supplied kNN index is out of range `[0, n_samples)`.
    KnnIndexOutOfRange { value: i32, n_samples: usize },
}

impl fmt::Display for UmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UmapError::NNDescent(e) => write!(f, "NNDescent error: {}", e),
            UmapError::NoNeighborGraph => write!(f, "Neighbor graph was not computed"),
            UmapError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            UmapError::KnnShapeMismatch { indices, distances } => write!(
                f,
                "kNN indices shape {:?} does not match distances shape {:?}",
                indices, distances
            ),
            UmapError::KnnIndexOutOfRange { value, n_samples } => write!(
                f,
                "kNN index {} out of range for {} samples",
                value, n_samples
            ),
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
    data: Option<&'a Array2<f32>>,
    /// Precomputed kNN graph (indices, distances). When set, NNDescent is skipped
    /// and `data` is unused. See [`Umap::builder_from_knn`].
    knn: Option<(Array2<i32>, Array2<f32>)>,
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
    gpu: bool,
    optimizer: Optimizer,
    progress: Option<ProgressCallback>,
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

    /// Enable GPU acceleration for nearest neighbor computation. Default: false.
    ///
    /// Requires the `gpu` crate feature. Falls back to CPU if no suitable GPU
    /// is available or if the metric is not supported on GPU.
    pub fn gpu(mut self, g: bool) -> Self {
        self.gpu = g;
        self
    }

    /// Set a progress callback. Default: none.
    pub fn progress(mut self, cb: ProgressCallback) -> Self {
        self.progress = Some(cb);
        self
    }

    /// Optimizer to use for the layout SGD. Default: [`Optimizer::Sgd`].
    pub fn optimizer(mut self, o: Optimizer) -> Self {
        self.optimizer = o;
        self
    }

    /// Run UMAP and return the result (embedding + KNN graph).
    pub fn build(self) -> Result<UmapResult, UmapError> {
        pollster::block_on(self.build_async())
    }

    /// Run UMAP asynchronously (for WASM with GPU): build the graph + initial
    /// embedding, then optimize to completion. Output matches [`build`](Self::build).
    ///
    /// On native, prefer [`build`](Self::build) which handles async internally.
    pub async fn build_async(self) -> Result<UmapResult, UmapError> {
        let Prepared {
            embedding,
            graph: _,
            knn_indices,
            knn_distances,
            head,
            tail,
            epochs_per_sample,
            n_epochs,
            n_neighbors,
            rng_state,
            params,
            optimizer,
            gpu,
            mut logger,
        } = self.prepare().await?;

        let mut engine = build_optimizer(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
            gpu,
            n_neighbors,
        )
        .await;

        logger.push_stage_with_message(
            STAGE_OPTIMIZATION,
            &format!("Optimizing layout for {} epochs...", n_epochs),
        );
        engine.run_to(n_epochs, &mut logger).await;
        logger.pop_stage();

        Ok(UmapResult {
            embedding: engine.into_embedding(),
            knn_indices,
            knn_distances,
        })
    }

    /// Build the graph + initial embedding and return a resumable [`UmapSetup`],
    /// **without** running the layout optimization. Use the returned
    /// [`UmapOptimizer`] to step/run interactively.
    pub async fn setup_async(self) -> Result<UmapSetup, UmapError> {
        let Prepared {
            embedding,
            graph,
            knn_indices,
            knn_distances,
            head,
            tail,
            epochs_per_sample,
            n_epochs,
            n_neighbors,
            rng_state,
            params,
            optimizer,
            gpu,
            logger: _,
        } = self.prepare().await?;

        let optimizer = build_optimizer(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
            gpu,
            n_neighbors,
        )
        .await;

        Ok(UmapSetup {
            optimizer,
            graph,
            knn_indices,
            knn_distances,
            n_epochs,
        })
    }

    /// Stages 1–7: nearest neighbors, fuzzy graph, prune, init, edge schedule.
    /// Shared by [`build_async`](Self::build_async) and [`setup_async`](Self::setup_async).
    async fn prepare(mut self) -> Result<Prepared, UmapError> {
        // Take ownership of any supplied kNN graph up front, so the Step 1 branch
        // below can move it without conflicting with the n_samples decision.
        let knn = self.knn.take();

        // n_samples comes from whichever input is present (kNN graph or data).
        let n_samples = match &knn {
            Some((idx, dist)) => {
                if idx.shape() != dist.shape() {
                    return Err(UmapError::KnnShapeMismatch {
                        indices: idx.dim(),
                        distances: dist.dim(),
                    });
                }
                idx.nrows()
            }
            None => self
                .data
                .expect("UmapBuilder has neither data nor a kNN graph")
                .nrows(),
        };

        let mut logger = Logger::new(self.verbose, self.progress, UMAP_STAGES);

        // Degenerate inputs have nothing to embed: with 0 or 1 points there are no
        // neighbor relationships to model. Return a trivial origin layout (empty
        // graph/knn, zero epochs) rather than erroring, so callers don't have to
        // special-case tiny datasets.
        if n_samples <= 1 {
            return Ok(Prepared {
                embedding: Array2::zeros((n_samples, self.n_components)),
                graph: SparseMatrix::from_coo(n_samples, n_samples, Vec::new()).to_csr(),
                knn_indices: Array2::zeros((n_samples, 0)),
                knn_distances: Array2::zeros((n_samples, 0)),
                head: Vec::new(),
                tail: Vec::new(),
                epochs_per_sample: Vec::new(),
                n_epochs: 0,
                n_neighbors: 0,
                rng_state: [0, 0, 0],
                params: LiveParams {
                    alpha: self.learning_rate,
                    gamma: self.repulsion_strength,
                    negative_sample_rate: self.negative_sample_rate as f32,
                    a: 1.0,
                    b: 1.0,
                },
                optimizer: self.optimizer,
                gpu: self.gpu,
                logger,
            });
        }

        // Compute curve parameters a, b (independent of the neighbor count).
        let (a, b) = find_ab_params(self.spread, self.min_dist);

        // Step 1: obtain the kNN graph — either supplied (skip NNDescent) or computed.
        // We keep the STAGE_NEIGHBORS push/pop in both paths so a progress callback's
        // stage fractions stay consistent (the supplied path reports it instantly).
        logger.push_stage_with_message(STAGE_NEIGHBORS, "Finding nearest neighbors...");
        let (knn_indices, knn_dists, n_neighbors) = match knn {
            Some((idx, dist)) => {
                // Validate indices; negatives are allowed and mean "missing".
                for &v in idx.iter() {
                    if v >= 0 && (v as usize) >= n_samples {
                        return Err(UmapError::KnnIndexOutOfRange {
                            value: v,
                            n_samples,
                        });
                    }
                }
                // Canonicalize to self-at-column-0, sorted ascending; n_neighbors is
                // taken from the (normalized) graph width, not self.n_neighbors.
                let (norm_indices, norm_dists) = normalize_knn_self_column(&idx, &dist);
                let n_neighbors = norm_indices.ncols();
                logger.log("Using precomputed neighbors");
                logger.pop_stage();
                (norm_indices, norm_dists, n_neighbors)
            }
            None => {
                let data = self
                    .data
                    .expect("UmapBuilder has neither data nor a kNN graph");

                // A point can have at most `n_samples - 1` neighbors. Clamp rather
                // than reject (mirrors umap-learn on small inputs).
                let n_neighbors = self.n_neighbors.min(n_samples - 1);
                if n_neighbors != self.n_neighbors {
                    logger.log(&format!(
                        "n_neighbors ({}) >= n_samples ({}); clamping to {}",
                        self.n_neighbors, n_samples, n_neighbors
                    ));
                }

                // Create nndescent callback that forwards to our logger's user callback.
                // We share the user callback between the UMAP logger and the NNDescent
                // sub-callback via Rc<RefCell<>>, avoiding unsafe pointer aliasing.
                let nn_cb: Option<nndescent::ProgressCallback> = if logger.callback.is_some() {
                    use std::cell::RefCell;
                    use std::rc::Rc;

                    let user_cb = Rc::new(RefCell::new(logger.callback.take().unwrap()));

                    // Replace logger's callback with a delegating wrapper
                    let logger_cb = user_cb.clone();
                    logger.callback = Some(Box::new(move |progress: f32, stage: &str| {
                        (*logger_cb.borrow_mut())(progress, stage);
                    }));

                    // Compute the NEIGHBORS stage progress mapping from UMAP_STAGES
                    let (stage_start, stage_frac) = {
                        let mut cumulative = 0.0f32;
                        let mut found = (0.0f32, 1.0f32);
                        for &(name, frac) in UMAP_STAGES {
                            if name == STAGE_NEIGHBORS {
                                found = (cumulative, frac);
                                break;
                            }
                            cumulative += frac;
                        }
                        found
                    };

                    // Create NNDescent callback that maps sub-progress to overall UMAP progress
                    let nn_cb_ref = user_cb.clone();
                    Some(Box::new(move |progress: f32, stage: &str| {
                        let label = format!("NNDescent: {}", stage);
                        let overall = (stage_start + stage_frac * progress).min(1.0);
                        (*nn_cb_ref.borrow_mut())(overall, &label);
                    }))
                } else {
                    None
                };

                let mut nnd_builder = NNDescent::builder(data.clone(), &self.metric, n_neighbors)
                    .verbose(self.verbose)
                    .gpu(self.gpu)
                    .progress_option(nn_cb);
                if let Some(seed) = self.random_state {
                    nnd_builder = nnd_builder.random_state(seed);
                }
                let nnd = nnd_builder.build_async().await?;
                let (knn_indices, knn_dists) =
                    nnd.neighbor_graph().ok_or(UmapError::NoNeighborGraph)?;
                logger.pop_stage();
                (knn_indices, knn_dists, n_neighbors)
            }
        };

        logger.log(&format!(
            "UMAP(n_neighbors={}, n_components={}, metric={})",
            n_neighbors, self.n_components, self.metric
        ));
        logger.log(&format!("Fitted a={:.4}, b={:.4}", a, b));

        // Step 2: Build fuzzy simplicial set
        logger.push_stage_with_message(STAGE_GRAPH, "Constructing fuzzy simplicial set...");
        let mut graph = fuzzy_simplicial_set(
            &knn_indices,
            &knn_dists,
            n_neighbors,
            self.set_op_mix_ratio,
            self.local_connectivity,
        );

        // Step 3: Determine number of epochs
        let n_epochs = self
            .n_epochs
            .unwrap_or(if n_samples <= 10000 { 500 } else { 200 });

        // Step 4: Prune weak edges
        if n_epochs > 10 {
            let max_weight = graph.max_weight();
            let threshold = max_weight / n_epochs as f32;
            graph.prune(threshold);
        }
        logger.pop_stage();

        // CSR of the pruned graph: used for spectral init and retained for reset.
        let csr = graph.to_csr();

        // Step 5: Initialize embedding
        logger.push_stage_with_message(STAGE_EMBEDDING_INIT, "Initializing embedding...");
        let mut rng = match self.random_state {
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
            None => Xoshiro256StarStar::seed_from_os(),
        };

        // Spectral init hurts momentum early (and costs LOBPCG time); prefer random
        // when momentum is selected. (Single-line guard; see plan §5.)
        let init = if self.optimizer == Optimizer::Momentum {
            Init::Random
        } else {
            self.init.clone()
        };

        let embedding = match init {
            Init::Spectral => {
                logger.log("Computing spectral initialization...");
                let mut emb = spectral_layout(&csr, self.n_components, &mut rng);
                let is_random_fallback = {
                    let min_val = emb.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = emb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    max_val - min_val > 15.0
                };
                if is_random_fallback {
                    logger
                        .log("WARNING: Spectral initialization failed, fell back to random layout");
                }
                noisy_scale_coords(&mut emb, &mut rng, 10.0, 0.0001);
                emb
            }
            Init::Random => random_layout(n_samples, self.n_components, &mut rng),
        };
        logger.pop_stage();

        // Step 6: Extract edges and compute sampling schedule
        let (heads, tails, weights) = graph.to_edge_list();
        let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);

        // Step 7: Set up RNG state for optimization
        let rng_state: [i64; 3] = [
            rng.random_i64().abs(),
            rng.random_i64().abs(),
            rng.random_i64().abs(),
        ];

        Ok(Prepared {
            embedding,
            graph: csr,
            knn_indices,
            knn_distances: knn_dists,
            head: heads,
            tail: tails,
            epochs_per_sample,
            n_epochs,
            n_neighbors,
            rng_state,
            params: LiveParams {
                alpha: self.learning_rate,
                gamma: self.repulsion_strength,
                negative_sample_rate: self.negative_sample_rate as f32,
                a: a as f32,
                b: b as f32,
            },
            optimizer: self.optimizer,
            gpu: self.gpu,
            logger,
        })
    }
}

/// Construct a [`UmapOptimizer`] from prepared pieces, using the resident GPU
/// context when `gpu` is requested and the `gpu` feature is enabled (falls back
/// to CPU otherwise).
#[allow(clippy::too_many_arguments)]
async fn build_optimizer(
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: Vec<f32>,
    embedding: Array2<f32>,
    params: LiveParams,
    optimizer: Optimizer,
    rng_state: [i64; 3],
    gpu: bool,
    n_neighbors: usize,
) -> UmapOptimizer {
    #[cfg(feature = "gpu")]
    {
        if gpu {
            return UmapOptimizer::new_gpu(
                head,
                tail,
                &epochs_per_sample,
                embedding,
                params,
                optimizer,
                rng_state,
                n_neighbors,
            )
            .await;
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (gpu, n_neighbors);
    }
    UmapOptimizer::new(
        head,
        tail,
        &epochs_per_sample,
        embedding,
        params,
        optimizer,
        rng_state,
    )
}

/// Output of [`UmapBuilder::prepare`]: everything needed to construct the
/// optimizer plus the retained pruned graph and kNN arrays.
struct Prepared {
    embedding: Array2<f32>,
    graph: CsrMatrix,
    knn_indices: Array2<i32>,
    knn_distances: Array2<f32>,
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: Vec<f32>,
    n_epochs: usize,
    n_neighbors: usize,
    rng_state: [i64; 3],
    params: LiveParams,
    optimizer: Optimizer,
    gpu: bool,
    logger: Logger,
}

/// A prepared, not-yet-optimized UMAP: the resumable optimizer plus the retained
/// graph and kNN arrays. Returned by [`UmapBuilder::setup_async`].
pub struct UmapSetup {
    /// The resumable layout optimizer (already holds the initial embedding).
    pub optimizer: UmapOptimizer,
    /// The pruned fuzzy graph (CSR), retained for spectral re-initialization.
    pub graph: CsrMatrix,
    /// KNN indices from the neighbor graph, shape (n_samples, n_neighbors).
    pub knn_indices: Array2<i32>,
    /// KNN distances from the neighbor graph, shape (n_samples, n_neighbors).
    pub knn_distances: Array2<f32>,
    /// The nominal optimization horizon (default 500/200).
    pub n_epochs: usize,
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
            data: Some(data),
            knn: None,
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
            gpu: false,
            optimizer: Optimizer::Sgd,
            progress: None,
        }
    }

    /// Create a builder from a precomputed kNN graph, skipping NNDescent. No
    /// high-dimensional data is needed — once the graph exists, UMAP uses only it.
    ///
    /// `knn_indices` and `knn_distances` are both shape (n_samples, k), where row
    /// `i` lists the neighbors of point `i` sorted by ascending distance. The point
    /// itself may be included (typically column 0, distance 0) or omitted — this is
    /// detected and normalized per row (see [`normalize_knn_self_column`]). `metric`
    /// and `n_neighbors` are ignored in this mode (k is taken from the supplied
    /// graph).
    ///
    /// The caller is responsible for supplying a well-formed graph; it is not
    /// validated beyond an index-range check, and bad input degrades silently:
    /// - every neighbor index must be valid (in `[0, n_samples)`);
    /// - point `i` may list itself at most once per row;
    /// - distances must be finite (a `NaN` propagates to a `NaN` embedding);
    /// - the real-neighbor count should be uniform across rows — the canonical
    ///   width is the per-row minimum, so one short row trims every row (see
    ///   [`normalize_knn_self_column`]).
    ///
    /// [`normalize_knn_self_column`]: crate::graph::normalize_knn_self_column
    pub fn builder_from_knn(
        knn_indices: Array2<i32>,
        knn_distances: Array2<f32>,
    ) -> UmapBuilder<'static> {
        UmapBuilder {
            data: None,
            knn: Some((knn_indices, knn_distances)),
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
            gpu: false,
            optimizer: Optimizer::Sgd,
            progress: None,
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
pub fn find_ab_params(spread: f32, min_dist: f32) -> (f64, f64) {
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
    use ndarray::{arr2, s, Array2};

    fn make_test_data(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |_| rng.random_f32())
    }

    /// Exact euclidean kNN in the umap-learn layout: row i lists point i itself in
    /// column 0 (distance 0), then its `k-1` nearest neighbors sorted ascending.
    fn make_knn(data: &Array2<f32>, k: usize) -> (Array2<i32>, Array2<f32>) {
        let n = data.nrows();
        let mut indices = Array2::<i32>::zeros((n, k));
        let mut distances = Array2::<f32>::zeros((n, k));
        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .map(|j| {
                    let d2: f32 = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (d2.sqrt(), j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            for c in 0..k {
                indices[[i, c]] = dists[c].1 as i32;
                distances[[i, c]] = dists[c].0;
            }
        }
        (indices, distances)
    }

    #[test]
    fn test_umap_from_knn_shape() {
        let data = make_test_data(200, 10, 42);
        let (idx, dist) = make_knn(&data, 11); // self + 10 neighbors
        let result = Umap::builder_from_knn(idx, dist)
            .n_epochs(50)
            .init_method(Init::Random)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 200);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.embedding.iter().all(|x| x.is_finite()));
        // Self already in column 0 -> the echoed graph keeps its width.
        assert_eq!(result.knn_indices.dim(), (200, 11));
        assert_eq!(result.knn_distances.dim(), (200, 11));
    }

    #[test]
    fn test_umap_from_knn_self_excluded_widens() {
        let data = make_test_data(150, 8, 7);
        let (idx, dist) = make_knn(&data, 11);
        // Drop column 0 (self) from every row -> self-excluded, width 10.
        let idx_ex = idx.slice(s![.., 1..]).to_owned();
        let dist_ex = dist.slice(s![.., 1..]).to_owned();
        let result = Umap::builder_from_knn(idx_ex, dist_ex)
            .n_epochs(30)
            .init_method(Init::Random)
            .random_state(7)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 150);
        // Self was re-added -> width grows back to 11.
        assert_eq!(result.knn_indices.ncols(), 11);
        assert!(result.embedding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_umap_from_knn_shape_mismatch() {
        let idx = Array2::<i32>::zeros((10, 5));
        let dist = Array2::<f32>::zeros((10, 6));
        let err = Umap::builder_from_knn(idx, dist)
            .build()
            .err()
            .expect("expected a shape-mismatch error");
        assert!(matches!(err, UmapError::KnnShapeMismatch { .. }));
    }

    #[test]
    fn test_umap_from_knn_index_out_of_range() {
        // n_samples = 3, but a row references index 5.
        let idx = arr2(&[[0, 1, 2], [1, 0, 2], [2, 5, 0]]);
        let dist = arr2(&[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]);
        let err = Umap::builder_from_knn(idx, dist)
            .build()
            .err()
            .expect("expected an out-of-range error");
        assert!(matches!(
            err,
            UmapError::KnnIndexOutOfRange {
                value: 5,
                n_samples: 3
            }
        ));
    }

    #[test]
    fn test_umap_from_knn_negative_index_ok() {
        // Negative indices are outside the documented contract (callers should
        // supply only valid neighbors), but are tolerated defensively: a `-1`
        // entry is skipped rather than crashing or being treated as vertex -1.
        let data = make_test_data(60, 6, 11);
        let (mut idx, dist) = make_knn(&data, 8);
        idx[[0, 7]] = -1; // out-of-contract entry; must be skipped, not crash
        let result = Umap::builder_from_knn(idx, dist)
            .n_epochs(20)
            .init_method(Init::Random)
            .random_state(11)
            .build();
        assert!(result.is_ok());
        assert!(result.unwrap().embedding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_umap_from_knn_single_point() {
        // 1 row: degenerate trivial origin layout (mirrors test_umap_single_point).
        let idx = Array2::<i32>::zeros((1, 1));
        let dist = Array2::<f32>::zeros((1, 1));
        let result = Umap::builder_from_knn(idx, dist).build().unwrap();
        assert_eq!(result.embedding.nrows(), 1);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.embedding.iter().all(|&x| x == 0.0));
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

    #[test]
    fn test_umap_small_dataset_clamps_neighbors() {
        // Fewer rows than the default n_neighbors (15). Previously this errored;
        // now n_neighbors is clamped to n_samples - 1 and the run succeeds.
        let data = make_test_data(5, 8, 42);
        let result = Umap::builder(&data)
            .metric("cosine")
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 5);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.embedding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_umap_two_points() {
        // Smallest non-degenerate input: n_neighbors clamps to 1.
        let data = make_test_data(2, 4, 42);
        let result = Umap::builder(&data).random_state(42).build().unwrap();

        assert_eq!(result.embedding.nrows(), 2);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.embedding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_umap_single_point() {
        // 1 row: nothing to embed, returns a trivial origin layout.
        let data = make_test_data(1, 4, 42);
        let result = Umap::builder(&data).build().unwrap();

        assert_eq!(result.embedding.nrows(), 1);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_umap_empty() {
        // 0 rows: returns an empty embedding rather than erroring.
        let data = make_test_data(0, 4, 42);
        let result = Umap::builder(&data).build().unwrap();

        assert_eq!(result.embedding.nrows(), 0);
        assert_eq!(result.embedding.ncols(), 2);
    }

    #[test]
    fn test_stepper_advances() {
        let data = make_test_data(200, 10, 42);
        let mut setup = pollster::block_on(
            Umap::builder(&data)
                .n_neighbors(10)
                .n_epochs(50)
                .init_method(Init::Random)
                .random_state(42)
                .setup_async(),
        )
        .unwrap();

        assert_eq!(setup.optimizer.current_epoch(), 0);
        let before = setup.optimizer.embedding().clone();

        pollster::block_on(setup.optimizer.step(10));

        assert_eq!(setup.optimizer.current_epoch(), 10);
        let after = setup.optimizer.embedding();
        assert_eq!(after.nrows(), 200);
        assert_eq!(after.ncols(), 2);
        assert!(!after.iter().any(|x| x.is_nan()));

        let moved = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(moved, "step did not move the embedding");
    }

    #[test]
    fn test_set_optimizer_and_reinit_finite() {
        let data = make_test_data(150, 8, 7);
        let mut setup = pollster::block_on(
            Umap::builder(&data)
                .n_neighbors(10)
                .n_epochs(30)
                .init_method(Init::Random)
                .random_state(7)
                .setup_async(),
        )
        .unwrap();

        setup.optimizer.set_optimizer(Optimizer::Momentum);
        pollster::block_on(setup.optimizer.step(20));
        assert!(setup.optimizer.embedding().iter().all(|x| x.is_finite()));

        let mut rng = Xoshiro256StarStar::seed_from_u64(7);
        let fresh = random_layout(150, 2, &mut rng);
        setup.optimizer.reinit_embedding(fresh);
        assert_eq!(setup.optimizer.current_epoch(), 0);

        pollster::block_on(setup.optimizer.step(5));
        assert!(setup.optimizer.embedding().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_live_negative_sample_rate_reenables_repulsion() {
        // Regression: starting at negative_sample_rate = 0 and then raising it via
        // set_params must actually turn repulsion back on. Previously the per-edge
        // negative-sample counter was parked at +inf (CPU) / a stale baseline (GPU)
        // and a live rate change never re-armed it, so repulsion stayed off.
        //
        // Two optimizers step identically with repulsion off; then one enables it
        // live. Repulsion spreads points apart, so the enabled run must end with a
        // visibly larger layout than the still-attract-only reference.
        let build = || {
            pollster::block_on(
                Umap::builder(&make_test_data(200, 10, 99))
                    .n_neighbors(10)
                    .n_epochs(100)
                    .init_method(Init::Random)
                    .negative_sample_rate(0)
                    .random_state(99)
                    .setup_async(),
            )
            .unwrap()
        };
        let mut on = build();
        let mut off = build();

        // Identical attract-only warmup.
        pollster::block_on(on.optimizer.step(30));
        pollster::block_on(off.optimizer.step(30));

        // Enable repulsion live on one of them.
        let mut p = on.optimizer.params();
        p.negative_sample_rate = 5.0;
        on.optimizer.set_params(p);

        pollster::block_on(on.optimizer.step(40));
        pollster::block_on(off.optimizer.step(40)); // stays attract-only

        // Total per-coordinate variance as a spread metric.
        let spread = |setup: &UmapSetup| -> f32 {
            let emb = setup.optimizer.embedding();
            let n = emb.nrows() as f32;
            (0..emb.ncols())
                .map(|c| {
                    let col = emb.column(c);
                    let mean = col.sum() / n;
                    col.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n
                })
                .sum()
        };
        let s_on = spread(&on);
        let s_off = spread(&off);

        assert!(on.optimizer.embedding().iter().all(|x| x.is_finite()));
        assert!(
            s_on > s_off * 1.5,
            "enabling negative_sample_rate live did not increase spread: on={s_on}, off={s_off}"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_live_negative_sample_rate_reenables_repulsion_gpu() {
        // Same regression as the CPU test, on the resident GPU path: a live
        // negative_sample_rate change must re-seed the GPU `eonns` buffer so
        // repulsion actually re-arms (it previously kept a stale baseline).
        let build = || {
            pollster::block_on(
                Umap::builder(&make_test_data(300, 10, 21))
                    .n_neighbors(15)
                    .n_epochs(200)
                    .init_method(Init::Random)
                    .negative_sample_rate(0)
                    .gpu(true)
                    .random_state(21)
                    .setup_async(),
            )
            .unwrap()
        };
        let mut on = build();
        let mut off = build();

        pollster::block_on(on.optimizer.step(60));
        pollster::block_on(off.optimizer.step(60));

        let mut p = on.optimizer.params();
        p.negative_sample_rate = 5.0;
        on.optimizer.set_params(p);

        pollster::block_on(on.optimizer.step(80));
        pollster::block_on(off.optimizer.step(80));

        let spread = |setup: &UmapSetup| -> f32 {
            let emb = setup.optimizer.embedding();
            let n = emb.nrows() as f32;
            (0..emb.ncols())
                .map(|c| {
                    let col = emb.column(c);
                    let mean = col.sum() / n;
                    col.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n
                })
                .sum()
        };
        let s_on = spread(&on);
        let s_off = spread(&off);

        assert!(on.optimizer.embedding().iter().all(|x| x.is_finite()));
        assert!(
            s_on > s_off * 1.5,
            "GPU: enabling negative_sample_rate live did not increase spread: on={s_on}, off={s_off}"
        );
    }

    #[test]
    fn test_run_to_reaches_horizon() {
        let data = make_test_data(150, 8, 1);
        let mut setup = pollster::block_on(
            Umap::builder(&data)
                .n_neighbors(10)
                .n_epochs(50)
                .init_method(Init::Random)
                .random_state(1)
                .setup_async(),
        )
        .unwrap();

        let mut logger = Logger::new(false, None, UMAP_STAGES);
        let n_epochs = setup.n_epochs;
        pollster::block_on(setup.optimizer.run_to(n_epochs, &mut logger));

        assert_eq!(setup.optimizer.current_epoch() as usize, n_epochs);
        let emb = setup.optimizer.embedding();
        assert_eq!(emb.nrows(), 150);
        assert!(!emb.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_run_uses_live_learning_rate() {
        // Regression (#4): run_to anneals from the *live* learning rate, not a
        // frozen build-time value. With alpha set to 0 via set_params, the decay
        // starts at 0 and must not move the layout; before the unification run_to
        // used the frozen initial alpha and would have moved it.
        let data = make_test_data(150, 8, 5);
        let mut setup = pollster::block_on(
            Umap::builder(&data)
                .n_neighbors(10)
                .n_epochs(50)
                .init_method(Init::Random)
                .random_state(5)
                .setup_async(),
        )
        .unwrap();

        let before = setup.optimizer.embedding().clone();

        let mut p = setup.optimizer.params();
        p.alpha = 0.0;
        setup.optimizer.set_params(p);

        let mut logger = Logger::new(false, None, UMAP_STAGES);
        let n_epochs = setup.n_epochs;
        pollster::block_on(setup.optimizer.run_to(n_epochs, &mut logger));

        assert_eq!(setup.optimizer.current_epoch() as usize, n_epochs);
        let after = setup.optimizer.embedding();
        let max_delta = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta < 1e-6,
            "run_to moved the layout despite learningRate=0 (max delta {max_delta}); \
             it is not honoring the live learning rate"
        );
    }

    #[test]
    fn test_momentum_build_finite() {
        let data = make_test_data(150, 8, 3);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_epochs(50)
            .optimizer(Optimizer::Momentum)
            .random_state(3)
            .build()
            .unwrap();

        assert_eq!(result.embedding.nrows(), 150);
        assert_eq!(result.embedding.ncols(), 2);
        assert!(result.embedding.iter().all(|x| x.is_finite()));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_sgd_bounded() {
        let data = make_test_data(300, 10, 11);
        let result = Umap::builder(&data)
            .n_neighbors(15)
            .n_epochs(200)
            .gpu(true)
            .init_method(Init::Random)
            .random_state(11)
            .build()
            .unwrap();

        assert!(result.embedding.iter().all(|x| x.is_finite()));
        let max_abs = result.embedding.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!(
            max_abs < 1000.0,
            "GPU SGD max|coord| = {} (unbounded?)",
            max_abs
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_momentum_finite() {
        let data = make_test_data(300, 10, 13);
        let result = Umap::builder(&data)
            .n_neighbors(15)
            .n_epochs(200)
            .gpu(true)
            .optimizer(Optimizer::Momentum)
            .random_state(13)
            .build()
            .unwrap();

        assert!(result.embedding.iter().all(|x| x.is_finite()));
    }
}
