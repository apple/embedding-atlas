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
mod init;
pub mod layout_modern;
pub mod optimize;
mod recursive_init;
pub mod spectral;

#[cfg(feature = "gpu")]
pub mod gpu_layout_modern;
#[cfg(feature = "gpu")]
pub mod gpu_optimize;

use std::fmt;
use std::sync::Arc;

use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;
use nndescent::NNDescent;

pub use nndescent::{Logger, ProgressCallback};

use crate::graph::{
    fuzzy_simplicial_set, make_epochs_per_sample, normalize_knn_self_column, CsrMatrix,
    SparseMatrix,
};
use crate::init::{
    jitter_duplicate_rows, pca_layout, rescale_embedding, rescale_embedding_min_max,
};
use crate::recursive_init::{recursive_init_with_device, GpuDevice};
use crate::spectral::{
    noisy_scale_coords, random_layout, spectral_layout_with_centroids, tsw_spectral_layout,
    ComponentCentroids,
};

pub use crate::init::InitRescale;
pub use crate::layout_modern::ModernOptions;
pub use crate::optimize::{LiveParams, Optimizer, UmapOptimizer};
pub use crate::recursive_init::{
    AnchorMethod, CoarseLayoutWeights, CurveExponents, CurveSchedule, GoodInitializationPolicy,
    HubSelection, NegativeSelectionRangeMode, RecursiveInitParams,
};

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
    /// [`Init::Spectral`] when `compatibility_layout` is true, else [`Init::Recursive`].
    Auto,
    /// Principal components of the data, scaled to a small range with a little noise.
    /// Needs the data, so it is rejected by [`Umap::builder_from_knn`].
    Pca,
    /// Spectral embedding of the fuzzy simplicial set graph.
    Spectral,
    /// Spectral embedding whose eigensolver is warm-started by a truncated SVD of
    /// the graph Laplacian (umap-learn `tswspectral`).
    TswSpectral,
    /// Random uniform initialization in [-10, 10].
    Random,
    /// Recursive label-propagation coarsening of the fuzzy graph, anchored on a
    /// PCA of the data (umap-learn 0.6). See [`RecursiveInitParams`].
    Recursive,
    /// A caller-supplied embedding of shape `(n_samples, n_components)`.
    Custom(Array2<f32>),
}

/// Parses the names `auto`, `spectral`, `tswspectral`, `random`, `recursive` and
/// `pca`. [`Init::Custom`] has no name.
impl std::str::FromStr for Init {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "auto" => Ok(Self::Auto),
            "spectral" => Ok(Self::Spectral),
            "tswspectral" => Ok(Self::TswSpectral),
            "random" => Ok(Self::Random),
            "recursive" => Ok(Self::Recursive),
            "pca" => Ok(Self::Pca),
            other => Err(format!("unknown init method \"{other}\"")),
        }
    }
}

/// The high-dimensional input of a builder: borrowed from the caller, or shared.
enum InputData<'a> {
    Borrowed(&'a Array2<f32>),
    Shared(Arc<Array2<f32>>),
}

impl InputData<'_> {
    fn view(&self) -> &Array2<f32> {
        match self {
            InputData::Borrowed(data) => data,
            InputData::Shared(data) => data,
        }
    }

    /// A shared handle; borrowed data has to be copied once.
    fn shared(&self) -> Arc<Array2<f32>> {
        match self {
            InputData::Borrowed(data) => Arc::new((*data).clone()),
            InputData::Shared(data) => Arc::clone(data),
        }
    }
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
    data: Option<InputData<'a>>,
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
    init_rescale: InitRescale,
    a: Option<f32>,
    b: Option<f32>,
    disconnection_distance: Option<f32>,
    negative_selection_range: usize,
    negative_sample_scale: Option<f32>,
    exclude_graph_neighbors: bool,
    negative_sample_scale_adaptation_samples: usize,
    recursive_init: RecursiveInitParams,
    gpu: bool,
    optimizer: Option<Optimizer>,
    compatibility_layout: Option<bool>,
    init_scale_factor: f32,
    progress: Option<ProgressCallback>,
}

impl<'a> UmapBuilder<'a> {
    fn with_inputs(data: Option<InputData<'a>>, knn: Option<(Array2<i32>, Array2<f32>)>) -> Self {
        UmapBuilder {
            data,
            knn,
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
            init: Init::Auto,
            init_rescale: InitRescale::default(),
            a: None,
            b: None,
            disconnection_distance: None,
            negative_selection_range: 200_000,
            negative_sample_scale: None,
            exclude_graph_neighbors: false,
            negative_sample_scale_adaptation_samples: 128,
            recursive_init: RecursiveInitParams::default(),
            gpu: false,
            optimizer: None,
            compatibility_layout: None,
            init_scale_factor: 1.0,
            progress: None,
        }
    }

    fn validate(&self, n_samples: usize) -> Result<(), UmapError> {
        let invalid = |message: &str| Err(UmapError::InvalidParameter(message.to_string()));
        if !(0.0..=1.0).contains(&self.set_op_mix_ratio) {
            return invalid("set_op_mix_ratio must be between 0.0 and 1.0");
        }
        if self.repulsion_strength.is_nan() || self.repulsion_strength < 0.0 {
            return invalid("repulsion_strength cannot be negative");
        }
        if self
            .negative_sample_scale
            .is_some_and(|s| s.is_nan() || s < 0.0)
        {
            return invalid("negative_sample_scale cannot be negative");
        }
        if self.min_dist.is_nan() || self.min_dist < 0.0 {
            return invalid("min_dist cannot be negative");
        }
        if self.min_dist > self.spread {
            return invalid("min_dist must be less than or equal to spread");
        }
        if self.learning_rate.is_nan() || self.learning_rate < 0.0 {
            return invalid("learning_rate must be positive");
        }
        if self.compatibility_layout == Some(true) && self.requests_modern_optimizer() {
            return Err(UmapError::InvalidParameter(format!(
                "optimizer={:?} requires compatibility_layout=false; \
                 compatibility_layout=true always uses the legacy optimizer",
                self.optimizer.expect("modern optimizer requested")
            )));
        }
        if self.knn.is_none() && self.n_neighbors < 2 {
            return invalid("n_neighbors must be greater than 1");
        }
        if self.n_components == 0 {
            return invalid("n_components must be greater than 0");
        }
        if !self.init_scale_factor.is_finite() || self.init_scale_factor <= 0.0 {
            return invalid("init_scale_factor must be finite and greater than zero");
        }
        if !matches!(self.recursive_init.coarsening_ratio, 2..=4) {
            return invalid("recursive_coarsening_ratio must be one of 2, 3, or 4");
        }
        if self.a.is_some_and(|a| !a.is_finite() || a <= 0.0)
            || self.b.is_some_and(|b| !b.is_finite() || b <= 0.0)
        {
            return invalid("a and b must be finite and greater than zero");
        }
        if self.disconnection_distance.is_some_and(f32::is_nan) {
            return invalid("disconnection_distance must be a number");
        }
        match &self.init {
            Init::Custom(embedding) if embedding.ncols() != self.n_components => {
                invalid("init array must match n_components value")
            }
            Init::Custom(embedding) if embedding.nrows() != n_samples => {
                invalid("init array must have one row per sample")
            }
            Init::Pca if self.data.is_none() => invalid("init \"pca\" requires the data"),
            _ => Ok(()),
        }
    }

    fn resolved_init(&self) -> Init {
        match &self.init {
            Init::Auto if self.uses_compatibility_layout() => Init::Spectral,
            Init::Auto => Init::Recursive,
            init => init.clone(),
        }
    }

    fn init_plan(&self, a: f64, b: f64) -> InitPlan {
        InitPlan {
            init: self.resolved_init(),
            init_rescale: self.init_rescale,
            repulsion_strength: self.repulsion_strength,
            init_scale_factor: self.init_scale_factor,
            compatibility_layout: self.uses_compatibility_layout(),
            n_components: self.n_components,
            a,
            b,
            recursive_init: self.recursive_init.clone(),
            metric: self.metric.clone(),
            random_state: self.random_state,
        }
    }

    fn resolved_disconnection_distance(&self) -> f32 {
        self.disconnection_distance.unwrap_or(match self.knn {
            Some(_) => f32::INFINITY,
            None => default_disconnection_distance(&self.metric),
        })
    }

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

    /// Initialization method. Default: [`Init::Auto`].
    pub fn init_method(mut self, init: Init) -> Self {
        self.init = init;
        self
    }

    /// How the initial embedding is scaled. Only used when `compatibility_layout`
    /// is false. Default: [`InitRescale::Isotropic`].
    pub fn init_rescale(mut self, method: InitRescale) -> Self {
        self.init_rescale = method;
        self
    }

    /// Parameter `a` of the low-dimensional curve `1 / (1 + a * x^(2b))`. Used
    /// only together with [`b`](Self::b); otherwise both are fitted from
    /// `min_dist` and `spread`.
    pub fn a(mut self, a: f32) -> Self {
        self.a = Some(a);
        self
    }

    /// Parameter `b` of the low-dimensional curve. See [`a`](Self::a).
    pub fn b(mut self, b: f32) -> Self {
        self.b = Some(b);
        self
    }

    /// Neighbors at this distance or farther are removed before the fuzzy graph is
    /// built. Default: infinity, except 2 for `correlation` and `cosine`, and 1
    /// for `hellinger`, `jaccard`, `bit_jaccard` and `dice`. Ignored for a
    /// precomputed kNN graph unless set.
    pub fn disconnection_distance(mut self, distance: f32) -> Self {
        self.disconnection_distance = Some(distance);
        self
    }

    /// Size of the window of nodes that negative samples are drawn from in the
    /// modern optimizer. Default: 200000.
    pub fn negative_selection_range(mut self, range: usize) -> Self {
        self.negative_selection_range = range;
        self
    }

    /// Fixed scale of the repulsive force of the modern optimizer. Default: derived
    /// from the selection range.
    pub fn negative_sample_scale(mut self, scale: f32) -> Self {
        self.negative_sample_scale = Some(scale);
        self
    }

    /// Skip negative samples that are graph neighbors of the source in the modern
    /// optimizer. Default: false.
    pub fn exclude_graph_neighbors(mut self, exclude: bool) -> Self {
        self.exclude_graph_neighbors = exclude;
        self
    }

    /// Sources sampled to adapt the negative sample scale of the Adam optimizer;
    /// 0 disables the adaptation. Default: 128.
    pub fn negative_sample_scale_adaptation_samples(mut self, samples: usize) -> Self {
        self.negative_sample_scale_adaptation_samples = samples;
        self
    }

    /// Parameters of [`Init::Recursive`]. Default: [`RecursiveInitParams::default`].
    pub fn recursive_init_params(mut self, params: RecursiveInitParams) -> Self {
        self.recursive_init = params;
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

    /// Optimizer to use for the layout. Default: [`Optimizer::Adam`] when
    /// `compatibility_layout` is false, else [`Optimizer::Sgd`]. Adam and Momentum
    /// need the modern pipeline and are rejected with `compatibility_layout(true)`.
    pub fn optimizer(mut self, o: Optimizer) -> Self {
        self.optimizer = Some(o);
        self
    }

    fn requests_modern_optimizer(&self) -> bool {
        matches!(self.optimizer, Some(Optimizer::Adam | Optimizer::Momentum))
    }

    /// Resolved like umap-learn's `_resolve_layout_options`: unset means the modern
    /// pipeline exactly when a modern optimizer is requested.
    fn uses_compatibility_layout(&self) -> bool {
        self.compatibility_layout
            .unwrap_or(!self.requests_modern_optimizer())
    }

    fn resolved_optimizer(&self) -> Optimizer {
        self.optimizer
            .unwrap_or(if self.uses_compatibility_layout() {
                Optimizer::Sgd
            } else {
                Optimizer::Adam
            })
    }

    /// Use the umap-learn 0.5 layout pipeline (`true`) or the 0.6 modern pipeline
    /// (`false`: modern NNDescent parameters, spectral noise and isotropic init
    /// rescaling). Default: unset, which selects the modern pipeline only when
    /// the optimizer is Adam or Momentum.
    pub fn compatibility_layout(self, compatibility_layout: bool) -> Self {
        self.compatibility_layout_option(Some(compatibility_layout))
    }

    /// Like [`compatibility_layout`](Self::compatibility_layout); `None` leaves the
    /// pipeline to be chosen from the optimizer.
    pub fn compatibility_layout_option(mut self, compatibility_layout: Option<bool>) -> Self {
        self.compatibility_layout = compatibility_layout;
        self
    }

    /// Multiplier of the isotropic init rescale. Only used when
    /// `compatibility_layout` is false. Default: 1.0.
    pub fn init_scale_factor(mut self, factor: f32) -> Self {
        self.init_scale_factor = factor;
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
        let (mut setup, mut logger) = self.prepare().await?;

        logger.push_stage_with_message(
            STAGE_OPTIMIZATION,
            &format!("Optimizing layout for {} epochs...", setup.n_epochs),
        );
        setup.optimizer.run_to(setup.n_epochs, &mut logger).await;
        logger.pop_stage();

        Ok(UmapResult {
            embedding: setup.optimizer.into_embedding(),
            knn_indices: setup.knn_indices,
            knn_distances: setup.knn_distances,
        })
    }

    /// Build the graph + initial embedding and return a resumable [`UmapSetup`],
    /// **without** running the layout optimization. Use the returned
    /// [`UmapOptimizer`] to step/run interactively.
    pub async fn setup_async(self) -> Result<UmapSetup, UmapError> {
        Ok(self.prepare().await?.0)
    }

    /// Stages 1–7: nearest neighbors, fuzzy graph, prune, init, edge schedule.
    /// Shared by [`build_async`](Self::build_async) and [`setup_async`](Self::setup_async).
    async fn prepare(mut self) -> Result<(UmapSetup, Logger), UmapError> {
        // Take ownership of any supplied kNN graph up front, so the Step 1 branch
        // below can move it without conflicting with the n_samples decision.
        let knn = self.knn.take();
        let optimizer = self.resolved_optimizer();

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
                .as_ref()
                .expect("UmapBuilder has neither data nor a kNN graph")
                .view()
                .nrows(),
        };

        self.validate(n_samples)?;
        let disconnection_distance = self.resolved_disconnection_distance();
        let mut logger = Logger::new(self.verbose, self.progress.take(), UMAP_STAGES);

        // Degenerate inputs have nothing to embed: with 0 or 1 points there are no
        // neighbor relationships to model. Return a trivial origin layout (empty
        // graph/knn, zero epochs) rather than erroring, so callers don't have to
        // special-case tiny datasets.
        if n_samples <= 1 {
            let optimizer = build_optimizer(
                Vec::new(),
                Vec::new(),
                &[],
                Array2::zeros((n_samples, self.n_components)),
                LiveParams {
                    alpha: self.learning_rate,
                    gamma: self.repulsion_strength,
                    negative_sample_rate: self.negative_sample_rate as f32,
                    a: 1.0,
                    b: 1.0,
                },
                optimizer,
                [0, 0, 0],
                0,
                None,
            );
            let setup = UmapSetup {
                optimizer,
                graph: SparseMatrix::from_coo(n_samples, n_samples, Vec::new()).to_csr(),
                knn_indices: Array2::zeros((n_samples, 0)),
                knn_distances: Array2::zeros((n_samples, 0)),
                n_epochs: 0,
                init_plan: self.init_plan(1.0, 1.0),
            };
            return Ok((setup, logger));
        }

        // Compute curve parameters a, b (independent of the neighbor count).
        let (a, b) = match (self.a, self.b) {
            (Some(a), Some(b)) => (a as f64, b as f64),
            _ => find_ab_params(self.spread, self.min_dist),
        };

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
                    .as_ref()
                    .expect("UmapBuilder has neither data nor a kNN graph")
                    .shared();

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

                let mut nnd_builder = NNDescent::builder_shared(data, &self.metric, n_neighbors)
                    .verbose(self.verbose)
                    .gpu(self.gpu)
                    .progress_option(nn_cb);
                if let Some(seed) = self.random_state {
                    nnd_builder = nnd_builder.random_state(seed);
                }
                let (n_trees, max_candidates, n_iters) =
                    nndescent_params(n_samples, n_neighbors, self.uses_compatibility_layout());
                nnd_builder = nnd_builder
                    .n_trees(n_trees)
                    .max_candidates(max_candidates)
                    .n_iters(n_iters);
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
        let (graph_indices, graph_dists) =
            disconnect_far_neighbors(&knn_indices, &knn_dists, disconnection_distance);
        let mut graph = fuzzy_simplicial_set(
            &graph_indices,
            &graph_dists,
            n_neighbors,
            self.set_op_mix_ratio,
            self.local_connectivity,
        );

        // Step 3: Determine number of epochs
        let n_epochs = self
            .n_epochs
            .unwrap_or(if n_samples <= 10000 { 500 } else { 200 });

        // Step 4: Prune weak edges
        let default_epochs = if n_samples <= 10000 { 500 } else { 200 };
        let pruning_epochs = if n_epochs > 10 {
            n_epochs
        } else {
            default_epochs
        };
        graph.prune(graph.max_weight() / pruning_epochs as f32);
        logger.pop_stage();

        // CSR of the pruned graph: used for spectral init and retained for reset.
        let csr = graph.to_csr();

        // Step 5: Initialize embedding
        logger.push_stage_with_message(STAGE_EMBEDDING_INIT, "Initializing embedding...");
        let mut rng = match self.random_state {
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
            None => Xoshiro256StarStar::seed_from_os(),
        };

        let init_plan = self.init_plan(a, b);
        let good_initialization = init_plan.good_initialization();
        #[cfg(feature = "gpu")]
        let gpu_device = if self.gpu {
            crate::gpu_optimize::request_device().await
        } else {
            None
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_device: Option<GpuDevice> = None;
        let embedding = init_plan
            .embedding(
                &csr,
                self.data.as_ref().map(InputData::view),
                &mut rng,
                &mut logger,
                gpu_device.clone(),
            )
            .await?;
        self.data = None;
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

        let mut optimizer = build_optimizer(
            heads,
            tails,
            &epochs_per_sample,
            embedding,
            LiveParams {
                alpha: self.learning_rate,
                gamma: self.repulsion_strength,
                negative_sample_rate: self.negative_sample_rate as f32,
                a: a as f32,
                b: b as f32,
            },
            optimizer,
            rng_state,
            n_neighbors,
            gpu_device,
        );
        optimizer.set_good_initialization(good_initialization);
        optimizer.set_modern_options(ModernOptions {
            negative_selection_range: self.negative_selection_range,
            negative_sample_scale: self.negative_sample_scale,
            scale_adaptation_samples: self.negative_sample_scale_adaptation_samples,
            exclude_graph_neighbors: self.exclude_graph_neighbors,
        });
        optimizer.set_n_epochs(n_epochs);

        let setup = UmapSetup {
            optimizer,
            graph: csr,
            knn_indices,
            knn_distances: knn_dists,
            n_epochs,
            init_plan,
        };
        Ok((setup, logger))
    }
}

/// Python's `DISCONNECTION_DISTANCES`: bounded metrics disconnect at their maximum.
fn default_disconnection_distance(metric: &str) -> f32 {
    match metric {
        "correlation" | "cosine" => 2.0,
        "hellinger" | "jaccard" | "bit_jaccard" | "dice" => 1.0,
        _ => f32::INFINITY,
    }
}

/// Copy of a kNN graph in which neighbors at `distance` or farther are marked
/// missing (index -1, distance infinity).
fn disconnect_far_neighbors(
    indices: &Array2<i32>,
    dists: &Array2<f32>,
    distance: f32,
) -> (Array2<i32>, Array2<f32>) {
    let mut indices = indices.clone();
    let mut dists = dists.clone();
    if distance.is_finite() {
        for (index, dist) in indices.iter_mut().zip(dists.iter_mut()) {
            if *dist >= distance {
                *index = -1;
                *dist = f32::INFINITY;
            }
        }
    }
    (indices, dists)
}

/// NNDescent parameters chosen by umap-learn's `nearest_neighbors`:
/// `(n_trees, max_candidates, n_iters)`.
fn nndescent_params(
    n_samples: usize,
    n_neighbors: usize,
    compatibility_layout: bool,
) -> (usize, usize, usize) {
    let n = n_samples as f64;
    let tree_bonus = (n.sqrt() / 20.0).round_ties_even();
    let (n_trees, max_candidates) = if compatibility_layout {
        ((5.0 + tree_bonus).min(64.0) as usize, 60)
    } else {
        (
            (5.0 + tree_bonus).clamp(4.0, 8.0) as usize,
            ((1.5 * n_neighbors as f64) as usize).clamp(20, 60),
        )
    };
    let n_iters = (n.log2().round_ties_even() as usize).max(5);
    (n_trees, max_candidates, n_iters)
}

/// Construct a [`UmapOptimizer`] from prepared pieces. A `gpu_device` is only
/// ever present when the `gpu` feature is enabled and a device was requested.
#[allow(clippy::too_many_arguments)]
fn build_optimizer(
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: &[f32],
    embedding: Array2<f32>,
    params: LiveParams,
    optimizer: Optimizer,
    rng_state: [i64; 3],
    n_neighbors: usize,
    gpu_device: Option<GpuDevice>,
) -> UmapOptimizer {
    #[cfg(feature = "gpu")]
    {
        UmapOptimizer::new_with_device(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
            n_neighbors,
            gpu_device,
        )
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (n_neighbors, gpu_device);
        UmapOptimizer::new(
            head,
            tail,
            epochs_per_sample,
            embedding,
            params,
            optimizer,
            rng_state,
        )
    }
}

/// How the initial embedding is computed and scaled. Retained after setup so the
/// embedding can be re-initialized with another method (see
/// [`InitPlan::initial_embedding`]).
#[derive(Clone, Debug)]
pub struct InitPlan {
    init: Init,
    init_rescale: InitRescale,
    repulsion_strength: f32,
    init_scale_factor: f32,
    compatibility_layout: bool,
    n_components: usize,
    a: f64,
    b: f64,
    recursive_init: RecursiveInitParams,
    metric: String,
    random_state: Option<u64>,
}

impl InitPlan {
    /// Replace the initialization method. [`Init::Auto`] resolves by
    /// `compatibility_layout`.
    pub fn set_init(&mut self, init: Init) {
        self.init = match init {
            Init::Auto if self.compatibility_layout => Init::Spectral,
            Init::Auto => Init::Recursive,
            init => init,
        };
    }

    /// Whether the initial layout is good enough for the shorter Adam warm-up.
    pub fn good_initialization(&self) -> bool {
        matches!(self.init, Init::Recursive)
    }

    /// Compute a fresh, rescaled initial embedding of the fuzzy `graph`. `data` is
    /// required for [`Init::Pca`] and improves [`Init::Recursive`] and the spectral
    /// methods on disconnected graphs. Seeded by `random_state`.
    pub fn initial_embedding(
        &self,
        graph: &CsrMatrix,
        data: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>, UmapError> {
        let mut rng = match self.random_state {
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
            None => Xoshiro256StarStar::seed_from_os(),
        };
        let mut logger = Logger::new(false, None, &[]);
        pollster::block_on(self.embedding(graph, data, &mut rng, &mut logger, None))
    }

    async fn embedding(
        &self,
        graph: &CsrMatrix,
        data: Option<&Array2<f32>>,
        rng: &mut Xoshiro256StarStar,
        logger: &mut Logger,
        device: Option<GpuDevice>,
    ) -> Result<Array2<f32>, UmapError> {
        let centroids = data.map(|data| ComponentCentroids {
            data,
            metric: &self.metric,
        });
        let embedding = match &self.init {
            Init::Spectral => {
                logger.log("Computing spectral initialization...");
                let mut emb =
                    spectral_layout_with_centroids(graph, centroids, self.n_components, rng);
                let is_random_fallback = {
                    let min_val = emb.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = emb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    max_val - min_val > 15.0
                };
                if is_random_fallback {
                    logger
                        .log("WARNING: Spectral initialization failed, fell back to random layout");
                }
                if self.compatibility_layout {
                    noisy_scale_coords(&mut emb, rng, 10.0, 0.0001);
                } else {
                    noisy_scale_coords(&mut emb, rng, 40.0, 0.01);
                }
                emb
            }
            Init::TswSpectral => {
                logger.log("Computing truncated-SVD spectral initialization...");
                let mut emb = tsw_spectral_layout(graph, centroids, self.n_components, rng);
                let max_coord = if self.compatibility_layout {
                    10.0
                } else {
                    20.0
                };
                noisy_scale_coords(&mut emb, rng, max_coord, 0.0001);
                emb
            }
            Init::Random => random_layout(graph.nrows, self.n_components, rng),
            Init::Pca => {
                logger.log("Computing PCA initialization...");
                let data = data.ok_or_else(|| {
                    UmapError::InvalidParameter("init \"pca\" requires the data".to_string())
                })?;
                let mut emb = pca_layout(data, self.n_components, rng);
                let max_coord = if self.compatibility_layout {
                    10.0
                } else {
                    20.0
                };
                noisy_scale_coords(&mut emb, rng, max_coord, 0.0001);
                emb
            }
            Init::Custom(embedding) => jitter_duplicate_rows(embedding.clone(), rng),
            Init::Auto => unreachable!("resolved_init never returns Auto"),
            Init::Recursive => {
                logger.log("Computing recursive initialization...");
                recursive_init_with_device(
                    graph,
                    data,
                    self.a,
                    self.b,
                    self.n_components,
                    &self.recursive_init,
                    rng,
                    device,
                )
                .await
            }
        };
        let mut embedding = embedding;
        if self.compatibility_layout {
            rescale_embedding_min_max(&mut embedding, 10.0);
        } else {
            rescale_embedding(
                &mut embedding,
                self.init_rescale,
                self.repulsion_strength,
                self.init_scale_factor,
                graph,
            );
        }
        Ok(embedding)
    }
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
    /// How the initial embedding was computed; reuse it to re-initialize.
    pub init_plan: InitPlan,
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
        UmapBuilder::with_inputs(Some(InputData::Borrowed(data)), None)
    }

    /// Like [`builder`](Self::builder), but takes shared ownership of the data. The
    /// neighbor search then reads it in place instead of copying it, and the builder
    /// frees its handle once the initial layout exists.
    pub fn builder_shared(data: Arc<Array2<f32>>) -> UmapBuilder<'static> {
        UmapBuilder::with_inputs(Some(InputData::Shared(data)), None)
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
        UmapBuilder::with_inputs(None, Some((knn_indices, knn_distances)))
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
    fn test_modern_layout_builds_finite_embedding() {
        let data = make_test_data(200, 10, 42);
        let result = Umap::builder(&data)
            .n_neighbors(10)
            .n_epochs(30)
            .compatibility_layout(false)
            .random_state(42)
            .build()
            .unwrap();

        assert_eq!(result.embedding.dim(), (200, 2));
        assert!(result.embedding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_modern_nndescent_params() {
        assert_eq!(nndescent_params(100, 15, false), (5, 22, 7));
        assert_eq!(nndescent_params(1_000_000, 15, false), (8, 22, 20));
        assert_eq!(nndescent_params(16, 5, false), (5, 20, 5));
        assert_eq!(nndescent_params(100, 100, false), (5, 60, 7));
    }

    #[test]
    fn test_compatibility_nndescent_params() {
        assert_eq!(nndescent_params(100, 15, true), (5, 60, 7));
        assert_eq!(nndescent_params(1_000_000, 15, true), (55, 60, 20));
        assert_eq!(nndescent_params(100_000_000, 15, true), (64, 60, 27));
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

    mod python_parameters {
        use super::*;
        use crate::init::quantile;

        fn setup(builder: UmapBuilder<'_>) -> UmapSetup {
            pollster::block_on(builder.setup_async()).unwrap()
        }

        fn invalid_message(builder: UmapBuilder<'_>) -> String {
            match builder.build() {
                Err(UmapError::InvalidParameter(message)) => message,
                other => panic!("expected InvalidParameter, got {:?}", other.map(|_| ())),
            }
        }

        fn column_spread(embedding: &Array2<f32>, column: usize) -> f32 {
            let values = embedding.column(column).to_vec();
            quantile(&values, 0.95) - quantile(&values, 0.05)
        }

        #[test]
        fn invalid_parameters_are_rejected_like_python() {
            let data = make_test_data(60, 5, 1);
            let cases: Vec<(UmapBuilder<'_>, &str)> = vec![
                (
                    Umap::builder(&data).set_op_mix_ratio(1.5),
                    "set_op_mix_ratio",
                ),
                (
                    Umap::builder(&data).set_op_mix_ratio(-0.1),
                    "set_op_mix_ratio",
                ),
                (
                    Umap::builder(&data).repulsion_strength(-1.0),
                    "repulsion_strength",
                ),
                (
                    Umap::builder(&data).negative_sample_scale(-1.0),
                    "negative_sample_scale",
                ),
                (
                    Umap::builder(&data).min_dist(-0.1),
                    "min_dist cannot be negative",
                ),
                (
                    Umap::builder(&data).min_dist(2.0).spread(1.0),
                    "less than or equal to spread",
                ),
                (Umap::builder(&data).learning_rate(-1.0), "learning_rate"),
                (Umap::builder(&data).n_neighbors(1), "n_neighbors"),
                (Umap::builder(&data).n_components(0), "n_components"),
                (
                    Umap::builder(&data).init_scale_factor(0.0),
                    "init_scale_factor",
                ),
                (
                    Umap::builder(&data).init_scale_factor(f32::NAN),
                    "init_scale_factor",
                ),
                (Umap::builder(&data).a(0.0).b(1.0), "a and b"),
                (
                    Umap::builder(&data).disconnection_distance(f32::NAN),
                    "disconnection_distance",
                ),
                (
                    Umap::builder(&data).init_method(Init::Custom(Array2::zeros((60, 3)))),
                    "match n_components",
                ),
                (
                    Umap::builder(&data).init_method(Init::Custom(Array2::zeros((59, 2)))),
                    "one row per sample",
                ),
            ];
            for (builder, expected) in cases {
                let message = invalid_message(builder);
                assert!(
                    message.contains(expected),
                    "{message} should mention {expected}"
                );
            }
        }

        #[test]
        fn recursive_coarsening_ratio_must_be_two_three_or_four() {
            let data = make_test_data(60, 5, 1);
            for ratio in [1, 5] {
                let params = RecursiveInitParams {
                    coarsening_ratio: ratio,
                    ..RecursiveInitParams::default()
                };
                let message = invalid_message(Umap::builder(&data).recursive_init_params(params));
                assert!(message.contains("recursive_coarsening_ratio"));
            }
            let params = RecursiveInitParams {
                coarsening_ratio: 2,
                ..RecursiveInitParams::default()
            };
            assert!(Umap::builder(&data)
                .n_epochs(5)
                .recursive_init_params(params)
                .build()
                .is_ok());
        }

        #[test]
        fn pca_init_needs_data() {
            let (indices, distances) = make_knn(&make_test_data(40, 3, 2), 6);
            let message =
                invalid_message(Umap::builder_from_knn(indices, distances).init_method(Init::Pca));
            assert!(message.contains("pca"));
        }

        #[test]
        fn explicit_a_and_b_override_the_fit() {
            let data = make_test_data(80, 5, 3);
            let fitted = setup(Umap::builder(&data).n_epochs(5).min_dist(0.5))
                .optimizer
                .params();
            let (a, b) = find_ab_params(1.0, 0.5);
            assert!((fitted.a - a as f32).abs() < 1e-6 && (fitted.b - b as f32).abs() < 1e-6);

            let overridden = setup(Umap::builder(&data).n_epochs(5).min_dist(0.5).a(2.5).b(0.7))
                .optimizer
                .params();
            assert_eq!((overridden.a, overridden.b), (2.5, 0.7));

            let only_a = setup(Umap::builder(&data).n_epochs(5).min_dist(0.5).a(2.5))
                .optimizer
                .params();
            assert_eq!((only_a.a, only_a.b), (fitted.a, fitted.b));
        }

        #[test]
        fn disconnection_distance_removes_far_edges() {
            let data = make_test_data(120, 4, 4);
            let (indices, distances) = make_knn(&data, 10);
            let edges =
                |builder: UmapBuilder<'static>| setup(builder.n_epochs(50)).graph.indices.len();

            let all = edges(Umap::builder_from_knn(indices.clone(), distances.clone()));
            let median = quantile(&distances.iter().copied().collect::<Vec<_>>(), 0.5);
            let some = edges(
                Umap::builder_from_knn(indices.clone(), distances.clone())
                    .disconnection_distance(median),
            );
            let none =
                edges(Umap::builder_from_knn(indices, distances).disconnection_distance(0.0));
            assert!(none < some && some < all, "{none} < {some} < {all}");
            assert_eq!(none, 0);
        }

        #[test]
        fn disconnection_distance_defaults_follow_the_metric() {
            assert_eq!(default_disconnection_distance("cosine"), 2.0);
            assert_eq!(default_disconnection_distance("correlation"), 2.0);
            for metric in ["hellinger", "jaccard", "bit_jaccard", "dice"] {
                assert_eq!(default_disconnection_distance(metric), 1.0);
            }
            assert!(default_disconnection_distance("euclidean").is_infinite());

            let data = make_test_data(10, 3, 5);
            let cosine = Umap::builder(&data).metric("cosine");
            assert_eq!(cosine.resolved_disconnection_distance(), 2.0);
            let overridden = Umap::builder(&data)
                .metric("cosine")
                .disconnection_distance(0.5);
            assert_eq!(overridden.resolved_disconnection_distance(), 0.5);
            let (indices, distances) = make_knn(&data, 4);
            assert!(Umap::builder_from_knn(indices, distances)
                .resolved_disconnection_distance()
                .is_infinite());
        }

        #[test]
        fn disconnecting_marks_neighbors_missing() {
            let indices = arr2(&[[0, 1, 2], [1, 0, 2]]);
            let distances = arr2(&[[0.0f32, 0.5, 2.0], [0.0, 1.5, 3.0]]);
            let (new_indices, new_distances) = disconnect_far_neighbors(&indices, &distances, 1.5);
            assert_eq!(new_indices, arr2(&[[0, 1, -1], [1, -1, -1]]));
            assert_eq!(new_distances[[0, 2]], f32::INFINITY);
            assert_eq!(new_distances[[0, 1]], 0.5);
            let (same, _) = disconnect_far_neighbors(&indices, &distances, f32::INFINITY);
            assert_eq!(same, indices);
        }

        #[test]
        fn auto_init_follows_compatibility_layout() {
            let data = make_test_data(20, 3, 6);
            let compat = Umap::builder(&data);
            assert!(matches!(compat.resolved_init(), Init::Spectral));
            let modern = Umap::builder(&data).compatibility_layout(false);
            assert!(matches!(modern.resolved_init(), Init::Recursive));
            let explicit = Umap::builder(&data).init_method(Init::Pca);
            assert!(matches!(explicit.resolved_init(), Init::Pca));
        }

        #[test]
        fn pca_init_follows_the_top_variance_axis() {
            let mut rng = Xoshiro256StarStar::seed_from_u64(9);
            let data = Array2::from_shape_fn((300, 6), |(_, d)| {
                rng.random_f32() * [0.1, 0.2, 30.0, 0.1, 0.3, 0.1][d]
            });
            let setup = setup(
                Umap::builder(&data)
                    .init_method(Init::Pca)
                    .random_state(1)
                    .n_neighbors(10),
            );
            let embedding = setup.optimizer.embedding();
            assert!(embedding.iter().all(|x| x.is_finite()));

            let correlation = correlation(&data.column(2).to_vec(), &embedding.column(0).to_vec());
            assert!(correlation.abs() > 0.99, "correlation {correlation}");
        }

        fn correlation(a: &[f32], b: &[f32]) -> f32 {
            let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            let (mean_a, mean_b) = (mean(a), mean(b));
            let (mut covariance, mut var_a, mut var_b) = (0.0f32, 0.0f32, 0.0f32);
            for (x, y) in a.iter().zip(b) {
                covariance += (x - mean_a) * (y - mean_b);
                var_a += (x - mean_a).powi(2);
                var_b += (y - mean_b).powi(2);
            }
            covariance / (var_a * var_b).sqrt()
        }

        #[test]
        fn custom_init_keeps_the_given_layout() {
            let data = make_test_data(50, 4, 7);
            let init = Array2::from_shape_fn((50, 2), |(i, d)| (i * 2 + d) as f32 * 0.1);
            let setup = setup(
                Umap::builder(&data)
                    .init_method(Init::Custom(init.clone()))
                    .n_neighbors(8),
            );
            for column in 0..2 {
                let correlation = correlation(
                    &init.column(column).to_vec(),
                    &setup.optimizer.embedding().column(column).to_vec(),
                );
                assert!(correlation > 0.9999, "correlation {correlation}");
            }
        }

        #[test]
        fn init_rescale_methods_set_their_target_spread() {
            let data = make_test_data(400, 5, 8);
            let target = 400f32.log10() * 3.0;
            let build = |method| {
                setup(
                    Umap::builder(&data)
                        .compatibility_layout(false)
                        .init_method(Init::Random)
                        .init_rescale(method)
                        .random_state(3),
                )
            };

            let isotropic = build(InitRescale::Isotropic);
            let embedding = isotropic.optimizer.embedding();
            let geometric_mean = (column_spread(embedding, 0) * column_spread(embedding, 1)).sqrt();
            assert!((geometric_mean - target).abs() < 1e-2);

            let per_axis = build(InitRescale::PerAxis);
            for column in 0..2 {
                let spread = column_spread(per_axis.optimizer.embedding(), column);
                assert!((spread - target).abs() < 1e-2);
            }

            let graph_edge = build(InitRescale::GraphEdge);
            let embedding = graph_edge.optimizer.embedding();
            let graph = &graph_edge.graph;
            let mut lengths_and_weights: Vec<(f32, f32)> = (0..graph.nrows)
                .flat_map(|row| (graph.indptr[row]..graph.indptr[row + 1]).map(move |e| (row, e)))
                .map(|(row, e)| {
                    let delta = &embedding.row(row) - &embedding.row(graph.indices[e]);
                    (delta.mapv(|x| x * x).sum().sqrt(), graph.data[e])
                })
                .collect();
            lengths_and_weights.sort_by(|a, b| a.0.total_cmp(&b.0));
            let total: f32 = lengths_and_weights.iter().map(|p| p.1).sum();
            let mut cumulative = 0.0;
            let median = lengths_and_weights
                .iter()
                .find(|p| {
                    cumulative += p.1;
                    cumulative >= total / 2.0
                })
                .unwrap()
                .0;
            assert!((median - 1.0).abs() < 1e-3, "median edge length {median}");
        }

        #[test]
        fn modern_optimizer_options_reach_the_adam_optimizer() {
            let large = make_test_data(1500, 6, 10);
            let small = make_test_data(300, 6, 11);
            let run = |data: &Array2<f32>, configure: fn(UmapBuilder<'_>) -> UmapBuilder<'_>| {
                configure(
                    Umap::builder(data)
                        .compatibility_layout(false)
                        .init_method(Init::Random)
                        .n_epochs(15)
                        .random_state(2),
                )
                .build()
                .unwrap()
                .embedding
            };
            let baseline = run(&large, |b| b);
            assert!(baseline.iter().all(|x| x.is_finite()));
            assert_eq!(baseline, run(&large, |b| b));
            assert_ne!(baseline, run(&large, |b| b.negative_sample_scale(0.25)));
            assert_ne!(baseline, run(&large, |b| b.negative_selection_range(100)));
            assert_ne!(
                run(&large, |b| b.negative_selection_range(100)),
                run(&large, |b| b
                    .negative_selection_range(100)
                    .negative_sample_scale_adaptation_samples(0))
            );
            assert_ne!(
                run(&small, |b| b),
                run(&small, |b| b.exclude_graph_neighbors(true))
            );
        }

        #[test]
        fn unset_compatibility_layout_follows_the_optimizer() {
            let data = make_test_data(20, 3, 12);
            let layout = |builder: UmapBuilder<'_>| {
                (
                    builder.uses_compatibility_layout(),
                    builder.resolved_optimizer(),
                )
            };
            assert_eq!(layout(Umap::builder(&data)), (true, Optimizer::Sgd));
            let sgd = Umap::builder(&data).optimizer(Optimizer::Sgd);
            assert_eq!(layout(sgd), (true, Optimizer::Sgd));
            for optimizer in [Optimizer::Adam, Optimizer::Momentum] {
                let modern = Umap::builder(&data).optimizer(optimizer);
                assert_eq!(layout(modern), (false, optimizer));
            }
            let explicit_modern = Umap::builder(&data).compatibility_layout(false);
            assert_eq!(layout(explicit_modern), (false, Optimizer::Adam));
            let modern_with_sgd = Umap::builder(&data)
                .compatibility_layout(false)
                .optimizer(Optimizer::Sgd);
            assert_eq!(layout(modern_with_sgd), (false, Optimizer::Sgd));
        }

        #[test]
        fn modern_optimizer_with_compatibility_layout_true_is_rejected() {
            let data = make_test_data(30, 3, 13);
            for optimizer in [Optimizer::Adam, Optimizer::Momentum] {
                let message = invalid_message(
                    Umap::builder(&data)
                        .compatibility_layout(true)
                        .optimizer(optimizer),
                );
                assert!(
                    message.contains("requires compatibility_layout=false"),
                    "{message}"
                );
            }
            assert!(Umap::builder(&data)
                .compatibility_layout(true)
                .optimizer(Optimizer::Sgd)
                .n_epochs(5)
                .build()
                .is_ok());
        }

        #[test]
        fn custom_init_with_coincident_rows_is_jittered() {
            let data = make_test_data(50, 4, 14);
            let init = Array2::from_shape_fn((50, 2), |(i, d)| (i.max(1) * 2 + d) as f32);
            let setup = setup(
                Umap::builder(&data)
                    .init_method(Init::Custom(init))
                    .n_neighbors(8)
                    .random_state(1),
            );
            let embedding = setup.optimizer.embedding();
            assert_ne!(embedding.row(0), embedding.row(1));
        }
    }
}
