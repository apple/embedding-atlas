use std::sync::Arc;

use ndarray::Array2;
use wasm_bindgen::prelude::*;

use nndescent::NNDescent;
use umap::graph::CsrMatrix;
use umap::{
    find_ab_params, AnchorMethod, CoarseLayoutWeights, CurveExponents, CurveSchedule,
    GoodInitializationPolicy, HubSelection, Init, InitPlan, InitRescale,
    NegativeSelectionRangeMode, Optimizer, RecursiveInitParams, Umap as UmapCore,
    UmapBuilder as UmapCoreBuilder, UmapOptimizer,
};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

#[wasm_bindgen(start)]
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        console_error(&format!("umap-wasm panicked: {info}"))
    }));
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_data(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Array2<f32>, JsError> {
    parse_owned_data(data.to_vec(), n_rows, n_cols)
}

fn parse_owned_data(data: Vec<f32>, n_rows: usize, n_cols: usize) -> Result<Array2<f32>, JsError> {
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
    Array2::from_shape_vec((n_rows, n_cols), data).map_err(|e| JsError::new(&e.to_string()))
}

fn parse<T: std::str::FromStr<Err = String>>(name: &str) -> Result<T, JsError> {
    name.parse()
        .map_err(|message: String| JsError::new(&message))
}

/// Layout and initialization options shared by both UMAP builders. `None` keeps
/// the core default.
struct Tuning {
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    learning_rate: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    local_connectivity: f32,
    set_op_mix_ratio: f32,
    init: Option<Init>,
    optimizer: Option<Optimizer>,
    compatibility_layout: Option<bool>,
    init_scale_factor: f32,
    random_state: Option<u64>,
    verbose: bool,
    gpu: bool,
    progress: Option<js_sys::Function>,
    a: Option<f32>,
    b: Option<f32>,
    disconnection_distance: Option<f32>,
    init_rescale: Option<InitRescale>,
    negative_selection_range: Option<usize>,
    negative_sample_scale: Option<f32>,
    exclude_graph_neighbors: Option<bool>,
    negative_sample_scale_adaptation_samples: Option<usize>,
    recursive_init: Option<RecursiveInitParams>,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            init: None,
            optimizer: None,
            compatibility_layout: None,
            init_scale_factor: 1.0,
            random_state: None,
            verbose: false,
            gpu: false,
            progress: None,
            a: None,
            b: None,
            disconnection_distance: None,
            init_rescale: None,
            negative_selection_range: None,
            negative_sample_scale: None,
            exclude_graph_neighbors: None,
            negative_sample_scale_adaptation_samples: None,
            recursive_init: None,
        }
    }
}

impl Tuning {
    fn apply<'a>(&self, mut builder: UmapCoreBuilder<'a>) -> UmapCoreBuilder<'a> {
        builder = builder
            .min_dist(self.min_dist)
            .spread(self.spread)
            .learning_rate(self.learning_rate)
            .negative_sample_rate(self.negative_sample_rate)
            .repulsion_strength(self.repulsion_strength)
            .local_connectivity(self.local_connectivity)
            .set_op_mix_ratio(self.set_op_mix_ratio)
            .compatibility_layout_option(self.compatibility_layout)
            .init_scale_factor(self.init_scale_factor)
            .gpu(self.gpu)
            .verbose(self.verbose);
        if let Some(a) = self.a {
            builder = builder.a(a);
        }
        if let Some(b) = self.b {
            builder = builder.b(b);
        }
        if let Some(distance) = self.disconnection_distance {
            builder = builder.disconnection_distance(distance);
        }
        if let Some(method) = self.init_rescale {
            builder = builder.init_rescale(method);
        }
        if let Some(range) = self.negative_selection_range {
            builder = builder.negative_selection_range(range);
        }
        if let Some(scale) = self.negative_sample_scale {
            builder = builder.negative_sample_scale(scale);
        }
        if let Some(exclude) = self.exclude_graph_neighbors {
            builder = builder.exclude_graph_neighbors(exclude);
        }
        if let Some(samples) = self.negative_sample_scale_adaptation_samples {
            builder = builder.negative_sample_scale_adaptation_samples(samples);
        }
        if let Some(params) = &self.recursive_init {
            builder = builder.recursive_init_params(params.clone());
        }
        if let Some(init) = &self.init {
            builder = builder.init_method(init.clone());
        }
        if let Some(optimizer) = self.optimizer {
            builder = builder.optimizer(optimizer);
        }
        if let Some(n) = self.n_epochs {
            builder = builder.n_epochs(n);
        }
        if let Some(seed) = self.random_state {
            builder = builder.random_state(seed);
        }
        if let Some(callback) = &self.progress {
            builder = builder.progress(js_to_progress_callback(callback.clone()));
        }
        builder
    }

    /// Apply the options to `builder`, run the setup and wrap the result. `n_cols`
    /// is the number of input features, 0 for a kNN graph.
    async fn finish(self, builder: UmapCoreBuilder<'_>, n_cols: usize) -> Result<Umap, JsError> {
        let setup = self
            .apply(builder)
            .setup_async()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let n_rows = setup.optimizer.embedding().nrows();
        let n_components = setup.optimizer.embedding().ncols();
        let n_neighbors_out = setup.knn_indices.ncols();

        Ok(Umap {
            optimizer: setup.optimizer,
            graph: setup.graph,
            init_plan: setup.init_plan,
            n_cols,
            knn_indices: setup.knn_indices.into_raw_vec_and_offset().0,
            knn_distances: setup.knn_distances.into_raw_vec_and_offset().0,
            n_neighbors_out,
            n_rows,
            n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            n_epochs: setup.n_epochs,
            progress: self.progress,
            verbose: self.verbose,
        })
    }
}

fn js_field(options: &JsValue, key: &str) -> Result<Option<JsValue>, JsError> {
    let value = js_sys::Reflect::get(options, &JsValue::from_str(key))
        .map_err(|_| JsError::new(&format!("cannot read recursiveInit.{key}")))?;
    Ok((!value.is_undefined() && !value.is_null()).then_some(value))
}

fn js_number(options: &JsValue, key: &str) -> Result<Option<f64>, JsError> {
    js_field(options, key)?
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| JsError::new(&format!("recursiveInit.{key} must be a number")))
        })
        .transpose()
}

fn js_bool(options: &JsValue, key: &str) -> Result<Option<bool>, JsError> {
    js_field(options, key)?
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| JsError::new(&format!("recursiveInit.{key} must be a boolean")))
        })
        .transpose()
}

fn js_choice<T: Copy>(
    options: &JsValue,
    key: &str,
    choices: &[(&str, T)],
) -> Result<Option<T>, JsError> {
    let Some(value) = js_field(options, key)? else {
        return Ok(None);
    };
    let name = value
        .as_string()
        .ok_or_else(|| JsError::new(&format!("recursiveInit.{key} must be a string")))?;
    choices
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, choice)| Some(*choice))
        .ok_or_else(|| JsError::new(&format!("unknown recursiveInit.{key} \"{name}\"")))
}

fn parse_curve_schedule(options: &JsValue) -> Result<Option<CurveSchedule>, JsError> {
    let Some(value) = js_field(options, "curveSchedule")? else {
        return Ok(None);
    };
    if let Some(name) = value.as_string() {
        return match name.as_str() {
            "strongToOne" => Ok(Some(CurveSchedule::StrongToOne)),
            "original" => Ok(Some(CurveSchedule::Original)),
            other => Err(JsError::new(&format!(
                "unknown recursiveInit.curveSchedule \"{other}\""
            ))),
        };
    }
    let exponent = |key: &str| {
        js_number(&value, key)?
            .ok_or_else(|| JsError::new(&format!("recursiveInit.curveSchedule.{key} is required")))
    };
    Ok(Some(CurveSchedule::Custom(CurveExponents {
        initial_a: exponent("initialA")?,
        initial_b: exponent("initialB")?,
        recursive_a: exponent("recursiveA")?,
        recursive_b: exponent("recursiveB")?,
    })))
}

fn parse_recursive_init(options: &JsValue) -> Result<RecursiveInitParams, JsError> {
    let mut params = RecursiveInitParams::default();
    if let Some(v) = js_number(options, "baseInitThreshold")? {
        params.base_init_threshold = v as usize;
    }
    if let Some(v) = js_number(options, "nEmbeddingEpochs")? {
        params.n_embedding_epochs = v as usize;
    }
    if let Some(v) = js_number(options, "approxNParts")? {
        params.approx_n_parts = Some(v as usize);
    }
    if let Some(v) = js_number(options, "coarseningRatio")? {
        params.coarsening_ratio = v as usize;
    }
    if let Some(v) = js_choice(
        options,
        "hubSelection",
        &[
            ("degree", HubSelection::Degree),
            ("diverse", HubSelection::Diverse),
            ("random", HubSelection::Random),
        ],
    )? {
        params.hub_selection = v;
    }
    if let Some(v) = js_number(options, "anchorSampleSize")? {
        params.anchor_sample_size = v as usize;
    }
    if let Some(v) = js_number(options, "repulsionStrength")? {
        params.repulsion_strength = v as f32;
    }
    if let Some(v) = js_number(options, "negativeSampleRate")? {
        params.negative_sample_rate = v as f32;
    }
    if let Some(v) = js_number(options, "negativeSelectionRangeScale")? {
        params.negative_selection_range_scale = v;
    }
    if let Some(v) = js_number(options, "labelPropagationShare")? {
        params.label_propagation_share = v;
    }
    if let Some(v) = js_choice(
        options,
        "coarseLayoutWeights",
        &[
            ("transition", CoarseLayoutWeights::Transition),
            ("fuzzyUnion", CoarseLayoutWeights::FuzzyUnion),
        ],
    )? {
        params.coarse_layout_weights = v;
    }
    if let Some(v) = js_bool(options, "coarsenOnLayoutWeights")? {
        params.coarsen_on_layout_weights = v;
    }
    if let Some(v) = js_number(options, "maxPropagationSweeps")? {
        params.max_propagation_sweeps = v as usize;
    }
    if let Some(v) = parse_curve_schedule(options)? {
        params.curve_schedule = v;
    }
    if let Some(v) = js_choice(
        options,
        "goodInitializationPolicy",
        &[
            ("heuristic", GoodInitializationPolicy::Heuristic),
            ("available", GoodInitializationPolicy::Available),
            ("always", GoodInitializationPolicy::Always),
            ("never", GoodInitializationPolicy::Never),
        ],
    )? {
        params.good_initialization_policy = v;
    }
    if let Some(v) = js_bool(options, "removeCoarseDiagonal")? {
        params.remove_coarse_diagonal = v;
    }
    if let Some(v) = js_choice(
        options,
        "anchorMethod",
        &[
            ("sampledPca", AnchorMethod::SampledPca),
            ("projectedPca", AnchorMethod::ProjectedPca),
            ("fullPca", AnchorMethod::FullPca),
        ],
    )? {
        params.anchor_method = v;
    }
    if let Some(v) = js_choice(
        options,
        "negativeSelectionRangeMode",
        &[
            ("coarseN", NegativeSelectionRangeMode::CoarseN),
            ("fixed200k", NegativeSelectionRangeMode::Fixed200k),
            ("scaledCoarse", NegativeSelectionRangeMode::ScaledCoarse),
        ],
    )? {
        params.negative_selection_range_mode = v;
    }
    Ok(params)
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
    metric: String,
    n_neighbors: usize,
    tuning: Tuning,
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
    pub fn new(data: Vec<f32>, n_rows: usize, n_cols: usize, n_components: usize) -> UMAPBuilder {
        UMAPBuilder {
            data,
            n_rows,
            n_cols,
            n_components,
            metric: "euclidean".to_string(),
            n_neighbors: 15,
            tuning: Tuning::default(),
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
        self.tuning.min_dist = d;
        self
    }

    /// Effective scale of embedded points. Default: 1.0.
    pub fn spread(mut self, s: f32) -> UMAPBuilder {
        self.tuning.spread = s;
        self
    }

    /// Number of optimization epochs. Default: auto.
    #[wasm_bindgen(js_name = "nEpochs")]
    pub fn n_epochs(mut self, n: usize) -> UMAPBuilder {
        self.tuning.n_epochs = Some(n);
        self
    }

    /// Initial learning rate. Default: 1.0.
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, lr: f32) -> UMAPBuilder {
        self.tuning.learning_rate = lr;
        self
    }

    /// Negative samples per positive sample. Default: 5.
    #[wasm_bindgen(js_name = "negativeSampleRate")]
    pub fn negative_sample_rate(mut self, r: usize) -> UMAPBuilder {
        self.tuning.negative_sample_rate = r;
        self
    }

    /// Weight of repulsive force. Default: 1.0.
    #[wasm_bindgen(js_name = "repulsionStrength")]
    pub fn repulsion_strength(mut self, s: f32) -> UMAPBuilder {
        self.tuning.repulsion_strength = s;
        self
    }

    /// Local connectivity constraint. Default: 1.0.
    #[wasm_bindgen(js_name = "localConnectivity")]
    pub fn local_connectivity(mut self, c: f32) -> UMAPBuilder {
        self.tuning.local_connectivity = c;
        self
    }

    /// Interpolation between fuzzy union and intersection. Default: 1.0.
    #[wasm_bindgen(js_name = "mixRatio")]
    pub fn mix_ratio(mut self, r: f32) -> UMAPBuilder {
        self.tuning.set_op_mix_ratio = r;
        self
    }

    /// Initialization method: "auto", "spectral", "tswspectral", "random",
    /// "recursive" or "pca" ("pca" needs the data, so it fails for a kNN graph).
    /// Default: "auto", which is "spectral" with `compatibilityLayout`, else
    /// "recursive".
    #[wasm_bindgen(js_name = "initMethod")]
    pub fn init_method(mut self, method: &str) -> Result<UMAPBuilder, JsError> {
        self.tuning.init = Some(parse(method)?);
        Ok(self)
    }

    /// Optimizer: "sgd", "momentum" or "adam". Default: "sgd" with
    /// `compatibilityLayout`, else "adam".
    pub fn optimizer(mut self, optimizer: &str) -> Result<UMAPBuilder, JsError> {
        self.tuning.optimizer = Some(parse(optimizer)?);
        Ok(self)
    }

    /// Use the umap-learn 0.5 layout pipeline (true) or the 0.6 pipeline (false).
    /// Default: unset, which selects the 0.6 pipeline only for the "adam" and
    /// "momentum" optimizers. Those optimizers fail with `true`.
    #[wasm_bindgen(js_name = "compatibilityLayout")]
    pub fn compatibility_layout(mut self, compatibility_layout: bool) -> UMAPBuilder {
        self.tuning.compatibility_layout = Some(compatibility_layout);
        self
    }

    /// Multiplier of the isotropic init rescale. Only used when
    /// `compatibilityLayout` is false. Default: 1.0.
    #[wasm_bindgen(js_name = "initScaleFactor")]
    pub fn init_scale_factor(mut self, factor: f32) -> UMAPBuilder {
        self.tuning.init_scale_factor = factor;
        self
    }

    /// Curve parameter `a`. Used only together with `b`; otherwise both are
    /// fitted from `minDist` and `spread`.
    pub fn a(mut self, a: f32) -> UMAPBuilder {
        self.tuning.a = Some(a);
        self
    }

    /// Curve parameter `b`. See `a`.
    pub fn b(mut self, b: f32) -> UMAPBuilder {
        self.tuning.b = Some(b);
        self
    }

    /// Neighbors at this distance or farther are dropped from the fuzzy graph.
    /// Default: depends on the metric.
    #[wasm_bindgen(js_name = "disconnectionDistance")]
    pub fn disconnection_distance(mut self, distance: f32) -> UMAPBuilder {
        self.tuning.disconnection_distance = Some(distance);
        self
    }

    /// How the initial layout is scaled with `compatibilityLayout` false:
    /// "isotropic" (default), "perAxis" or "graphEdge".
    #[wasm_bindgen(js_name = "initRescale")]
    pub fn init_rescale(mut self, method: &str) -> Result<UMAPBuilder, JsError> {
        self.tuning.init_rescale = Some(parse(method)?);
        Ok(self)
    }

    /// Window of nodes that negative samples are drawn from. Default: 200000.
    #[wasm_bindgen(js_name = "negativeSelectionRange")]
    pub fn negative_selection_range(mut self, range: usize) -> UMAPBuilder {
        self.tuning.negative_selection_range = Some(range);
        self
    }

    /// Fixed scale of the repulsive force. Default: derived from the selection range.
    #[wasm_bindgen(js_name = "negativeSampleScale")]
    pub fn negative_sample_scale(mut self, scale: f32) -> UMAPBuilder {
        self.tuning.negative_sample_scale = Some(scale);
        self
    }

    /// Skip negative samples that are graph neighbors of the source. Default: false.
    #[wasm_bindgen(js_name = "excludeGraphNeighbors")]
    pub fn exclude_graph_neighbors(mut self, exclude: bool) -> UMAPBuilder {
        self.tuning.exclude_graph_neighbors = Some(exclude);
        self
    }

    /// Sources sampled to adapt the negative sample scale of Adam; 0 disables
    /// the adaptation. Default: 128.
    #[wasm_bindgen(js_name = "negativeSampleScaleAdaptationSamples")]
    pub fn negative_sample_scale_adaptation_samples(mut self, samples: usize) -> UMAPBuilder {
        self.tuning.negative_sample_scale_adaptation_samples = Some(samples);
        self
    }

    /// Options of the recursive initialization, an object with camelCase keys of
    /// the core `RecursiveInitParams`. Missing keys keep their defaults.
    #[wasm_bindgen(js_name = "recursiveInit")]
    pub fn recursive_init(mut self, options: JsValue) -> Result<UMAPBuilder, JsError> {
        self.tuning.recursive_init = Some(parse_recursive_init(&options)?);
        Ok(self)
    }

    /// Random seed for reproducibility. Default: None (random).
    #[wasm_bindgen(js_name = "randomState")]
    pub fn random_state(mut self, seed: i64) -> UMAPBuilder {
        self.tuning.random_state = parse_seed(seed);
        self
    }

    /// Enable verbose output. Default: false.
    pub fn verbose(mut self, v: bool) -> UMAPBuilder {
        self.tuning.verbose = v;
        self
    }

    /// Enable GPU acceleration via WebGPU. Default: false.
    /// Requires the `gpu` feature.
    pub fn gpu(mut self, g: bool) -> UMAPBuilder {
        self.tuning.gpu = g;
        self
    }

    /// Set a progress callback: `(progress: number, stage: string) => void`.
    /// `progress` is in [0, 1], `stage` describes the current processing phase.
    pub fn progress(mut self, callback: js_sys::Function) -> UMAPBuilder {
        self.tuning.progress = Some(callback);
        self
    }

    /// Build the graph + initial embedding (eager) and return a stateful [`Umap`].
    ///
    /// The embedding and kNN graph are valid immediately at epoch 0; call
    /// [`Umap::run`] to anneal to completion or [`Umap::step`] to advance
    /// interactively.
    pub async fn build(mut self) -> Result<Umap, JsError> {
        let array = parse_owned_data(std::mem::take(&mut self.data), self.n_rows, self.n_cols)?;
        let builder = UmapCore::builder_shared(Arc::new(array))
            .n_components(self.n_components)
            .n_neighbors(self.n_neighbors)
            .metric(&self.metric);
        self.tuning.finish(builder, self.n_cols).await
    }
}

/// A stateful, resumable UMAP. The embedding/kNN are valid immediately after
/// [`UMAPBuilder::build`] (eager setup); `run`/`step` advance the layout in place.
#[wasm_bindgen]
pub struct Umap {
    optimizer: UmapOptimizer,
    /// Pruned fuzzy graph, retained for re-initialization in `reset`.
    graph: CsrMatrix,
    init_plan: InitPlan,
    /// Number of input features; 0 when built from a kNN graph.
    n_cols: usize,
    knn_indices: Vec<i32>,
    knn_distances: Vec<f32>,
    n_neighbors_out: usize,
    n_rows: usize,
    n_components: usize,
    min_dist: f32,
    spread: f32,
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

    /// Switch optimizer: "sgd", "momentum" (umap-learn 0.6 heavy-ball momentum) or
    /// "adam". The new optimizer starts with fresh state.
    #[wasm_bindgen(js_name = "setOptimizer")]
    pub fn set_optimizer(&mut self, optimizer: &str) -> Result<(), JsError> {
        self.optimizer.set_optimizer(parse(optimizer)?);
        Ok(())
    }

    /// Re-initialize the embedding and restart the schedule from epoch 0.
    /// `method` is "auto", "spectral", "tswspectral", "random", "recursive" or
    /// "pca". `data` is the input data of the build (n_rows * n_cols); the
    /// instance does not keep it. "pca" needs it. "recursive" uses it for its PCA
    /// anchor and falls back to a random anchor without it, like an instance
    /// built from a kNN graph. The others ignore it.
    pub fn reset(&mut self, method: &str, data: Option<Vec<f32>>) -> Result<(), JsError> {
        let init = parse(method)?;
        if data.is_none() && matches!(init, Init::Pca) {
            return Err(JsError::new(&format!(
                "reset(\"{method}\") needs the input data as the second argument"
            )));
        }
        let array = match data {
            Some(_) if self.n_cols == 0 => {
                return Err(JsError::new(
                    "reset data was given, but this instance was built from a kNN graph",
                ))
            }
            Some(data) => Some(parse_owned_data(data, self.n_rows, self.n_cols)?),
            None => None,
        };
        self.init_plan.set_init(init);
        let embedding = self
            .init_plan
            .initial_embedding(&self.graph, array.as_ref())
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.optimizer
            .set_good_initialization(self.init_plan.good_initialization());
        self.optimizer.reinit_embedding(embedding);
        Ok(())
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
    tuning: Tuning,
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
            tuning: Tuning::default(),
        }
    }

    /// Minimum distance between points in embedding. Default: 0.1.
    #[wasm_bindgen(js_name = "minDist")]
    pub fn min_dist(mut self, d: f32) -> UMAPFromKnnBuilder {
        self.tuning.min_dist = d;
        self
    }

    /// Effective scale of embedded points. Default: 1.0.
    pub fn spread(mut self, s: f32) -> UMAPFromKnnBuilder {
        self.tuning.spread = s;
        self
    }

    /// Number of optimization epochs. Default: auto.
    #[wasm_bindgen(js_name = "nEpochs")]
    pub fn n_epochs(mut self, n: usize) -> UMAPFromKnnBuilder {
        self.tuning.n_epochs = Some(n);
        self
    }

    /// Initial learning rate. Default: 1.0.
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, lr: f32) -> UMAPFromKnnBuilder {
        self.tuning.learning_rate = lr;
        self
    }

    /// Negative samples per positive sample. Default: 5.
    #[wasm_bindgen(js_name = "negativeSampleRate")]
    pub fn negative_sample_rate(mut self, r: usize) -> UMAPFromKnnBuilder {
        self.tuning.negative_sample_rate = r;
        self
    }

    /// Weight of repulsive force. Default: 1.0.
    #[wasm_bindgen(js_name = "repulsionStrength")]
    pub fn repulsion_strength(mut self, s: f32) -> UMAPFromKnnBuilder {
        self.tuning.repulsion_strength = s;
        self
    }

    /// Local connectivity constraint. Default: 1.0.
    #[wasm_bindgen(js_name = "localConnectivity")]
    pub fn local_connectivity(mut self, c: f32) -> UMAPFromKnnBuilder {
        self.tuning.local_connectivity = c;
        self
    }

    /// Interpolation between fuzzy union and intersection. Default: 1.0.
    #[wasm_bindgen(js_name = "mixRatio")]
    pub fn mix_ratio(mut self, r: f32) -> UMAPFromKnnBuilder {
        self.tuning.set_op_mix_ratio = r;
        self
    }

    /// Initialization method: "auto", "spectral", "tswspectral", "random",
    /// "recursive" or "pca" ("pca" needs the data, so it fails for a kNN graph).
    /// Default: "auto", which is "spectral" with `compatibilityLayout`, else
    /// "recursive".
    #[wasm_bindgen(js_name = "initMethod")]
    pub fn init_method(mut self, method: &str) -> Result<UMAPFromKnnBuilder, JsError> {
        self.tuning.init = Some(parse(method)?);
        Ok(self)
    }

    /// Optimizer: "sgd", "momentum" or "adam". Default: "sgd" with
    /// `compatibilityLayout`, else "adam".
    pub fn optimizer(mut self, optimizer: &str) -> Result<UMAPFromKnnBuilder, JsError> {
        self.tuning.optimizer = Some(parse(optimizer)?);
        Ok(self)
    }

    /// Use the umap-learn 0.5 layout pipeline (true) or the 0.6 pipeline (false).
    /// Default: unset, which selects the 0.6 pipeline only for the "adam" and
    /// "momentum" optimizers. Those optimizers fail with `true`.
    #[wasm_bindgen(js_name = "compatibilityLayout")]
    pub fn compatibility_layout(mut self, compatibility_layout: bool) -> UMAPFromKnnBuilder {
        self.tuning.compatibility_layout = Some(compatibility_layout);
        self
    }

    /// Multiplier of the isotropic init rescale. Only used when
    /// `compatibilityLayout` is false. Default: 1.0.
    #[wasm_bindgen(js_name = "initScaleFactor")]
    pub fn init_scale_factor(mut self, factor: f32) -> UMAPFromKnnBuilder {
        self.tuning.init_scale_factor = factor;
        self
    }

    /// Curve parameter `a`. Used only together with `b`; otherwise both are
    /// fitted from `minDist` and `spread`.
    pub fn a(mut self, a: f32) -> UMAPFromKnnBuilder {
        self.tuning.a = Some(a);
        self
    }

    /// Curve parameter `b`. See `a`.
    pub fn b(mut self, b: f32) -> UMAPFromKnnBuilder {
        self.tuning.b = Some(b);
        self
    }

    /// Neighbors at this distance or farther are dropped from the fuzzy graph.
    /// Default: depends on the metric.
    #[wasm_bindgen(js_name = "disconnectionDistance")]
    pub fn disconnection_distance(mut self, distance: f32) -> UMAPFromKnnBuilder {
        self.tuning.disconnection_distance = Some(distance);
        self
    }

    /// How the initial layout is scaled with `compatibilityLayout` false:
    /// "isotropic" (default), "perAxis" or "graphEdge".
    #[wasm_bindgen(js_name = "initRescale")]
    pub fn init_rescale(mut self, method: &str) -> Result<UMAPFromKnnBuilder, JsError> {
        self.tuning.init_rescale = Some(parse(method)?);
        Ok(self)
    }

    /// Window of nodes that negative samples are drawn from. Default: 200000.
    #[wasm_bindgen(js_name = "negativeSelectionRange")]
    pub fn negative_selection_range(mut self, range: usize) -> UMAPFromKnnBuilder {
        self.tuning.negative_selection_range = Some(range);
        self
    }

    /// Fixed scale of the repulsive force. Default: derived from the selection range.
    #[wasm_bindgen(js_name = "negativeSampleScale")]
    pub fn negative_sample_scale(mut self, scale: f32) -> UMAPFromKnnBuilder {
        self.tuning.negative_sample_scale = Some(scale);
        self
    }

    /// Skip negative samples that are graph neighbors of the source. Default: false.
    #[wasm_bindgen(js_name = "excludeGraphNeighbors")]
    pub fn exclude_graph_neighbors(mut self, exclude: bool) -> UMAPFromKnnBuilder {
        self.tuning.exclude_graph_neighbors = Some(exclude);
        self
    }

    /// Sources sampled to adapt the negative sample scale of Adam; 0 disables
    /// the adaptation. Default: 128.
    #[wasm_bindgen(js_name = "negativeSampleScaleAdaptationSamples")]
    pub fn negative_sample_scale_adaptation_samples(
        mut self,
        samples: usize,
    ) -> UMAPFromKnnBuilder {
        self.tuning.negative_sample_scale_adaptation_samples = Some(samples);
        self
    }

    /// Options of the recursive initialization, an object with camelCase keys of
    /// the core `RecursiveInitParams`. Missing keys keep their defaults.
    #[wasm_bindgen(js_name = "recursiveInit")]
    pub fn recursive_init(mut self, options: JsValue) -> Result<UMAPFromKnnBuilder, JsError> {
        self.tuning.recursive_init = Some(parse_recursive_init(&options)?);
        Ok(self)
    }

    /// Random seed for reproducibility. Default: None (random).
    #[wasm_bindgen(js_name = "randomState")]
    pub fn random_state(mut self, seed: i64) -> UMAPFromKnnBuilder {
        self.tuning.random_state = parse_seed(seed);
        self
    }

    /// Enable verbose output. Default: false.
    pub fn verbose(mut self, v: bool) -> UMAPFromKnnBuilder {
        self.tuning.verbose = v;
        self
    }

    /// Enable GPU acceleration via WebGPU for the layout optimization. Default: false.
    pub fn gpu(mut self, g: bool) -> UMAPFromKnnBuilder {
        self.tuning.gpu = g;
        self
    }

    /// Set a progress callback: `(progress: number, stage: string) => void`.
    pub fn progress(mut self, callback: js_sys::Function) -> UMAPFromKnnBuilder {
        self.tuning.progress = Some(callback);
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

        let builder = UmapCore::builder_from_knn(idx, dist).n_components(self.n_components);
        self.tuning.finish(builder, 0).await
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
        data: Vec<f32>,
        n_rows: usize,
        n_cols: usize,
        metric: &str,
        n_neighbors: usize,
    ) -> NNDescentBuilder {
        NNDescentBuilder {
            data,
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
    pub async fn build(mut self) -> Result<NNDescentIndex, JsError> {
        let array = parse_owned_data(std::mem::take(&mut self.data), self.n_rows, self.n_cols)?;

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
