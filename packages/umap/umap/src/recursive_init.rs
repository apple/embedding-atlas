//! Recursive label-propagation initialization (umap-learn 0.6 `init="recursive"`).
//!
//! The fuzzy graph is coarsened by label propagation from high-degree hubs into
//! about `1 / coarsening_ratio` as many parts, recursively, until it has at most
//! `base_init_threshold` nodes. That graph is placed from a PCA anchor of the
//! data. Each coarse layout is refined with a short Adam optimization, expanded
//! back to the finer level, and Procrustes-aligned with the anchor.

use std::future::Future;
use std::pin::Pin;

use ndarray::{s, Array2, ArrayView2, Axis, Zip};
use nndescent::rng::{TauRng, Xoshiro256StarStar};
use rayon::prelude::*;

use crate::graph::CsrMatrix;
use crate::init::quantile;
use crate::layout_modern::{ModernLayout, ModernLayoutParams, ModernOptimizerKind};
use crate::spectral::eigh_jacobi;

const RNG_STATE_MIN: i64 = i32::MIN as i64 + 1;
const RNG_STATE_MAX: i64 = i32::MAX as i64 - 1;
const ANCHOR_SPAN: f32 = 10.0;
const PCA_OVERSAMPLES: usize = 8;
const PCA_POWER_ITERATIONS: usize = 1;
const OUTLIER_SEARCH_LIMIT: usize = 100;
const STALL_CHECK_INTERVAL: usize = 5;
const COARSE_LAYOUT_LEARNING_RATE: f32 = 0.5;
const SCALE_ADAPTATION_SAMPLES: usize = 128;
/// Coarse layouts with at least this many nodes run on the GPU when a device is
/// available; below it the GPU setup cost outweighs the speedup.
#[cfg(feature = "gpu")]
const GPU_COARSE_LAYOUT_MIN_NODES: usize = 2048;

/// GPU device and queue shared with the optimizer.
#[cfg(feature = "gpu")]
pub(crate) type GpuDevice = (wgpu::Device, wgpu::Queue);
/// Placeholder so the init path has one signature with and without the `gpu` feature.
#[cfg(not(feature = "gpu"))]
pub(crate) type GpuDevice = std::convert::Infallible;

/// How label-propagation seeds are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HubSelection {
    /// The nodes with the highest total edge weight.
    Degree,
    /// High-degree nodes that are not adjacent to each other where possible.
    Diverse,
    /// Uniformly random nodes; collisions can yield slightly fewer parts.
    Random,
}

/// Edge weights of the coarse graphs that are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseLayoutWeights {
    /// Fuzzy union of the random-walk transition probabilities between parts.
    Transition,
    /// The coarsening graph `1 - prod(1 - w_ij)` over all edges between two parts.
    FuzzyUnion,
}

/// Exponents applied to the curve parameters `a` and `b` of the coarse layouts:
/// the first level uses `a^initial_a` and `b^initial_b`, every deeper level raises
/// its parent's parameters to `recursive_a` and `recursive_b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveExponents {
    pub initial_a: f64,
    pub initial_b: f64,
    pub recursive_a: f64,
    pub recursive_b: f64,
}

/// The exponents of the coarse layouts, by name or given explicitly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurveSchedule {
    /// All four exponents are 1/4.
    StrongToOne,
    /// All four exponents are 1: the output curve is used at every level.
    Original,
    Custom(CurveExponents),
}

impl CurveSchedule {
    fn exponents(self) -> CurveExponents {
        let uniform = |exponent| CurveExponents {
            initial_a: exponent,
            initial_b: exponent,
            recursive_a: exponent,
            recursive_b: exponent,
        };
        match self {
            CurveSchedule::StrongToOne => uniform(0.25),
            CurveSchedule::Original => uniform(1.0),
            CurveSchedule::Custom(exponents) => exponents,
        }
    }
}

/// When the coarse layouts use the shorter Adam warm-up for a good initialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GoodInitializationPolicy {
    /// When the next level has more than `base_init_threshold` parts.
    #[default]
    Heuristic,
    /// When the coordinates are finite and every axis has a spread.
    Available,
    Always,
    Never,
}

/// How the PCA anchor of the data is computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnchorMethod {
    /// PCA scores of at most `anchor_sample_size` sampled rows; other rows are
    /// placed from their neighbors.
    #[default]
    SampledPca,
    /// Components fitted on the sample, projected onto every row.
    ProjectedPca,
    /// Behaves like [`AnchorMethod::ProjectedPca`], as in umap-learn.
    FullPca,
}

/// Size of the window that coarse layouts draw negative samples from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NegativeSelectionRangeMode {
    /// All coarse nodes.
    CoarseN,
    /// A fixed window of 200000 nodes.
    Fixed200k,
    /// `negative_selection_range_scale` times the number of coarse nodes.
    #[default]
    ScaledCoarse,
}

/// Parameters of [`crate::Init::Recursive`]. The defaults match umap-learn 0.6.
#[derive(Clone, Debug)]
pub struct RecursiveInitParams {
    /// Graphs with at most this many nodes are placed from the anchor directly.
    pub base_init_threshold: usize,
    /// Epochs of the finest coarse layout; each coarser level uses `sqrt(2)` times more.
    pub n_embedding_epochs: usize,
    /// Number of parts at the first level; `None` uses
    /// `max(base_init_threshold, n_samples / coarsening_ratio)`.
    pub approx_n_parts: Option<usize>,
    /// Approximate reduction in size per level.
    pub coarsening_ratio: usize,
    pub hub_selection: HubSelection,
    /// Number of rows used for the anchor PCA.
    pub anchor_sample_size: usize,
    /// Repulsion strength (gamma) of the coarse layouts.
    pub repulsion_strength: f32,
    /// Negative samples per positive sample in the coarse layouts.
    pub negative_sample_rate: f32,
    /// Fraction of a coarse graph used as its negative selection range.
    pub negative_selection_range_scale: f64,
    /// Minimum share of a node's edge weight that a label needs to be adopted,
    /// on levels that use transition weights.
    pub label_propagation_share: f64,
    pub coarse_layout_weights: CoarseLayoutWeights,
    /// Coarsen, propagate and expand the deeper levels on the layout weights.
    pub coarsen_on_layout_weights: bool,
    /// Maximum number of label-propagation sweeps per level.
    pub max_propagation_sweeps: usize,
    pub curve_schedule: CurveSchedule,
    pub good_initialization_policy: GoodInitializationPolicy,
    /// Drop the edges inside a part from the coarsening graph.
    pub remove_coarse_diagonal: bool,
    pub anchor_method: AnchorMethod,
    pub negative_selection_range_mode: NegativeSelectionRangeMode,
}

impl Default for RecursiveInitParams {
    fn default() -> Self {
        RecursiveInitParams {
            base_init_threshold: 1024,
            n_embedding_epochs: 50,
            approx_n_parts: None,
            coarsening_ratio: 4,
            hub_selection: HubSelection::Degree,
            anchor_sample_size: 16384,
            repulsion_strength: 4.0,
            negative_sample_rate: 1.0,
            negative_selection_range_scale: 0.5,
            label_propagation_share: 0.1,
            coarse_layout_weights: CoarseLayoutWeights::Transition,
            coarsen_on_layout_weights: true,
            max_propagation_sweeps: 100,
            curve_schedule: CurveSchedule::StrongToOne,
            good_initialization_policy: GoodInitializationPolicy::Heuristic,
            remove_coarse_diagonal: false,
            anchor_method: AnchorMethod::SampledPca,
            negative_selection_range_mode: NegativeSelectionRangeMode::ScaledCoarse,
        }
    }
}

/// Initial coordinates of shape `(graph.nrows, n_components)` from recursive
/// label-propagation coarsening of the symmetric fuzzy `graph`. Without `data`
/// the anchor is uniform random.
#[cfg(test)]
pub(crate) fn recursive_init(
    graph: &CsrMatrix,
    data: Option<&Array2<f32>>,
    a: f64,
    b: f64,
    n_components: usize,
    params: &RecursiveInitParams,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    pollster::block_on(recursive_init_with_device(
        graph,
        data,
        a,
        b,
        n_components,
        params,
        rng,
        None,
    ))
}

/// Like [`recursive_init`], but runs the large coarse layouts on `device` when given.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn recursive_init_with_device(
    graph: &CsrMatrix,
    data: Option<&Array2<f32>>,
    a: f64,
    b: f64,
    n_components: usize,
    params: &RecursiveInitParams,
    rng: &mut Xoshiro256StarStar,
    device: Option<GpuDevice>,
) -> Array2<f32> {
    let anchor = anchor_reference(
        data,
        graph.nrows,
        n_components,
        params.anchor_method,
        params.anchor_sample_size,
        rng,
    );
    let exponents = params.curve_schedule.exponents();
    let level = Level {
        graph,
        anchor: &anchor,
        summed_layout_weights: None,
        a: a.powf(exponents.initial_a),
        b: b.powf(exponents.initial_b),
        approx_n_parts: params.approx_n_parts,
        n_embedding_epochs: params.n_embedding_epochs,
        depth: 1,
    };
    let mut layouter = CoarseLayouter::new(device);
    label_propagation_init(level, params, rng, &mut layouter).await
}

/// Runs the coarse layouts: large ones on the GPU while it works, the rest and
/// all after a GPU failure on the CPU.
struct CoarseLayouter {
    #[cfg(feature = "gpu")]
    device: Option<GpuDevice>,
}

impl CoarseLayouter {
    fn new(device: Option<GpuDevice>) -> Self {
        #[cfg(feature = "gpu")]
        return CoarseLayouter { device };
        #[cfg(not(feature = "gpu"))]
        {
            let _ = device;
            CoarseLayouter {}
        }
    }

    async fn optimize(
        &mut self,
        embedding: &mut Array2<f32>,
        graph: &CsrMatrix,
        params: &ModernLayoutParams,
    ) {
        #[cfg(feature = "gpu")]
        if let Some((device, queue)) = &self.device {
            if graph.nrows >= GPU_COARSE_LAYOUT_MIN_NODES {
                match optimize_on_gpu(device, queue, embedding, graph, params).await {
                    Ok(()) => return,
                    Err(error) => {
                        crate::gpu_optimize::report_gpu_error(&format!(
                            "GPU coarse layout failed ({error}); continuing the initialization on the CPU"
                        ));
                        self.device = None;
                    }
                }
            }
        }
        ModernLayout::new(graph, embedding.ncols(), params.clone()).run(embedding);
    }
}

/// The GPU counterpart of [`ModernLayout::run`]; leaves `embedding`
/// untouched on failure.
#[cfg(feature = "gpu")]
async fn optimize_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    embedding: &mut Array2<f32>,
    graph: &CsrMatrix,
    params: &ModernLayoutParams,
) -> Result<(), String> {
    let epochs_per_sample = crate::graph::make_epochs_per_sample(&graph.data, params.n_epochs)
        .into_iter()
        .map(f64::from)
        .collect();
    let dim = embedding.ncols();
    let start = embedding.as_standard_layout();
    let mut layout = crate::gpu_layout_modern::GpuModernLayout::new(
        device.clone(),
        queue.clone(),
        graph.indptr.clone(),
        graph.indices.clone(),
        epochs_per_sample,
        start.as_slice().expect("standard layout is contiguous"),
        dim,
        params.clone(),
        0,
    )
    .ok_or("the layout exceeds the device limits")?;
    #[cfg(test)]
    if tests::SIMULATE_DEVICE_LOSS.get() {
        layout.simulate_device_loss();
    }
    layout.run().await?;
    layout.sync_embedding().await;
    *embedding = Array2::from_shape_vec(embedding.raw_dim(), layout.embedding().to_vec())
        .expect("same shape as the input");
    Ok(())
}

// ---------------------------------------------------------------------------
// Anchor
// ---------------------------------------------------------------------------

/// Coordinates of the sampled nodes, in node order.
struct Anchor {
    coords: Array2<f32>,
    sampled: Vec<bool>,
}

impl Anchor {
    fn all_sampled(&self) -> bool {
        self.sampled.iter().all(|&s| s)
    }
}

fn anchor_reference(
    data: Option<&Array2<f32>>,
    n_samples: usize,
    n_components: usize,
    method: AnchorMethod,
    sample_size: usize,
    rng: &mut Xoshiro256StarStar,
) -> Anchor {
    let (sampled, sample_rows) = if sample_size < n_samples {
        let rows = sorted_sample(n_samples, sample_size, rng);
        let mut mask = vec![false; n_samples];
        rows.iter().for_each(|&r| mask[r] = true);
        (mask, Some(rows))
    } else {
        (vec![true; n_samples], None)
    };
    let n_sampled = sample_rows.as_ref().map_or(n_samples, Vec::len);

    let mut sampled = sampled;
    let mut coords = match data {
        Some(data) => {
            let sample = match &sample_rows {
                Some(rows) => data.select(Axis(0), rows),
                None => data.clone(),
            };
            let means = finite_column_means(&sample);
            let centered = centered_columns(sample);
            let scores = randomized_pca_scores(&centered, n_components, rng);
            if method == AnchorMethod::SampledPca {
                scores
            } else {
                sampled = vec![true; n_samples];
                project_onto_components(data, &means, &centered, &scores)
            }
        }
        None => Array2::zeros((n_sampled, 0)),
    };
    scale_columns_to_anchor_span(&mut coords);
    Anchor {
        coords: pad_with_uniform_columns(coords, n_components, rng),
        sampled,
    }
}

fn sorted_sample(n: usize, k: usize, rng: &mut Xoshiro256StarStar) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u64() % (n - i) as u64) as usize;
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool.sort_unstable();
    pool
}

/// Mean of the finite entries of each column (0 for a column without any).
fn finite_column_means(sample: &Array2<f32>) -> Vec<f64> {
    sample
        .columns()
        .into_iter()
        .map(|column| {
            let (sum, count) = column
                .iter()
                .filter(|x| x.is_finite())
                .fold((0.0f64, 0usize), |(s, c), &x| (s + x as f64, c + 1));
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// Scores of every row of `data` on the principal axes that produced `scores`
/// from the centered `sample`. Non-finite entries take the column mean.
fn project_onto_components(
    data: &Array2<f32>,
    means: &[f64],
    centered_sample: &Array2<f32>,
    scores: &Array2<f32>,
) -> Array2<f32> {
    let mut axes = centered_sample.t().dot(scores);
    for (mut axis, score) in axes.columns_mut().into_iter().zip(scores.columns()) {
        let squared_norm = score.dot(&score);
        axis.mapv_inplace(|x| {
            if squared_norm > 0.0 {
                x / squared_norm
            } else {
                0.0
            }
        });
    }
    let centered = Array2::from_shape_fn(data.dim(), |(row, column)| {
        let value = data[[row, column]];
        if value.is_finite() {
            (value as f64 - means[column]) as f32
        } else {
            0.0
        }
    });
    centered.dot(&axes)
}

/// Subtract the mean of the finite entries of each column; non-finite entries become 0.
pub(crate) fn centered_columns(mut sample: Array2<f32>) -> Array2<f32> {
    let means = finite_column_means(&sample);
    for (mut column, mean) in sample.columns_mut().into_iter().zip(means) {
        column.mapv_inplace(|x| {
            if x.is_finite() {
                (x as f64 - mean) as f32
            } else {
                0.0
            }
        });
    }
    sample
}

pub(crate) fn standard_normal(rng: &mut Xoshiro256StarStar) -> f64 {
    let u1 = rng.random_f64().max(1e-300);
    let u2 = rng.random_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Principal-component scores `U * S` of a centered matrix by randomized SVD
/// (Gaussian sketch, `PCA_POWER_ITERATIONS` unnormalized power iterations). Returns
/// `min(n_components, n_rows, n_cols)` columns, each signed so its largest-magnitude
/// entry is positive.
pub(crate) fn randomized_pca_scores(
    matrix: &Array2<f32>,
    n_components: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let (n_rows, n_cols) = matrix.dim();
    let rank = n_rows.min(n_cols);
    let n_scores = n_components.min(rank);
    if n_scores == 0 {
        return Array2::zeros((n_rows, 0));
    }
    let sketch_size = (n_components + PCA_OVERSAMPLES).min(rank);
    let transpose = n_rows < n_cols;
    let a = if transpose { matrix.t() } else { matrix.view() };

    let gaussian =
        Array2::from_shape_simple_fn((a.ncols(), sketch_size), || standard_normal(rng) as f32);
    let mut range = a.dot(&gaussian);
    for _ in 0..PCA_POWER_ITERATIONS {
        range = a.dot(&a.t().dot(&range));
    }
    let q = orthonormal_columns(range.mapv(f64::from)).mapv(|x| x as f32);
    let projected = q.t().dot(&a);

    let gram = projected
        .mapv(f64::from)
        .dot(&projected.t().mapv(f64::from));
    let (eigenvalues, eigenvectors) = symmetric_eigen_descending(&gram);
    let top_vectors = eigenvectors.slice(s![.., ..n_scores]).mapv(|x| x as f32);

    let mut scores = if transpose {
        projected.t().dot(&top_vectors)
    } else {
        let singular_values: Vec<f32> = eigenvalues[..n_scores]
            .iter()
            .map(|&l| l.max(0.0).sqrt() as f32)
            .collect();
        let mut basis = q.dot(&top_vectors);
        for (mut column, &sv) in basis.columns_mut().into_iter().zip(&singular_values) {
            column *= sv;
        }
        basis
    };
    for mut column in scores.columns_mut() {
        let largest = column
            .iter()
            .copied()
            .fold(0.0f32, |m, x| if x.abs() > m.abs() { x } else { m });
        if largest < 0.0 {
            column.mapv_inplace(|x| -x);
        }
    }
    scores
}

/// Modified Gram-Schmidt with one re-orthogonalization pass. Columns that are
/// numerically dependent on earlier ones become zero.
fn orthonormal_columns(mut m: Array2<f64>) -> Array2<f64> {
    let n_cols = m.ncols();
    for j in 0..n_cols {
        let initial_norm = m.column(j).dot(&m.column(j)).sqrt();
        for _ in 0..2 {
            for k in 0..j {
                let projection = m.column(k).dot(&m.column(j));
                let basis = m.column(k).to_owned();
                m.column_mut(j).scaled_add(-projection, &basis);
            }
        }
        let norm = m.column(j).dot(&m.column(j)).sqrt();
        if norm > 1e-10 * initial_norm && norm > 0.0 {
            m.column_mut(j).mapv_inplace(|x| x / norm);
        } else {
            m.column_mut(j).fill(0.0);
        }
    }
    m
}

/// Eigen-decomposition of a small symmetric matrix, eigenvalues descending,
/// eigenvectors as columns.
fn symmetric_eigen_descending(matrix: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let m = matrix.nrows();
    let scale = matrix.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let flat: Vec<f64> = matrix.iter().map(|x| x / scale).collect();
    let (values, vectors) = eigh_jacobi(&flat, m);
    let order: Vec<usize> = (0..m).rev().collect();
    let eigenvalues = order.iter().map(|&k| values[k] * scale).collect();
    let eigenvectors = Array2::from_shape_fn((m, m), |(row, col)| vectors[row * m + order[col]]);
    (eigenvalues, eigenvectors)
}

fn scale_columns_to_anchor_span(coords: &mut Array2<f32>) {
    for mut column in coords.columns_mut() {
        let min = column.iter().copied().fold(f32::INFINITY, f32::min);
        let max = column.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span = if max - min == 0.0 { 1.0 } else { max - min };
        column.mapv_inplace(|x| (x - min) / span * ANCHOR_SPAN);
    }
}

/// Append uniform `[0, ANCHOR_SPAN)` columns up to `n_components`.
fn pad_with_uniform_columns(
    coords: Array2<f32>,
    n_components: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let n_present = coords.ncols();
    if n_present >= n_components {
        return coords;
    }
    let mut padded = Array2::zeros((coords.nrows(), n_components));
    padded.slice_mut(s![.., ..n_present]).assign(&coords);
    for mut row in padded.rows_mut() {
        for value in row.iter_mut().skip(n_present) {
            *value = rng.random_f32() * ANCHOR_SPAN;
        }
    }
    padded
}

// ---------------------------------------------------------------------------
// Label propagation
// ---------------------------------------------------------------------------

fn tau_rng_offset(state: [i64; 3], offset: usize) -> TauRng {
    let offset = offset as i64;
    TauRng::from_state(state.map(|s| s.wrapping_add(offset)))
}

fn degrees_ascending_order(graph: &CsrMatrix) -> Vec<usize> {
    let degrees: Vec<f64> = (0..graph.nrows)
        .map(|i| {
            graph.data[graph.indptr[i]..graph.indptr[i + 1]]
                .iter()
                .map(|&w| w as f64)
                .sum()
        })
        .collect();
    let mut order: Vec<usize> = (0..graph.nrows).collect();
    order.sort_by(|&i, &j| degrees[i].total_cmp(&degrees[j]).then(i.cmp(&j)));
    order
}

fn seed_labels_from_hubs(graph: &CsrMatrix, n_parts: usize) -> Vec<i32> {
    let order = degrees_ascending_order(graph);
    let mut labels = vec![-1; graph.nrows];
    for (label, &hub) in order[graph.nrows - n_parts..].iter().enumerate() {
        labels[hub] = label as i32;
    }
    labels
}

/// High-degree hubs, skipping nodes next to an already chosen hub while possible;
/// the remaining labels go to the highest-degree unchosen nodes.
fn seed_labels_from_diverse_hubs(graph: &CsrMatrix, n_parts: usize) -> Vec<i32> {
    let order = degrees_ascending_order(graph);
    let mut labels = vec![-1; graph.nrows];
    let mut blocked = vec![false; graph.nrows];
    let mut next_label = 0;
    for &candidate in order.iter().rev() {
        if next_label == n_parts {
            return labels;
        }
        if !blocked[candidate] {
            labels[candidate] = next_label as i32;
            next_label += 1;
            blocked[candidate] = true;
            for &neighbor in &graph.indices[graph.indptr[candidate]..graph.indptr[candidate + 1]] {
                blocked[neighbor] = true;
            }
        }
    }
    for &candidate in order.iter().rev() {
        if next_label == n_parts {
            break;
        }
        if labels[candidate] < 0 {
            labels[candidate] = next_label as i32;
            next_label += 1;
        }
    }
    labels
}

fn seed_labels_at_random(n_nodes: usize, n_parts: usize, rng: &mut TauRng) -> Vec<i32> {
    let mut labels = vec![-1; n_nodes];
    for label in 0..n_parts {
        let node = (rng.tau_rand_int() as i64).rem_euclid(n_nodes as i64) as usize;
        labels[node] = label as i32;
    }
    labels
}

/// One Jacobi sweep: every unlabeled node adopts the label with the largest summed
/// edge weight among its neighbours (ties go to the label met last), provided that
/// weight reaches the threshold: `share` times the node's total edge weight, or 1.
fn propagation_sweep(
    graph: &CsrMatrix,
    labels: &[i32],
    n_labels: usize,
    share: Option<f64>,
) -> Vec<i32> {
    (0..graph.nrows)
        .into_par_iter()
        .map_init(
            || (vec![0.0f32; n_labels], vec![false; n_labels], Vec::new()),
            |(votes, seen, met), i| {
                if labels[i] >= 0 {
                    return labels[i];
                }
                let mut total_weight = 0.0f64;
                for edge in graph.indptr[i]..graph.indptr[i + 1] {
                    let weight = graph.data[edge];
                    total_weight += weight as f64;
                    let label = labels[graph.indices[edge]];
                    if label >= 0 {
                        let label = label as usize;
                        if !seen[label] {
                            seen[label] = true;
                            met.push(label);
                        }
                        votes[label] += weight;
                    }
                }
                let mut best_vote = share.map_or(1.0, |s| s * total_weight);
                let mut best_label = -1;
                for &label in met.iter() {
                    let vote = votes[label] as f64;
                    if vote >= best_vote {
                        best_vote = vote;
                        best_label = label as i32;
                    }
                    votes[label] = 0.0;
                    seen[label] = false;
                }
                met.clear();
                best_label
            },
        )
        .collect()
}

fn count_unlabeled(labels: &[i32]) -> usize {
    labels.iter().filter(|&&l| l < 0).count()
}

/// Sweep until every node is labeled or the unlabeled count stops changing
/// between checks `STALL_CHECK_INTERVAL` sweeps apart.
fn propagate_labels(
    graph: &CsrMatrix,
    mut labels: Vec<i32>,
    n_labels: usize,
    share: Option<f64>,
    max_sweeps: usize,
) -> Vec<i32> {
    let mut previous_unlabeled = count_unlabeled(&labels);
    for sweep in 0..max_sweeps {
        labels = propagation_sweep(graph, &labels, n_labels, share);
        if sweep % STALL_CHECK_INTERVAL == 0 {
            let unlabeled = count_unlabeled(&labels);
            if unlabeled == 0 || unlabeled == previous_unlabeled {
                break;
            }
            previous_unlabeled = unlabeled;
        }
    }
    labels
}

/// Each unlabeled node takes the label of the first labeled node met in a
/// depth-first walk from it (at most `OUTLIER_SEARCH_LIMIT` expansions), else a
/// random label.
fn label_outliers(graph: &CsrMatrix, labels: &mut [i32], rng_state: [i64; 3]) {
    let n_labels = (labels.iter().copied().max().unwrap_or(-1) as i64 + 1).max(1);
    let mut stack = Vec::new();
    for i in 0..graph.nrows {
        if labels[i] >= 0 {
            continue;
        }
        stack.clear();
        stack.push(i);
        let mut expansions = 0;
        let mut found = None;
        while found.is_none() && expansions < OUTLIER_SEARCH_LIMIT {
            let Some(node) = stack.pop() else { break };
            expansions += 1;
            for &neighbor in &graph.indices[graph.indptr[node]..graph.indptr[node + 1]] {
                if labels[neighbor] >= 0 {
                    found = Some(labels[neighbor]);
                    break;
                }
                stack.push(neighbor);
            }
        }
        labels[i] = found.unwrap_or_else(|| {
            (tau_rng_offset(rng_state, i).tau_rand_int() as i64).rem_euclid(n_labels) as i32
        });
    }
}

/// Map labels to `0..n_labels` in ascending order; negative labels each get a new label.
fn remap_labels(labels: &[i32]) -> Vec<usize> {
    let mut unique: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    unique.sort_unstable();
    unique.dedup();
    let mut next_label = unique.len();
    labels
        .iter()
        .map(|&l| {
            if l < 0 {
                next_label += 1;
                next_label - 1
            } else {
                unique.binary_search(&l).unwrap()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Coarsening
// ---------------------------------------------------------------------------

/// Node-to-part assignment with the members of each part in ascending order.
struct Partition {
    labels: Vec<usize>,
    member_indptr: Vec<usize>,
    members: Vec<usize>,
}

impl Partition {
    fn new(labels: Vec<usize>) -> Self {
        let n_parts = labels.iter().max().map_or(0, |&l| l + 1);
        let mut member_indptr = vec![0usize; n_parts + 1];
        for &l in &labels {
            member_indptr[l + 1] += 1;
        }
        for p in 0..n_parts {
            member_indptr[p + 1] += member_indptr[p];
        }
        let mut next = member_indptr[..n_parts].to_vec();
        let mut members = vec![0usize; labels.len()];
        for (node, &l) in labels.iter().enumerate() {
            members[next[l]] = node;
            next[l] += 1;
        }
        Partition {
            labels,
            member_indptr,
            members,
        }
    }

    fn n_parts(&self) -> usize {
        self.member_indptr.len() - 1
    }

    fn members(&self, part: usize) -> &[usize] {
        &self.members[self.member_indptr[part]..self.member_indptr[part + 1]]
    }
}

/// Square sparse matrix with f64 weights and sorted columns per row.
#[derive(Debug, PartialEq)]
struct WeightSums {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl WeightSums {
    fn from_rows(rows: Vec<(Vec<usize>, Vec<f64>)>) -> Self {
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        indptr.push(0);
        let nnz = rows.iter().map(|(c, _)| c.len()).sum();
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for (cols, values) in rows {
            indices.extend(cols);
            data.extend(values);
            indptr.push(indices.len());
        }
        WeightSums {
            indptr,
            indices,
            data,
        }
    }

    fn n(&self) -> usize {
        self.indptr.len() - 1
    }

    fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let range = self.indptr[i]..self.indptr[i + 1];
        (&self.indices[range.clone()], &self.data[range])
    }

    fn transpose(&self) -> WeightSums {
        let n = self.n();
        let mut counts = vec![0usize; n + 1];
        for &c in &self.indices {
            counts[c + 1] += 1;
        }
        for c in 0..n {
            counts[c + 1] += counts[c];
        }
        let mut next = counts[..n].to_vec();
        let mut indices = vec![0usize; self.indices.len()];
        let mut data = vec![0.0f64; self.data.len()];
        for row in 0..n {
            let (cols, values) = self.row(row);
            for (&c, &v) in cols.iter().zip(values) {
                indices[next[c]] = row;
                data[next[c]] = v;
                next[c] += 1;
            }
        }
        WeightSums {
            indptr: counts,
            indices,
            data,
        }
    }
}

/// `sum_{i in P, j in Q} weight(edge i->j)` for every pair of parts, summed in
/// member order then edge order.
fn sum_between_parts(
    partition: &Partition,
    indptr: &[usize],
    indices: &[usize],
    weight: impl Fn(usize) -> f64 + Sync,
) -> WeightSums {
    let n_parts = partition.n_parts();
    let rows = (0..n_parts)
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; n_parts], vec![false; n_parts], Vec::new()),
            |(sums, seen, targets), part| {
                for &node in partition.members(part) {
                    for edge in indptr[node]..indptr[node + 1] {
                        let target = partition.labels[indices[edge]];
                        if !seen[target] {
                            seen[target] = true;
                            targets.push(target);
                        }
                        sums[target] += weight(edge);
                    }
                }
                targets.sort_unstable();
                let values = targets
                    .iter()
                    .map(|&t| {
                        seen[t] = false;
                        std::mem::take(&mut sums[t])
                    })
                    .collect();
                (std::mem::take(targets), values)
            },
        )
        .collect();
    WeightSums::from_rows(rows)
}

fn to_f32_graph(sums: WeightSums, keep: impl Fn(usize, usize, f64) -> bool) -> CsrMatrix {
    let n = sums.n();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    let mut indices = Vec::with_capacity(sums.indices.len());
    let mut data = Vec::with_capacity(sums.data.len());
    for row in 0..n {
        let (cols, values) = sums.row(row);
        for (&c, &v) in cols.iter().zip(values) {
            if keep(row, c, v) {
                indices.push(c);
                data.push(v as f32);
            }
        }
        indptr.push(indices.len());
    }
    CsrMatrix {
        indptr,
        indices,
        data,
        nrows: n,
        ncols: n,
    }
}

/// Fuzzy union of all edges between two parts: `1 - prod(1 - w_ij)`, diagonal kept.
fn fuzzy_union_coarse_graph(
    graph: &CsrMatrix,
    partition: &Partition,
    remove_diagonal: bool,
) -> CsrMatrix {
    let mut log_complements = sum_between_parts(partition, &graph.indptr, &graph.indices, |edge| {
        (-(graph.data[edge] as f64).clamp(0.0, 1.0 - 1e-16)).ln_1p()
    });
    log_complements
        .data
        .iter_mut()
        .for_each(|s| *s = 1.0 - s.exp());
    to_f32_graph(log_complements, |row, col, v| {
        v != 0.0 && !(remove_diagonal && row == col)
    })
}

/// Total edge weight between parts, `S = Rᵀ W R`, from the summed weights of the
/// finer level or, at the top level, from the fuzzy graph itself.
fn summed_layout_weights(
    graph: &CsrMatrix,
    finer_sums: Option<&WeightSums>,
    partition: &Partition,
) -> WeightSums {
    match finer_sums {
        Some(finer) => sum_between_parts(partition, &finer.indptr, &finer.indices, |edge| {
            finer.data[edge]
        }),
        None => sum_between_parts(partition, &graph.indptr, &graph.indices, |edge| {
            graph.data[edge] as f64
        }),
    }
}

/// `f_PQ = S_PQ / sum_{Q' != P} S_PQ'` (diagonal excluded), combined by the fuzzy
/// union `f_PQ + f_QP - f_PQ f_QP` and clipped to `[0, 1]`.
fn transition_edges(summed: &WeightSums) -> CsrMatrix {
    let n = summed.n();
    let rows = (0..n)
        .into_par_iter()
        .map(|p| {
            let (cols, values) = summed.row(p);
            let external: Vec<(usize, f64)> = cols
                .iter()
                .zip(values)
                .filter(|&(&q, &v)| q != p && v != 0.0)
                .map(|(&q, &v)| (q, v))
                .collect();
            let out_weight: f64 = external.iter().map(|&(_, v)| v).sum();
            let inverse = if out_weight > 0.0 {
                1.0 / out_weight
            } else {
                1.0
            };
            external.into_iter().map(|(q, v)| (q, v * inverse)).unzip()
        })
        .collect();
    let directed = WeightSums::from_rows(rows);
    let reverse = directed.transpose();

    let merged = (0..n)
        .into_par_iter()
        .map(|p| {
            let (forward_cols, forward) = directed.row(p);
            let (reverse_cols, backward) = reverse.row(p);
            let (mut i, mut j) = (0, 0);
            let mut cols = Vec::with_capacity(forward_cols.len().max(reverse_cols.len()));
            let mut values = Vec::with_capacity(cols.capacity());
            while i < forward_cols.len() || j < reverse_cols.len() {
                let fc = forward_cols.get(i).copied().unwrap_or(usize::MAX);
                let rc = reverse_cols.get(j).copied().unwrap_or(usize::MAX);
                let col = fc.min(rc);
                let f = if fc == col { forward[i] } else { 0.0 };
                let r = if rc == col { backward[j] } else { 0.0 };
                i += usize::from(fc == col);
                j += usize::from(rc == col);
                cols.push(col);
                values.push((f + r - f * r).clamp(0.0, 1.0));
            }
            (cols, values)
        })
        .collect();
    to_f32_graph(WeightSums::from_rows(merged), |_, _, _| true)
}

/// Mean anchor coordinates per part over its sampled members. Parts without a
/// sampled member stay unsampled.
fn coarsen_anchor(anchor: &Anchor, partition: &Partition) -> Anchor {
    let n_parts = partition.n_parts();
    let dim = anchor.coords.ncols();
    let mut sums = Array2::<f64>::zeros((n_parts, dim));
    let mut counts = vec![0usize; n_parts];
    let sampled_nodes = anchor
        .sampled
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s)
        .map(|(node, _)| node);
    for (row, node) in sampled_nodes.enumerate() {
        let part = partition.labels[node];
        counts[part] += 1;
        let mut target = sums.row_mut(part);
        Zip::from(&mut target)
            .and(anchor.coords.row(row))
            .for_each(|t, &x| *t += x as f64);
    }
    let sampled: Vec<bool> = counts.iter().map(|&c| c > 0).collect();
    let kept: Vec<usize> = (0..n_parts).filter(|&p| sampled[p]).collect();
    let coords = Array2::from_shape_fn((kept.len(), dim), |(r, d)| {
        let part = kept[r];
        (sums[[part, d]] / counts[part] as f64) as f32
    });
    Anchor { coords, sampled }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// `x_i = (sum_e w_e Y[label(j_e)] / sum_e w_e + Y[label(i)]) / 2` over the edges of
/// node `i`; the first term is 0 for nodes without edge weight.
fn expand_layout(graph: &CsrMatrix, labels: &[usize], coarse: &Array2<f32>) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((graph.nrows, coarse.ncols()));
    Zip::indexed(result.rows_mut()).par_for_each(|i, mut row| {
        let mut total_weight = 0.0f64;
        for edge in graph.indptr[i]..graph.indptr[i + 1] {
            let weight = graph.data[edge] as f64;
            total_weight += weight;
            let neighbor = coarse.row(labels[graph.indices[edge]]);
            Zip::from(&mut row)
                .and(neighbor)
                .for_each(|x, &y| *x += weight * y as f64);
        }
        let own = coarse.row(labels[i]);
        Zip::from(&mut row).and(own).for_each(|x, &y| {
            let neighbor_mean = if total_weight > 0.0 {
                *x / total_weight
            } else {
                *x
            };
            *x = 0.5 * (neighbor_mean + y as f64);
        });
    });
    result
}

fn center_columns_f64(m: &mut Array2<f64>) {
    if let Some(mean) = m.mean_axis(Axis(0)) {
        *m -= &mean;
    }
}

fn centered_unit_rms(m: ArrayView2<f32>) -> Option<Array2<f64>> {
    let mut centered = m.mapv(f64::from);
    center_columns_f64(&mut centered);
    let rms = (centered.iter().map(|x| x * x).sum::<f64>() / centered.len() as f64).sqrt();
    (rms > 0.0).then(|| centered / rms)
}

fn determinant(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut m = matrix.clone();
    let mut det = 1.0;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[[i, col]].abs().total_cmp(&m[[j, col]].abs()))
            .unwrap();
        if m[[pivot, col]] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            for k in 0..n {
                m.swap([pivot, k], [col, k]);
            }
            det = -det;
        }
        det *= m[[col, col]];
        for row in col + 1..n {
            let factor = m[[row, col]] / m[[col, col]];
            for k in col..n {
                m[[row, k]] -= factor * m[[col, k]];
            }
        }
    }
    det
}

/// Proper rotation `R` that best maps `moving @ R` onto `reference` after both are
/// centered and scaled to unit RMS: `R = U Vᵀ` from the SVD `movingᵀ reference = U S Vᵀ`,
/// with the last column of `U` negated when `det(U Vᵀ) < 0`. Identity if either is collapsed.
fn procrustes_rotation(reference: ArrayView2<f32>, moving: ArrayView2<f32>) -> Array2<f64> {
    let dim = reference.ncols();
    let (Some(reference), Some(moving)) = (centered_unit_rms(reference), centered_unit_rms(moving))
    else {
        return Array2::eye(dim);
    };
    let covariance = moving.t().dot(&reference);
    let (eigenvalues, v) = symmetric_eigen_descending(&covariance.t().dot(&covariance));
    let largest_singular_value = eigenvalues[0].max(0.0).sqrt();

    let mut u = Array2::<f64>::zeros((dim, dim));
    let mut completed = 0;
    for (k, &eigenvalue) in eigenvalues.iter().enumerate() {
        let singular_value = eigenvalue.max(0.0).sqrt();
        if singular_value > 1e-12 * largest_singular_value && singular_value > 0.0 {
            let column = covariance.dot(&v.column(k)) / singular_value;
            u.column_mut(k).assign(&column);
            completed = k + 1;
        } else {
            break;
        }
    }
    let mut candidate = 0;
    for k in completed..dim {
        loop {
            let mut column = Array2::<f64>::eye(dim).column(candidate).to_owned();
            candidate += 1;
            for _ in 0..2 {
                for j in 0..k {
                    let projection = u.column(j).dot(&column);
                    column.scaled_add(-projection, &u.column(j));
                }
            }
            let norm = column.dot(&column).sqrt();
            if norm > 1e-6 {
                u.column_mut(k).assign(&(column / norm));
                break;
            }
        }
    }

    let mut rotation = u.dot(&v.t());
    if determinant(&rotation) < 0.0 {
        u.column_mut(dim - 1).mapv_inplace(|x| -x);
        rotation = u.dot(&v.t());
    }
    rotation
}

/// Fill unsampled nodes from their sampled neighbours (edge-weighted mean, else the
/// mean of all sampled nodes), center, and scale each axis to a 5%-95% spread of
/// `log10(n) * 3 * log2(depth + 1)`.
fn base_case_layout(graph: &CsrMatrix, anchor: &Anchor, depth: usize) -> Array2<f32> {
    let mut result = if anchor.all_sampled() {
        anchor.coords.mapv(f64::from)
    } else {
        fill_unsampled(graph, anchor)
    };
    let n = result.nrows();
    let target_spread = (n as f64).log10() * 3.0 * ((depth + 1) as f64).log2();
    center_columns_f64(&mut result);
    let mut result = result.mapv(|x| x as f32);
    for mut column in result.columns_mut() {
        let values = column.to_vec();
        let spread = quantile(&values, 0.95) - quantile(&values, 0.05);
        let spread = if spread == 0.0 { 1.0 } else { spread };
        let factor = (target_spread / spread as f64) as f32;
        column.mapv_inplace(|x| x * factor);
    }
    result
}

fn fill_unsampled(graph: &CsrMatrix, anchor: &Anchor) -> Array2<f64> {
    let dim = anchor.coords.ncols();
    let mut result = Array2::<f64>::zeros((graph.nrows, dim));
    let mut sampled_row = vec![usize::MAX; graph.nrows];
    let mut row = 0;
    for (node, &sampled) in anchor.sampled.iter().enumerate() {
        if sampled {
            sampled_row[node] = row;
            result
                .row_mut(node)
                .assign(&anchor.coords.row(row).mapv(f64::from));
            row += 1;
        }
    }
    let fallback = anchor
        .coords
        .mapv(f64::from)
        .mean_axis(Axis(0))
        .unwrap_or_else(|| ndarray::Array1::zeros(dim));
    for node in 0..graph.nrows {
        if anchor.sampled[node] {
            continue;
        }
        let mut total_weight = 0.0f64;
        let mut weighted = ndarray::Array1::<f64>::zeros(dim);
        for edge in graph.indptr[node]..graph.indptr[node + 1] {
            let neighbor = graph.indices[edge];
            if sampled_row[neighbor] != usize::MAX {
                let weight = graph.data[edge] as f64;
                total_weight += weight;
                Zip::from(&mut weighted)
                    .and(anchor.coords.row(sampled_row[neighbor]))
                    .for_each(|w, &x| *w += weight * x as f64);
            }
        }
        if total_weight > 0.0 {
            result.row_mut(node).assign(&(weighted / total_weight));
        } else {
            result.row_mut(node).assign(&fallback);
        }
    }
    result
}

fn negative_selection_range(n_vertices: usize, params: &RecursiveInitParams) -> usize {
    match params.negative_selection_range_mode {
        NegativeSelectionRangeMode::CoarseN => n_vertices,
        NegativeSelectionRangeMode::Fixed200k => 200_000,
        NegativeSelectionRangeMode::ScaledCoarse => {
            let scaled = params.negative_selection_range_scale * n_vertices as f64;
            (scaled.round_ties_even().max(1.0) as usize).min(n_vertices)
        }
    }
}

fn is_good_initialization(
    coarse_layout: &Array2<f32>,
    n_next_parts: usize,
    params: &RecursiveInitParams,
) -> bool {
    match params.good_initialization_policy {
        GoodInitializationPolicy::Heuristic => n_next_parts > params.base_init_threshold,
        GoodInitializationPolicy::Always => true,
        GoodInitializationPolicy::Never => false,
        GoodInitializationPolicy::Available => {
            coarse_layout.iter().all(|x| x.is_finite())
                && coarse_layout.columns().into_iter().all(|column| {
                    let values = column.to_vec();
                    quantile(&values, 0.95) - quantile(&values, 0.05) > 0.0
                })
        }
    }
}

/// One level of the recursion.
struct Level<'a> {
    graph: &'a CsrMatrix,
    anchor: &'a Anchor,
    summed_layout_weights: Option<&'a WeightSums>,
    a: f64,
    b: f64,
    approx_n_parts: Option<usize>,
    n_embedding_epochs: usize,
    depth: usize,
}

fn label_propagation_init<'a>(
    level: Level<'a>,
    params: &'a RecursiveInitParams,
    rng: &'a mut Xoshiro256StarStar,
    layouter: &'a mut CoarseLayouter,
) -> Pin<Box<dyn Future<Output = Array2<f32>> + 'a>> {
    Box::pin(label_propagation_level(level, params, rng, layouter))
}

async fn label_propagation_level(
    level: Level<'_>,
    params: &RecursiveInitParams,
    rng: &mut Xoshiro256StarStar,
    layouter: &mut CoarseLayouter,
) -> Array2<f32> {
    let graph = level.graph;
    let n = graph.nrows;
    let exponents = params.curve_schedule.exponents();
    if n <= params.base_init_threshold {
        return base_case_layout(graph, level.anchor, level.depth);
    }

    let n_parts = level
        .approx_n_parts
        .unwrap_or_else(|| params.base_init_threshold.max(n / params.coarsening_ratio))
        .min(n / 2)
        .max(2);

    let rng_state: [i64; 3] =
        std::array::from_fn(|_| rng.random_range_i64(RNG_STATE_MIN, RNG_STATE_MAX));
    let mut tau = TauRng::from_state(rng_state);
    let seeds = match params.hub_selection {
        HubSelection::Degree => seed_labels_from_hubs(graph, n_parts),
        HubSelection::Diverse => seed_labels_from_diverse_hubs(graph, n_parts),
        HubSelection::Random => seed_labels_at_random(n, n_parts, &mut tau),
    };
    let transition_weighted = params.coarsen_on_layout_weights
        && level.summed_layout_weights.is_some()
        && params.coarse_layout_weights == CoarseLayoutWeights::Transition;
    let share = transition_weighted.then_some(params.label_propagation_share);
    let mut labels = propagate_labels(graph, seeds, n_parts, share, params.max_propagation_sweeps);
    label_outliers(graph, &mut labels, tau.state);
    let partition = Partition::new(remap_labels(&labels));

    let summed = summed_layout_weights(graph, level.summed_layout_weights, &partition);
    let fuzzy_graph = (params.coarse_layout_weights == CoarseLayoutWeights::FuzzyUnion
        || !params.coarsen_on_layout_weights)
        .then(|| fuzzy_union_coarse_graph(graph, &partition, params.remove_coarse_diagonal));
    let transition_graph = (params.coarse_layout_weights == CoarseLayoutWeights::Transition)
        .then(|| transition_edges(&summed));
    let layout_edges = transition_graph
        .as_ref()
        .or(fuzzy_graph.as_ref())
        .expect("one coarse graph is always built");
    let next_graph = if params.coarsen_on_layout_weights {
        layout_edges
    } else {
        fuzzy_graph
            .as_ref()
            .expect("built when not coarsening on layout weights")
    };
    let coarse_anchor = coarsen_anchor(level.anchor, &partition);

    let mut coarse_layout = label_propagation_init(
        Level {
            graph: next_graph,
            anchor: &coarse_anchor,
            summed_layout_weights: Some(&summed),
            a: level.a.powf(exponents.recursive_a),
            b: level.b.powf(exponents.recursive_b),
            approx_n_parts: Some(n_parts / params.coarsening_ratio),
            n_embedding_epochs: (level.n_embedding_epochs as f64 * std::f64::consts::SQRT_2)
                as usize,
            depth: level.depth + 1,
        },
        params,
        rng,
        layouter,
    )
    .await;
    let n_coarse = partition.n_parts();
    let good_initialization =
        is_good_initialization(&coarse_layout, n_parts / params.coarsening_ratio, params);
    let layout_params = ModernLayoutParams {
        kind: ModernOptimizerKind::Adam,
        n_epochs: level.n_embedding_epochs,
        a: level.a as f32,
        b: level.b as f32,
        gamma: params.repulsion_strength,
        initial_alpha: COARSE_LAYOUT_LEARNING_RATE,
        negative_sample_rate: params.negative_sample_rate,
        negative_selection_range: negative_selection_range(n_coarse, params),
        negative_sample_scale: None,
        scale_adaptation_samples: SCALE_ADAPTATION_SAMPLES,
        good_initialization,
        exclude_graph_neighbors: false,
        seed: rng.next_u64(),
    };
    layouter
        .optimize(&mut coarse_layout, layout_edges, &layout_params)
        .await;

    let mut expanded = expand_layout(graph, &partition.labels, &coarse_layout);
    center_columns_f64(&mut expanded);
    let expanded = expanded.mapv(|x| x as f32);

    let sampled_rows: Vec<usize> = (0..n).filter(|&i| level.anchor.sampled[i]).collect();
    let rotation = procrustes_rotation(
        level.anchor.coords.view(),
        expanded.select(Axis(0), &sampled_rows).view(),
    );
    expanded.mapv(f64::from).dot(&rotation).mapv(|x| x as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SparseMatrix;
    use ndarray::arr2;

    fn csr(n: usize, edges: &[(usize, usize, f32)]) -> CsrMatrix {
        let triples = edges
            .iter()
            .flat_map(|&(i, j, w)| [(i, j, w), (j, i, w)])
            .collect();
        SparseMatrix::from_coo(n, n, triples).to_csr()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
    }

    #[test]
    fn plain_propagation_needs_unit_vote_and_takes_last_tie() {
        // Node 1 sees label 0 with weight 0.6 < 1 and stays unlabeled. Node 2 meets
        // label 1 (via 3) before label 0 (via 4), both with weight 1; the last one wins.
        let graph = csr(5, &[(0, 1, 0.6), (2, 4, 1.0), (2, 3, 1.0), (1, 2, 0.1)]);
        let labels = vec![0, -1, -1, 1, 0];
        let swept = propagation_sweep(&graph, &labels, 2, None);
        assert_eq!(swept, vec![0, -1, 0, 1, 0]);
    }

    #[test]
    fn share_propagation_uses_fraction_of_total_weight() {
        let graph = csr(4, &[(0, 1, 0.05), (1, 2, 0.9), (1, 3, 0.05)]);
        let labels = vec![0, -1, -1, 1];
        let swept = propagation_sweep(&graph, &labels, 2, Some(0.1));
        // Node 1: total 1.0, label 0 has 0.05, label 1 has 0.05 -> both below 0.1.
        assert_eq!(swept[1], -1);
        let labels = vec![0, -1, 1, 1];
        let swept = propagation_sweep(&graph, &labels, 2, Some(0.1));
        assert_eq!(swept[1], 1);
    }

    #[test]
    fn propagation_on_two_cliques_labels_each_clique_from_its_hub() {
        let mut edges = Vec::new();
        for offset in [0, 5] {
            for i in 0..5 {
                for j in i + 1..5 {
                    edges.push((offset + i, offset + j, 1.0));
                }
            }
        }
        edges.push((4, 5, 0.1));
        let graph = csr(10, &edges);
        let mut seeds = vec![-1; 10];
        seeds[0] = 0;
        seeds[9] = 1;
        let mut labels = propagate_labels(&graph, seeds, 2, None, 100);
        label_outliers(&graph, &mut labels, [1, 2, 3]);
        assert_eq!(labels, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn outliers_take_first_labeled_node_of_depth_first_walk() {
        let graph = csr(5, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)]);
        let mut labels = vec![-1, -1, 7, -1, -1];
        label_outliers(&graph, &mut labels, [11, 22, 33]);
        assert_eq!(&labels[..3], &[7, 7, 7]);
        assert!((0..8).contains(&labels[3]) && (0..8).contains(&labels[4]));
        assert_eq!(remap_labels(&[5, 2, 5, -1, 9, -1]), vec![1, 0, 1, 3, 2, 4]);
    }

    #[test]
    fn coarse_graphs_follow_their_formulas() {
        let graph = csr(4, &[(0, 1, 0.5), (1, 2, 0.5), (2, 3, 0.25), (0, 3, 0.5)]);
        let partition = Partition::new(vec![0, 0, 1, 1]);

        let fuzzy = fuzzy_union_coarse_graph(&graph, &partition, false);
        assert_eq!(fuzzy.indptr, vec![0, 2, 4]);
        // Within part 0: edges 0-1 and 1-0, each 0.5 -> 1 - 0.25.
        assert_close(fuzzy.data[0] as f64, 0.75, 1e-6);
        // Between: 1-2 (0.5) and 0-3 (0.5) -> 1 - 0.25.
        assert_close(fuzzy.data[1] as f64, 0.75, 1e-6);
        assert_close(fuzzy.data[3] as f64, 1.0 - 0.75 * 0.75, 1e-6);

        let summed = summed_layout_weights(&graph, None, &partition);
        assert_eq!(summed.data, vec![1.0, 1.0, 1.0, 0.5]);

        let coarser = Partition::new(vec![0, 0]);
        let total = summed_layout_weights(&fuzzy, Some(&summed), &coarser);
        assert_eq!(total.data, vec![3.5]);
    }

    #[test]
    fn transition_edges_are_fuzzy_union_of_row_normalized_external_weights() {
        let summed = WeightSums {
            indptr: vec![0, 3, 6, 9],
            indices: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            data: vec![9.0, 1.0, 3.0, 1.0, 5.0, 1.0, 3.0, 1.0, 2.0],
        };
        let edges = transition_edges(&summed);
        assert_eq!(edges.indptr, vec![0, 2, 4, 6]);
        assert_eq!(edges.indices, vec![1, 2, 0, 2, 0, 1]);
        let union = |f: f64, r: f64| f + r - f * r;
        let expected = [
            union(0.25, 0.5),
            union(0.75, 0.75),
            union(0.5, 0.25),
            union(0.5, 0.25),
            union(0.75, 0.75),
            union(0.25, 0.5),
        ];
        for (&got, want) in edges.data.iter().zip(expected) {
            assert_close(got as f64, want, 1e-6);
        }
    }

    #[test]
    fn expand_averages_neighbour_parts_with_own_part() {
        let graph = csr(3, &[(0, 1, 1.0), (0, 2, 3.0)]);
        let coarse = arr2(&[[0.0f32, 0.0], [4.0, 8.0]]);
        let expanded = expand_layout(&graph, &[0, 1, 1], &coarse);
        // Node 0: neighbours both in part 1 -> mean (4, 8); own part 0 -> (2, 4).
        assert_eq!(expanded.row(0).to_vec(), vec![2.0, 4.0]);
        assert_eq!(expanded.row(1).to_vec(), vec![2.0, 4.0]);
        let isolated = expand_layout(&csr(1, &[]), &[1], &coarse);
        assert_eq!(isolated.row(0).to_vec(), vec![2.0, 4.0]);
    }

    #[test]
    fn procrustes_recovers_known_rotation() {
        let reference = Array2::from_shape_fn((50, 3), |(i, d)| {
            ((i * (d + 3)) as f32 * 0.37).sin() * (d + 1) as f32
        });
        let (theta, phi) = (0.7f64, -1.1f64);
        let rz = arr2(&[
            [theta.cos(), -theta.sin(), 0.0],
            [theta.sin(), theta.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let rx = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, phi.cos(), -phi.sin()],
            [0.0, phi.sin(), phi.cos()],
        ]);
        let rotation = rz.dot(&rx);
        let moving = (reference.mapv(f64::from).dot(&rotation) * 3.0 + 5.0).mapv(|x| x as f32);
        let recovered = procrustes_rotation(reference.view(), moving.view());
        let product = rotation.dot(&recovered);
        for ((i, j), &x) in product.indexed_iter() {
            assert_close(x, if i == j { 1.0 } else { 0.0 }, 1e-4);
        }

        let mirrored = reference.mapv(f64::from) * &ndarray::arr1(&[1.0, 1.0, -1.0]);
        let proper = procrustes_rotation(reference.view(), mirrored.mapv(|x| x as f32).view());
        assert_close(determinant(&proper), 1.0, 1e-9);

        let collapsed = Array2::<f32>::ones((50, 3));
        assert_eq!(
            procrustes_rotation(reference.view(), collapsed.view()),
            Array2::eye(3)
        );
    }

    #[test]
    fn randomized_pca_matches_exact_scores_on_low_rank_matrix() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(3);
        let left = Array2::from_shape_simple_fn((60, 3), || standard_normal(&mut rng));
        let right = Array2::from_shape_simple_fn((3, 20), || standard_normal(&mut rng));
        let scales = ndarray::arr1(&[10.0, 3.0, 1.0]);
        let matrix = (&left * &scales).dot(&right);
        let matrix = centered_columns(matrix.mapv(|x| x as f32));

        let exact = {
            let m = matrix.mapv(f64::from);
            let (values, vectors) = symmetric_eigen_descending(&m.dot(&m.t()));
            Array2::from_shape_fn((60, 2), |(i, k)| vectors[[i, k]] * values[k].sqrt())
        };
        for transposed in [false, true] {
            let input = if transposed {
                matrix.t().to_owned()
            } else {
                matrix.clone()
            };
            let mut rng = Xoshiro256StarStar::seed_from_u64(9);
            let scores = randomized_pca_scores(&input, 2, &mut rng);
            let reference = if transposed {
                let m = input.mapv(f64::from);
                let (values, vectors) = symmetric_eigen_descending(&m.dot(&m.t()));
                Array2::from_shape_fn((20, 2), |(i, k)| vectors[[i, k]] * values[k].sqrt())
            } else {
                exact.clone()
            };
            assert_eq!(scores.dim(), reference.dim());
            for (score, expected) in scores.columns().into_iter().zip(reference.columns()) {
                let dot: f64 = score
                    .iter()
                    .zip(expected)
                    .map(|(&a, &b)| a as f64 * b)
                    .sum();
                let sign = dot.signum();
                for (&a, &b) in score.iter().zip(expected) {
                    assert_close(
                        a as f64,
                        sign * b,
                        1e-2 * expected.iter().fold(0.0, |m: f64, x| m.max(x.abs())),
                    );
                }
            }
        }
    }

    #[test]
    fn anchor_pads_missing_axes_with_uniform_columns() {
        let data = Array2::from_shape_fn((40, 1), |(i, _)| i as f32);
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let anchor = anchor_reference(
            Some(&data),
            40,
            3,
            AnchorMethod::SampledPca,
            16384,
            &mut rng,
        );
        assert_eq!(anchor.coords.dim(), (40, 3));
        let first = anchor.coords.column(0);
        let (min, max) = first
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert_close(min as f64, 0.0, 1e-5);
        assert_close(max as f64, 10.0, 1e-4);
        for column in 1..3 {
            let padded = anchor.coords.column(column);
            assert!(padded.iter().all(|&x| (0.0..10.0).contains(&x)));
            assert!(padded.iter().any(|&x| x != padded[0]));
        }

        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let sampled = anchor_reference(Some(&data), 40, 2, AnchorMethod::SampledPca, 10, &mut rng);
        assert_eq!(sampled.coords.nrows(), 10);
        assert_eq!(sampled.sampled.iter().filter(|&&s| s).count(), 10);
    }

    #[test]
    fn base_case_fills_unsampled_nodes_and_scales_spread() {
        let graph = csr(4, &[(0, 1, 1.0), (1, 2, 3.0), (2, 3, 1.0)]);
        let anchor = Anchor {
            coords: arr2(&[[0.0f32, 0.0], [4.0, 8.0]]),
            sampled: vec![true, false, true, false],
        };
        let filled = fill_unsampled(&graph, &anchor);
        assert_eq!(filled.row(1).to_vec(), vec![3.0, 6.0]);
        assert_eq!(filled.row(3).to_vec(), vec![4.0, 8.0]);

        let layout = base_case_layout(&graph, &anchor, 1);
        let target = 4f64.log10() * 3.0;
        for column in layout.columns() {
            let values = column.to_vec();
            assert_close(values.iter().map(|&x| x as f64).sum::<f64>(), 0.0, 1e-5);
            let spread = quantile(&values, 0.95) - quantile(&values, 0.05);
            assert_close(spread as f64, target, 1e-5);
        }
    }

    fn blobs(n: usize, centers: &[[f32; 8]], seed: u64) -> (Array2<f32>, Vec<usize>) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let labels: Vec<usize> = (0..n).map(|i| i % centers.len()).collect();
        let data = Array2::from_shape_fn((n, 8), |(i, d)| {
            centers[labels[i]][d] + standard_normal(&mut rng) as f32
        });
        (data, labels)
    }

    fn knn_graph(data: &Array2<f32>, k: usize) -> CsrMatrix {
        let n = data.nrows();
        let rows: Vec<(Vec<i32>, Vec<f32>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut d: Vec<(f32, usize)> = (0..n)
                    .map(|j| {
                        let dist = data
                            .row(i)
                            .iter()
                            .zip(data.row(j))
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            .sqrt();
                        (dist, j)
                    })
                    .collect();
                d.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                d.truncate(k);
                d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                (
                    d.iter().map(|x| x.1 as i32).collect(),
                    d.iter().map(|x| x.0).collect(),
                )
            })
            .collect();
        let indices = Array2::from_shape_fn((n, k), |(i, c)| rows[i].0[c]);
        let distances = Array2::from_shape_fn((n, k), |(i, c)| rows[i].1[c]);
        let mut graph = crate::graph::fuzzy_simplicial_set(&indices, &distances, k, 1.0, 1.0);
        graph.prune(graph.max_weight() / 500.0);
        graph.to_csr()
    }

    fn cluster_separation(embedding: &Array2<f32>, labels: &[usize], n_clusters: usize) -> f64 {
        let dim = embedding.ncols();
        let mut centroids = Array2::<f64>::zeros((n_clusters, dim));
        let mut counts = vec![0.0; n_clusters];
        for (row, &l) in embedding.rows().into_iter().zip(labels) {
            counts[l] += 1.0;
            Zip::from(centroids.row_mut(l))
                .and(row)
                .for_each(|c, &x| *c += x as f64);
        }
        for (mut c, &count) in centroids.rows_mut().into_iter().zip(&counts) {
            c /= count;
        }
        let distance = |a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f32>| {
            a.iter()
                .zip(b)
                .map(|(x, &y)| (x - y as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let within: f64 = embedding
            .rows()
            .into_iter()
            .zip(labels)
            .map(|(row, &l)| distance(centroids.row(l), row))
            .sum::<f64>()
            / labels.len() as f64;
        let mut between = f64::INFINITY;
        for i in 0..n_clusters {
            for j in i + 1..n_clusters {
                let d = distance(centroids.row(i), centroids.row(j).mapv(|x| x as f32).view());
                between = between.min(d);
            }
        }
        between / within
    }

    const CENTERS: [[f32; 8]; 4] = [
        [0.0; 8],
        [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    #[test]
    fn recursive_init_separates_blobs_deterministically() {
        let (data, labels) = blobs(5000, &CENTERS, 5);
        let graph = knn_graph(&data, 15);
        let (a, b) = crate::find_ab_params(1.0, 0.1);
        let params = RecursiveInitParams::default();

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut rng = Xoshiro256StarStar::seed_from_u64(42);
                let start = std::time::Instant::now();
                let embedding = recursive_init(&graph, Some(&data), a, b, 2, &params, &mut rng);
                eprintln!(
                    "recursive_init on 5000 points with {threads} threads: {:?}",
                    start.elapsed()
                );
                embedding
            })
        };
        let embedding = run(4);
        assert_eq!(embedding.dim(), (5000, 2));
        assert!(embedding.iter().all(|x| x.is_finite()));
        assert_eq!(embedding, run(1));
        assert_eq!(embedding, run(4));

        let separation = cluster_separation(&embedding, &labels, CENTERS.len());
        assert!(separation > 2.0, "cluster separation {separation}");
    }

    #[test]
    fn recursive_init_small_graph_is_base_case() {
        let (data, labels) = blobs(600, &CENTERS, 8);
        let graph = knn_graph(&data, 15);
        let (a, b) = crate::find_ab_params(1.0, 0.1);
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let embedding = recursive_init(
            &graph,
            Some(&data),
            a,
            b,
            3,
            &RecursiveInitParams::default(),
            &mut rng,
        );
        assert_eq!(embedding.dim(), (600, 3));
        assert!(embedding.iter().all(|x| x.is_finite()));
        let target = 600f64.log10() * 3.0;
        for column in embedding.columns() {
            let values = column.to_vec();
            let spread = quantile(&values, 0.95) - quantile(&values, 0.05);
            assert_close(spread as f64, target, 1e-3);
        }
        assert!(cluster_separation(&embedding, &labels, CENTERS.len()) > 2.0);

        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let without_data = recursive_init(
            &graph,
            None,
            a,
            b,
            2,
            &RecursiveInitParams::default(),
            &mut rng,
        );
        assert!(without_data.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn modern_builder_defaults_to_recursive_init() {
        let (data, _) = blobs(1500, &CENTERS, 2);
        let build = || {
            crate::Umap::builder(&data)
                .compatibility_layout(false)
                .optimizer(crate::Optimizer::Adam)
                .n_epochs(30)
                .random_state(3)
                .build()
                .unwrap()
                .embedding
        };
        let embedding = build();
        assert_eq!(embedding.dim(), (1500, 2));
        assert!(embedding.iter().all(|x| x.is_finite()));
        assert_eq!(embedding, build());
    }

    #[test]
    fn diverse_hubs_avoid_adjacent_hubs_then_fill_by_degree() {
        let graph = csr(5, &[(0, 1, 5.0), (0, 2, 5.0), (3, 4, 1.0)]);
        let diverse = seed_labels_from_diverse_hubs(&graph, 2);
        assert_eq!(diverse[0], 0);
        assert_eq!(diverse.iter().filter(|&&l| l >= 0).count(), 2);
        assert!(diverse[1] < 0 && diverse[2] < 0);
        assert!(diverse[3] == 1 || diverse[4] == 1);

        let filled = seed_labels_from_diverse_hubs(&graph, 4);
        assert_eq!(filled.iter().filter(|&&l| l >= 0).count(), 4);
        assert_eq!(filled[0], 0);
    }

    #[test]
    fn projected_and_full_anchors_cover_every_row() {
        let data = Array2::from_shape_fn((40, 3), |(i, d)| (i * (d + 1)) as f32 + (i % 3) as f32);
        for method in [AnchorMethod::ProjectedPca, AnchorMethod::FullPca] {
            let mut rng = Xoshiro256StarStar::seed_from_u64(1);
            let anchor = anchor_reference(Some(&data), 40, 2, method, 10, &mut rng);
            assert_eq!(anchor.coords.dim(), (40, 2));
            assert!(anchor.all_sampled());
            assert!(anchor.coords.iter().all(|x| x.is_finite()));
        }
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let full = anchor_reference(Some(&data), 40, 2, AnchorMethod::FullPca, 40, &mut rng);
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let sampled = anchor_reference(Some(&data), 40, 2, AnchorMethod::SampledPca, 40, &mut rng);
        for (a, b) in full
            .coords
            .column(0)
            .iter()
            .zip(sampled.coords.column(0).iter())
        {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
    }

    #[test]
    fn curve_schedules_resolve_to_exponents() {
        let strong = CurveSchedule::StrongToOne.exponents();
        assert_eq!((strong.initial_a, strong.recursive_b), (0.25, 0.25));
        let original = CurveSchedule::Original.exponents();
        assert_eq!((original.initial_b, original.recursive_a), (1.0, 1.0));
        let custom = CurveExponents {
            initial_a: 0.5,
            initial_b: 0.6,
            recursive_a: 0.7,
            recursive_b: 0.8,
        };
        assert_eq!(CurveSchedule::Custom(custom).exponents(), custom);
    }

    #[test]
    fn negative_selection_range_follows_its_mode() {
        let mode = |mode| RecursiveInitParams {
            negative_selection_range_mode: mode,
            ..RecursiveInitParams::default()
        };
        assert_eq!(
            negative_selection_range(300, &mode(NegativeSelectionRangeMode::CoarseN)),
            300
        );
        assert_eq!(
            negative_selection_range(300, &mode(NegativeSelectionRangeMode::Fixed200k)),
            200_000
        );
        assert_eq!(
            negative_selection_range(300, &mode(NegativeSelectionRangeMode::ScaledCoarse)),
            150
        );
    }

    #[test]
    fn good_initialization_policies() {
        let policy = |policy| RecursiveInitParams {
            good_initialization_policy: policy,
            ..RecursiveInitParams::default()
        };
        let spread = Array2::from_shape_fn((20, 2), |(i, d)| (i * (d + 1)) as f32);
        let flat = Array2::zeros((20, 2));
        let heuristic = policy(GoodInitializationPolicy::Heuristic);
        assert!(is_good_initialization(&spread, 2000, &heuristic));
        assert!(!is_good_initialization(&spread, 1000, &heuristic));
        let available = policy(GoodInitializationPolicy::Available);
        assert!(is_good_initialization(&spread, 1, &available));
        assert!(!is_good_initialization(&flat, 1, &available));
        assert!(is_good_initialization(
            &flat,
            1,
            &policy(GoodInitializationPolicy::Always)
        ));
        assert!(!is_good_initialization(
            &spread,
            5000,
            &policy(GoodInitializationPolicy::Never)
        ));
    }

    #[test]
    fn removing_the_coarse_diagonal_drops_in_part_edges() {
        let graph = csr(4, &[(0, 1, 0.5), (2, 3, 0.5), (1, 2, 0.5)]);
        let partition = Partition::new(vec![0, 0, 1, 1]);
        let kept = fuzzy_union_coarse_graph(&graph, &partition, false);
        let removed = fuzzy_union_coarse_graph(&graph, &partition, true);
        assert_eq!(kept.indices.len(), 4);
        assert_eq!(removed.indices.len(), 2);
        for row in 0..2 {
            assert!(!removed.indices[removed.indptr[row]..removed.indptr[row + 1]].contains(&row));
        }
    }

    #[test]
    fn every_recursive_option_yields_a_separated_layout() {
        let (data, labels) = blobs(2500, &CENTERS, 6);
        let graph = knn_graph(&data, 15);
        let (a, b) = crate::find_ab_params(1.0, 0.1);
        let variants = [
            RecursiveInitParams {
                curve_schedule: CurveSchedule::Original,
                ..RecursiveInitParams::default()
            },
            RecursiveInitParams {
                good_initialization_policy: GoodInitializationPolicy::Always,
                ..RecursiveInitParams::default()
            },
            RecursiveInitParams {
                remove_coarse_diagonal: true,
                ..RecursiveInitParams::default()
            },
            RecursiveInitParams {
                anchor_method: AnchorMethod::ProjectedPca,
                anchor_sample_size: 500,
                ..RecursiveInitParams::default()
            },
            RecursiveInitParams {
                negative_selection_range_mode: NegativeSelectionRangeMode::CoarseN,
                ..RecursiveInitParams::default()
            },
            RecursiveInitParams {
                coarsening_ratio: 2,
                ..RecursiveInitParams::default()
            },
        ];
        for params in variants {
            let mut rng = Xoshiro256StarStar::seed_from_u64(11);
            let embedding = recursive_init(&graph, Some(&data), a, b, 2, &params, &mut rng);
            assert!(embedding.iter().all(|x| x.is_finite()));
            let separation = cluster_separation(&embedding, &labels, CENTERS.len());
            assert!(separation > 2.0, "{params:?}: separation {separation}");
        }
    }

    #[cfg(feature = "gpu")]
    thread_local! {
        pub(super) static SIMULATE_DEVICE_LOSS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    #[cfg(feature = "gpu")]
    fn gpu_device() -> Option<GpuDevice> {
        let device = pollster::block_on(crate::gpu_optimize::request_device());
        if device.is_none() {
            eprintln!("skipping: no GPU adapter");
        }
        device
    }

    /// Mean share of the `k` nearest data neighbors that stay among the `k`
    /// nearest embedding neighbors, over every `stride`-th row.
    #[cfg(feature = "gpu")]
    fn knn_recall(data: &Array2<f32>, embedding: &Array2<f32>, k: usize, stride: usize) -> f64 {
        let nearest = |points: &Array2<f32>, i: usize| -> std::collections::HashSet<usize> {
            let mut d: Vec<(f32, usize)> = (0..points.nrows())
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = points
                        .row(i)
                        .iter()
                        .zip(points.row(j))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>();
                    (dist, j)
                })
                .collect();
            d.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
            d[..k].iter().map(|x| x.1).collect()
        };
        let rows: Vec<usize> = (0..data.nrows()).step_by(stride).collect();
        let total: usize = rows
            .par_iter()
            .map(|&i| {
                nearest(data, i)
                    .intersection(&nearest(embedding, i))
                    .count()
            })
            .sum();
        total as f64 / (rows.len() * k) as f64
    }

    #[cfg(feature = "gpu")]
    fn init_on(device: Option<GpuDevice>, graph: &CsrMatrix, data: &Array2<f32>) -> Array2<f32> {
        let (a, b) = crate::find_ab_params(1.0, 0.1);
        let mut rng = Xoshiro256StarStar::seed_from_u64(5);
        pollster::block_on(recursive_init_with_device(
            graph,
            Some(data),
            a,
            b,
            2,
            &RecursiveInitParams::default(),
            &mut rng,
            device,
        ))
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_coarse_layouts_match_the_cpu_init_quality() {
        let Some(device) = gpu_device() else { return };
        let (data, labels) = blobs(20_000, &CENTERS, 21);
        let graph = knn_graph(&data, 15);
        let cpu = init_on(None, &graph, &data);
        let gpu = init_on(Some(device), &graph, &data);
        assert!(gpu.iter().all(|x| x.is_finite()));
        let (cpu_recall, gpu_recall) = (
            knn_recall(&data, &cpu, 15, 40),
            knn_recall(&data, &gpu, 15, 40),
        );
        eprintln!("init recall: cpu {cpu_recall:.3}, gpu {gpu_recall:.3}");
        assert!((cpu_recall - gpu_recall).abs() < 0.05);
        assert!(cluster_separation(&gpu, &labels, CENTERS.len()) > 2.0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_init_keeps_the_final_embedding_quality() {
        if gpu_device().is_none() {
            return;
        }
        let (data, _) = blobs(5_000, &CENTERS, 22);
        let run = |gpu: bool| {
            crate::Umap::builder(&data)
                .compatibility_layout(false)
                .random_state(3)
                .gpu(gpu)
                .build()
                .unwrap()
                .embedding
        };
        let (cpu, gpu) = (run(false), run(true));
        assert!(gpu.iter().all(|x| x.is_finite()));
        let (cpu_recall, gpu_recall) = (
            knn_recall(&data, &cpu, 15, 10),
            knn_recall(&data, &gpu, 15, 10),
        );
        eprintln!("final recall: cpu {cpu_recall:.3}, gpu {gpu_recall:.3}");
        assert!((cpu_recall - gpu_recall).abs() < 0.05);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn lost_device_falls_back_to_the_cpu_for_the_init() {
        let Some(device) = gpu_device() else { return };
        let (data, labels) = blobs(12_000, &CENTERS, 23);
        let graph = knn_graph(&data, 15);
        SIMULATE_DEVICE_LOSS.set(true);
        let embedding = init_on(Some(device), &graph, &data);
        SIMULATE_DEVICE_LOSS.set(false);
        assert_eq!(embedding.dim(), (12_000, 2));
        assert!(embedding.iter().all(|x| x.is_finite()));
        assert!(cluster_separation(&embedding, &labels, CENTERS.len()) > 2.0);
    }
}
