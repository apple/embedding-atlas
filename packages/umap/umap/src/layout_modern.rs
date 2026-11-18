//! Modern UMAP layout optimizers (Adam and momentum).
//!
//! Each CSR edge `i -> j` moves only `i`; the symmetric graph supplies `j -> i`.
//! Negative samples are chosen by a deterministic hash of (edge index, epoch,
//! sample index) inside a window of a random-projection ordering, and all
//! position updates are Jacobi-style (positions are frozen while gradients of a
//! block are summed), so results do not depend on the number of threads.

use crate::graph::{make_epochs_per_sample, CsrMatrix};
use crate::optimize::{rdist, LiveParams};
use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;
use rayon::prelude::*;

pub(crate) const SMALL_LAYOUT_MAX_VERTICES: usize = 1024;
pub(crate) const SMALL_LAYOUT_ATTRACTION_SCALE: f32 = 0.5;
pub(crate) const ADAM_BETA1: f32 = 0.9;
pub(crate) const ADAM_BETA2: f32 = 0.99;
pub(crate) const ADAM_EPSILON: f32 = 1e-4;
const NEGATIVE_SAMPLE_INTERVAL_STRETCH: f64 = 1.5;
const MIN_NEGATIVE_SELECTION_RANGE: usize = 200;
pub(crate) const MAX_REPULSION_NORM: f32 = 16.0;
const SCALE_ADAPTATION_SEED: u64 = 42;
pub(crate) const SCALE_ADAPTATION_NEGATIVES_PER_SOURCE: usize = 5;
const SCALE_ADAPTATION_INTERVAL: usize = 10;

/// The update rule of the modern layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModernOptimizerKind {
    /// Lazy Adam (β1 = 0.9, β2 = 0.99, ε = 1e-4).
    Adam,
    /// Heavy-ball momentum with a scheduled momentum coefficient.
    Momentum,
}

/// Parameters of [`ModernLayout`].
#[derive(Clone, Debug)]
pub struct ModernLayoutParams {
    pub kind: ModernOptimizerKind,
    /// Schedule horizon (number of epochs of a full run).
    pub n_epochs: usize,
    pub a: f32,
    pub b: f32,
    /// Repulsion strength (γ).
    pub gamma: f32,
    /// Peak learning rate (α0) of the schedule.
    pub initial_alpha: f32,
    pub negative_sample_rate: f32,
    /// Width of the projection window that negative samples come from.
    pub negative_selection_range: usize,
    /// Fixed repulsion scale; `None` derives it from the selection range.
    pub negative_sample_scale: Option<f32>,
    /// Number of sources for the Adam negative-scale adaptation; 0 disables it.
    pub scale_adaptation_samples: usize,
    /// Use the shorter warm-up schedules for an already good initial layout.
    pub good_initialization: bool,
    /// Skip negative samples that are graph neighbors of the source.
    pub exclude_graph_neighbors: bool,
    pub seed: u64,
}

/// Negative-selection options of the modern layout (umap-learn 0.6 defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModernOptions {
    /// Width of the projection window that negative samples come from.
    pub negative_selection_range: usize,
    /// Fixed repulsion scale; `None` derives it from the selection range.
    pub negative_sample_scale: Option<f32>,
    /// Number of sources for the Adam negative-scale adaptation; 0 disables it.
    pub scale_adaptation_samples: usize,
    /// Skip negative samples that are graph neighbors of the source.
    pub exclude_graph_neighbors: bool,
}

impl Default for ModernOptions {
    fn default() -> Self {
        ModernOptions {
            negative_selection_range: 200_000,
            negative_sample_scale: None,
            scale_adaptation_samples: 128,
            exclude_graph_neighbors: false,
        }
    }
}

impl ModernLayoutParams {
    pub fn options(&self) -> ModernOptions {
        ModernOptions {
            negative_selection_range: self.negative_selection_range,
            negative_sample_scale: self.negative_sample_scale,
            scale_adaptation_samples: self.scale_adaptation_samples,
            exclude_graph_neighbors: self.exclude_graph_neighbors,
        }
    }

    pub fn set_options(&mut self, options: ModernOptions) {
        self.negative_selection_range = options.negative_selection_range;
        self.negative_sample_scale = options.negative_sample_scale;
        self.scale_adaptation_samples = options.scale_adaptation_samples;
        self.exclude_graph_neighbors = options.exclude_graph_neighbors;
    }
}

impl Default for ModernLayoutParams {
    fn default() -> Self {
        ModernLayoutParams {
            kind: ModernOptimizerKind::Adam,
            n_epochs: 200,
            a: 1.576_943_5,
            b: 0.895_060_9,
            gamma: 1.0,
            initial_alpha: 1.0,
            negative_sample_rate: 5.0,
            negative_selection_range: 200_000,
            negative_sample_scale: None,
            scale_adaptation_samples: 128,
            good_initialization: false,
            exclude_graph_neighbors: false,
            seed: 0,
        }
    }
}

fn adam_learning_rate(epoch: f64, n_epochs: f64, peak: f64, n_warm_up_epochs: usize) -> f64 {
    let warm_up = n_warm_up_epochs as f64;
    if epoch < warm_up {
        (peak - 0.1) * (1.0 - epoch / warm_up).powi(2) + 0.1
    } else {
        0.15 * (1.0 - (epoch - warm_up) / (n_epochs - warm_up)) + 0.05
    }
}

fn momentum_good_initialization_warm_up(n_epochs: usize) -> usize {
    n_epochs.min((n_epochs as f64 / 8.0).max(200.0) as usize)
}

/// Learning rate at `epoch` of a run with `n_epochs` epochs. Epochs past the
/// horizon keep the final value.
pub fn learning_rate_at(
    kind: ModernOptimizerKind,
    epoch: usize,
    n_epochs: usize,
    initial_alpha: f32,
    good_initialization: bool,
) -> f32 {
    let n_epochs = n_epochs.max(1);
    let epoch = epoch.min(n_epochs - 1);
    let n = epoch as f64;
    let total = n_epochs as f64;
    let alpha0 = initial_alpha as f64;
    let rate = match (kind, good_initialization) {
        (ModernOptimizerKind::Adam, true) => {
            adam_learning_rate(n, total, 0.5 * alpha0, (total / 8.0).min(100.0) as usize)
        }
        (ModernOptimizerKind::Adam, false) => {
            adam_learning_rate(n, total, 2.0 * alpha0, (total / 2.0).min(100.0) as usize)
        }
        (ModernOptimizerKind::Momentum, true) => {
            let warm_up = momentum_good_initialization_warm_up(n_epochs);
            let decay = if epoch < warm_up {
                (1.0 - n / warm_up as f64).powi(2)
            } else {
                0.0
            };
            0.25 * decay * alpha0 + 0.005
        }
        (ModernOptimizerKind::Momentum, false) => 0.25 * (1.0 - n / total).powi(2) * alpha0,
    };
    rate as f32
}

/// How an epoch picks its learning rate (and momentum coefficient).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateMode {
    /// The umap-learn 0.6 warm-up/decay schedule over `n_epochs`.
    Schedule,
    /// Flat rates from the live `initial_alpha`: Adam uses α, momentum 0.25·α
    /// with momentum 0.5 (the 0.6 values after the warm-up).
    Flat,
}

/// Learning rate and momentum coefficient of an epoch.
pub fn epoch_rates(
    mode: RateMode,
    kind: ModernOptimizerKind,
    epoch: usize,
    n_epochs: usize,
    initial_alpha: f32,
    good_initialization: bool,
) -> (f32, f32) {
    match (mode, kind) {
        (RateMode::Flat, ModernOptimizerKind::Adam) => (initial_alpha, 0.0),
        (RateMode::Flat, ModernOptimizerKind::Momentum) => (0.25 * initial_alpha, 0.5),
        (RateMode::Schedule, _) => (
            learning_rate_at(kind, epoch, n_epochs, initial_alpha, good_initialization),
            momentum_at(kind, epoch, n_epochs, good_initialization),
        ),
    }
}

/// Momentum coefficient at `epoch` (0 for Adam). Epochs past the horizon keep
/// the final value.
pub fn momentum_at(
    kind: ModernOptimizerKind,
    epoch: usize,
    n_epochs: usize,
    good_initialization: bool,
) -> f32 {
    let n_epochs = n_epochs.max(1);
    let epoch = epoch.min(n_epochs - 1);
    let n = epoch as f64;
    let momentum = match (kind, good_initialization) {
        (ModernOptimizerKind::Adam, _) => 0.0,
        (ModernOptimizerKind::Momentum, true) => {
            let warm_up = momentum_good_initialization_warm_up(n_epochs);
            if epoch < warm_up {
                0.5 * (1.0 - (1.0 - n / warm_up as f64).powi(2))
            } else {
                0.5
            }
        }
        (ModernOptimizerKind::Momentum, false) => {
            0.5 * (1.0 - 0.25 * (1.0 - n / n_epochs as f64).powi(2))
        }
    };
    momentum as f32
}

pub(crate) fn negative_sample_interval(epochs_per_sample: f64, negative_sample_rate: f32) -> f64 {
    if negative_sample_rate > 0.0 {
        epochs_per_sample / negative_sample_rate as f64 * NEGATIVE_SAMPLE_INTERVAL_STRETCH
    } else {
        f64::INFINITY
    }
}

fn shuffle(values: &mut [usize], rng: &mut Xoshiro256StarStar) {
    for i in (1..values.len()).rev() {
        let j = rng.random_range_i64(0, i as i64 + 1) as usize;
        values.swap(i, j);
    }
}

fn standard_normal(rng: &mut Xoshiro256StarStar) -> f64 {
    let u1 = 1.0 - rng.random_f64();
    let u2 = rng.random_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn random_unit_vector(rng: &mut Xoshiro256StarStar, dim: usize) -> Vec<f64> {
    let mut direction: Vec<f64> = (0..dim).map(|_| standard_normal(rng)).collect();
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        direction.iter_mut().for_each(|x| *x /= norm);
    } else if let Some(first) = direction.first_mut() {
        *first = 1.0;
    }
    direction
}

/// Linear-interpolation quantile of sorted values (numpy's default method).
fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// Split `data` into consecutive per-row slices following `indptr`.
fn split_rows<'a, T>(mut data: &'a mut [T], indptr: &[usize]) -> Vec<&'a mut [T]> {
    let mut rows = Vec::with_capacity(indptr.len().saturating_sub(1));
    for bounds in indptr.windows(2) {
        let (row, rest) = std::mem::take(&mut data).split_at_mut(bounds[1] - bounds[0]);
        rows.push(row);
        data = rest;
    }
    rows
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EdgeSchedule {
    pub(crate) next_sample: f64,
    pub(crate) next_negative: f64,
}

/// Advance the schedule of one edge at `epoch`. Returns the number of negative
/// samples (may be negative, which means none) when the edge is due, else `None`.
fn advance_edge_schedule(
    edge: &mut EdgeSchedule,
    epoch: f64,
    epochs_per_sample: f64,
    epochs_per_negative_sample: f64,
) -> Option<i64> {
    if edge.next_sample > epoch {
        return None;
    }
    edge.next_sample += epochs_per_sample;
    if !epochs_per_negative_sample.is_finite() {
        return Some(0);
    }
    let n_negative = ((epoch - edge.next_negative) / epochs_per_negative_sample) as i64;
    edge.next_negative += n_negative as f64 * epochs_per_negative_sample;
    Some(n_negative)
}

/// First projection rank of the negative-selection window of a source.
fn window_start(source_rank: usize, selection_range: usize, n_vertices: usize) -> usize {
    (source_rank as i64 - (selection_range / 2) as i64)
        .clamp(0, (n_vertices - selection_range) as i64) as usize
}

/// Projection rank of negative sample `sample` of edge `raw_index` (large layouts).
fn windowed_candidate_rank(
    raw_index: usize,
    epoch: usize,
    sample: u64,
    range_start: usize,
    selection_range: usize,
    n_vertices: usize,
) -> usize {
    let selection = (raw_index as u64 * (epoch as u64 + sample + 1)) % selection_range as u64;
    (range_start + selection as usize) % n_vertices
}

/// Rank of the next negative candidate of edge `raw_index` (small layouts).
fn small_candidate_rank(
    raw_index: usize,
    epoch: usize,
    accepted: u64,
    attempts: u64,
    n_vertices: usize,
    exclude_graph_neighbors: bool,
) -> usize {
    let (raw_index, epoch) = (raw_index as u64, epoch as u64);
    let rank = if exclude_graph_neighbors {
        raw_index * (epoch + 1) + attempts
    } else {
        raw_index * (epoch + accepted + 1)
    };
    (rank % n_vertices as u64) as usize
}

/// Factor of `x_i - x_j` in the attractive update (0 for coincident points).
fn attraction_coefficient(dist_squared: f32, a: f32, b: f32) -> f32 {
    if dist_squared <= 0.0 {
        return 0.0;
    }
    -2.0 * a * b * dist_squared.powf(b - 1.0) / (a * dist_squared.powf(b) + 1.0)
}

/// Factor of `x_i - x_j` in the repulsive update, including the smooth norm
/// clip `min(γ, 16)·tanh(‖g‖/γ)/‖g‖` (0 for coincident points).
fn repulsion_coefficient(dist_squared: f32, a: f32, b: f32, gamma: f32, scale: f32) -> f32 {
    if dist_squared <= 0.0 {
        return 0.0;
    }
    let grad_coeff =
        scale * 2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_squared.powf(b) + 1.0));
    if grad_coeff <= 0.0 {
        return 0.0;
    }
    let grad_norm = (grad_coeff * grad_coeff * dist_squared).sqrt();
    grad_coeff * gamma.min(MAX_REPULSION_NORM) * (grad_norm / gamma).tanh() / grad_norm
}

/// Bias corrections `(1 - β1^(n+1), 1 - β2^(n+1))` of Adam step `n` (counted
/// from the last moment reset).
pub(crate) fn adam_bias_corrections(step: usize) -> (f32, f32) {
    let exponent = step as f64 + 1.0;
    (
        (1.0 - (ADAM_BETA1 as f64).powf(exponent)) as f32,
        (1.0 - (ADAM_BETA2 as f64).powf(exponent)) as f32,
    )
}

/// One lazy Adam update of a coordinate with nonzero gradient `update`.
/// Returns the position delta.
fn adam_update(
    m: &mut f32,
    v: &mut f32,
    update: f32,
    alpha: f32,
    bias_corrections: (f32, f32),
) -> f32 {
    *m = ADAM_BETA1 * *m + (1.0 - ADAM_BETA1) * update;
    *v = ADAM_BETA2 * *v + (1.0 - ADAM_BETA2) * update * update;
    let m_estimate = *m / bias_corrections.0;
    let v_estimate = *v / bias_corrections.1;
    alpha * m_estimate / (v_estimate.sqrt() + ADAM_EPSILON)
}

/// Repulsion scale state, including the Adam adaptation of the scale.
#[derive(Clone, Debug)]
struct NegativeScale {
    /// `None`: use `effective selection range / n_vertices`.
    value: Option<f64>,
    prior: f64,
    force_ratio_anchor: Option<f64>,
    rng: Xoshiro256StarStar,
}

impl NegativeScale {
    fn new(params: &ModernLayoutParams, n_vertices: usize) -> Self {
        let mut value = params.negative_sample_scale.map(f64::from);
        if params.scale_adaptation_samples > 0 && value.is_none() && n_vertices > 0 {
            let covered = params.negative_selection_range.min(n_vertices) as f64;
            value = Some((covered / n_vertices as f64).powf(0.75));
        }
        NegativeScale {
            value,
            prior: value.unwrap_or(1.0),
            force_ratio_anchor: None,
            rng: Xoshiro256StarStar::seed_from_u64(SCALE_ADAPTATION_SEED),
        }
    }

    fn resolve(&self, effective_range: usize, n_vertices: usize) -> f32 {
        self.value
            .unwrap_or(effective_range as f64 / n_vertices as f64) as f32
    }

    fn adapt(&mut self, force_ratio: f64) {
        let Some(scale) = self.value else { return };
        match self.force_ratio_anchor {
            None => self.force_ratio_anchor = Some(scale / force_ratio),
            Some(anchor) => {
                let raw_scale = (anchor * force_ratio).clamp(0.5 * self.prior, 2.0 * self.prior);
                let log_change = (raw_scale / scale).ln().clamp(0.9f64.ln(), 1.1f64.ln());
                self.value = Some(scale * (0.1 * log_change).exp());
            }
        }
    }
}

pub(crate) fn uses_projection(params: &ModernLayoutParams, n_vertices: usize) -> bool {
    params.negative_selection_range < n_vertices
}

pub(crate) fn effective_selection_range(params: &ModernLayoutParams, n_vertices: usize) -> usize {
    params
        .negative_selection_range
        .max(MIN_NEGATIVE_SELECTION_RANGE)
        .min(n_vertices)
}

/// Source order, negative-selection order (projection ranks), and repulsion
/// scale of a modern layout, with the RNGs that drive them.
#[derive(Clone, Debug)]
pub(crate) struct NegativeOrdering {
    node_order: Vec<usize>,
    pub(crate) to_node_order: Vec<usize>,
    pub(crate) source_ranks: Vec<usize>,
    rng: Xoshiro256StarStar,
    negative_scale: NegativeScale,
    orders_ready: bool,
}

impl NegativeOrdering {
    pub(crate) fn new(params: &ModernLayoutParams, n_vertices: usize) -> Self {
        let identity: Vec<usize> = (0..n_vertices).collect();
        NegativeOrdering {
            node_order: identity.clone(),
            to_node_order: identity.clone(),
            source_ranks: identity,
            rng: Xoshiro256StarStar::seed_from_u64(params.seed),
            negative_scale: NegativeScale::new(params, n_vertices),
            orders_ready: false,
        }
    }

    pub(crate) fn node_order(&self) -> &[usize] {
        &self.node_order
    }

    /// Prior of the adapted repulsion scale.
    #[cfg(feature = "gpu")]
    pub(crate) fn scale_prior(&self) -> f32 {
        self.negative_scale.prior as f32
    }

    /// Repulsion scale of the large kernels at the current state.
    pub(crate) fn negative_scale(&self, params: &ModernLayoutParams, n_vertices: usize) -> f32 {
        self.negative_scale
            .resolve(effective_selection_range(params, n_vertices), n_vertices)
    }

    /// Mark the orderings ready; returns the first projection direction if this
    /// layout needs one. `None` leaves the identity orderings.
    pub(crate) fn prepare(
        &mut self,
        dim: usize,
        params: &ModernLayoutParams,
    ) -> Option<OrderUpdate> {
        if self.orders_ready {
            return None;
        }
        self.orders_ready = true;
        let n_vertices = self.to_node_order.len();
        if n_vertices > SMALL_LAYOUT_MAX_VERTICES && uses_projection(params, n_vertices) {
            return Some(OrderUpdate::Project(random_unit_vector(&mut self.rng, dim)));
        }
        Some(OrderUpdate::Shuffled)
    }

    /// Shuffle the source order and advance the negative-selection order: either
    /// shuffled in place, or a projection direction for the caller to sort by.
    pub(crate) fn advance(&mut self, dim: usize, params: &ModernLayoutParams) -> OrderUpdate {
        shuffle(&mut self.node_order, &mut self.rng);
        if uses_projection(params, self.to_node_order.len()) {
            return OrderUpdate::Project(random_unit_vector(&mut self.rng, dim));
        }
        if params.kind == ModernOptimizerKind::Adam {
            shuffle(&mut self.to_node_order, &mut self.rng);
        } else {
            self.to_node_order.copy_from_slice(&self.node_order);
        }
        OrderUpdate::Shuffled
    }

    /// Compute the first projection ordering if this layout needs one.
    pub(crate) fn prepare_with(
        &mut self,
        embedding: &[f32],
        dim: usize,
        params: &ModernLayoutParams,
    ) {
        if let Some(OrderUpdate::Project(direction)) = self.prepare(dim, params) {
            self.apply_projection(embedding, dim, &direction);
        }
    }

    /// Shuffle the source order and refresh the negative-selection order.
    pub(crate) fn reorder_after_epoch(
        &mut self,
        embedding: &[f32],
        dim: usize,
        params: &ModernLayoutParams,
    ) {
        if let OrderUpdate::Project(direction) = self.advance(dim, params) {
            self.apply_projection(embedding, dim, &direction);
        }
    }

    /// Order all vertices by their projection on `direction`.
    fn apply_projection(&mut self, embedding: &[f32], dim: usize, direction: &[f64]) {
        let projection: Vec<f64> = embedding
            .par_chunks(dim.max(1))
            .map(|row| row.iter().zip(direction).map(|(&x, &d)| x as f64 * d).sum())
            .collect();
        self.to_node_order
            .iter_mut()
            .enumerate()
            .for_each(|(i, node)| *node = i);
        self.to_node_order
            .par_sort_by(|&i, &j| projection[i].total_cmp(&projection[j]));
        for (rank, &node) in self.to_node_order.iter().enumerate() {
            self.source_ranks[node] = rank;
        }
    }

    /// Projection ranks for the Adam repulsion-scale adaptation, when `epoch`
    /// is an adaptation epoch.
    pub(crate) fn adaptation_ranks(
        &mut self,
        params: &ModernLayoutParams,
        epoch: usize,
    ) -> Option<AdaptationRanks> {
        let n_vertices = self.to_node_order.len();
        if params.scale_adaptation_samples == 0
            || !uses_projection(params, n_vertices)
            || epoch < SCALE_ADAPTATION_INTERVAL
            || epoch > params.n_epochs / 2
            || !epoch.is_multiple_of(SCALE_ADAPTATION_INTERVAL)
        {
            return None;
        }
        let n_sources = params.scale_adaptation_samples.min(n_vertices);
        let per_source = SCALE_ADAPTATION_NEGATIVES_PER_SOURCE;
        let range = params.negative_selection_range.max(1);
        let rng = &mut self.negative_scale.rng;
        let sources: Vec<usize> = (0..n_sources)
            .map(|_| rng.random_range_i64(0, n_vertices as i64) as usize)
            .collect();
        let local: Vec<usize> = sources
            .iter()
            .flat_map(|&rank| {
                let start = window_start(rank, range, n_vertices);
                (0..per_source)
                    .map(|_| start + rng.random_range_i64(0, range as i64) as usize)
                    .collect::<Vec<_>>()
            })
            .collect();
        let global: Vec<usize> = (0..n_sources * per_source)
            .map(|_| rng.random_range_i64(0, n_vertices as i64) as usize)
            .collect();
        Some(AdaptationRanks {
            sources,
            local,
            global,
        })
    }

    /// Adapt the Adam repulsion scale at the scheduled epochs (large layouts).
    pub(crate) fn adapt_negative_scale(
        &mut self,
        embedding: &[f32],
        dim: usize,
        graph: GraphRows,
        params: &ModernLayoutParams,
        epoch: usize,
    ) {
        let Some(ranks) = self.adaptation_ranks(params, epoch) else {
            return;
        };
        let force_ratio = self.negative_force_ratio(embedding, dim, graph, params, &ranks);
        self.negative_scale.adapt(force_ratio);
    }

    /// Robust estimate of the global/local resultant repulsion ratio.
    fn negative_force_ratio(
        &self,
        embedding: &[f32],
        dim: usize,
        graph: GraphRows,
        params: &ModernLayoutParams,
        ranks: &AdaptationRanks,
    ) -> f64 {
        let per_source = SCALE_ADAPTATION_NEGATIVES_PER_SOURCE;
        let source_ranks = &ranks.sources;
        let resultants = |ranks: &[usize]| -> Vec<f64> {
            source_ranks
                .iter()
                .zip(ranks.chunks(per_source))
                .map(|(&rank, candidate_ranks)| {
                    let candidates: Vec<usize> = candidate_ranks
                        .iter()
                        .map(|&r| self.to_node_order[r])
                        .collect();
                    repulsion_resultant(
                        embedding,
                        dim,
                        graph,
                        params,
                        self.to_node_order[rank],
                        &candidates,
                    )
                })
                .collect()
        };
        let global = resultants(&ranks.global);
        let local = resultants(&ranks.local);

        let mut log_ratios: Vec<f64> = global
            .iter()
            .zip(&local)
            .filter(|(&g, &l)| g > 0.0 && l > 0.0)
            .map(|(&g, &l)| g.ln() - l.ln())
            .collect();
        if log_ratios.is_empty() {
            return 1.0;
        }
        let mut sorted = log_ratios.clone();
        sorted.sort_by(f64::total_cmp);
        let lower = sorted_quantile(&sorted, 0.1);
        let upper = sorted_quantile(&sorted, 0.9);
        log_ratios
            .iter_mut()
            .for_each(|r| *r = r.clamp(lower, upper));
        (log_ratios.iter().sum::<f64>() / log_ratios.len() as f64).exp()
    }
}

/// How the negative-selection order changed.
pub(crate) enum OrderUpdate {
    /// `to_node_order` was shuffled (or copied) on the host.
    Shuffled,
    /// Sort all vertices by their projection on this unit direction.
    Project(Vec<f64>),
}

/// Projection ranks sampled for one repulsion-scale adaptation:
/// `SCALE_ADAPTATION_NEGATIVES_PER_SOURCE` local and global candidates per source.
pub(crate) struct AdaptationRanks {
    pub(crate) sources: Vec<usize>,
    pub(crate) local: Vec<usize>,
    pub(crate) global: Vec<usize>,
}

/// Sorted CSR rows of the layout graph.
#[derive(Clone, Copy)]
pub(crate) struct GraphRows<'a> {
    pub(crate) indptr: &'a [usize],
    pub(crate) indices: &'a [usize],
}

impl GraphRows<'_> {
    fn neighbors(&self, node: usize) -> &[usize] {
        &self.indices[self.indptr[node]..self.indptr[node + 1]]
    }
}

/// Unscaled, unclipped-by-cap resultant repulsion of `source` against `candidates`.
fn repulsion_resultant(
    embedding: &[f32],
    dim: usize,
    graph: GraphRows,
    params: &ModernLayoutParams,
    source: usize,
    candidates: &[usize],
) -> f64 {
    let (a, b, gamma) = (params.a as f64, params.b as f64, params.gamma as f64);
    let current = &embedding[source * dim..(source + 1) * dim];
    let neighbors = graph.neighbors(source);
    let mut resultant = vec![0.0f64; dim];
    for &candidate in candidates {
        if params.exclude_graph_neighbors
            && (candidate == source || neighbors.binary_search(&candidate).is_ok())
        {
            continue;
        }
        let other = &embedding[candidate * dim..(candidate + 1) * dim];
        let delta: Vec<f64> = current
            .iter()
            .zip(other)
            .map(|(&c, &o)| (c - o) as f64)
            .collect();
        let dist_squared: f64 = delta.iter().map(|d| d * d).sum();
        let coefficient =
            2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_squared.powf(b) + 1.0));
        let gradient_norm = coefficient * dist_squared.sqrt();
        let clip = if gradient_norm > 0.0 {
            gamma * (gradient_norm / gamma).tanh() / gradient_norm
        } else {
            1.0
        };
        for (r, d) in resultant.iter_mut().zip(&delta) {
            *r += coefficient * d * clip;
        }
    }
    resultant.iter().map(|r| r * r).sum::<f64>().sqrt()
}

/// Read-only inputs of one gradient accumulation pass.
struct EpochContext<'a> {
    embedding: &'a [f32],
    dim: usize,
    n_vertices: usize,
    graph: GraphRows<'a>,
    epochs_per_sample: &'a [f64],
    epochs_per_negative_sample: &'a [f64],
    to_node_order: &'a [usize],
    source_ranks: &'a [usize],
    epoch: usize,
    a: f32,
    b: f32,
    gamma: f32,
    attraction_factor: f32,
    repulsion_factor: f32,
    negative_scale: f32,
    selection_range: usize,
    small: bool,
    exclude_graph_neighbors: bool,
}

impl EpochContext<'_> {
    fn position(&self, node: usize) -> &[f32] {
        &self.embedding[node * self.dim..(node + 1) * self.dim]
    }

    fn is_excluded(&self, from: usize, to: usize) -> bool {
        to == from || self.graph.neighbors(from).binary_search(&to).is_ok()
    }

    fn add_scaled_difference(factor: f32, current: &[f32], other: &[f32], update: &mut [f32]) {
        if factor == 0.0 {
            return;
        }
        for ((u, &c), &o) in update.iter_mut().zip(current).zip(other) {
            *u += factor * (c - o);
        }
    }

    fn attract(&self, current: &[f32], other: &[f32], update: &mut [f32]) {
        let coefficient = attraction_coefficient(rdist(current, other), self.a, self.b);
        Self::add_scaled_difference(self.attraction_factor * coefficient, current, other, update);
    }

    fn repel(&self, current: &[f32], other: &[f32], scale: f32, update: &mut [f32]) {
        let coefficient =
            repulsion_coefficient(rdist(current, other), self.a, self.b, self.gamma, scale);
        Self::add_scaled_difference(self.repulsion_factor * coefficient, current, other, update);
    }

    fn repel_windowed(&self, from: usize, raw_index: usize, n_negative: i64, update: &mut [f32]) {
        let current = self.position(from);
        let range = self.selection_range;
        let range_start = window_start(self.source_ranks[from], range, self.n_vertices);
        for sample in 0..n_negative.max(0) as u64 {
            let rank = windowed_candidate_rank(
                raw_index,
                self.epoch,
                sample,
                range_start,
                range,
                self.n_vertices,
            );
            let to = self.to_node_order[rank];
            if self.exclude_graph_neighbors && self.is_excluded(from, to) {
                continue;
            }
            self.repel(current, self.position(to), self.negative_scale, update);
        }
    }

    fn repel_small(&self, from: usize, raw_index: usize, n_negative: i64, update: &mut [f32]) {
        let current = self.position(from);
        let n_negative = n_negative.max(0) as u64;
        let max_attempts = n_negative.max(1) * self.n_vertices as u64;
        let mut accepted = 0u64;
        let mut attempts = 0u64;
        while accepted < n_negative && attempts < max_attempts {
            let rank = small_candidate_rank(
                raw_index,
                self.epoch,
                accepted,
                attempts,
                self.n_vertices,
                self.exclude_graph_neighbors,
            );
            attempts += 1;
            let to = self.to_node_order[rank];
            if self.exclude_graph_neighbors && self.is_excluded(from, to) {
                continue;
            }
            accepted += 1;
            self.repel(current, self.position(to), 1.0, update);
        }
    }

    fn accumulate_source(&self, from: usize, edges: &mut [EdgeSchedule], update: &mut [f32]) {
        let n = self.epoch as f64;
        let first_edge = self.graph.indptr[from];
        let current = self.position(from);
        for (offset, edge) in edges.iter_mut().enumerate() {
            let raw_index = first_edge + offset;
            let Some(n_negative) = advance_edge_schedule(
                edge,
                n,
                self.epochs_per_sample[raw_index],
                self.epochs_per_negative_sample[raw_index],
            ) else {
                continue;
            };
            self.attract(
                current,
                self.position(self.graph.indices[raw_index]),
                update,
            );
            if self.small {
                self.repel_small(from, raw_index, n_negative, update);
            } else {
                self.repel_windowed(from, raw_index, n_negative, update);
            }
        }
    }
}

/// Resumable state of a modern layout optimization. [`step`](Self::step) runs
/// one epoch; [`run`](Self::run) runs to the schedule horizon.
#[derive(Clone, Debug)]
pub struct ModernLayout {
    params: ModernLayoutParams,
    n_vertices: usize,
    dim: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    epochs_per_sample: Vec<f64>,
    epochs_per_negative_sample: Vec<f64>,
    edge_schedule: Vec<EdgeSchedule>,
    updates: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    /// Adam steps since the moments were last reset.
    adam_steps: usize,
    ordering: NegativeOrdering,
    epoch: usize,
}

impl ModernLayout {
    /// Prepare a layout of `graph` (pruned, symmetric, sorted CSR) in `dim` dimensions.
    pub fn new(graph: &CsrMatrix, dim: usize, params: ModernLayoutParams) -> Self {
        let epochs_per_sample = make_epochs_per_sample(&graph.data, params.n_epochs)
            .into_iter()
            .map(f64::from)
            .collect();
        Self::from_csr_parts(
            graph.indptr.clone(),
            graph.indices.clone(),
            epochs_per_sample,
            dim,
            params,
            0,
        )
    }

    /// Build from CSR structure plus per-edge sampling intervals, starting the
    /// schedule at `start_epoch`.
    pub(crate) fn from_csr_parts(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        epochs_per_sample: Vec<f64>,
        dim: usize,
        params: ModernLayoutParams,
        start_epoch: usize,
    ) -> Self {
        let n_vertices = indptr.len().saturating_sub(1);
        assert_eq!(indices.len(), epochs_per_sample.len());
        let n_edges = indices.len();
        let mut layout = ModernLayout {
            ordering: NegativeOrdering::new(&params, n_vertices),
            params,
            n_vertices,
            dim,
            indptr,
            indices,
            epochs_per_sample,
            epochs_per_negative_sample: vec![0.0; n_edges],
            edge_schedule: Vec::new(),
            updates: Vec::new(),
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_steps: 0,
            epoch: start_epoch,
        };
        layout.restart(start_epoch);
        layout
    }

    /// Reset all optimizer state (moments, sampling schedules, orderings, RNGs)
    /// and continue the schedule from `start_epoch`.
    pub fn restart(&mut self, start_epoch: usize) {
        let buffer_len = self.n_vertices * self.dim;
        self.updates = vec![0.0; buffer_len];
        let uses_adam = self.params.kind == ModernOptimizerKind::Adam;
        self.adam_m = if uses_adam {
            vec![0.0; buffer_len]
        } else {
            Vec::new()
        };
        self.adam_v = self.adam_m.clone();
        self.adam_steps = 0;
        self.ordering = NegativeOrdering::new(&self.params, self.n_vertices);
        self.epoch = start_epoch;
        self.edge_schedule = vec![
            EdgeSchedule {
                next_sample: 0.0,
                next_negative: 0.0
            };
            self.epochs_per_sample.len()
        ];
        let from = start_epoch as f64;
        for (edge, &interval) in self.edge_schedule.iter_mut().zip(&self.epochs_per_sample) {
            edge.next_sample = from + interval;
        }
        self.reseed_negative_schedule(from);
    }

    fn reseed_negative_schedule(&mut self, from_epoch: f64) {
        let rate = self.params.negative_sample_rate;
        for ((edge, negative_interval), &interval) in self
            .edge_schedule
            .iter_mut()
            .zip(self.epochs_per_negative_sample.iter_mut())
            .zip(&self.epochs_per_sample)
        {
            *negative_interval = negative_sample_interval(interval, rate);
            edge.next_negative = from_epoch + *negative_interval;
        }
    }

    pub fn params(&self) -> &ModernLayoutParams {
        &self.params
    }

    /// The number of epochs run so far (including the start offset).
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Change the parameters that can vary between epochs. A changed
    /// negative-sample rate re-arms the negative schedule from the current epoch.
    pub fn set_live_params(&mut self, live: LiveParams) {
        self.params.initial_alpha = live.alpha;
        self.params.gamma = live.gamma;
        self.params.a = live.a;
        self.params.b = live.b;
        if live.negative_sample_rate != self.params.negative_sample_rate {
            self.params.negative_sample_rate = live.negative_sample_rate;
            self.reseed_negative_schedule(self.epoch as f64);
        }
    }

    /// Change the schedule horizon.
    pub fn set_n_epochs(&mut self, n_epochs: usize) {
        self.params.n_epochs = n_epochs;
    }

    pub fn set_good_initialization(&mut self, good_initialization: bool) {
        self.params.good_initialization = good_initialization;
    }

    /// Change the negative-selection options; this resets the orderings, their
    /// RNGs, and the repulsion scale.
    pub fn set_options(&mut self, options: ModernOptions) {
        self.params.set_options(options);
        self.ordering = NegativeOrdering::new(&self.params, self.n_vertices);
    }

    /// Run epochs until the schedule horizon is reached.
    pub fn run(&mut self, embedding: &mut Array2<f32>) {
        while self.epoch < self.params.n_epochs {
            self.step(embedding);
        }
    }

    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn edge_schedule(&self) -> &[EdgeSchedule] {
        &self.edge_schedule
    }

    /// Zero the Adam moments (after a change of the gradient field, whose stale
    /// second moments would damp the response).
    pub fn reset_moments(&mut self) {
        self.adam_m.fill(0.0);
        self.adam_v.fill(0.0);
        self.adam_steps = 0;
    }

    /// Scale the momentum velocity, which holds the learning rate it was built
    /// with, to a new learning rate.
    pub fn scale_velocity(&mut self, factor: f32) {
        if self.params.kind == ModernOptimizerKind::Momentum {
            self.updates.iter_mut().for_each(|u| *u *= factor);
        }
    }

    /// Run one epoch on the 0.6 schedule, updating `embedding` (shape
    /// `n_vertices × dim`, standard layout).
    pub fn step(&mut self, embedding: &mut Array2<f32>) {
        self.step_with(embedding, RateMode::Schedule);
    }

    /// Run one epoch with rates from `mode`.
    pub fn step_with(&mut self, embedding: &mut Array2<f32>, mode: RateMode) {
        assert_eq!(embedding.nrows(), self.n_vertices);
        assert_eq!(embedding.ncols(), self.dim);
        let embedding = embedding
            .as_slice_mut()
            .expect("embedding must be in standard layout");
        let small = self.n_vertices <= SMALL_LAYOUT_MAX_VERTICES;

        self.ordering
            .prepare_with(embedding, self.dim, &self.params);

        let kind = self.params.kind;
        let (alpha, momentum) = epoch_rates(
            mode,
            kind,
            self.epoch,
            self.params.n_epochs,
            self.params.initial_alpha,
            self.params.good_initialization,
        );

        match kind {
            ModernOptimizerKind::Adam => {
                if !small {
                    let graph = GraphRows {
                        indptr: &self.indptr,
                        indices: &self.indices,
                    };
                    self.ordering.adapt_negative_scale(
                        embedding,
                        self.dim,
                        graph,
                        &self.params,
                        self.epoch,
                    );
                }
                self.accumulate(embedding, None, alpha, small);
                self.apply_adam(embedding, alpha);
                self.updates.fill(0.0);
            }
            ModernOptimizerKind::Momentum => {
                if small {
                    self.accumulate(embedding, None, alpha, small);
                    self.apply_momentum(embedding, None);
                } else {
                    let block_size = (self.n_vertices / 2).max(1);
                    let mut in_block = vec![false; self.n_vertices];
                    for block_start in (0..self.n_vertices).step_by(block_size) {
                        let block_end = (block_start + block_size).min(self.n_vertices);
                        in_block.fill(false);
                        for &node in &self.ordering.node_order()[block_start..block_end] {
                            in_block[node] = true;
                        }
                        self.accumulate(embedding, Some(&in_block), alpha, small);
                        self.apply_momentum(embedding, Some(&in_block));
                    }
                }
                self.updates.par_iter_mut().for_each(|u| *u *= momentum);
            }
        }

        self.ordering
            .reorder_after_epoch(embedding, self.dim, &self.params);
        self.epoch += 1;
    }

    fn accumulate(&mut self, embedding: &[f32], active: Option<&[bool]>, alpha: f32, small: bool) {
        let uses_momentum = self.params.kind == ModernOptimizerKind::Momentum;
        let momentum_alpha = if uses_momentum { alpha } else { 1.0 };
        let small_scale = if small {
            SMALL_LAYOUT_ATTRACTION_SCALE
        } else {
            1.0
        };
        let selection_range = effective_selection_range(&self.params, self.n_vertices);
        let context = EpochContext {
            embedding,
            dim: self.dim,
            n_vertices: self.n_vertices,
            graph: GraphRows {
                indptr: &self.indptr,
                indices: &self.indices,
            },
            epochs_per_sample: &self.epochs_per_sample,
            epochs_per_negative_sample: &self.epochs_per_negative_sample,
            to_node_order: &self.ordering.to_node_order,
            source_ranks: &self.ordering.source_ranks,
            epoch: self.epoch,
            a: self.params.a,
            b: self.params.b,
            gamma: self.params.gamma,
            attraction_factor: small_scale * momentum_alpha,
            repulsion_factor: momentum_alpha,
            negative_scale: self.ordering.negative_scale(&self.params, self.n_vertices),
            selection_range,
            small,
            exclude_graph_neighbors: self.params.exclude_graph_neighbors,
        };
        let dim = self.dim.max(1);
        split_rows(&mut self.edge_schedule, &self.indptr)
            .into_par_iter()
            .zip(self.updates.par_chunks_mut(dim))
            .enumerate()
            .filter(|(from, _)| active.is_none_or(|mask| mask[*from]))
            .for_each(|(from, (edges, update))| context.accumulate_source(from, edges, update));
    }

    fn apply_adam(&mut self, embedding: &mut [f32], alpha: f32) {
        let bias_corrections = adam_bias_corrections(self.adam_steps);
        self.adam_steps += 1;
        let dim = self.dim.max(1);
        embedding
            .par_chunks_mut(dim)
            .zip(self.updates.par_chunks(dim))
            .zip(self.adam_m.par_chunks_mut(dim))
            .zip(self.adam_v.par_chunks_mut(dim))
            .for_each(|(((position, update), m), v)| {
                for d in 0..position.len() {
                    if update[d] != 0.0 {
                        position[d] +=
                            adam_update(&mut m[d], &mut v[d], update[d], alpha, bias_corrections);
                    }
                }
            });
    }

    fn apply_momentum(&mut self, embedding: &mut [f32], active: Option<&[bool]>) {
        let dim = self.dim.max(1);
        embedding
            .par_chunks_mut(dim)
            .zip(self.updates.par_chunks(dim))
            .enumerate()
            .filter(|(node, _)| active.is_none_or(|mask| mask[*node]))
            .for_each(|(_, (position, update))| {
                for (x, u) in position.iter_mut().zip(update) {
                    *x += u;
                }
            });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::graph::fuzzy_simplicial_set;

    fn optimize(embedding: &mut Array2<f32>, graph: &CsrMatrix, params: &ModernLayoutParams) {
        ModernLayout::new(graph, embedding.ncols(), params.clone()).run(embedding);
    }

    pub(crate) fn gaussian_blobs(
        n_per_blob: usize,
        dim: usize,
        offset: f32,
        seed: u64,
    ) -> Array2<f32> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        Array2::from_shape_fn((2 * n_per_blob, dim), |(i, d)| {
            let center = if i < n_per_blob || d != 0 {
                0.0
            } else {
                offset
            };
            center + standard_normal(&mut rng) as f32
        })
    }

    fn squared_distance(data: &Array2<f32>, i: usize, j: usize) -> f32 {
        rdist(
            data.row(i).as_slice().unwrap(),
            data.row(j).as_slice().unwrap(),
        )
    }

    /// Brute-force neighbor ranking of every point (self excluded), nearest first.
    fn neighbor_ranking(data: &Array2<f32>) -> Vec<Vec<usize>> {
        let n = data.nrows();
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut others: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                others.sort_by(|&x, &y| {
                    squared_distance(data, i, x).total_cmp(&squared_distance(data, i, y))
                });
                others
            })
            .collect()
    }

    pub(crate) fn pruned_graph(
        data: &Array2<f32>,
        n_neighbors: usize,
        n_epochs: usize,
    ) -> CsrMatrix {
        let ranking = neighbor_ranking(data);
        let n = data.nrows();
        let knn_indices = Array2::from_shape_fn((n, n_neighbors), |(i, k)| {
            if k == 0 {
                i as i32
            } else {
                ranking[i][k - 1] as i32
            }
        });
        let knn_dists = Array2::from_shape_fn((n, n_neighbors), |(i, k)| {
            squared_distance(data, i, knn_indices[[i, k]] as usize).sqrt()
        });
        let mut graph = fuzzy_simplicial_set(&knn_indices, &knn_dists, n_neighbors, 1.0, 1.0);
        let threshold = graph.max_weight() / n_epochs as f32;
        graph.prune(threshold);
        graph.to_csr()
    }

    pub(crate) fn random_init(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |_| rng.random_f32() * 20.0 - 10.0)
    }

    /// Trustworthiness (Venna & Kaski) with `k` neighbors.
    pub(crate) fn trustworthiness(data: &Array2<f32>, embedding: &Array2<f32>, k: usize) -> f64 {
        let n = data.nrows();
        let data_ranking = neighbor_ranking(data);
        let embedding_ranking = neighbor_ranking(embedding);
        let penalty: f64 = (0..n)
            .map(|i| {
                let mut rank_of = vec![0usize; n];
                for (rank, &j) in data_ranking[i].iter().enumerate() {
                    rank_of[j] = rank + 1;
                }
                embedding_ranking[i][..k]
                    .iter()
                    .map(|&j| rank_of[j].saturating_sub(k) as f64)
                    .sum::<f64>()
            })
            .sum();
        let (n, k) = (n as f64, k as f64);
        1.0 - 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)) * penalty
    }

    fn run_with_threads(
        threads: usize,
        graph: &CsrMatrix,
        init: &Array2<f32>,
        params: &ModernLayoutParams,
    ) -> Array2<f32> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut embedding = init.clone();
            optimize(&mut embedding, graph, params);
            embedding
        })
    }

    fn assert_thread_count_invariant(n_per_blob: usize, params: ModernLayoutParams) {
        let data = gaussian_blobs(n_per_blob, 8, 6.0, 3);
        let graph = pruned_graph(&data, 10, params.n_epochs);
        let init = random_init(data.nrows(), 2, 5);
        let single = run_with_threads(1, &graph, &init, &params);
        let multi = run_with_threads(4, &graph, &init, &params);
        assert!(single.iter().all(|x| x.is_finite()));
        assert_eq!(single, multi);
        assert_ne!(single, init);
    }

    #[test]
    fn small_adam_is_thread_count_invariant() {
        assert_thread_count_invariant(150, ModernLayoutParams::default());
    }

    #[test]
    fn large_adam_with_window_is_thread_count_invariant() {
        assert_thread_count_invariant(
            800,
            ModernLayoutParams {
                n_epochs: 60,
                negative_selection_range: 300,
                ..ModernLayoutParams::default()
            },
        );
    }

    #[test]
    fn large_adam_full_range_with_exclusion_is_thread_count_invariant() {
        assert_thread_count_invariant(
            600,
            ModernLayoutParams {
                n_epochs: 30,
                exclude_graph_neighbors: true,
                ..ModernLayoutParams::default()
            },
        );
    }

    #[test]
    fn momentum_is_thread_count_invariant() {
        for n_per_blob in [150, 700] {
            assert_thread_count_invariant(
                n_per_blob,
                ModernLayoutParams {
                    kind: ModernOptimizerKind::Momentum,
                    n_epochs: 40,
                    negative_selection_range: 400,
                    good_initialization: true,
                    ..ModernLayoutParams::default()
                },
            );
        }
    }

    #[test]
    fn small_layout_with_exclusion_is_finite() {
        let data = gaussian_blobs(100, 6, 5.0, 11);
        let graph = pruned_graph(&data, 10, 100);
        let mut embedding = random_init(data.nrows(), 2, 1);
        let params = ModernLayoutParams {
            n_epochs: 100,
            exclude_graph_neighbors: true,
            negative_selection_range: 50,
            ..ModernLayoutParams::default()
        };
        optimize(&mut embedding, &graph, &params);
        assert!(embedding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn adam_separates_two_blobs() {
        let n_per_blob = 200;
        let data = gaussian_blobs(n_per_blob, 10, 12.0, 7);
        let graph = pruned_graph(&data, 15, 200);
        let mut embedding = random_init(data.nrows(), 2, 9);
        optimize(&mut embedding, &graph, &ModernLayoutParams::default());

        let centroid = |range: std::ops::Range<usize>| -> [f32; 2] {
            let count = range.len() as f32;
            let mut sum = [0.0f32; 2];
            for i in range {
                sum[0] += embedding[[i, 0]];
                sum[1] += embedding[[i, 1]];
            }
            [sum[0] / count, sum[1] / count]
        };
        let first = centroid(0..n_per_blob);
        let second = centroid(n_per_blob..2 * n_per_blob);
        let nearest_is_own_blob = (0..2 * n_per_blob)
            .filter(|&i| {
                let p = [embedding[[i, 0]], embedding[[i, 1]]];
                let to_first = rdist(&p, &first);
                let to_second = rdist(&p, &second);
                (i < n_per_blob) == (to_first < to_second)
            })
            .count();
        assert_eq!(nearest_is_own_blob, 2 * n_per_blob);
    }

    #[test]
    fn large_adam_layout_is_trustworthy() {
        let data = gaussian_blobs(700, 6, 4.0, 13);
        let params = ModernLayoutParams {
            negative_selection_range: 500,
            ..ModernLayoutParams::default()
        };
        let graph = pruned_graph(&data, 15, params.n_epochs);
        let mut embedding = random_init(data.nrows(), 2, 17);
        optimize(&mut embedding, &graph, &params);
        assert!(embedding.iter().all(|x| x.is_finite()));
        let random_trust = trustworthiness(&data, &random_init(data.nrows(), 2, 17), 10);
        let trust = trustworthiness(&data, &embedding, 10);
        assert!(
            trust > 0.9,
            "trustworthiness {trust} (random {random_trust})"
        );
    }

    #[test]
    fn negative_scale_adaptation_changes_scale() {
        let data = gaussian_blobs(700, 6, 4.0, 19);
        let params = ModernLayoutParams {
            n_epochs: 60,
            negative_selection_range: 300,
            ..ModernLayoutParams::default()
        };
        let graph = pruned_graph(&data, 10, params.n_epochs);
        let mut embedding = random_init(data.nrows(), 2, 23);
        let mut layout = ModernLayout::new(&graph, 2, params);
        let initial_scale = layout.ordering.negative_scale.value.unwrap();
        assert!((initial_scale - (300.0f64 / 1400.0).powf(0.75)).abs() < 1e-12);
        layout.run(&mut embedding);
        assert!(layout.ordering.negative_scale.force_ratio_anchor.is_some());
        let final_scale = layout.ordering.negative_scale.value.unwrap();
        assert!(final_scale.is_finite() && final_scale > 0.0);
        assert_ne!(final_scale, initial_scale);
        assert!(final_scale >= 0.5 * initial_scale && final_scale <= 2.0 * initial_scale);
    }

    #[test]
    fn step_matches_run() {
        let data = gaussian_blobs(80, 5, 5.0, 29);
        let params = ModernLayoutParams {
            n_epochs: 50,
            ..ModernLayoutParams::default()
        };
        let graph = pruned_graph(&data, 10, params.n_epochs);
        let init = random_init(data.nrows(), 2, 31);
        let mut ran = init.clone();
        optimize(&mut ran, &graph, &params);
        let mut stepped = init;
        let mut layout = ModernLayout::new(&graph, 2, params);
        for _ in 0..50 {
            layout.step(&mut stepped);
        }
        assert_eq!(layout.epoch(), 50);
        assert_eq!(ran, stepped);
    }

    #[test]
    fn negative_selection_hash_matches_reference_formulas() {
        assert_eq!(window_start(3, 200, 1000), 0);
        assert_eq!(window_start(500, 200, 1000), 400);
        assert_eq!(window_start(990, 200, 1000), 800);
        // (raw_index · (epoch + sample + 1)) % range, offset by the window start.
        assert_eq!(windowed_candidate_rank(7, 4, 2, 400, 200, 1000), 400 + 49);
        assert_eq!(windowed_candidate_rank(1000, 9, 0, 800, 200, 1000), 800);
        assert_eq!(small_candidate_rank(7, 4, 2, 9, 30, false), 49 % 30);
        assert_eq!(small_candidate_rank(7, 4, 2, 9, 30, true), (35 + 9) % 30);
    }

    #[test]
    fn edge_schedule_follows_reference_recurrence() {
        let mut edge = EdgeSchedule {
            next_sample: 1.0,
            next_negative: 0.3,
        };
        assert_eq!(advance_edge_schedule(&mut edge, 0.0, 1.0, 0.3), None);
        // Epoch 1: int((1 - 0.3) / 0.3) = 2 negatives.
        assert_eq!(advance_edge_schedule(&mut edge, 1.0, 1.0, 0.3), Some(2));
        assert_eq!(edge.next_sample, 2.0);
        assert!((edge.next_negative - 0.9).abs() < 1e-12);
        assert_eq!(
            advance_edge_schedule(&mut edge, 2.0, 1.0, f64::INFINITY),
            Some(0)
        );
    }

    fn assert_close(actual: f32, expected: f64) {
        assert!(
            (actual as f64 - expected).abs() < 1e-6,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn adam_schedules_match_reference_formulas() {
        let adam = ModernOptimizerKind::Adam;
        // Not good init, 200 epochs, α0 = 1: warm-up int(min(100, 100)) = 100.
        assert_close(learning_rate_at(adam, 0, 200, 1.0, false), 2.0);
        assert_close(
            learning_rate_at(adam, 50, 200, 1.0, false),
            1.9 * 0.25 + 0.1,
        );
        assert_close(learning_rate_at(adam, 100, 200, 1.0, false), 0.2);
        assert_close(
            learning_rate_at(adam, 150, 200, 1.0, false),
            0.15 * 0.5 + 0.05,
        );
        assert_close(
            learning_rate_at(adam, 199, 200, 1.0, false),
            0.15 * 0.01 + 0.05,
        );
        // Good init, 500 epochs, α0 = 0.5: warm-up int(min(100, 62.5)) = 62.
        let w = 62.0;
        assert_close(learning_rate_at(adam, 0, 500, 0.5, true), 0.25);
        assert_close(
            learning_rate_at(adam, 31, 500, 0.5, true),
            0.15 * (1.0f64 - 31.0 / w).powi(2) + 0.1,
        );
        assert_close(learning_rate_at(adam, 62, 500, 0.5, true), 0.2);
        assert_close(
            learning_rate_at(adam, 300, 500, 0.5, true),
            0.15 * (1.0 - (300.0 - w) / (500.0 - w)) + 0.05,
        );
        // Epochs past the horizon hold the last value.
        assert_eq!(
            learning_rate_at(adam, 900, 200, 1.0, false),
            learning_rate_at(adam, 199, 200, 1.0, false)
        );
        // Small horizon: warm-up int(min(1.5, 100)) = 1.
        assert_close(learning_rate_at(adam, 0, 3, 1.0, false), 2.0);
        assert_close(learning_rate_at(adam, 1, 3, 1.0, false), 0.2);
        assert_eq!(momentum_at(adam, 5, 200, false), 0.0);
    }

    #[test]
    fn momentum_schedules_match_reference_formulas() {
        let momentum = ModernOptimizerKind::Momentum;
        // Not good init, 200 epochs, α0 = 1.
        assert_close(learning_rate_at(momentum, 0, 200, 1.0, false), 0.25);
        assert_close(learning_rate_at(momentum, 100, 200, 1.0, false), 0.0625);
        assert_close(momentum_at(momentum, 0, 200, false), 0.375);
        assert_close(momentum_at(momentum, 100, 200, false), 0.5 * (1.0 - 0.0625));
        // Good init, 300 epochs: warm-up min(300, int(max(200, 37.5))) = 200.
        assert_close(learning_rate_at(momentum, 0, 300, 1.0, true), 0.255);
        assert_close(
            learning_rate_at(momentum, 100, 300, 1.0, true),
            0.0625 + 0.005,
        );
        assert_close(learning_rate_at(momentum, 250, 300, 1.0, true), 0.005);
        assert_close(momentum_at(momentum, 0, 300, true), 0.0);
        assert_close(momentum_at(momentum, 100, 300, true), 0.375);
        assert_close(momentum_at(momentum, 250, 300, true), 0.5);
        // Good init, 50 epochs: warm-up min(50, 200) = 50.
        assert_close(learning_rate_at(momentum, 25, 50, 2.0, true), 0.125 + 0.005);
    }

    #[test]
    fn rate_zero_disables_repulsion_without_nan() {
        let data = gaussian_blobs(60, 4, 4.0, 37);
        let graph = pruned_graph(&data, 8, 30);
        let mut embedding = random_init(data.nrows(), 2, 41);
        let params = ModernLayoutParams {
            n_epochs: 30,
            negative_sample_rate: 0.0,
            ..ModernLayoutParams::default()
        };
        optimize(&mut embedding, &graph, &params);
        assert!(embedding.iter().all(|x| x.is_finite()));
    }
}
