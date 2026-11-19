use ndarray::{s, Array2, Axis};
use nndescent::rng::Xoshiro256StarStar;

use crate::graph::CsrMatrix;
use crate::recursive_init::{centered_columns, randomized_pca_scores, standard_normal};

/// Linear-interpolated quantile of an unsorted slice (matches `numpy.quantile`).
pub(crate) fn quantile(values: &[f32], q: f64) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = (position - lower as f64) as f32;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

/// How an initial embedding is centered and scaled before optimization
/// (`init_rescale` in umap-learn).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitRescale {
    /// Scale each axis so its 5%-95% spread equals the target spread.
    PerAxis,
    /// One uniform factor so the geometric mean of the per-axis spreads equals
    /// the target spread, which is `log10(N) * 3 * log2(gamma + 1) * factor`.
    #[default]
    Isotropic,
    /// Scale so the weighted median length of the graph edges equals `factor`.
    GraphEdge,
}

impl std::str::FromStr for InitRescale {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "isotropic" => Ok(Self::Isotropic),
            "perAxis" => Ok(Self::PerAxis),
            "graphEdge" => Ok(Self::GraphEdge),
            other => Err(format!("unknown initRescale \"{other}\"")),
        }
    }
}

/// Center `embedding` and scale it by `method`. `graph` is only read by
/// [`InitRescale::GraphEdge`].
pub(crate) fn rescale_embedding(
    embedding: &mut Array2<f32>,
    method: InitRescale,
    gamma: f32,
    factor: f32,
    graph: &CsrMatrix,
) {
    if embedding.nrows() == 0 || embedding.ncols() == 0 {
        return;
    }
    if let Some(mean) = embedding.mean_axis(Axis(0)) {
        *embedding -= &mean;
    }
    match method {
        InitRescale::GraphEdge => {
            if let Some(length) = weighted_median_edge_length(embedding, graph) {
                let multiplier = factor / if length == 0.0 { 1.0 } else { length };
                embedding.mapv_inplace(|x| x * multiplier);
            }
        }
        InitRescale::PerAxis | InitRescale::Isotropic => {
            let target = target_spread(embedding.nrows(), gamma, factor);
            let spreads = robust_spreads(embedding);
            let multipliers = match method {
                InitRescale::PerAxis => spreads.iter().map(|&s| target / s).collect(),
                _ => {
                    let log_mean =
                        spreads.iter().map(|s| s.ln()).sum::<f64>() / spreads.len() as f64;
                    vec![target / log_mean.exp(); spreads.len()]
                }
            };
            for (mut column, multiplier) in embedding.columns_mut().into_iter().zip(multipliers) {
                column.mapv_inplace(|x| x * multiplier as f32);
            }
        }
    }
}

fn target_spread(n_samples: usize, gamma: f32, factor: f32) -> f64 {
    (n_samples as f64).log10() * 3.0 * (gamma as f64 + 1.0).log2() * factor as f64
}

/// Per-axis 5%-95% spread; axes without spread count as 1.
fn robust_spreads(embedding: &Array2<f32>) -> Vec<f64> {
    embedding
        .columns()
        .into_iter()
        .map(|column| {
            let values = column.to_vec();
            let spread = quantile(&values, 0.95) - quantile(&values, 0.05);
            if spread == 0.0 {
                1.0
            } else {
                spread as f64
            }
        })
        .collect()
}

/// Length of the edge at which the cumulative edge weight, ordered by length,
/// first reaches half the total weight. `None` without positive total weight.
fn weighted_median_edge_length(embedding: &Array2<f32>, graph: &CsrMatrix) -> Option<f32> {
    let total_weight: f64 = graph.data.iter().map(|&w| w as f64).sum();
    if graph.data.is_empty() || total_weight <= 0.0 {
        return None;
    }
    let mut edges: Vec<(f32, f32)> = (0..graph.nrows)
        .flat_map(|row| (graph.indptr[row]..graph.indptr[row + 1]).map(move |edge| (row, edge)))
        .map(|(row, edge)| {
            let squared: f32 = embedding
                .row(row)
                .iter()
                .zip(embedding.row(graph.indices[edge]).iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (squared.sqrt(), graph.data[edge])
        })
        .collect();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cumulative = 0.0f64;
    for &(length, weight) in &edges {
        cumulative += weight as f64;
        if cumulative >= total_weight / 2.0 {
            return Some(length);
        }
    }
    edges.last().map(|&(length, _)| length)
}

const JITTER_NEIGHBOR_QUERIES: usize = 2048;
const JITTER_SCALE: f64 = 0.001;

/// When rows of `embedding` coincide, add Gaussian noise of scale
/// `0.001 * mean nearest-neighbour distance`, so that coincident points can separate.
/// The mean is estimated from an evenly spaced subset of the rows.
pub(crate) fn jitter_duplicate_rows(
    mut embedding: Array2<f32>,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let n = embedding.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    let row_key = |i: usize| {
        embedding
            .row(i)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    order.sort_by_cached_key(|&i| row_key(i));
    let has_duplicates = order.windows(2).any(|w| row_key(w[0]) == row_key(w[1]));
    if !has_duplicates {
        return embedding;
    }

    let stride = n.div_ceil(JITTER_NEIGHBOR_QUERIES);
    let nearest: Vec<f64> = (0..n)
        .step_by(stride)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let squared: f32 = embedding
                        .row(i)
                        .iter()
                        .zip(embedding.row(j).iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    squared as f64
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .collect();
    let mean_distance = nearest.iter().sum::<f64>() / nearest.len() as f64;
    let scale = JITTER_SCALE * mean_distance;
    embedding.mapv_inplace(|x| x + (scale * standard_normal(rng)) as f32);
    embedding
}

/// Scores of the first `n_components` principal components of `data`, padded
/// with zero columns when the data has fewer axes.
pub(crate) fn pca_layout(
    data: &Array2<f32>,
    n_components: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let scores = randomized_pca_scores(&centered_columns(data.clone()), n_components, rng);
    let mut layout = Array2::zeros((data.nrows(), n_components));
    layout.slice_mut(s![.., ..scores.ncols()]).assign(&scores);
    layout
}

/// Rescale every axis independently to `[0, max_coord]`. A constant axis maps to 0.
pub(crate) fn rescale_embedding_min_max(embedding: &mut Array2<f32>, max_coord: f32) {
    for mut column in embedding.columns_mut() {
        let min = column.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let span = max - min;
        let scale = if span > 0.0 { max_coord / span } else { 0.0 };
        column.mapv_inplace(|x| (x - min) * scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn graph_without_edges(n: usize) -> CsrMatrix {
        CsrMatrix {
            indptr: vec![0; n + 1],
            indices: Vec::new(),
            data: Vec::new(),
            nrows: n,
            ncols: n,
        }
    }

    fn spreads(embedding: &Array2<f32>) -> Vec<f32> {
        embedding
            .columns()
            .into_iter()
            .map(|c| {
                let v = c.to_vec();
                quantile(&v, 0.95) - quantile(&v, 0.05)
            })
            .collect()
    }

    fn skewed_embedding() -> Array2<f32> {
        Array2::from_shape_fn((1000, 2), |(i, d)| {
            let t = i as f32 / 999.0;
            if d == 0 {
                100.0 + 50.0 * t
            } else {
                -3.0 + 0.5 * t
            }
        })
    }

    #[test]
    fn isotropic_rescale_centers_and_hits_target_spread() {
        let mut embedding = skewed_embedding();
        rescale_embedding(
            &mut embedding,
            InitRescale::Isotropic,
            1.0,
            1.0,
            &graph_without_edges(1000),
        );

        for column in embedding.columns() {
            assert!(column.mean().unwrap().abs() < 1e-3);
        }
        let spreads = spreads(&embedding);
        let target = 1000f32.log10() * 3.0;
        let geomean = (spreads[0] * spreads[1]).sqrt();
        assert!((geomean - target).abs() < 1e-2, "{geomean} vs {target}");
        assert!((spreads[0] / spreads[1] - 100.0).abs() < 0.5);
    }

    #[test]
    fn per_axis_rescale_gives_every_axis_the_target_spread() {
        let mut embedding = skewed_embedding();
        rescale_embedding(
            &mut embedding,
            InitRescale::PerAxis,
            1.0,
            2.0,
            &graph_without_edges(1000),
        );

        let target = 1000f32.log10() * 3.0 * 2.0;
        for spread in spreads(&embedding) {
            assert!((spread - target).abs() < 1e-2, "{spread} vs {target}");
        }
    }

    #[test]
    fn graph_edge_rescale_sets_weighted_median_edge_length_to_factor() {
        let mut embedding = arr2(&[[0.0f32, 0.0], [10.0, 0.0], [10.0, 40.0]]);
        let graph = CsrMatrix {
            indptr: vec![0, 1, 3, 3],
            indices: vec![1, 0, 2],
            data: vec![1.0, 1.0, 0.1],
            nrows: 3,
            ncols: 3,
        };
        rescale_embedding(&mut embedding, InitRescale::GraphEdge, 1.0, 3.0, &graph);

        let length = |a: usize, b: usize| {
            (&embedding.row(a) - &embedding.row(b))
                .mapv(|x| x * x)
                .sum()
                .sqrt()
        };
        assert!((length(0, 1) - 3.0).abs() < 1e-4);
        assert!((length(1, 2) - 12.0).abs() < 1e-3);
    }

    #[test]
    fn graph_edge_rescale_without_edges_only_centers() {
        let mut embedding = arr2(&[[1.0f32, 2.0], [3.0, 6.0]]);
        rescale_embedding(
            &mut embedding,
            InitRescale::GraphEdge,
            1.0,
            1.0,
            &graph_without_edges(2),
        );
        assert_eq!(embedding, arr2(&[[-1.0f32, -2.0], [1.0, 2.0]]));
    }

    #[test]
    fn rescale_handles_constant_axis() {
        for method in [
            InitRescale::PerAxis,
            InitRescale::Isotropic,
            InitRescale::GraphEdge,
        ] {
            let mut embedding = arr2(&[[1.0f32, 5.0], [2.0, 5.0], [3.0, 5.0]]);
            rescale_embedding(&mut embedding, method, 1.0, 1.0, &graph_without_edges(3));
            assert!(embedding.iter().all(|x| x.is_finite()));
            assert!(embedding.column(1).iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn jitter_separates_coincident_rows_only_when_present() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(3);
        let distinct = arr2(&[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(jitter_duplicate_rows(distinct.clone(), &mut rng), distinct);

        let coincident = arr2(&[[0.0f32, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let jittered = jitter_duplicate_rows(coincident.clone(), &mut rng);
        assert_ne!(jittered.row(0), jittered.row(1));
        assert!((&jittered - &coincident).iter().all(|d| d.abs() < 0.05));
    }

    #[test]
    fn pca_layout_orders_axes_by_variance_and_pads_missing_axes() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(7);
        let data = Array2::from_shape_fn((200, 3), |(i, d)| {
            let t = (i as f32 * 0.37).sin() + (i as f32 * 0.11).cos();
            t * [10.0, 3.0, 0.1][d]
        });
        let layout = pca_layout(&data, 2, &mut rng);
        assert_eq!(layout.dim(), (200, 2));
        let variance = |c: usize| layout.column(c).mapv(|x| x * x).sum();
        assert!(variance(0) > variance(1));

        let narrow = Array2::from_shape_fn((50, 1), |(i, _)| i as f32);
        let padded = pca_layout(&narrow, 2, &mut rng);
        assert_eq!(padded.dim(), (50, 2));
        assert!(padded.column(1).iter().all(|&x| x == 0.0));
        assert!(padded.column(0).iter().any(|&x| x != 0.0));
    }

    #[test]
    fn min_max_rescale_maps_each_axis_to_range() {
        let mut embedding = Array2::from_shape_vec(
            (3, 3),
            vec![-3.0f32, 7.0, 1.0, 1.0, 9.0, 1.0, 5.0, 8.0, 1.0],
        )
        .unwrap();
        rescale_embedding_min_max(&mut embedding, 10.0);
        assert_eq!(embedding.column(0).to_vec(), vec![0.0, 5.0, 10.0]);
        assert_eq!(embedding.column(1).to_vec(), vec![0.0, 10.0, 5.0]);
        assert_eq!(embedding.column(2).to_vec(), vec![0.0, 0.0, 0.0]);
    }
}
