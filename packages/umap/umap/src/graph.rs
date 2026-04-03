use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;

const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;

/// Sparse matrix in COO-like format backed by a HashMap for efficient lookups.
pub struct SparseMatrix {
    pub entries: HashMap<(usize, usize), f32>,
    pub shape: (usize, usize),
}

impl SparseMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        SparseMatrix {
            entries: HashMap::new(),
            shape: (nrows, ncols),
        }
    }

    pub fn insert(&mut self, row: usize, col: usize, val: f32) {
        if val != 0.0 {
            self.entries.insert((row, col), val);
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        *self.entries.get(&(row, col)).unwrap_or(&0.0)
    }

    /// Apply fuzzy set union: result = mix*(A + A^T - A*A^T) + (1-mix)*(A*A^T)
    /// where A*A^T is element-wise product (Hadamard), not matrix multiplication.
    pub fn symmetrize(&self, set_op_mix_ratio: f32) -> SparseMatrix {
        let mut result = SparseMatrix::new(self.shape.0, self.shape.1);
        let mix = set_op_mix_ratio as f64;

        // Collect all unique (i,j) pairs considering both A[i,j] and A[j,i]
        let mut pairs: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for &(r, c) in self.entries.keys() {
            pairs.insert((r, c));
            pairs.insert((c, r));
        }

        for (i, j) in pairs {
            let a_ij = self.get(i, j) as f64;
            let a_ji = self.get(j, i) as f64;
            let prod = a_ij * a_ji;
            let val = mix * (a_ij + a_ji - prod) + (1.0 - mix) * prod;
            if val > 0.0 {
                result.insert(i, j, val as f32);
            }
        }

        result
    }

    /// Remove entries below threshold.
    pub fn prune(&mut self, threshold: f32) {
        self.entries.retain(|_, v| *v >= threshold);
    }

    /// Convert to CSR format for efficient row access.
    pub fn to_csr(&self) -> CsrMatrix {
        let n = self.shape.0;
        let mut row_entries: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (&(r, c), &v) in &self.entries {
            row_entries[r].push((c, v));
        }
        // Sort each row by column index
        for row in &mut row_entries {
            row.sort_by_key(|&(c, _)| c);
        }

        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for row in &row_entries {
            for &(c, v) in row {
                indices.push(c);
                data.push(v);
            }
            indptr.push(indices.len());
        }

        CsrMatrix {
            indptr,
            indices,
            data,
            nrows: self.shape.0,
            ncols: self.shape.1,
        }
    }

    /// Extract edges as (head, tail, weight) arrays for the optimizer.
    pub fn to_edge_list(&self) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let mut edges: Vec<_> = self.entries.iter().map(|(&(r, c), &v)| (r, c, v)).collect();
        edges.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut heads = Vec::with_capacity(edges.len());
        let mut tails = Vec::with_capacity(edges.len());
        let mut weights = Vec::with_capacity(edges.len());
        for (r, c, v) in edges {
            heads.push(r);
            tails.push(c);
            weights.push(v);
        }
        (heads, tails, weights)
    }
}

/// CSR sparse matrix.
pub struct CsrMatrix {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f32>,
    pub nrows: usize,
    pub ncols: usize,
}

impl CsrMatrix {
    /// Multiply this matrix by a dense vector.
    pub fn mul_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.nrows];
        for i in 0..self.nrows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            let mut sum = 0.0f32;
            for idx in start..end {
                sum += self.data[idx] * x[self.indices[idx]];
            }
            result[i] = sum;
        }
        result
    }

    /// Compute row sums (degree vector).
    pub fn row_sums(&self) -> Vec<f32> {
        let mut sums = vec![0.0f32; self.nrows];
        for i in 0..self.nrows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            for idx in start..end {
                sums[i] += self.data[idx];
            }
        }
        sums
    }

    /// Multiply this matrix by a dense f64 vector, accumulating in f64.
    /// The matrix data (f32) is promoted to f64 for the computation.
    pub fn mul_vec_f64(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0f64; self.nrows];
        for i in 0..self.nrows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            let mut sum = 0.0f64;
            for idx in start..end {
                sum += self.data[idx] as f64 * x[self.indices[idx]];
            }
            result[i] = sum;
        }
        result
    }
}

/// Compute sigma (bandwidth) and rho (local connectivity distance) for each point
/// via binary search, matching the UMAP smooth_knn_dist algorithm.
pub fn smooth_knn_dist(
    distances: &Array2<f32>,
    k: f32,
    n_iter: usize,
    local_connectivity: f32,
    bandwidth: f32,
) -> (Vec<f32>, Vec<f32>) {
    let target = (k.ln() / 2.0f32.ln()) * bandwidth; // log2(k) * bandwidth
    let n_samples = distances.nrows();
    let n_neighbors = distances.ncols();

    let mean_distances: f32 = distances.mean().unwrap_or(0.0);

    let results: Vec<(f32, f32)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            // Collect non-zero distances (sorted, since kNN distances are sorted)
            let non_zero_dists: Vec<f32> = (0..n_neighbors)
                .filter_map(|j| {
                    let d = distances[[i, j]];
                    if d > 0.0 {
                        Some(d)
                    } else {
                        None
                    }
                })
                .collect();

            // Compute rho: distance to the local_connectivity-th nearest neighbor
            let lc = local_connectivity as usize;
            let mut rho = 0.0f32;
            if non_zero_dists.len() >= lc {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - index as f32;
                if index > 0 {
                    rho = non_zero_dists[index - 1];
                    if interpolation > SMOOTH_K_TOLERANCE as f32 {
                        rho += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1]);
                    }
                } else {
                    rho = interpolation * non_zero_dists[0];
                }
            } else if !non_zero_dists.is_empty() {
                rho = *non_zero_dists.last().unwrap();
            }

            // Binary search for sigma
            let mut lo = 0.0f64;
            let mut hi = f64::INFINITY;
            let mut mid = 1.0f64;
            let rho_i = rho as f64;

            for _ in 0..n_iter {
                let mut psum = 0.0f64;
                // Start at j=1 to skip the self-neighbor at position 0,
                // consistent with the reference UMAP implementation.
                for j in 1..n_neighbors {
                    let d = distances[[i, j]] as f64 - rho_i;
                    if d > 0.0 {
                        psum += (-d / mid).exp();
                    } else {
                        psum += 1.0;
                    }
                }

                if (psum - target as f64).abs() < SMOOTH_K_TOLERANCE {
                    break;
                }

                if psum > target as f64 {
                    hi = mid;
                    mid = (lo + hi) / 2.0;
                } else {
                    lo = mid;
                    if hi == f64::INFINITY {
                        mid *= 2.0;
                    } else {
                        mid = (lo + hi) / 2.0;
                    }
                }
            }

            let mut sigma = mid as f32;

            // Apply minimum distance scale
            if rho > 0.0 {
                let mean_ith: f32 = distances.row(i).iter().sum::<f32>() / n_neighbors as f32;
                if sigma < MIN_K_DIST_SCALE * mean_ith {
                    sigma = MIN_K_DIST_SCALE * mean_ith;
                }
            } else if sigma < MIN_K_DIST_SCALE * mean_distances {
                sigma = MIN_K_DIST_SCALE * mean_distances;
            }

            (sigma, rho)
        })
        .collect();

    let sigmas: Vec<f32> = results.iter().map(|r| r.0).collect();
    let rhos: Vec<f32> = results.iter().map(|r| r.1).collect();

    (sigmas, rhos)
}

/// Convert kNN distances to fuzzy set membership strengths.
pub fn compute_membership_strengths(
    knn_indices: &Array2<i32>,
    knn_dists: &Array2<f32>,
    sigmas: &[f32],
    rhos: &[f32],
) -> SparseMatrix {
    let n_samples = knn_indices.nrows();
    let n_neighbors = knn_indices.ncols();

    let triples: Vec<Vec<(usize, usize, f32)>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut local = Vec::new();
            for j in 0..n_neighbors {
                let idx = knn_indices[[i, j]];
                if idx < 0 {
                    continue;
                }
                let idx = idx as usize;

                // Self-loops get weight 0
                if idx == i {
                    continue;
                }

                let val = if knn_dists[[i, j]] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                    1.0
                } else {
                    (-(knn_dists[[i, j]] - rhos[i]) / sigmas[i]).exp()
                };

                if val > 0.0 {
                    local.push((i, idx, val));
                }
            }
            local
        })
        .collect();

    let mut graph = SparseMatrix::new(n_samples, n_samples);
    for batch in triples {
        for (r, c, v) in batch {
            graph.insert(r, c, v);
        }
    }

    graph
}

/// Build the fuzzy simplicial set from kNN data.
///
/// This computes bandwidths, membership strengths, and applies the fuzzy
/// set union operation to produce a symmetric weighted graph.
pub fn fuzzy_simplicial_set(
    knn_indices: &Array2<i32>,
    knn_dists: &Array2<f32>,
    n_neighbors: usize,
    set_op_mix_ratio: f32,
    local_connectivity: f32,
) -> SparseMatrix {
    let (sigmas, rhos) =
        smooth_knn_dist(knn_dists, n_neighbors as f32, 64, local_connectivity, 1.0);

    let graph = compute_membership_strengths(knn_indices, knn_dists, &sigmas, &rhos);

    graph.symmetrize(set_op_mix_ratio)
}

/// Compute epochs_per_sample for each edge based on weight.
/// Edges with higher weight are sampled more frequently.
pub fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_weight <= 0.0 {
        return vec![-1.0; weights.len()];
    }

    weights
        .iter()
        .map(|&w| {
            let n_samples = n_epochs as f32 * (w / max_weight);
            if n_samples > 0.0 {
                n_epochs as f32 / n_samples
            } else {
                -1.0
            }
        })
        .collect()
}
