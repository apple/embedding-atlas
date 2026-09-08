use ndarray::Array2;
use rayon::prelude::*;

const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;

/// Sparse weighted graph in CSR format (row-major; columns sorted ascending
/// within each row). Built once from COO triples — the UMAP graph pipeline never
/// needs incremental insertion — which lets `symmetrize`/`to_csr`/`to_edge_list`
/// run on flat arrays instead of hashing millions of `(row, col)` keys.
pub struct SparseMatrix {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f32>,
    pub shape: (usize, usize),
}

impl SparseMatrix {
    /// Build from COO triples. Columns are sorted ascending within each row, and
    /// duplicate `(row, col)` pairs are collapsed (max weight) so each row has
    /// unique, sorted columns — the pipeline never emits duplicates, but this
    /// keeps `get`'s binary search and the CSR contract valid for any input.
    pub fn from_coo(nrows: usize, ncols: usize, mut triples: Vec<(usize, usize, f32)>) -> Self {
        triples.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let nnz = triples.len();
        let mut indptr = vec![0usize; nrows + 1];
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        let mut t = 0;
        for r in 0..nrows {
            while t < nnz && triples[t].0 == r {
                let c = triples[t].1;
                let mut v = triples[t].2;
                t += 1;
                while t < nnz && triples[t].0 == r && triples[t].1 == c {
                    v = v.max(triples[t].2);
                    t += 1;
                }
                indices.push(c);
                data.push(v);
            }
            indptr[r + 1] = indices.len();
        }
        SparseMatrix {
            indptr,
            indices,
            data,
            shape: (nrows, ncols),
        }
    }

    /// Look up A[row, col] (0.0 if absent). Columns are sorted, so binary search.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        let r = &self.indices[self.indptr[row]..self.indptr[row + 1]];
        match r.binary_search(&col) {
            Ok(k) => self.data[self.indptr[row] + k],
            Err(_) => 0.0,
        }
    }

    /// Apply fuzzy set union: result = mix*(A + A^T - A∘A^T) + (1-mix)*(A∘A^T),
    /// where A∘A^T is the element-wise (Hadamard) product. The result is
    /// symmetric, so for each stored edge (i,j) the value at (j,i) is identical;
    /// we emit the transpose entry only when A has no (j,i) of its own.
    pub fn symmetrize(&self, set_op_mix_ratio: f32) -> SparseMatrix {
        let mix = set_op_mix_ratio as f64;
        let n = self.shape.0;
        let mut triples: Vec<(usize, usize, f32)> = Vec::with_capacity(self.data.len() * 2);
        for i in 0..n {
            for p in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[p];
                let a_ij = self.data[p] as f64;
                let a_ji = self.get(j, i) as f64;
                let prod = a_ij * a_ji;
                let val = mix * (a_ij + a_ji - prod) + (1.0 - mix) * prod;
                if val > 0.0 {
                    triples.push((i, j, val as f32));
                    // The reverse edge isn't in A, so it won't be visited on its
                    // own pass — emit it here to keep the result symmetric.
                    if a_ji == 0.0 {
                        triples.push((j, i, val as f32));
                    }
                }
            }
        }
        SparseMatrix::from_coo(n, self.shape.1, triples)
    }

    /// Remove entries below threshold.
    pub fn prune(&mut self, threshold: f32) {
        let n = self.shape.0;
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(self.indices.len());
        let mut data = Vec::with_capacity(self.data.len());
        indptr.push(0);
        for i in 0..n {
            for p in self.indptr[i]..self.indptr[i + 1] {
                if self.data[p] >= threshold {
                    indices.push(self.indices[p]);
                    data.push(self.data[p]);
                }
            }
            indptr.push(indices.len());
        }
        self.indptr = indptr;
        self.indices = indices;
        self.data = data;
    }

    /// Largest edge weight (NEG_INFINITY if empty).
    pub fn max_weight(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Number of stored edges.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Convert to CSR format for efficient row access. The graph is already CSR,
    /// so this is a copy into the standalone [`CsrMatrix`] type.
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.clone(),
            nrows: self.shape.0,
            ncols: self.shape.1,
        }
    }

    /// Extract edges as (head, tail, weight) arrays for the optimizer, in
    /// row-major / column-sorted order.
    pub fn to_edge_list(&self) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let nnz = self.data.len();
        let mut heads = Vec::with_capacity(nnz);
        let mut tails = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);
        for i in 0..self.shape.0 {
            for p in self.indptr[i]..self.indptr[i + 1] {
                heads.push(i);
                tails.push(self.indices[p]);
                weights.push(self.data[p]);
            }
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

    SparseMatrix::from_coo(
        n_samples,
        n_samples,
        triples.into_iter().flatten().collect(),
    )
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

/// Canonicalize a precomputed kNN graph into the form [`fuzzy_simplicial_set`]
/// expects: column 0 is the point itself (distance 0) and the remaining columns
/// are its neighbors, sorted ascending by distance.
///
/// # Caller's contract
///
/// This function canonicalizes the *layout* of a well-formed graph; it does **not**
/// repair bad data and does not validate the input. The caller is responsible that,
/// for every row `i`:
/// - all neighbor indices are valid — in `[0, n_samples)` (no negative/`-1`
///   "missing" padding);
/// - `i` appears among its own neighbors **at most once** (self-inclusion is
///   optional — this library and umap-learn put self in column 0, while
///   faiss / scikit-learn / hnswlib exclude it — but it must not appear twice);
/// - distances are finite (no `NaN`/`inf`).
///
/// Violating the contract degrades silently rather than erroring: a `NaN` distance
/// sorts arbitrarily and can survive into the graph (yielding a `NaN` embedding),
/// and because the output width is the *minimum* real-neighbor count across all
/// rows (see below), a single short/ragged row trims every row down to its length.
/// Keep the real-neighbor count uniform across rows.
///
/// # Canonicalization
///
/// Self-inclusion is detected and the graph is rewritten without ever fabricating a
/// neighbor:
/// - A row's *real* neighbors are the entries with `idx != i`, sorted ascending by
///   `(distance, index)`. (Entries with `idx < 0` are skipped defensively, but per
///   the contract there should be none.)
/// - `m` is the minimum real-neighbor count across all rows.
/// - The output is `(n, m + 1)`: column 0 is `(i, 0.0)`; columns `1..=m` are the
///   `m` nearest real neighbors. Rows with more than `m` real neighbors drop their
///   farthest ones, so every row keeps the same width with no padding.
///
/// For a contract-conforming graph this subsumes the three cases the boundary can
/// see:
/// - all rows exclude self → `m = k`, output width `k + 1` (self prepended);
/// - all rows include self → `m = k - 1`, output width `k` (reordered, self leads);
/// - mixed → `m = k - 1`, output width `k` (self-excluded rows drop one neighbor).
///
/// Already-canonical input (self in column 0, sorted) round-trips unchanged.
pub fn normalize_knn_self_column(
    indices: &Array2<i32>,
    distances: &Array2<f32>,
) -> (Array2<i32>, Array2<f32>) {
    let n_samples = indices.nrows();
    let n_neighbors = indices.ncols();

    // Per row: real neighbors (idx >= 0, idx != self), sorted by ascending distance.
    let per_row: Vec<Vec<(i32, f32)>> = (0..n_samples)
        .map(|i| {
            let mut neighbors: Vec<(i32, f32)> = (0..n_neighbors)
                .filter_map(|j| {
                    let idx = indices[[i, j]];
                    if idx >= 0 && idx as usize != i {
                        Some((idx, distances[[i, j]]))
                    } else {
                        None
                    }
                })
                .collect();
            // Sort by (distance, index): the index tiebreak gives a deterministic
            // order for equal distances. Distances are assumed finite per the
            // contract; a NaN compares as Equal and would sort arbitrarily.
            neighbors.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            neighbors
        })
        .collect();

    // m = the largest neighbor count keepable for every row without padding.
    let m = per_row.iter().map(|r| r.len()).min().unwrap_or(0);

    let out_cols = m + 1;
    let mut out_indices = Array2::<i32>::zeros((n_samples, out_cols));
    let mut out_distances = Array2::<f32>::zeros((n_samples, out_cols));
    for (i, row) in per_row.iter().enumerate() {
        out_indices[[i, 0]] = i as i32;
        out_distances[[i, 0]] = 0.0;
        for (col, &(idx, dist)) in row.iter().take(m).enumerate() {
            out_indices[[i, col + 1]] = idx;
            out_distances[[i, col + 1]] = dist;
        }
    }

    (out_indices, out_distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn normalize_passthrough_when_already_canonical() {
        // Self in column 0 (distance 0), neighbors already sorted ascending.
        let idx = arr2(&[[0, 1, 2], [1, 0, 2], [2, 1, 0]]);
        let dist = arr2(&[[0.0, 1.0, 2.0], [0.0, 1.0, 3.0], [0.0, 1.5, 2.5]]);
        let (oi, od) = normalize_knn_self_column(&idx, &dist);
        assert_eq!(oi, idx);
        assert_eq!(od, dist);
    }

    #[test]
    fn normalize_adds_self_when_excluded() {
        // No row lists itself: width grows from k=2 to k+1=3, self prepended.
        let idx = arr2(&[[1, 2], [0, 2], [0, 1]]);
        let dist = arr2(&[[1.0, 2.0], [1.0, 3.0], [1.5, 2.5]]);
        let (oi, od) = normalize_knn_self_column(&idx, &dist);
        assert_eq!(oi.dim(), (3, 3));
        assert_eq!(oi, arr2(&[[0, 1, 2], [1, 0, 2], [2, 0, 1]]));
        assert_eq!(
            od,
            arr2(&[[0.0, 1.0, 2.0], [0.0, 1.0, 3.0], [0.0, 1.5, 2.5]])
        );
    }

    #[test]
    fn normalize_moves_self_to_front_and_sorts() {
        // Self in the middle and neighbors out of order -> self leads, sorted.
        let idx = arr2(&[[2, 0, 1], [2, 1, 0]]);
        let dist = arr2(&[[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]);
        let (oi, od) = normalize_knn_self_column(&idx, &dist);
        assert_eq!(oi, arr2(&[[0, 1, 2], [1, 0, 2]]));
        assert_eq!(od, arr2(&[[0.0, 1.0, 2.0], [0.0, 1.0, 3.0]]));
    }

    #[test]
    fn normalize_mixed_trims_to_min_no_padding() {
        // Row 0 includes self (2 real neighbors); row 1 excludes self (3 real).
        // m = 2 -> output width 3 (= k); row 1 drops its farthest neighbor (idx 3, d=9).
        let idx = arr2(&[[0, 1, 2], [0, 2, 3]]);
        let dist = arr2(&[[0.0, 1.0, 2.0], [1.0, 4.0, 9.0]]);
        let (oi, od) = normalize_knn_self_column(&idx, &dist);
        assert_eq!(oi.dim(), (2, 3));
        assert_eq!(oi, arr2(&[[0, 1, 2], [1, 0, 2]]));
        assert_eq!(od, arr2(&[[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]]));
        // No fabricated (-1) entries anywhere.
        assert!(oi.iter().all(|&v| v >= 0));
    }
}
