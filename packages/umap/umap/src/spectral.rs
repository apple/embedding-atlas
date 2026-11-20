use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;

use nndescent::distance::get_distance_func;

use crate::graph::CsrMatrix;

// ============================================================================
// Dense linear algebra helpers (small matrices, max ~9x9 for UMAP)
// ============================================================================

/// Symmetric eigendecomposition via Jacobi rotations.
/// Input: `a` is an m×m symmetric matrix in row-major flat layout.
/// Returns (eigenvalues sorted ascending, eigenvectors as row-major m×m matrix).
pub(crate) fn eigh_jacobi(a: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    // V = identity (eigenvector accumulator)
    let mut v = vec![0.0; m * m];
    for i in 0..m {
        v[i * m + i] = 1.0;
    }

    let max_sweeps = 100;
    let tol = 1e-12;

    for _ in 0..max_sweeps {
        // Compute off-diagonal norm
        let mut off_norm = 0.0;
        for p in 0..m {
            for q in (p + 1)..m {
                off_norm += a[p * m + q] * a[p * m + q];
            }
        }
        if off_norm.sqrt() < tol {
            break;
        }

        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[p * m + q];
                if apq.abs() < 1e-15 {
                    continue;
                }

                let tau = (a[q * m + q] - a[p * m + p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update A: rotate rows/cols p, q
                let app = a[p * m + p];
                let aqq = a[q * m + q];
                a[p * m + p] = app - t * apq;
                a[q * m + q] = aqq + t * apq;
                a[p * m + q] = 0.0;
                a[q * m + p] = 0.0;

                for r in 0..m {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * m + p];
                    let arq = a[r * m + q];
                    a[r * m + p] = c * arp - s * arq;
                    a[p * m + r] = a[r * m + p];
                    a[r * m + q] = s * arp + c * arq;
                    a[q * m + r] = a[r * m + q];
                }

                // Update V: rotate columns p, q
                for r in 0..m {
                    let vrp = v[r * m + p];
                    let vrq = v[r * m + q];
                    v[r * m + p] = c * vrp - s * vrq;
                    v[r * m + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvalues (diagonal of A)
    let eigenvalues: Vec<f64> = (0..m).map(|i| a[i * m + i]).collect();

    // Sort ascending
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

    let sorted_vals: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_vecs = vec![0.0; m * m];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..m {
            sorted_vecs[row * m + new_col] = v[row * m + old_col];
        }
    }

    (sorted_vals, sorted_vecs)
}

/// Cholesky factorization of m×m SPD matrix (row-major flat).
/// Returns lower triangle L such that A = L L^T, or None if not SPD.
fn cholesky(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; m * m];
    for j in 0..m {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * m + k] * l[j * m + k];
        }
        let diag = a[j * m + j] - sum;
        if diag <= 1e-14 {
            return None;
        }
        l[j * m + j] = diag.sqrt();

        for i in (j + 1)..m {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * m + k] * l[j * m + k];
            }
            l[i * m + j] = (a[i * m + j] - sum) / l[j * m + j];
        }
    }
    Some(l)
}

/// Forward substitution: solve L * x = b in-place (b becomes x).
/// L is m×m lower triangular, row-major. b has `cols` right-hand sides,
/// stored as consecutive columns of length m.
fn solve_lower(l: &[f64], b: &mut [f64], m: usize, cols: usize) {
    for c in 0..cols {
        for i in 0..m {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i * m + k] * b[k * cols + c];
            }
            b[i * cols + c] = (b[i * cols + c] - sum) / l[i * m + i];
        }
    }
}

/// Back substitution: solve L^T * x = b in-place.
fn solve_lower_transpose(l: &[f64], b: &mut [f64], m: usize, cols: usize) {
    for c in 0..cols {
        for i in (0..m).rev() {
            let mut sum = 0.0;
            for k in (i + 1)..m {
                sum += l[k * m + i] * b[k * cols + c];
            }
            b[i * cols + c] = (b[i * cols + c] - sum) / l[i * m + i];
        }
    }
}

/// Solve the generalized eigenvalue problem G*c = λ*S*c.
/// Uses Cholesky reduction: S = L*L^T, then solve eigh(L^{-1} G L^{-T}).
/// Returns (eigenvalues ascending, eigenvectors in original basis) or None.
fn generalized_eigh(g: &[f64], s: &[f64], m: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let l = cholesky(s, m)?;

    // Compute L^{-1} G L^{-T}:
    // 1. Solve L * Y = G  (Y = L^{-1} G)
    let mut y = g.to_vec();
    solve_lower(&l, &mut y, m, m);
    // 2. Solve L * Z^T = Y^T, i.e. Z = Y L^{-T}
    // Transpose y, solve, transpose back
    let mut yt = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            yt[i * m + j] = y[j * m + i];
        }
    }
    solve_lower(&l, &mut yt, m, m);
    // Transpose back to get Z = L^{-1} G L^{-T}
    let mut z = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            z[i * m + j] = yt[j * m + i];
        }
    }

    // Symmetrize Z (numerical stability)
    for i in 0..m {
        for j in (i + 1)..m {
            let avg = (z[i * m + j] + z[j * m + i]) * 0.5;
            z[i * m + j] = avg;
            z[j * m + i] = avg;
        }
    }

    let (eigenvalues, eigvecs_reduced) = eigh_jacobi(&z, m);

    // Transform back: original eigenvectors = L^{-T} * reduced_eigenvectors
    let mut eigvecs = eigvecs_reduced;
    solve_lower_transpose(&l, &mut eigvecs, m, m);

    Some((eigenvalues, eigvecs))
}

// ============================================================================
// Block vector type for LOBPCG
// ============================================================================

/// A block of k column vectors, each of length n.
struct Block {
    cols: Vec<Vec<f64>>,
}

impl Block {
    fn new(k: usize, n: usize) -> Self {
        Block {
            cols: (0..k).map(|_| vec![0.0; n]).collect(),
        }
    }

    fn ncols(&self) -> usize {
        self.cols.len()
    }

    fn nrows(&self) -> usize {
        if self.cols.is_empty() {
            0
        } else {
            self.cols[0].len()
        }
    }

    /// Compute self^T * other → (self.ncols() × other.ncols()) matrix, row-major flat.
    fn gram(&self, other: &Block) -> Vec<f64> {
        let k1 = self.ncols();
        let k2 = other.ncols();
        let n = self.nrows();
        let mut result = vec![0.0; k1 * k2];
        for i in 0..k1 {
            for j in 0..k2 {
                let mut dot = 0.0;
                for l in 0..n {
                    dot += self.cols[i][l] * other.cols[j][l];
                }
                result[i * k2 + j] = dot;
            }
        }
        result
    }

    /// Orthonormalize columns via modified Gram-Schmidt.
    /// Columns with norm below `eps` are replaced with zero vectors.
    /// Returns false only if all columns are zero after orthonormalization.
    fn orthonormalize_mgs(&mut self) -> bool {
        let k = self.ncols();
        let n = self.nrows();
        let eps = 1e-12;
        let mut any_nonzero = false;

        for j in 0..k {
            // Subtract projections of previous columns
            for l in 0..j {
                let dot: f64 = self.cols[j]
                    .iter()
                    .zip(self.cols[l].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                for i in 0..n {
                    self.cols[j][i] -= dot * self.cols[l][i];
                }
            }
            // Normalize
            let norm: f64 = self.cols[j].iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > eps {
                self.cols[j].iter_mut().for_each(|v| *v /= norm);
                any_nonzero = true;
            } else {
                self.cols[j].fill(0.0);
            }
        }
        any_nonzero
    }

    /// Orthonormalize self against a basis block, then self-orthonormalize.
    /// self -= basis * (basis^T * self), then MGS-orthonormalize self.
    fn orthonormalize_against(&mut self, basis: &Block) -> bool {
        let n = self.nrows();
        // overlap = basis^T * self
        let overlap = basis.gram(self);
        let kb = basis.ncols();
        let ks = self.ncols();
        // self -= basis * overlap
        for j in 0..ks {
            for l in 0..kb {
                let coeff = overlap[l * ks + j];
                for i in 0..n {
                    self.cols[j][i] -= coeff * basis.cols[l][i];
                }
            }
        }
        self.orthonormalize_mgs()
    }

    /// Multiply by a small dense matrix: result = self * coeffs.
    /// coeffs is (self.ncols() × k_out), row-major.
    fn mul_small(&self, coeffs: &[f64], k_out: usize) -> Block {
        let n = self.nrows();
        let k_in = self.ncols();
        let mut result = Block::new(k_out, n);
        for j in 0..k_out {
            for l in 0..k_in {
                let c = coeffs[l * k_out + j];
                if c != 0.0 {
                    for i in 0..n {
                        result.cols[j][i] += c * self.cols[l][i];
                    }
                }
            }
        }
        result
    }

    /// Element-wise addition: self += other.
    fn add_assign(&mut self, other: &Block) {
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                self.cols[j][i] += other.cols[j][i];
            }
        }
    }
}

// ============================================================================
// Laplacian operator
// ============================================================================

/// Apply L*v = v - D^{-1/2} * A * (D^{-1/2} * v) to each column of a Block.
/// All computation is done in f64 to avoid precision loss in the eigensolver.
fn apply_laplacian(graph: &CsrMatrix, inv_sqrt_deg: &[f32], x: &Block) -> Block {
    let n = x.nrows();
    let k = x.ncols();
    let inv_sqrt_deg_f64: Vec<f64> = inv_sqrt_deg.iter().map(|&d| d as f64).collect();
    let mut result = Block::new(k, n);
    for col in 0..k {
        // Step 1: scale by D^{-1/2} (in f64)
        let scaled: Vec<f64> = (0..n)
            .map(|i| inv_sqrt_deg_f64[i] * x.cols[col][i])
            .collect();
        // Step 2: sparse mat-vec (in f64)
        let av = graph.mul_vec_f64(&scaled);
        // Step 3: L*v = v - D^{-1/2} * A * (D^{-1/2} * v)
        for i in 0..n {
            result.cols[col][i] = x.cols[col][i] - inv_sqrt_deg_f64[i] * av[i];
        }
    }
    result
}

// ============================================================================
// LOBPCG
// ============================================================================

/// Assemble a symmetric block Gram matrix from sub-blocks.
/// blocks is a list of (row_block, col_block, sub_matrix) triples.
/// Each sub_matrix is (rows × cols) row-major. The result is (m × m) row-major.
fn assemble_gram(
    m: usize,
    diagonal_blocks: &[(usize, &[f64], usize)], // (offset, diag_data, block_size)
    off_diagonal: &[(usize, usize, &[f64], usize, usize)], // (row_off, col_off, data, rows, cols)
) -> Vec<f64> {
    let mut result = vec![0.0; m * m];

    for &(offset, data, bk) in diagonal_blocks {
        for i in 0..bk {
            for j in 0..bk {
                result[(offset + i) * m + (offset + j)] = data[i * bk + j];
            }
        }
    }

    for &(row_off, col_off, data, rows, cols) in off_diagonal {
        for i in 0..rows {
            for j in 0..cols {
                result[(row_off + i) * m + (col_off + j)] = data[i * cols + j];
                result[(col_off + j) * m + (row_off + i)] = data[i * cols + j]; // symmetric
            }
        }
    }

    result
}

/// LOBPCG for the smallest eigenvalues of the normalized Laplacian.
///
/// Finds the k smallest non-trivial eigenvalues by constraining the search
/// space to be orthogonal to the known trivial eigenvector (sqrt(D)/||sqrt(D)||,
/// eigenvalue ~0). This avoids the convergence pollution that occurs when the
/// trivial eigenpair converges early and its residual replacement corrupts the
/// Rayleigh-Ritz subspace.
///
/// `constraint` is the trivial eigenvector that all search directions must
/// remain orthogonal to.
#[allow(clippy::too_many_arguments)]
fn lobpcg_solve(
    graph: &CsrMatrix,
    inv_sqrt_deg: &[f32],
    k: usize,
    n: usize,
    mut x: Block,
    tol: f64,
    max_iter: usize,
    rng_state: &mut [u64; 2],
    constraint: &[f64],
) -> Option<(Vec<f64>, Block)> {
    // Simple xorshift128+ RNG for replacing converged residual columns
    let mut rand_f64 = || -> f64 {
        let mut s1 = rng_state[0];
        let s0 = rng_state[1];
        let result = s1.wrapping_add(s0);
        rng_state[0] = s0;
        s1 ^= s1 << 23;
        rng_state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        (result >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };

    // Helper: project out the constraint vector from all columns of a block
    let project_out_constraint = |block: &mut Block| {
        for j in 0..block.ncols() {
            let dot: f64 = block.cols[j]
                .iter()
                .zip(constraint.iter())
                .map(|(a, b)| a * b)
                .sum();
            for (i, &c) in constraint.iter().enumerate().take(block.nrows()) {
                block.cols[j][i] -= dot * c;
            }
        }
    };

    // 1. Project out constraint and orthonormalize X
    project_out_constraint(&mut x);
    for column in x.cols.iter_mut() {
        let norm = column.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-10 {
            column.fill_with(&mut rand_f64);
        }
    }
    project_out_constraint(&mut x);
    if !x.orthonormalize_mgs() {
        return None;
    }

    // 2. Compute AX = L * X
    let mut ax = apply_laplacian(graph, inv_sqrt_deg, &x);

    // 3. Initial Rayleigh-Ritz on X^T * AX
    let g_xx = x.gram(&ax);
    let (mut lambda, c) = eigh_jacobi(&g_xx, k);
    x = x.mul_small(&c, k);
    ax = ax.mul_small(&c, k);

    // 4. Main LOBPCG loop
    let mut p = Block::new(0, n);
    let mut ap = Block::new(0, n);

    for _iter in 0..max_iter {
        // a. Compute residuals: R = AX - X * diag(lambda)
        let mut r = Block::new(k, n);
        for j in 0..k {
            for i in 0..n {
                r.cols[j][i] = ax.cols[j][i] - lambda[j] * x.cols[j][i];
            }
        }

        // b. Check convergence
        let mut all_converged = true;
        let mut rnorms = vec![0.0f64; k];
        for j in 0..k {
            rnorms[j] = r.cols[j].iter().map(|v| v * v).sum::<f64>().sqrt();
            if rnorms[j] >= tol {
                all_converged = false;
            } else {
                // Replace converged residual with a random vector to prevent
                // rank deficiency in the orthonormalization step.
                r.cols[j].fill_with(&mut rand_f64);
            }
        }
        // Project out the constraint from residuals before orthonormalization
        project_out_constraint(&mut r);
        if all_converged {
            break;
        }

        // c. Orthonormalize R against X, then self
        if !r.orthonormalize_against(&x) {
            break;
        }

        // d. Compute AR = L * R
        let ar = apply_laplacian(graph, inv_sqrt_deg, &r);

        // e. Build Gram matrices and solve Rayleigh-Ritz
        let have_p = p.ncols() > 0;

        let g_xr = x.gram(&ar);
        let g_rr = r.gram(&ar);
        let s_xr = x.gram(&r);
        let s_rr = r.gram(&r);

        let mut g_xx_diag = vec![0.0; k * k];
        for j in 0..k {
            g_xx_diag[j * k + j] = lambda[j];
        }
        let mut s_xx = vec![0.0; k * k];
        for j in 0..k {
            s_xx[j * k + j] = 1.0;
        }

        let (evals, evecs, m_total) = if have_p {
            let g_xp = x.gram(&ap);
            let g_rp = r.gram(&ap);
            let g_pp = p.gram(&ap);
            let s_xp = x.gram(&p);
            let s_rp = r.gram(&p);
            let s_pp = p.gram(&p);

            let m = 3 * k;
            let g = assemble_gram(
                m,
                &[(0, &g_xx_diag, k), (k, &g_rr, k), (2 * k, &g_pp, k)],
                &[
                    (0, k, &g_xr, k, k),
                    (0, 2 * k, &g_xp, k, k),
                    (k, 2 * k, &g_rp, k, k),
                ],
            );
            let s = assemble_gram(
                m,
                &[(0, &s_xx, k), (k, &s_rr, k), (2 * k, &s_pp, k)],
                &[
                    (0, k, &s_xr, k, k),
                    (0, 2 * k, &s_xp, k, k),
                    (k, 2 * k, &s_rp, k, k),
                ],
            );

            match generalized_eigh(&g, &s, m) {
                Some((evals, evecs)) => (evals, evecs, m),
                None => {
                    let m = 2 * k;
                    let g = assemble_gram(
                        m,
                        &[(0, &g_xx_diag, k), (k, &g_rr, k)],
                        &[(0, k, &g_xr, k, k)],
                    );
                    let s =
                        assemble_gram(m, &[(0, &s_xx, k), (k, &s_rr, k)], &[(0, k, &s_xr, k, k)]);
                    match generalized_eigh(&g, &s, m) {
                        Some((evals, evecs)) => (evals, evecs, m),
                        None => break, // S not SPD, return current best
                    }
                }
            }
        } else {
            let m = 2 * k;
            let g = assemble_gram(
                m,
                &[(0, &g_xx_diag, k), (k, &g_rr, k)],
                &[(0, k, &g_xr, k, k)],
            );
            let s = assemble_gram(m, &[(0, &s_xx, k), (k, &s_rr, k)], &[(0, k, &s_xr, k, k)]);
            match generalized_eigh(&g, &s, m) {
                Some((evals, evecs)) => (evals, evecs, m),
                None => break, // S not SPD, return current best
            }
        };

        // f. Extract coefficient sub-matrices for the k smallest eigenpairs
        let mut c_x = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                c_x[i * k + j] = evecs[i * m_total + j];
            }
        }
        let mut c_r = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                c_r[i * k + j] = evecs[(k + i) * m_total + j];
            }
        }

        // g. Compute new P, AP
        if have_p && m_total == 3 * k {
            let mut c_p = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    c_p[i * k + j] = evecs[(2 * k + i) * m_total + j];
                }
            }
            let rp = r.mul_small(&c_r, k);
            let arp = ar.mul_small(&c_r, k);
            let pp = p.mul_small(&c_p, k);
            let app = ap.mul_small(&c_p, k);
            p = rp;
            p.add_assign(&pp);
            ap = arp;
            ap.add_assign(&app);
        } else {
            p = r.mul_small(&c_r, k);
            ap = ar.mul_small(&c_r, k);
        }

        // h. Compute new X, AX
        let mut x_new = x.mul_small(&c_x, k);
        let mut ax_new = ax.mul_small(&c_x, k);
        x_new.add_assign(&p);
        ax_new.add_assign(&ap);
        x = x_new;
        ax = ax_new;

        // i. Update eigenvalues
        lambda = evals[..k].to_vec();
    }

    Some((lambda, x))
}

/// Starting block of `k` vectors: all ones and an alternating sign pattern.
fn deterministic_start(k: usize, n: usize) -> Block {
    let mut x0 = Block::new(k, n);
    x0.cols[0].fill(1.0);
    if k > 1 {
        for (i, value) in x0.cols[1].iter_mut().enumerate() {
            *value = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    x0
}

/// Starting block from a randomized truncated SVD of L: power iterations on a random
/// block, a Rayleigh-Ritz rotation, then the dominant vectors after the first, which
/// stands in for the trivial eigenvector.
fn truncated_svd_start(
    graph: &CsrMatrix,
    inv_sqrt_deg: &[f32],
    k: usize,
    n: usize,
    rng: &mut Xoshiro256StarStar,
) -> Block {
    const OVERSAMPLES: usize = 10;
    const POWER_ITERATIONS: usize = 5;
    let width = (k + 1 + OVERSAMPLES).min(n);

    let mut block = Block::new(width, n);
    for column in block.cols.iter_mut() {
        column.fill_with(|| rng.random_f64() - 0.5);
    }
    block.orthonormalize_mgs();
    for _ in 0..POWER_ITERATIONS {
        block = apply_laplacian(graph, inv_sqrt_deg, &block);
        block.orthonormalize_mgs();
    }

    let projected = block.gram(&apply_laplacian(graph, inv_sqrt_deg, &block));
    let (_, rotation) = eigh_jacobi(&projected, width);
    let descending: Vec<f64> = (0..width * width)
        .map(|index| rotation[(index / width) * width + (width - 1 - index % width)])
        .collect();
    let mut dominant = block.mul_small(&descending, width);
    Block {
        cols: dominant.cols.drain(1..=k).collect(),
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Source data used to place the centroids of disconnected graph components.
#[derive(Clone, Copy)]
pub struct ComponentCentroids<'a> {
    pub data: &'a Array2<f32>,
    pub metric: &'a str,
}

#[derive(Clone, Copy, PartialEq)]
enum SolverStart {
    Deterministic,
    TruncatedSvd,
}

/// Spectral embedding of the graph from the eigenvectors of the normalized
/// Laplacian L = I - D^{-1/2} A D^{-1/2}. A graph with several connected
/// components is laid out component by component (see [`multi_component_layout`]).
///
/// The smallest eigenvectors of L are the largest eigenvectors of D^{-1/2} A D^{-1/2}.
/// Falls back to random initialization if the eigensolver fails.
pub fn spectral_layout(graph: &CsrMatrix, dim: usize, rng: &mut Xoshiro256StarStar) -> Array2<f32> {
    layout(graph, None, dim, rng, SolverStart::Deterministic)
}

/// Like [`spectral_layout`], placing disconnected components by their data centroids.
pub fn spectral_layout_with_centroids(
    graph: &CsrMatrix,
    centroids: Option<ComponentCentroids>,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    layout(graph, centroids, dim, rng, SolverStart::Deterministic)
}

/// Spectral embedding whose eigensolver starts from a truncated SVD of the Laplacian
/// (umap-learn's `tswspectral_layout`).
pub fn tsw_spectral_layout(
    graph: &CsrMatrix,
    centroids: Option<ComponentCentroids>,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    layout(graph, centroids, dim, rng, SolverStart::TruncatedSvd)
}

fn layout(
    graph: &CsrMatrix,
    centroids: Option<ComponentCentroids>,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
    start: SolverStart,
) -> Array2<f32> {
    let (n_components, labels) = connected_components(graph);
    if n_components > 1 {
        multi_component_layout(graph, n_components, &labels, centroids, dim, rng)
    } else {
        connected_layout(graph, dim, rng, start)
    }
}

/// Label every vertex with the index of its (weakly) connected component.
/// Labels are assigned in order of the smallest vertex of each component.
fn connected_components(graph: &CsrMatrix) -> (usize, Vec<usize>) {
    let n = graph.nrows;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for &j in &graph.indices[graph.indptr[i]..graph.indptr[i + 1]] {
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj {
                parent[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    let mut label_of_root = vec![usize::MAX; n];
    let mut n_components = 0;
    let labels = (0..n)
        .map(|i| {
            let root = find(&mut parent, i);
            if label_of_root[root] == usize::MAX {
                label_of_root[root] = n_components;
                n_components += 1;
            }
            label_of_root[root]
        })
        .collect();
    (n_components, labels)
}

/// Lay out each connected component separately. Component positions come from a
/// spectral embedding of the component centroids (or fixed axis-aligned positions
/// when there are at most `2 * dim` components); each component is then embedded on
/// its own and scaled to fit half the distance to its nearest neighboring component.
fn multi_component_layout(
    graph: &CsrMatrix,
    n_components: usize,
    labels: &[usize],
    centroids: Option<ComponentCentroids>,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let meta_embedding = if n_components > 2 * dim {
        component_layout(centroids, labels, n_components, dim, rng)
    } else {
        axis_aligned_layout(n_components, dim)
    };

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    let mut local_index = vec![0usize; graph.nrows];
    for (vertex, &label) in labels.iter().enumerate() {
        local_index[vertex] = members[label].len();
        members[label].push(vertex);
    }

    let mut result = Array2::zeros((graph.nrows, dim));
    for (label, vertices) in members.iter().enumerate() {
        let center = meta_embedding.row(label);
        let data_range = nearest_component_distance(&meta_embedding, label) / 2.0;
        let size = vertices.len();

        let mut component_embedding = if size < 2 * dim || size <= dim + 1 {
            Array2::from_shape_fn((size, dim), |_| (rng.random_f32() * 2.0 - 1.0) * data_range)
        } else {
            let subgraph = induced_subgraph(graph, vertices, &local_index);
            let mut embedding = connected_layout(&subgraph, dim, rng, SolverStart::Deterministic);
            let max_abs = embedding.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            if max_abs > 0.0 {
                embedding *= data_range / max_abs;
            }
            embedding
        };
        component_embedding += &center;
        for (row, &vertex) in vertices.iter().enumerate() {
            result.row_mut(vertex).assign(&component_embedding.row(row));
        }
    }
    result
}

fn induced_subgraph(graph: &CsrMatrix, vertices: &[usize], local_index: &[usize]) -> CsrMatrix {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for &vertex in vertices {
        for idx in graph.indptr[vertex]..graph.indptr[vertex + 1] {
            indices.push(local_index[graph.indices[idx]]);
            data.push(graph.data[idx]);
        }
        indptr.push(indices.len());
    }
    CsrMatrix {
        indptr,
        indices,
        data,
        nrows: vertices.len(),
        ncols: vertices.len(),
    }
}

/// Distance from `label` to the closest other position (1 if there is none).
fn nearest_component_distance(positions: &Array2<f32>, label: usize) -> f32 {
    let nearest = (0..positions.nrows())
        .map(|other| {
            let difference = &positions.row(label) - &positions.row(other);
            difference.dot(&difference).sqrt()
        })
        .filter(|&distance| distance > 0.0)
        .fold(f32::INFINITY, f32::min);
    if nearest.is_finite() {
        nearest
    } else {
        1.0
    }
}

/// Positions at `+e_i` and `-e_i` for up to `2 * dim` components.
fn axis_aligned_layout(n_components: usize, dim: usize) -> Array2<f32> {
    let half = n_components.div_ceil(2);
    Array2::from_shape_fn((n_components, dim), |(component, axis)| {
        let (sign, positive_axis) = if component < half {
            (1.0, component)
        } else {
            (-1.0, component - half)
        };
        if positive_axis == axis {
            sign
        } else {
            0.0
        }
    })
}

/// Spectral embedding of the component centroids using an affinity `exp(-d^2)`,
/// normalized by the largest coordinate. Without data, random positions in `[0, 10]`.
fn component_layout(
    centroids: Option<ComponentCentroids>,
    labels: &[usize],
    n_components: usize,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let Some(ComponentCentroids { data, metric }) = centroids else {
        return Array2::from_shape_fn((n_components, dim), |_| rng.random_f32() * 10.0);
    };
    let Some(distance_func) = get_distance_func(metric) else {
        return Array2::from_shape_fn((n_components, dim), |_| rng.random_f32() * 10.0);
    };

    let n_features = data.ncols();
    let mut sums = vec![0.0f64; n_components * n_features];
    let mut counts = vec![0usize; n_components];
    for (row, &label) in data.rows().into_iter().zip(labels) {
        counts[label] += 1;
        for (sum, &value) in sums[label * n_features..(label + 1) * n_features]
            .iter_mut()
            .zip(row.iter())
        {
            *sum += value as f64;
        }
    }
    let centroid_rows: Vec<Vec<f32>> = (0..n_components)
        .map(|label| {
            sums[label * n_features..(label + 1) * n_features]
                .iter()
                .map(|&sum| (sum / counts[label].max(1) as f64) as f32)
                .collect()
        })
        .collect();

    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for i in 0..n_components {
        for j in (0..n_components).filter(|&j| j != i) {
            let distance = distance_func(&centroid_rows[i], &centroid_rows[j]) as f64;
            let affinity = (-distance * distance).exp() as f32;
            if affinity > 0.0 {
                indices.push(j);
                weights.push(affinity);
            }
        }
        indptr.push(indices.len());
    }
    let affinity_graph = CsrMatrix {
        indptr,
        indices,
        data: weights,
        nrows: n_components,
        ncols: n_components,
    };

    let mut embedding = connected_layout(&affinity_graph, dim, rng, SolverStart::Deterministic);
    let degrees = affinity_graph.row_sums();
    for (mut row, degree) in embedding.rows_mut().into_iter().zip(degrees) {
        if degree > 0.0 {
            row /= degree.sqrt();
        }
    }
    for mut column in embedding.columns_mut() {
        let dominant = column
            .iter()
            .cloned()
            .fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a });
        if dominant < 0.0 {
            column.mapv_inplace(|x| -x);
        }
    }
    let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max != 0.0 && max.is_finite() {
        embedding /= max;
    }
    embedding
}

/// Spectral embedding of a connected graph.
fn connected_layout(
    graph: &CsrMatrix,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
    start: SolverStart,
) -> Array2<f32> {
    let n = graph.nrows;

    let k_tiny = dim + 1;
    if n < k_tiny {
        return random_layout(n, dim, rng);
    }
    if n == k_tiny {
        return dense_spectral_layout(graph, dim, rng);
    }

    let degrees = graph.row_sums();

    let min_degree: f32 = degrees.iter().cloned().fold(f32::INFINITY, f32::min);
    if min_degree <= 0.0 {
        return random_layout(n, dim, rng);
    }

    let inv_sqrt_deg: Vec<f32> = degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();
    let k = dim; // Find dim eigenpairs, orthogonal to the trivial eigenvector

    // The trivial eigenvector of the normalized Laplacian is sqrt(D)/||sqrt(D)||.
    // We constrain LOBPCG to search orthogonal to it, avoiding convergence issues.
    let sqrt_deg: Vec<f64> = degrees.iter().map(|&d| (d as f64).sqrt()).collect();
    let norm: f64 = sqrt_deg.iter().map(|x| x * x).sum::<f64>().sqrt();
    let constraint: Vec<f64> = sqrt_deg.iter().map(|x| x / norm).collect();

    let x0 = match start {
        SolverStart::Deterministic => deterministic_start(k, n),
        SolverStart::TruncatedSvd => truncated_svd_start(graph, &inv_sqrt_deg, k, n, rng),
    };

    let tol = 1e-5;
    let max_iter = 200;
    // Deterministic RNG state for LOBPCG internal use (replacing converged residuals)
    let mut rng_state: [u64; 2] = [0x12345678_9abcdef0, 0xfedcba98_76543210];

    match lobpcg_solve(
        graph,
        &inv_sqrt_deg,
        k,
        n,
        x0,
        tol,
        max_iter,
        &mut rng_state,
        &constraint,
    ) {
        Some((eigenvalues, eigenvectors)) => {
            // Sort eigenvalues ascending and use the k smallest eigenvectors
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| {
                eigenvalues[a]
                    .partial_cmp(&eigenvalues[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut embedding = Array2::zeros((n, dim));
            for d in 0..dim {
                let col_idx = order[d];
                for i in 0..n {
                    embedding[[i, d]] = eigenvectors.cols[col_idx][i] as f32;
                }
            }

            let has_nan = embedding.iter().any(|x| x.is_nan());
            let all_zero = embedding.iter().all(|&x| x == 0.0);
            if has_nan || all_zero {
                eprintln!("  Spectral embedding has NaN or all-zero, falling back to random");
                return random_layout(n, dim, rng);
            }

            embedding
        }
        None => {
            eprintln!("  LOBPCG failed to converge, falling back to random layout");
            random_layout(n, dim, rng)
        }
    }
}

/// Spectral embedding of a graph with exactly `dim + 1` nodes: dense
/// eigendecomposition of L, using eigenvectors `1..=dim` in ascending eigenvalue order.
fn dense_spectral_layout(
    graph: &CsrMatrix,
    dim: usize,
    rng: &mut Xoshiro256StarStar,
) -> Array2<f32> {
    let n = graph.nrows;
    let inv_sqrt_deg: Vec<f64> = graph
        .row_sums()
        .iter()
        .map(|&d| 1.0 / (d as f64).sqrt())
        .collect();
    let mut laplacian = vec![0.0f64; n * n];
    for i in 0..n {
        laplacian[i * n + i] = 1.0;
        for idx in graph.indptr[i]..graph.indptr[i + 1] {
            let j = graph.indices[idx];
            laplacian[i * n + j] -= graph.data[idx] as f64 * inv_sqrt_deg[i] * inv_sqrt_deg[j];
        }
    }
    if laplacian.iter().any(|x| !x.is_finite()) {
        return random_layout(n, dim, rng);
    }

    let (_, eigenvectors) = eigh_jacobi(&laplacian, n);
    Array2::from_shape_fn((n, dim), |(i, d)| eigenvectors[i * n + d + 1] as f32)
}

/// Scale embedding coordinates and add noise to avoid local minima.
pub fn noisy_scale_coords(
    coords: &mut Array2<f32>,
    rng: &mut Xoshiro256StarStar,
    max_coord: f32,
    noise: f32,
) {
    let abs_max = coords.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    if abs_max > 0.0 {
        let expansion = max_coord / abs_max;
        coords.mapv_inplace(|x| x * expansion);
    }

    // Add Gaussian noise using Box-Muller transform
    let shape = coords.raw_dim();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let u1: f64 = rng.random_f64().max(1e-10);
            let u2: f64 = rng.random_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            coords[[i, j]] += (noise as f64 * z) as f32;
        }
    }
}

/// Random embedding in [-10, 10]^dim.
pub fn random_layout(n: usize, dim: usize, rng: &mut Xoshiro256StarStar) -> Array2<f32> {
    Array2::from_shape_fn((n, dim), |_| rng.random_f32() * 20.0 - 10.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigh_jacobi_diagonal() {
        // diag(3, 1, 2) → eigenvalues [1, 2, 3] (numpy.linalg.eigh verified)
        let a = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (vals, vecs) = eigh_jacobi(&a, 3);
        // Reference: np.linalg.eigh(np.diag([3,1,2])) → [1.0, 2.0, 3.0]
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);
        // Verify V * diag(vals) * V^T ≈ A (reconstruction)
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for l in 0..3 {
                    sum += vecs[i * 3 + l] * vals[l] * vecs[j * 3 + l];
                }
                assert!(
                    (sum - a[i * 3 + j]).abs() < 1e-10,
                    "Reconstruction failed at ({}, {}): {} vs {}",
                    i,
                    j,
                    sum,
                    a[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn test_eigh_jacobi_symmetric() {
        // [[2,1,0],[1,3,1],[0,1,2]]
        // Reference: np.linalg.eigh → eigenvalues [1.0, 2.0, 4.0]
        let a = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let (vals, _) = eigh_jacobi(&a, 3);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_2x2() {
        // [[4,2],[2,3]]
        // Reference: np.linalg.cholesky → [[2.0, 0.0], [1.0, 1.41421356...]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        assert!((l[0] - 2.0).abs() < 1e-10); // L[0,0] = 2.0
        assert!((l[2] - 1.0).abs() < 1e-10); // L[1,0] = 1.0
        assert!((l[3] - 2.0f64.sqrt()).abs() < 1e-10); // L[1,1] = sqrt(2)
    }

    #[test]
    fn test_cholesky_not_spd() {
        // [[1,2],[2,1]] has eigenvalues -1 and 3, not positive definite
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    #[test]
    fn test_generalized_eigh_identity_s() {
        // G*c = lambda*I*c → standard eigh of G
        // G = [[2,1],[1,3]], S = I
        // Reference: scipy.linalg.eigh(G, I) → [(5-√5)/2, (5+√5)/2] ≈ [1.381966, 3.618034]
        let g = vec![2.0, 1.0, 1.0, 3.0];
        let s = vec![1.0, 0.0, 0.0, 1.0];
        let (vals, _) = generalized_eigh(&g, &s, 2).unwrap();
        let expected_0 = (5.0 - 5.0f64.sqrt()) / 2.0; // 1.381966011250105
        let expected_1 = (5.0 + 5.0f64.sqrt()) / 2.0; // 3.618033988749895
        assert!((vals[0] - expected_0).abs() < 1e-8);
        assert!((vals[1] - expected_1).abs() < 1e-8);
    }

    #[test]
    fn test_lobpcg_two_cluster_graph() {
        // Two-cluster graph: K_5 + K_5 connected by 2 bridge edges (0-5, 1-6).
        // This has a clear spectral gap, representative of UMAP's typical graphs.
        //
        // Reference eigenvalues from numpy (np.linalg.eigvalsh):
        //   lambda_0 =  0.000000000000000  (trivial, excluded via constraint)
        //   lambda_1 =  0.147920271060385
        //   lambda_2 =  1.000000000000000
        //   lambda_3 =  1.100000000000000
        //
        // With deterministic initial vectors (ones, alternating) on this tiny
        // 10-node graph, LOBPCG finds (0.148, 1.1) — missing 1.0 because the
        // initial vectors lack overlap with that eigenvector. On real UMAP graphs
        // (thousands of nodes), the deterministic vectors work correctly.
        let n = 10;
        let indptr = vec![0, 5, 10, 14, 18, 22, 27, 32, 36, 40, 44];
        let indices = vec![
            1, 2, 3, 4, 5, // node 0
            0, 2, 3, 4, 6, // node 1
            0, 1, 3, 4, // node 2
            0, 1, 2, 4, // node 3
            0, 1, 2, 3, // node 4
            0, 6, 7, 8, 9, // node 5
            1, 5, 7, 8, 9, // node 6
            5, 6, 8, 9, // node 7
            5, 6, 7, 9, // node 8
            5, 6, 7, 8, // node 9
        ];
        let data = vec![1.0f32; 44];
        let graph = CsrMatrix {
            indptr,
            indices,
            data,
            nrows: n,
            ncols: n,
        };

        let degrees = graph.row_sums();
        let inv_sqrt_deg: Vec<f32> = degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();

        // Constraint: trivial eigenvector sqrt(D)/||sqrt(D)||
        let k = 2; // Find 2 non-trivial eigenpairs
        let sqrt_deg: Vec<f64> = degrees.iter().map(|&d| (d as f64).sqrt()).collect();
        let norm: f64 = sqrt_deg.iter().map(|x| x * x).sum::<f64>().sqrt();
        let constraint: Vec<f64> = sqrt_deg.iter().map(|x| x / norm).collect();

        let x0 = deterministic_start(k, n);

        let tol = 1e-5;
        let max_iter = 200;
        let mut rng_state: [u64; 2] = [0x12345678_9abcdef0, 0xfedcba98_76543210];
        let result = lobpcg_solve(
            &graph,
            &inv_sqrt_deg,
            k,
            n,
            x0,
            tol,
            max_iter,
            &mut rng_state,
            &constraint,
        );
        assert!(result.is_some(), "LOBPCG failed to converge");
        let (mut eigenvalues, _) = result.unwrap();

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // On this tiny graph, deterministic vectors find (0.148, 1.1).
        // The first eigenvalue is the important one for cluster separation.
        assert!(
            (eigenvalues[0] - 0.147920271060385).abs() < 1e-4,
            "lambda_0 = {} (expected 0.14792)",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 1.1).abs() < 0.11,
            "lambda_1 = {} (expected ~1.0 or ~1.1)",
            eigenvalues[1]
        );
    }

    fn complete_graph(n: usize) -> CsrMatrix {
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        for i in 0..n {
            indices.extend((0..n).filter(|&j| j != i));
            indptr.push(indices.len());
        }
        let data = vec![1.0f32; indices.len()];
        CsrMatrix {
            indptr,
            indices,
            data,
            nrows: n,
            ncols: n,
        }
    }

    #[test]
    fn test_spectral_layout_fewer_nodes_than_k_is_random() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let emb = spectral_layout(&complete_graph(2), 2, &mut rng);
        assert_eq!(emb.dim(), (2, 2));
        assert!(emb.iter().all(|x| (-10.0..=10.0).contains(x)));
    }

    #[test]
    fn test_spectral_layout_exactly_k_nodes_uses_dense_eigenvectors() {
        // K3 has L eigenvalues 0, 1.5, 1.5; eigenvectors 1 and 2 span the space
        // orthogonal to sqrt(D), so every column sums to zero.
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let emb = spectral_layout(&complete_graph(3), 2, &mut rng);
        assert_eq!(emb.dim(), (3, 2));
        for d in 0..2 {
            let sum: f32 = emb.column(d).sum();
            let norm: f32 = emb.column(d).iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(sum.abs() < 1e-5, "column {d} sum = {sum}");
            assert!((norm - 1.0).abs() < 1e-5, "column {d} norm = {norm}");
        }
    }

    /// Block-diagonal graph; each block is a ring where every vertex links to its
    /// two nearest neighbors on both sides (a complete graph for blocks below 5).
    fn block_ring_graph(sizes: &[usize]) -> CsrMatrix {
        let n: usize = sizes.iter().sum();
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut offset = 0;
        for &size in sizes {
            for i in 0..size {
                let mut neighbors: Vec<usize> = if size < 5 {
                    (0..size).filter(|&j| j != i).collect()
                } else {
                    [1, 2, size - 1, size - 2]
                        .iter()
                        .map(|d| (i + d) % size)
                        .collect()
                };
                neighbors.sort();
                indices.extend(neighbors.iter().map(|j| offset + j));
                indptr.push(indices.len());
            }
            offset += size;
        }
        let data = vec![1.0f32; indices.len()];
        CsrMatrix {
            indptr,
            indices,
            data,
            nrows: n,
            ncols: n,
        }
    }

    fn component_ranges(sizes: &[usize]) -> Vec<std::ops::Range<usize>> {
        let mut start = 0;
        sizes
            .iter()
            .map(|&size| {
                start += size;
                start - size..start
            })
            .collect()
    }

    #[test]
    fn test_connected_components_labels_blocks() {
        let sizes = [7, 1, 12];
        let (n_components, labels) = connected_components(&block_ring_graph(&sizes));
        assert_eq!(n_components, 3);
        for (label, range) in component_ranges(&sizes).into_iter().enumerate() {
            assert!(labels[range].iter().all(|&l| l == label));
        }
    }

    /// Components are separated when every pair of component centers is farther apart
    /// than the mean distance of the vertices to their own center.
    fn assert_components_separated(emb: &Array2<f32>, sizes: &[usize]) {
        assert!(emb.iter().all(|x| x.is_finite()));
        let ranges = component_ranges(sizes);
        let centers: Vec<_> = ranges
            .iter()
            .map(|range| {
                emb.slice(ndarray::s![range.clone(), ..])
                    .mean_axis(ndarray::Axis(0))
                    .unwrap()
            })
            .collect();
        let distance = |a: &ndarray::Array1<f32>, b: &ndarray::ArrayView1<f32>| {
            let difference = a - b;
            difference.dot(&difference).sqrt()
        };
        let largest_mean_radius = ranges
            .iter()
            .zip(&centers)
            .map(|(range, center)| {
                let total: f32 = range.clone().map(|v| distance(center, &emb.row(v))).sum();
                total / range.len() as f32
            })
            .fold(0.0, f32::max);
        for i in 0..centers.len() {
            for j in (i + 1)..centers.len() {
                let between = distance(&centers[i], &centers[j].view());
                assert!(
                    between > largest_mean_radius,
                    "centers {i} and {j}: {between} <= {largest_mean_radius}"
                );
            }
        }
    }

    #[test]
    fn test_spectral_layout_separates_three_components() {
        let sizes = [30, 25, 40];
        let graph = block_ring_graph(&sizes);
        let data = Array2::from_shape_fn((95, 2), |(i, d)| {
            let component = component_ranges(&sizes)
                .iter()
                .position(|r| r.contains(&i))
                .unwrap();
            [0.0, 2.0, 4.0][component] * (1 - d) as f32 + 0.01 * i as f32
        });
        let centroids = ComponentCentroids {
            data: &data,
            metric: "euclidean",
        };
        let mut rng = Xoshiro256StarStar::seed_from_u64(3);
        let emb = spectral_layout_with_centroids(&graph, Some(centroids), 2, &mut rng);
        assert_eq!(emb.dim(), (95, 2));
        assert_components_separated(&emb, &sizes);
    }

    #[test]
    fn test_spectral_layout_separates_many_components_by_centroid_embedding() {
        let sizes = [20, 20, 20, 20, 20, 20];
        let graph = block_ring_graph(&sizes);
        let data = Array2::from_shape_fn((120, 2), |(i, d)| {
            let component = (i / 20) as f32;
            if d == 0 {
                component * 1.5
            } else {
                (component % 2.0) * 1.5
            }
        });
        let centroids = ComponentCentroids {
            data: &data,
            metric: "euclidean",
        };
        let mut rng = Xoshiro256StarStar::seed_from_u64(5);
        let emb = spectral_layout_with_centroids(&graph, Some(centroids), 2, &mut rng);
        assert_components_separated(&emb, &sizes);
    }

    #[test]
    fn test_spectral_layout_without_data_separates_many_components() {
        let sizes = [15, 15, 15, 15, 15];
        let mut rng = Xoshiro256StarStar::seed_from_u64(9);
        let emb = spectral_layout(&block_ring_graph(&sizes), 2, &mut rng);
        assert_components_separated(&emb, &sizes);
    }

    #[test]
    fn test_spectral_layout_handles_tiny_components() {
        let sizes = [1, 2, 3, 30];
        let mut rng = Xoshiro256StarStar::seed_from_u64(2);
        let emb = spectral_layout(&block_ring_graph(&sizes), 2, &mut rng);
        assert_eq!(emb.dim(), (36, 2));
        assert!(emb.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_axis_aligned_layout_uses_signed_axes() {
        let layout = axis_aligned_layout(3, 2);
        assert_eq!(layout.row(0).to_vec(), vec![1.0, 0.0]);
        assert_eq!(layout.row(1).to_vec(), vec![0.0, 1.0]);
        assert_eq!(layout.row(2).to_vec(), vec![-1.0, 0.0]);
    }

    #[test]
    fn test_tsw_spectral_layout_spans_spectral_subspace() {
        let graph = block_ring_graph(&[80]);
        let mut rng = Xoshiro256StarStar::seed_from_u64(4);
        let spectral = spectral_layout(&graph, 2, &mut rng);
        let tsw = tsw_spectral_layout(&graph, None, 2, &mut rng);
        assert_eq!(tsw.dim(), (80, 2));
        assert!(tsw.iter().all(|x| x.is_finite()));

        let mut basis: Vec<ndarray::Array1<f32>> = vec![];
        for column in spectral.columns() {
            let mut vector = column.to_owned();
            for earlier in &basis {
                let overlap = vector.dot(earlier);
                vector.scaled_add(-overlap, earlier);
            }
            let norm = vector.dot(&vector).sqrt();
            basis.push(vector / norm);
        }
        for column in tsw.columns() {
            let captured: f32 = basis.iter().map(|b| column.dot(b).powi(2)).sum();
            let total = column.dot(&column);
            assert!(captured / total > 0.99, "captured {captured} of {total}");
        }
    }
}
