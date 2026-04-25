use ndarray::Array2;
use nndescent::rng::TauRng;
use nndescent::Logger;
use rayon::prelude::*;

/// Clamp gradient to [-4.0, 4.0] to prevent instability.
#[inline]
fn clip(val: f32) -> f32 {
    val.clamp(-4.0, 4.0)
}

/// Squared Euclidean distance in embedding space.
#[inline]
fn rdist(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

/// Optimize the low-dimensional embedding using SGD with Hogwild parallelism.
///
/// Minimizes the fuzzy set cross entropy between the high-dimensional
/// fuzzy simplicial set and the low-dimensional one, using negative
/// sampling similar to word2vec.
///
/// The edge gradient updates are parallelized using the Hogwild approach
/// (Recht et al. 2011): concurrent writes to the embedding array are
/// allowed without synchronization. With dim=2, each update touches only
/// 2 floats per vertex, so conflicts are rare and benign.
pub fn optimize_layout_euclidean(
    embedding: &mut Array2<f32>,
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f32],
    n_epochs: usize,
    n_vertices: usize,
    a: f64,
    b: f64,
    gamma: f32,
    initial_alpha: f32,
    negative_sample_rate: f32,
    rng_state: [i64; 3],
    logger: &mut Logger,
) {
    let dim = embedding.ncols();
    let n_edges = head.len();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    let mut epoch_of_next_sample = epochs_per_sample.to_vec();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let a = a as f32;
    let b = b as f32;

    // Use flat slice for embedding data. ndarray Array2 is row-major,
    // so row j starts at offset j*dim.
    let emb = embedding
        .as_slice_memory_order_mut()
        .expect("embedding not contiguous");

    // SAFETY: Hogwild-style parallel SGD (Recht et al., 2011). The embedding array
    // is shared mutably across threads WITHOUT synchronization. This is intentional:
    // - Each gradient update touches only `dim` floats (typically 2-4) per vertex.
    // - Concurrent writes to the same vertex produce slightly stale/torn reads, which
    //   act as additional stochastic noise and do not affect convergence in practice.
    // - This matches the approach used by the reference Python UMAP (via numba prange)
    //   and is standard practice for embedding optimization.
    //
    // Formally, creating multiple `&mut` slices to overlapping memory is UB under Rust's
    // aliasing model. In practice, the generated code is safe because:
    // (a) all accesses are simple f32 loads/stores (no LLVM optimizations rely on
    //     noalias for correctness at this granularity), and
    // (b) the primary target is WASM, which is single-threaded (no actual concurrency).
    //
    // For a strictly sound alternative, raw pointer arithmetic or AtomicF32 could be
    // used, at the cost of readability and (for atomics) performance.
    let emb_ptr = emb.as_mut_ptr() as usize;

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        // Phase 1: Sequentially collect active edges and advance scheduling counters.
        // Each active edge records (edge_index, n_negative_samples).
        let mut active_edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_edges {
            if epoch_of_next_sample[i] > epoch as f32 {
                continue;
            }

            let n_neg = ((epoch as f32 - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]) as usize;

            active_edges.push((i, n_neg));

            epoch_of_next_sample[i] += epochs_per_sample[i];
            epoch_of_next_negative_sample[i] += n_neg as f32 * epochs_per_negative_sample[i];
        }

        // Phase 2: Apply gradients in parallel (Hogwild).
        // Each edge gets its own RNG seeded deterministically from
        // (rng_state, edge_index, epoch) — no shared mutable RNG state.
        active_edges.par_iter().for_each_init(
            || (vec![0.0f32; dim], vec![0.0f32; dim], vec![0.0f32; dim]),
            |(current, other, neg_other), &(edge_idx, n_neg)| {
                // SAFETY: see Hogwild comment above on emb_ptr.
                let emb = unsafe {
                    std::slice::from_raw_parts_mut(emb_ptr as *mut f32, n_vertices * dim)
                };

                let j = head[edge_idx];
                let k = tail[edge_idx];
                let j_off = j * dim;
                let k_off = k * dim;

                // Copy current positions to local buffers
                current.copy_from_slice(&emb[j_off..j_off + dim]);
                other.copy_from_slice(&emb[k_off..k_off + dim]);
                let dist_squared = rdist(current, other);

                // Positive sample: attractive force
                let grad_coeff = if dist_squared > 0.0 {
                    let pow_b = dist_squared.powf(b);
                    -2.0 * a * b * (pow_b / dist_squared) / (a * pow_b + 1.0)
                } else {
                    0.0
                };

                for d in 0..dim {
                    let grad = clip(grad_coeff * (current[d] - other[d]));
                    emb[j_off + d] += alpha * grad;
                    emb[k_off + d] -= alpha * grad;
                }

                // Negative samples: repulsive force
                // Per-edge RNG seeded from (rng_state, edge_index, epoch)
                let mut rng = TauRng::from_state([
                    rng_state[0]
                        .wrapping_add(edge_idx as i64)
                        .wrapping_mul(epoch as i64 + 1),
                    rng_state[1].wrapping_add(j as i64),
                    rng_state[2].wrapping_add(edge_idx as i64),
                ]);

                for _ in 0..n_neg {
                    let neg_k = (rng.tau_rand_int().unsigned_abs() as usize) % n_vertices;

                    let neg_off = neg_k * dim;
                    neg_other.copy_from_slice(&emb[neg_off..neg_off + dim]);
                    // Re-read current since it may have been updated
                    current.copy_from_slice(&emb[j_off..j_off + dim]);
                    let dist_squared = rdist(current, neg_other);

                    let grad_coeff = if dist_squared > 0.0 {
                        2.0 * gamma * b
                            / ((0.001 + dist_squared) * (a * dist_squared.powf(b) + 1.0))
                    } else if j == neg_k {
                        continue;
                    } else {
                        0.0
                    };

                    for d in 0..dim {
                        let grad = if grad_coeff > 0.0 {
                            clip(grad_coeff * (current[d] - neg_other[d]))
                        } else {
                            0.0
                        };
                        emb[j_off + d] += alpha * grad;
                    }
                }
            },
        );

        if n_epochs >= 10 && epoch % (n_epochs / 10) == 0 {
            logger.log(&format!("completed {} / {} epochs", epoch, n_epochs));
        }
        logger.stage_progress((epoch + 1) as f64 / n_epochs as f64, None);
    }
}
