/// Core NN-descent algorithm implementation.
use ndarray::Array2;
use rayon::prelude::*;

use crate::distance::DistanceFunc;
use crate::heap::{
    checked_flagged_heap_push, checked_heap_push, deheap_sort, make_heap, NeighborHeap,
};
use crate::rng::TauRng;
use crate::{Logger, STAGE_NN_DESCENT, STAGE_SORTING};

#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;

/// Initialize the heap from RP tree leaf co-occurrences.
/// For each leaf, compute distances between all pairs and insert into heap.
/// Processes leaves in batches to limit peak memory, with bucketed parallel
/// application within each batch.
pub fn init_rp_tree(
    data: &Array2<f32>,
    dist: DistanceFunc,
    heap: &mut NeighborHeap,
    leaf_array: &Array2<i32>,
) {
    let n_leaves = leaf_array.nrows();
    let leaf_width = leaf_array.ncols();
    let n_vertices = heap.indices.nrows();
    let n_neighbors = heap.indices.ncols();
    let n_threads = rayon::current_num_threads();
    let block_size = (n_vertices + n_threads - 1) / n_threads;

    // SAFETY: Raw pointers cast to usize to cross the Send boundary into par_iter.
    // Soundness relies on the bucketed parallel scheme: in Phase 1, each thread only
    // writes to heap rows within its own block [block_start..block_end). Cross-block
    // updates are buffered into per-target-thread buckets and applied in a separate
    // parallel phase where, again, each thread only writes to its own row range.
    // The arrays remain live and pinned for the duration of each parallel section.
    let dist_ptr = heap.distances.as_mut_ptr() as usize;
    let idx_ptr = heap.indices.as_mut_ptr() as usize;
    let flags_ptr = heap.flags.as_mut_ptr() as usize;
    let batch_size = 4096;
    for batch_start in (0..n_leaves).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n_leaves);

        // Compute distances in parallel, bucketed by target thread block
        let all_buckets: Vec<Vec<Vec<(i32, i32, f32)>>> = (batch_start..batch_end)
            .into_par_iter()
            .fold(
                || -> Vec<Vec<(i32, i32, f32)>> { (0..n_threads).map(|_| Vec::new()).collect() },
                |mut buckets, leaf_idx| {
                    for i in 0..leaf_width {
                        let p = leaf_array[[leaf_idx, i]];
                        if p < 0 {
                            break;
                        }

                        for j in (i + 1)..leaf_width {
                            let q = leaf_array[[leaf_idx, j]];
                            if q < 0 {
                                break;
                            }

                            let d = dist(
                                data.row(p as usize).as_slice().unwrap(),
                                data.row(q as usize).as_slice().unwrap(),
                            );

                            let p_block = ((p as usize) / block_size).min(n_threads - 1);
                            buckets[p_block].push((p, q, d));
                            let q_block = ((q as usize) / block_size).min(n_threads - 1);
                            buckets[q_block].push((q, p, d));
                        }
                    }
                    buckets
                },
            )
            .collect();

        // Apply: each thread reads its own bucket from all fold results
        (0..n_threads).into_par_iter().for_each(|t| {
            for chunk_buckets in &all_buckets {
                for &(target, source, d) in &chunk_buckets[t] {
                    let tu = target as usize;
                    let off = tu * n_neighbors;
                    let dists = unsafe {
                        std::slice::from_raw_parts_mut((dist_ptr as *mut f32).add(off), n_neighbors)
                    };
                    let inds = unsafe {
                        std::slice::from_raw_parts_mut((idx_ptr as *mut i32).add(off), n_neighbors)
                    };
                    let flgs = unsafe {
                        std::slice::from_raw_parts_mut((flags_ptr as *mut u8).add(off), n_neighbors)
                    };
                    checked_flagged_heap_push(dists, inds, flgs, d, source, 1);
                }
            }
        });
    }
}

/// Fill remaining empty neighbor slots with random points.
pub fn init_random(
    n_neighbors: usize,
    data: &Array2<f32>,
    heap: &mut NeighborHeap,
    dist: DistanceFunc,
    rng: &mut TauRng,
) {
    let n = data.nrows();
    for i in 0..n {
        // Check if the root index is -1 (heap not fully filled)
        if heap.indices[[i, 0]] < 0 {
            // Count filled slots
            let filled: usize = (0..n_neighbors)
                .filter(|&j| heap.indices[[i, j]] >= 0)
                .count();
            let needed = n_neighbors - filled;
            for _ in 0..needed {
                let idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
                let d = dist(
                    data.row(idx).as_slice().unwrap(),
                    data.row(i).as_slice().unwrap(),
                );
                checked_flagged_heap_push(
                    heap.distances.row_mut(i).as_slice_mut().unwrap(),
                    heap.indices.row_mut(i).as_slice_mut().unwrap(),
                    heap.flags.row_mut(i).as_slice_mut().unwrap(),
                    d,
                    idx as i32,
                    1,
                );
            }
        }
    }
}

/// Build new and old candidate neighbor lists from the current graph.
/// Returns (new_candidate_indices, old_candidate_indices), each shape (n_vertices, max_candidates).
///
/// Phase 1: Each thread scans only its own block of graph rows, pushing candidates
///           directly into its own candidate rows and buffering cross-block pushes.
/// Phase 2: Cross-block pushes are applied in parallel (each thread applies to its own rows).
/// Phase 3: Mark matched edges as old.
pub fn new_build_candidates(
    graph: &mut NeighborHeap,
    max_candidates: usize,
    rng: &mut TauRng,
) -> (Array2<i32>, Array2<i32>) {
    let n_vertices = graph.indices.nrows();
    let n_neighbors = graph.indices.ncols();

    let mut new_candidate_indices = Array2::from_elem((n_vertices, max_candidates), -1i32);
    let mut new_candidate_priority = Array2::from_elem((n_vertices, max_candidates), f32::INFINITY);
    let mut old_candidate_indices = Array2::from_elem((n_vertices, max_candidates), -1i32);
    let mut old_candidate_priority = Array2::from_elem((n_vertices, max_candidates), f32::INFINITY);

    let n_threads = rayon::current_num_threads();
    let block_size = (n_vertices + n_threads - 1) / n_threads;

    // Capture base RNG state; each thread derives its own RNG from base_state + thread_id
    let base_state = rng.state;
    // Advance the main RNG past this call so subsequent uses don't overlap
    for _ in 0..(n_threads as i64) {
        rng.tau_rand_int();
    }

    // SAFETY: Raw pointers cast to usize to cross the Send boundary into par_iter.
    // Phase 1: each thread reads graph rows in its own block and writes candidate
    // rows within that same block directly; cross-block pushes are buffered.
    // Phase 2: each thread applies buffered cross-block pushes to its own rows only.
    // Phase 3: each thread writes flags for its own row range only.
    // No two threads write to the same row in any phase, so accesses are disjoint.
    let new_idx_ptr = new_candidate_indices.as_mut_ptr() as usize;
    let new_pri_ptr = new_candidate_priority.as_mut_ptr() as usize;
    let old_idx_ptr = old_candidate_indices.as_mut_ptr() as usize;
    let old_pri_ptr = old_candidate_priority.as_mut_ptr() as usize;
    let graph_idx_ptr = graph.indices.as_ptr() as usize;
    let graph_flags_ptr = graph.flags.as_ptr() as usize;

    // Phase 1: Each thread scans only its own block, buffers cross-block pushes
    // Each thread produces n_threads buckets of cross-block updates, keyed by target block
    // Bucket entry: (target_row as i32, candidate_idx as i32, priority, is_new)
    let cross_buckets: Vec<Vec<Vec<(i32, i32, f32, bool)>>> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let block_start = t * block_size;
            let block_end = (block_start + block_size).min(n_vertices);
            let mut buckets: Vec<Vec<(i32, i32, f32, bool)>> =
                (0..n_threads).map(|_| Vec::new()).collect();

            let mut local_rng = TauRng::from_state([
                base_state[0] + t as i64,
                base_state[1] + t as i64,
                base_state[2] + t as i64,
            ]);

            for i in block_start..block_end {
                for j in 0..n_neighbors {
                    let idx = unsafe { *(graph_idx_ptr as *const i32).add(i * n_neighbors + j) };
                    if idx < 0 {
                        continue;
                    }

                    let iu = idx as usize;
                    let is_new =
                        unsafe { *(graph_flags_ptr as *const u8).add(i * n_neighbors + j) } != 0;
                    let d = local_rng.tau_rand();

                    if is_new {
                        // Push idx into row i (in our block - direct)
                        let row_off = i * max_candidates;
                        let pri = unsafe {
                            std::slice::from_raw_parts_mut(
                                (new_pri_ptr as *mut f32).add(row_off),
                                max_candidates,
                            )
                        };
                        let ind = unsafe {
                            std::slice::from_raw_parts_mut(
                                (new_idx_ptr as *mut i32).add(row_off),
                                max_candidates,
                            )
                        };
                        checked_heap_push(pri, ind, d, idx);

                        // Push i into row idx (may be cross-block)
                        if iu >= block_start && iu < block_end {
                            let row_off = iu * max_candidates;
                            let pri = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (new_pri_ptr as *mut f32).add(row_off),
                                    max_candidates,
                                )
                            };
                            let ind = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (new_idx_ptr as *mut i32).add(row_off),
                                    max_candidates,
                                )
                            };
                            checked_heap_push(pri, ind, d, i as i32);
                        } else {
                            let target_thread = iu / block_size;
                            buckets[target_thread.min(n_threads - 1)]
                                .push((idx, i as i32, d, true));
                        }
                    } else {
                        let row_off = i * max_candidates;
                        let pri = unsafe {
                            std::slice::from_raw_parts_mut(
                                (old_pri_ptr as *mut f32).add(row_off),
                                max_candidates,
                            )
                        };
                        let ind = unsafe {
                            std::slice::from_raw_parts_mut(
                                (old_idx_ptr as *mut i32).add(row_off),
                                max_candidates,
                            )
                        };
                        checked_heap_push(pri, ind, d, idx);

                        if iu >= block_start && iu < block_end {
                            let row_off = iu * max_candidates;
                            let pri = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (old_pri_ptr as *mut f32).add(row_off),
                                    max_candidates,
                                )
                            };
                            let ind = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (old_idx_ptr as *mut i32).add(row_off),
                                    max_candidates,
                                )
                            };
                            checked_heap_push(pri, ind, d, i as i32);
                        } else {
                            let target_thread = iu / block_size;
                            buckets[target_thread.min(n_threads - 1)]
                                .push((idx, i as i32, d, false));
                        }
                    }
                }
            }
            buckets
        })
        .collect();

    // Phase 2: Each thread applies cross-block updates from all source threads' buckets for its block
    (0..n_threads).into_par_iter().for_each(|t| {
        // Gather all updates targeting thread t's block from all source threads
        for source_thread in 0..n_threads {
            for &(target, candidate, d, is_new) in &cross_buckets[source_thread][t] {
                let tu = target as usize;
                let row_off = tu * max_candidates;
                if is_new {
                    let pri = unsafe {
                        std::slice::from_raw_parts_mut(
                            (new_pri_ptr as *mut f32).add(row_off),
                            max_candidates,
                        )
                    };
                    let ind = unsafe {
                        std::slice::from_raw_parts_mut(
                            (new_idx_ptr as *mut i32).add(row_off),
                            max_candidates,
                        )
                    };
                    checked_heap_push(pri, ind, d, candidate);
                } else {
                    let pri = unsafe {
                        std::slice::from_raw_parts_mut(
                            (old_pri_ptr as *mut f32).add(row_off),
                            max_candidates,
                        )
                    };
                    let ind = unsafe {
                        std::slice::from_raw_parts_mut(
                            (old_idx_ptr as *mut i32).add(row_off),
                            max_candidates,
                        )
                    };
                    checked_heap_push(pri, ind, d, candidate);
                }
            }
        }
    });

    // Phase 3: Mark matched edges as old (parallel per row)
    let graph_flags_mut_ptr = graph.flags.as_mut_ptr() as usize;
    let new_cand_ptr = new_candidate_indices.as_ptr() as usize;

    (0..n_vertices).into_par_iter().for_each(|i| {
        for j in 0..n_neighbors {
            let idx = unsafe { *(graph_idx_ptr as *const i32).add(i * n_neighbors + j) };
            if idx < 0 {
                continue;
            }
            for k in 0..max_candidates {
                if unsafe { *(new_cand_ptr as *const i32).add(i * max_candidates + k) } == idx {
                    unsafe {
                        *(graph_flags_mut_ptr as *mut u8).add(i * n_neighbors + j) = 0;
                    }
                    break;
                }
            }
        }
    });

    (new_candidate_indices, old_candidate_indices)
}

/// Process candidates: fused distance computation and heap update.
/// Phase 1: Each thread processes its own block of candidate rows, pushes into
///           its own heap rows directly, and buckets cross-block updates by target thread.
/// Phase 2: Each thread applies cross-block updates from all source threads' buckets.
fn process_candidates(
    data: &Array2<f32>,
    dist: DistanceFunc,
    heap: &mut NeighborHeap,
    new_candidates: &Array2<i32>,
    old_candidates: &Array2<i32>,
) -> usize {
    let n_vertices = new_candidates.nrows();
    let n_neighbors = heap.indices.ncols();
    let max_new = new_candidates.ncols();
    let max_old = old_candidates.ncols();
    let n_threads = rayon::current_num_threads();
    let block_size = (n_vertices + n_threads - 1) / n_threads;

    // SAFETY: Raw pointers cast to usize for the bucketed parallel scheme.
    // Phase 1: each thread writes only to heap rows in its own block; cross-block
    // updates are buffered. Phase 2: each thread applies its bucket to its own rows.
    // No two threads write to the same row in any phase.
    let dist_ptr = heap.distances.as_mut_ptr() as usize;
    let idx_ptr = heap.indices.as_mut_ptr() as usize;
    let flags_ptr = heap.flags.as_mut_ptr() as usize;

    // Phase 1: Each thread processes its own candidate rows
    // Returns (local_changes, per-target-thread buckets of cross-block updates)
    let results: Vec<(usize, Vec<Vec<(i32, i32, f32)>>)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let block_start = t * block_size;
            let block_end = (block_start + block_size).min(n_vertices);
            let mut local_changes = 0usize;
            let mut buckets: Vec<Vec<(i32, i32, f32)>> =
                (0..n_threads).map(|_| Vec::new()).collect();

            macro_rules! heap_row {
                ($row:expr) => {{
                    let off = $row * n_neighbors;
                    let dists = unsafe {
                        std::slice::from_raw_parts_mut((dist_ptr as *mut f32).add(off), n_neighbors)
                    };
                    let inds = unsafe {
                        std::slice::from_raw_parts_mut((idx_ptr as *mut i32).add(off), n_neighbors)
                    };
                    let flgs = unsafe {
                        std::slice::from_raw_parts_mut((flags_ptr as *mut u8).add(off), n_neighbors)
                    };
                    (dists, inds, flgs)
                }};
            }

            #[inline]
            fn get_thresh(dist_ptr: usize, row: usize, n_neighbors: usize) -> f32 {
                unsafe { *(dist_ptr as *const f32).add(row * n_neighbors) }
            }

            for i in block_start..block_end {
                for j in 0..max_new {
                    let p = new_candidates[[i, j]];
                    if p < 0 {
                        continue;
                    }
                    let pu = p as usize;
                    let data_p = data.row(pu);
                    let data_p_slice = data_p.as_slice().unwrap();

                    // New-New comparisons
                    for k in j..max_new {
                        let q = new_candidates[[i, k]];
                        if q < 0 {
                            continue;
                        }
                        let qu = q as usize;

                        let d = dist(data_p_slice, data.row(qu).as_slice().unwrap());

                        if d < get_thresh(dist_ptr, pu, n_neighbors) {
                            if pu >= block_start && pu < block_end {
                                let (dists, inds, flgs) = heap_row!(pu);
                                local_changes +=
                                    checked_flagged_heap_push(dists, inds, flgs, d, q, 1) as usize;
                            } else {
                                buckets[(pu / block_size).min(n_threads - 1)].push((p, q, d));
                            }
                        }
                        if d < get_thresh(dist_ptr, qu, n_neighbors) {
                            if qu >= block_start && qu < block_end {
                                let (dists, inds, flgs) = heap_row!(qu);
                                local_changes +=
                                    checked_flagged_heap_push(dists, inds, flgs, d, p, 1) as usize;
                            } else {
                                buckets[(qu / block_size).min(n_threads - 1)].push((q, p, d));
                            }
                        }
                    }

                    // New-Old comparisons
                    for k in 0..max_old {
                        let q = old_candidates[[i, k]];
                        if q < 0 {
                            continue;
                        }
                        let qu = q as usize;

                        let d = dist(data_p_slice, data.row(qu).as_slice().unwrap());

                        if d < get_thresh(dist_ptr, pu, n_neighbors) {
                            if pu >= block_start && pu < block_end {
                                let (dists, inds, flgs) = heap_row!(pu);
                                local_changes +=
                                    checked_flagged_heap_push(dists, inds, flgs, d, q, 1) as usize;
                            } else {
                                buckets[(pu / block_size).min(n_threads - 1)].push((p, q, d));
                            }
                        }
                        if d < get_thresh(dist_ptr, qu, n_neighbors) {
                            if qu >= block_start && qu < block_end {
                                let (dists, inds, flgs) = heap_row!(qu);
                                local_changes +=
                                    checked_flagged_heap_push(dists, inds, flgs, d, p, 1) as usize;
                            } else {
                                buckets[(qu / block_size).min(n_threads - 1)].push((q, p, d));
                            }
                        }
                    }
                }
            }
            (local_changes, buckets)
        })
        .collect();

    let mut total_changes: usize = results.iter().map(|(c, _)| c).sum();

    // Phase 2: Each thread applies cross-block updates from all source threads' buckets
    let cross_changes: Vec<usize> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let mut local_changes = 0usize;
            for source in 0..n_threads {
                for &(target, source_idx, d) in &results[source].1[t] {
                    let tu = target as usize;
                    let off = tu * n_neighbors;
                    let dists = unsafe {
                        std::slice::from_raw_parts_mut((dist_ptr as *mut f32).add(off), n_neighbors)
                    };
                    let inds = unsafe {
                        std::slice::from_raw_parts_mut((idx_ptr as *mut i32).add(off), n_neighbors)
                    };
                    let flgs = unsafe {
                        std::slice::from_raw_parts_mut((flags_ptr as *mut u8).add(off), n_neighbors)
                    };
                    local_changes +=
                        checked_flagged_heap_push(dists, inds, flgs, d, source_idx, 1) as usize;
                }
            }
            local_changes
        })
        .collect();

    total_changes += cross_changes.iter().sum::<usize>();
    total_changes
}

/// The internal NN-descent iteration loop.
pub async fn nn_descent_internal(
    heap: &mut NeighborHeap,
    data: &Array2<f32>,
    n_neighbors: usize,
    rng: &mut TauRng,
    max_candidates: usize,
    dist: DistanceFunc,
    n_iters: usize,
    delta: f32,
    logger: &mut Logger,
    #[cfg(feature = "gpu")] gpu_ctx: Option<&GpuContext>,
) {
    let n_vertices = data.nrows();

    for n in 0..n_iters {
        logger.log(&format!("{} / {}", n + 1, n_iters));

        let (new_candidates, old_candidates) = new_build_candidates(heap, max_candidates, rng);

        #[cfg(feature = "gpu")]
        let c = if let Some(gpu) = gpu_ctx {
            crate::nn_descent_gpu::process_candidates_gpu(
                gpu,
                heap,
                &new_candidates,
                &old_candidates,
            )
            .await
        } else {
            process_candidates(data, dist, heap, &new_candidates, &old_candidates)
        };

        #[cfg(not(feature = "gpu"))]
        let c = process_candidates(data, dist, heap, &new_candidates, &old_candidates);

        logger.stage_progress((n + 1) as f64 / n_iters as f64, None);

        if c as f32 <= delta * n_neighbors as f32 * n_vertices as f32 {
            logger.log(&format!(
                "Stopping threshold met -- exiting after {} iterations",
                n + 1
            ));
            return;
        }
    }
}

/// Top-level NN-descent: initialize heap, run iterations, sort result.
/// Returns (indices, distances) sorted by ascending distance.
pub async fn nn_descent(
    data: &Array2<f32>,
    n_neighbors: usize,
    rng: &mut TauRng,
    max_candidates: usize,
    dist: DistanceFunc,
    n_iters: usize,
    delta: f32,
    rp_tree_init: bool,
    leaf_array: Option<&Array2<i32>>,
    logger: &mut Logger,
    #[cfg(feature = "gpu")] gpu_ctx: Option<&GpuContext>,
) -> (Array2<i32>, Array2<f32>) {
    let mut heap = make_heap(data.nrows(), n_neighbors);

    if rp_tree_init {
        if let Some(leaves) = leaf_array {
            #[cfg(feature = "gpu")]
            if let Some(gpu) = gpu_ctx {
                crate::nn_descent_gpu::init_rp_tree_gpu(gpu, &mut heap, leaves).await;
            } else {
                init_rp_tree(data, dist, &mut heap, leaves);
            }

            #[cfg(not(feature = "gpu"))]
            init_rp_tree(data, dist, &mut heap, leaves);
        }
    }

    init_random(n_neighbors, data, &mut heap, dist, rng);

    logger.push_stage_with_message(
        STAGE_NN_DESCENT,
        &format!("NN descent for {} iterations", n_iters),
    );
    nn_descent_internal(
        &mut heap,
        data,
        n_neighbors,
        rng,
        max_candidates,
        dist,
        n_iters,
        delta,
        logger,
        #[cfg(feature = "gpu")]
        gpu_ctx,
    )
    .await;
    logger.pop_stage();

    logger.push_stage(STAGE_SORTING);
    deheap_sort(&mut heap.indices, &mut heap.distances);

    // Ensure self-neighbor is always at position 0 for each point,
    // consistent with pynndescent behavior.
    let n_pts = data.nrows();
    let n_cols = heap.indices.ncols();
    for i in 0..n_pts {
        let self_idx = i as i32;
        // Find if self is already in the neighbor list
        let self_pos = (0..n_cols).find(|&j| heap.indices[[i, j]] == self_idx);

        if self_pos == Some(0) {
            // Already at position 0, ensure distance is 0
            heap.distances[[i, 0]] = 0.0;
            continue;
        }

        // Shift end: if self was found at pos, shift [0..pos] right; otherwise shift all right (drop last)
        let shift_end = self_pos.unwrap_or(n_cols - 1);
        for j in (1..=shift_end).rev() {
            heap.indices[[i, j]] = heap.indices[[i, j - 1]];
            heap.distances[[i, j]] = heap.distances[[i, j - 1]];
        }
        heap.indices[[i, 0]] = self_idx;
        heap.distances[[i, 0]] = 0.0;
    }
    logger.pop_stage();

    (heap.indices, heap.distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;
    use crate::rp_trees;

    fn make_test_data(n: usize, dim: usize) -> Array2<f32> {
        // Simple deterministic test data
        let mut rng = TauRng::new(189212);
        let mut data = Array2::zeros((n, dim));
        for i in 0..n {
            for j in 0..dim {
                data[[i, j]] = rng.tau_rand();
            }
        }
        data
    }

    #[test]
    fn test_nn_descent_runs() {
        let data = make_test_data(100, 5);
        let mut rng = TauRng::new(42);
        let mut logger = Logger::new(false, None, &[]);
        let (indices, distances) = pollster::block_on(nn_descent(
            &data,
            10,
            &mut rng,
            20,
            distance::euclidean,
            5,
            0.001,
            false,
            None,
            &mut logger,
            #[cfg(feature = "gpu")]
            None,
        ));

        assert_eq!(indices.nrows(), 100);
        assert_eq!(indices.ncols(), 10);
        // Check that distances are sorted ascending for each row
        for i in 0..100 {
            for j in 1..10 {
                assert!(distances[[i, j]] >= distances[[i, j - 1]]);
            }
        }
    }

    #[test]
    fn test_nn_descent_with_rp_tree() {
        let data = make_test_data(200, 5);
        let mut rng = TauRng::new(42);
        let rng_state = [12345i64, 67890, 11111];
        let forest = rp_trees::make_forest(&data, 10, 3, None, &rng_state, false, 200);
        let leaf_array = rp_trees::rptree_leaf_array(&forest);

        let mut logger = Logger::new(false, None, &[]);
        let (indices, _distances) = pollster::block_on(nn_descent(
            &data,
            10,
            &mut rng,
            20,
            distance::euclidean,
            10,
            0.001,
            true,
            Some(&leaf_array),
            &mut logger,
            #[cfg(feature = "gpu")]
            None,
        ));

        assert_eq!(indices.nrows(), 200);
        assert_eq!(indices.ncols(), 10);
    }
}
