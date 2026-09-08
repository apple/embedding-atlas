use crate::gpu::GpuContext;
use crate::heap::{checked_flagged_heap_push, NeighborHeap};
/// GPU-accelerated variants of the NN-descent hot paths.
///
/// These functions replace `init_rp_tree` and `process_candidates` when a GPU
/// is available, offloading distance computation to a compute shader while
/// keeping heap management on the CPU.
use ndarray::Array2;
use rayon::prelude::*;

/// GPU-accelerated RP tree initialization.
///
/// Collects all pairwise indices from leaf co-occurrences, dispatches them
/// to the GPU for distance computation, then pushes results into the heap.
pub async fn init_rp_tree_gpu(gpu: &GpuContext, heap: &mut NeighborHeap, leaf_array: &Array2<i32>) {
    let n_leaves = leaf_array.nrows();
    let leaf_width = leaf_array.ncols();

    // Collect pairs in parallel across leaves
    let pairs: Vec<u32> = (0..n_leaves)
        .into_par_iter()
        .flat_map(|leaf_idx| {
            let mut local_pairs = Vec::new();
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
                    local_pairs.push(p as u32);
                    local_pairs.push(q as u32);
                }
            }
            local_pairs
        })
        .collect();

    if pairs.is_empty() {
        return;
    }

    // Dispatch to GPU — no threshold filtering needed (heap is empty)
    let results = gpu.compute_all_distances(&pairs).await;

    // Push results into heap on CPU
    for (target, source, d) in &results {
        let tu = *target as usize;
        checked_flagged_heap_push(
            heap.distances.row_mut(tu).as_slice_mut().unwrap(),
            heap.indices.row_mut(tu).as_slice_mut().unwrap(),
            heap.flags.row_mut(tu).as_slice_mut().unwrap(),
            *d,
            *source as i32,
            1,
        );
    }
}

/// GPU-accelerated candidate processing.
///
/// Snapshots heap thresholds, collects candidate pairs, dispatches distance
/// computation with threshold filtering to the GPU, then applies results to
/// the heap on the CPU.
///
/// Returns the number of heap changes (same semantics as CPU `process_candidates`).
pub async fn process_candidates_gpu(
    gpu: &GpuContext,
    heap: &mut NeighborHeap,
    new_candidates: &Array2<i32>,
    old_candidates: &Array2<i32>,
) -> usize {
    let n_vertices = new_candidates.nrows();
    let max_new = new_candidates.ncols();
    let max_old = old_candidates.ncols();

    // Snapshot heap thresholds (max distance = root of each row's max-heap)
    let thresholds: Vec<f32> = (0..n_vertices).map(|i| heap.distances[[i, 0]]).collect();

    // Collect candidate pairs in parallel across vertices using fold+reduce
    // to minimize allocation overhead (each thread builds one Vec, then concatenate)
    let pairs: Vec<u32> = (0..n_vertices)
        .into_par_iter()
        .fold(Vec::new, |mut acc, i| {
            for j in 0..max_new {
                let p = new_candidates[[i, j]];
                if p < 0 {
                    continue;
                }
                for k in (j + 1)..max_new {
                    let q = new_candidates[[i, k]];
                    if q < 0 {
                        continue;
                    }
                    acc.push(p as u32);
                    acc.push(q as u32);
                }
                for k in 0..max_old {
                    let q = old_candidates[[i, k]];
                    if q < 0 {
                        continue;
                    }
                    acc.push(p as u32);
                    acc.push(q as u32);
                }
            }
            acc
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    if pairs.is_empty() {
        return 0;
    }

    // Dispatch to GPU with threshold filtering
    let results = gpu.compute_filtered_distances(&pairs, &thresholds).await;

    // Apply results to heap
    let mut total_changes = 0usize;
    for (target, source, d) in &results {
        let tu = *target as usize;
        total_changes += checked_flagged_heap_push(
            heap.distances.row_mut(tu).as_slice_mut().unwrap(),
            heap.indices.row_mut(tu).as_slice_mut().unwrap(),
            heap.flags.row_mut(tu).as_slice_mut().unwrap(),
            *d,
            *source as i32,
            1,
        ) as usize;
    }

    total_changes
}
