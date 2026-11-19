use crate::gpu::{result_triples, GpuContext};
use crate::heap::{checked_flagged_heap_push, NeighborHeap};
/// GPU-accelerated variants of the NN-descent hot paths.
///
/// These functions replace `init_rp_tree` and `process_candidates` when a GPU
/// is available, offloading distance computation to a compute shader while
/// keeping heap management on the CPU.
use ndarray::Array2;
use rayon::prelude::*;
use std::ops::Range;

/// Upper bound on the pairs that one block holds on the host (8 bytes each).
const MAX_PAIRS_PER_BLOCK: usize = 8 * 1024 * 1024;

fn pair_budget(gpu: &GpuContext) -> usize {
    MAX_PAIRS_PER_BLOCK.min(gpu.max_pairs_per_chunk())
}

/// Split `0..counts.len()` into consecutive ranges whose counts sum to at most
/// `budget`. A single item above the budget gets a range of its own.
fn block_ranges(counts: &[usize], budget: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut total = 0;
    for (i, &count) in counts.iter().enumerate() {
        if total + count > budget && i > start {
            ranges.push(start..i);
            start = i;
            total = 0;
        }
        total += count;
    }
    if start < counts.len() {
        ranges.push(start..counts.len());
    }
    ranges
}

fn count_valid(row: ndarray::ArrayView1<i32>) -> usize {
    row.iter().filter(|&&p| p >= 0).count()
}

fn push_results(heap: &mut NeighborHeap, results: &[u32]) -> usize {
    let mut changes = 0usize;
    for (target, source, d) in result_triples(results) {
        let tu = target as usize;
        changes += checked_flagged_heap_push(
            heap.distances.row_mut(tu).as_slice_mut().unwrap(),
            heap.indices.row_mut(tu).as_slice_mut().unwrap(),
            heap.flags.row_mut(tu).as_slice_mut().unwrap(),
            d,
            source as i32,
            1,
        ) as usize;
    }
    changes
}

/// Max distance in each point's heap (the root of its max-heap).
fn heap_thresholds(heap: &NeighborHeap) -> Vec<f32> {
    heap.distances.column(0).to_vec()
}

/// Split `dst` into one slice of `counts[i] * 2` entries per item `i` of `items`
/// and call `write(i, slice)` for each, in parallel.
fn fill_pairs_per_item(
    dst: &mut [u32],
    items: Range<usize>,
    counts: &[usize],
    write: impl Fn(usize, &mut [u32]) + Sync,
) {
    let mut rest = dst;
    let mut parts = Vec::with_capacity(items.len());
    for i in items {
        let (head, tail) = rest.split_at_mut(counts[i] * 2);
        parts.push((i, head));
        rest = tail;
    }
    parts.into_par_iter().for_each(|(i, out)| write(i, out));
}

/// GPU-accelerated RP tree initialization.
///
/// Collects pairwise indices from leaf co-occurrences in blocks of leaves,
/// dispatches each block to the GPU for distance computation, then pushes
/// results into the heap. Each block filters against the heap thresholds
/// left by the previous blocks.
pub async fn init_rp_tree_gpu(gpu: &GpuContext, heap: &mut NeighborHeap, leaf_array: &Array2<i32>) {
    let n_leaves = leaf_array.nrows();
    let pairs_per_leaf: Vec<usize> = (0..n_leaves)
        .map(|l| {
            let m = count_valid(leaf_array.row(l));
            m * m.saturating_sub(1) / 2
        })
        .collect();

    for block in block_ranges(&pairs_per_leaf, pair_budget(gpu)) {
        let n_pairs: usize = pairs_per_leaf[block.clone()].iter().sum();
        let thresholds = heap_thresholds(heap);
        gpu.compute_filtered_block(
            n_pairs,
            &thresholds,
            |dst| {
                fill_pairs_per_item(dst, block, &pairs_per_leaf, |leaf, out| {
                    let members: Vec<u32> = leaf_array
                        .row(leaf)
                        .iter()
                        .take_while(|&&p| p >= 0)
                        .map(|&p| p as u32)
                        .collect();
                    let mut w = 0;
                    for (i, &p) in members.iter().enumerate() {
                        for &q in &members[i + 1..] {
                            out[w] = p;
                            out[w + 1] = q;
                            w += 2;
                        }
                    }
                })
            },
            |results| {
                push_results(heap, results);
            },
        )
        .await;
    }
}

/// GPU-accelerated candidate processing.
///
/// Snapshots heap thresholds, then walks the vertices in blocks: each block
/// collects its candidate pairs, dispatches distance computation with threshold
/// filtering to the GPU, and applies the results to the heap on the CPU before
/// the next block starts. Host memory for pairs is bounded by one block.
///
/// Returns the number of heap changes (same semantics as CPU `process_candidates`).
pub async fn process_candidates_gpu(
    gpu: &GpuContext,
    heap: &mut NeighborHeap,
    new_candidates: &Array2<i32>,
    old_candidates: &Array2<i32>,
) -> usize {
    process_candidates_gpu_in_blocks(gpu, heap, new_candidates, old_candidates, pair_budget(gpu))
        .await
}

/// `process_candidates_gpu` with an explicit upper bound on the pairs per block.
pub async fn process_candidates_gpu_in_blocks(
    gpu: &GpuContext,
    heap: &mut NeighborHeap,
    new_candidates: &Array2<i32>,
    old_candidates: &Array2<i32>,
    max_pairs_per_block: usize,
) -> usize {
    let n_vertices = new_candidates.nrows();
    let max_new = new_candidates.ncols();

    // Snapshot heap thresholds (max distance = root of each row's max-heap).
    // Blocks do not refresh them, so the result does not depend on the block size.
    let thresholds = heap_thresholds(heap);

    let pairs_per_vertex: Vec<usize> = (0..n_vertices)
        .map(|i| {
            let new = count_valid(new_candidates.row(i));
            let old = count_valid(old_candidates.row(i));
            new * new.saturating_sub(1) / 2 + new * old
        })
        .collect();

    let mut total_changes = 0usize;
    for block in block_ranges(&pairs_per_vertex, max_pairs_per_block) {
        let n_pairs: usize = pairs_per_vertex[block.clone()].iter().sum();
        gpu.compute_filtered_block(
            n_pairs,
            &thresholds,
            |dst| {
                fill_pairs_per_item(dst, block, &pairs_per_vertex, |i, out| {
                    let new_row = new_candidates.row(i);
                    let old_row = old_candidates.row(i);
                    let mut w = 0;
                    for j in 0..max_new {
                        let p = new_row[j];
                        if p < 0 {
                            continue;
                        }
                        let later_new = new_row.iter().skip(j + 1);
                        for &q in later_new.chain(old_row.iter()) {
                            if q >= 0 {
                                out[w] = p as u32;
                                out[w + 1] = q as u32;
                                w += 2;
                            }
                        }
                    }
                })
            },
            |results| {
                total_changes += push_results(heap, results);
            },
        )
        .await;
    }

    total_changes
}
