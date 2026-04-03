use ndarray::Array2;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
/// Search/query functionality for the NN-descent index.
use std::collections::BinaryHeap;

use crate::distance::DistanceFunc;
use crate::heap::simple_heap_push;
use crate::rng::TauRng;
use crate::rp_trees::{search_flat_tree, FlatTree};
use crate::utils::VisitedTable;

/// CSR (Compressed Sparse Row) graph representation for the search graph.
#[derive(Clone)]
pub struct CsrGraph {
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
}

impl CsrGraph {
    pub fn new(n_vertices: usize) -> Self {
        CsrGraph {
            indptr: vec![0i32; n_vertices + 1],
            indices: Vec::new(),
        }
    }

    /// Build CSR from COO (coordinate) format
    pub fn from_coo(
        n_vertices: usize,
        rows: &[i32],
        cols: &[i32],
        data: &[f32],
    ) -> (Self, Vec<f32>) {
        // Count entries per row
        let mut row_counts = vec![0usize; n_vertices];
        for &r in rows {
            if r >= 0 && (r as usize) < n_vertices {
                row_counts[r as usize] += 1;
            }
        }

        // Build indptr
        let mut indptr = vec![0i32; n_vertices + 1];
        for i in 0..n_vertices {
            indptr[i + 1] = indptr[i] + row_counts[i] as i32;
        }

        // Fill indices and data
        let nnz = indptr[n_vertices] as usize;
        let mut indices = vec![0i32; nnz];
        let mut csr_data = vec![0.0f32; nnz];
        let mut offsets = vec![0usize; n_vertices];

        for k in 0..rows.len() {
            let r = rows[k] as usize;
            if r < n_vertices {
                let pos = indptr[r] as usize + offsets[r];
                indices[pos] = cols[k];
                csr_data[pos] = data[k];
                offsets[r] += 1;
            }
        }

        (CsrGraph { indptr, indices }, csr_data)
    }

    /// Get the neighbor indices for a vertex.
    pub fn neighbors(&self, vertex: i32) -> &[i32] {
        let v = vertex as usize;
        let start = self.indptr[v] as usize;
        let end = self.indptr[v + 1] as usize;
        &self.indices[start..end]
    }
}

/// Search for k nearest neighbors of a single query point using greedy graph traversal.
fn search_one(
    query: &[f32],
    data: &Array2<f32>,
    search_graph: &CsrGraph,
    search_forest: &[FlatTree],
    dist_fn: DistanceFunc,
    k: usize,
    epsilon: f32,
    n_neighbors: usize,
    min_distance: f32,
    rng: &mut TauRng,
) -> (Vec<i32>, Vec<f32>) {
    let n = data.nrows();
    let mut visited = VisitedTable::new(n);

    // Result heap (max-heap, root = farthest)
    let mut heap_priorities = vec![f32::INFINITY; k];
    let mut heap_indices = vec![-1i32; k];

    // Seed set (min-heap for frontier expansion)
    let mut seed_set: BinaryHeap<Reverse<(OrderedFloat<f32>, i32)>> = BinaryHeap::new();

    // Initialize from tree search
    let mut n_initial = 0;
    if !search_forest.is_empty() {
        let (start, end) = search_flat_tree(&search_forest[0], query, rng);
        let tree_indices = &search_forest[0].indices;
        for j in start..end {
            if j < tree_indices.len() {
                let candidate = tree_indices[j];
                if candidate >= 0 && (candidate as usize) < n {
                    let d = dist_fn(query, data.row(candidate as usize).as_slice().unwrap());
                    simple_heap_push(&mut heap_priorities, &mut heap_indices, d, candidate);
                    seed_set.push(Reverse((OrderedFloat(d), candidate)));
                    visited.mark_visited(candidate);
                    n_initial += 1;
                }
            }
        }
    }

    // Add random samples if needed
    let n_random_needed = k.min(n_neighbors) as i64 - n_initial as i64;
    if n_random_needed > 0 {
        for _ in 0..n_random_needed {
            let candidate = (rng.tau_rand_int().unsigned_abs() as usize % n) as i32;
            if !visited.check_and_mark_visited(candidate) {
                let d = dist_fn(query, data.row(candidate as usize).as_slice().unwrap());
                simple_heap_push(&mut heap_priorities, &mut heap_indices, d, candidate);
                seed_set.push(Reverse((OrderedFloat(d), candidate)));
            }
        }
    }

    // Greedy graph search
    let mut distance_bound = heap_priorities[0] + epsilon * (heap_priorities[0] - min_distance);

    while let Some(Reverse((OrderedFloat(d_vertex), vertex))) = seed_set.pop() {
        if d_vertex >= distance_bound {
            break;
        }

        // Expand neighbors
        let neighbors = search_graph.neighbors(vertex);
        for &candidate in neighbors {
            if candidate >= 0 && !visited.check_and_mark_visited(candidate) {
                let d = dist_fn(query, data.row(candidate as usize).as_slice().unwrap());
                if d < distance_bound {
                    simple_heap_push(&mut heap_priorities, &mut heap_indices, d, candidate);
                    seed_set.push(Reverse((OrderedFloat(d), candidate)));
                    // Update bound
                    distance_bound =
                        heap_priorities[0] + epsilon * (heap_priorities[0] - min_distance);
                }
            }
        }
    }

    (heap_indices, heap_priorities)
}

/// Search for k nearest neighbors of multiple query points (parallel).
pub fn batch_search(
    query_points: &Array2<f32>,
    data: &Array2<f32>,
    search_graph: &CsrGraph,
    search_forest: &[FlatTree],
    dist_fn: DistanceFunc,
    k: usize,
    epsilon: f32,
    n_neighbors: usize,
    min_distance: f32,
    rng_state: &[i64; 3],
) -> (Array2<i32>, Array2<f32>) {
    let n_queries = query_points.nrows();
    let mut result_indices = Array2::from_elem((n_queries, k), -1i32);
    let mut result_distances = Array2::from_elem((n_queries, k), f32::INFINITY);

    // Process each query (could parallelize with rayon for large batches)
    for i in 0..n_queries {
        let mut rng = TauRng::from_state([
            rng_state[0].wrapping_add(i as i64),
            rng_state[1].wrapping_add(i as i64),
            rng_state[2].wrapping_add(i as i64),
        ]);

        let query = query_points.row(i);
        let (indices, distances) = search_one(
            query.as_slice().unwrap(),
            data,
            search_graph,
            search_forest,
            dist_fn,
            k,
            epsilon,
            n_neighbors,
            min_distance,
            &mut rng,
        );

        // Sort by ascending distance using deheap_sort logic
        let mut pairs: Vec<(f32, i32)> = distances
            .iter()
            .zip(indices.iter())
            .map(|(&d, &i)| (d, i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for j in 0..k {
            result_indices[[i, j]] = pairs[j].1;
            result_distances[[i, j]] = pairs[j].0;
        }
    }

    (result_indices, result_distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_graph() {
        let rows = vec![0i32, 0, 1, 1, 2];
        let cols = vec![1i32, 2, 0, 2, 0];
        let data = vec![1.0f32, 2.0, 1.0, 3.0, 2.0];

        let (graph, _) = CsrGraph::from_coo(3, &rows, &cols, &data);
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.neighbors(1), &[0, 2]);
        assert_eq!(graph.neighbors(2), &[0]);
    }
}
