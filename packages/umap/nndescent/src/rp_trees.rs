/// Random Projection Trees for NN-descent initialization and search.
use ndarray::Array2;
use rayon::prelude::*;

use crate::rng::TauRng;

const EPS: f32 = 1e-8;

/// Dot product of two slices. Written as a simple loop so the compiler
/// can auto-vectorize it (LLVM generates SIMD for contiguous slice iteration).
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// A flattened RP tree for efficient storage and query traversal.
#[derive(Clone)]
pub struct FlatTree {
    /// Hyperplane vectors for each internal node, shape (n_nodes, dim)
    pub hyperplanes: Array2<f32>,
    /// Offset values for each node
    pub offsets: Vec<f32>,
    /// Children indices: (left, right). Negative values indicate leaf bounds into `indices`.
    pub children: Array2<i32>,
    /// Flat array of point indices (leaves store ranges into this)
    pub indices: Vec<i32>,
    /// Maximum number of points in any leaf
    pub leaf_size: usize,
}

// ===== Tree construction (recursive node lists) =====

struct TreeBuilder {
    hyperplanes: Vec<Vec<f32>>,
    offsets: Vec<f32>,
    children: Vec<(i32, i32)>,
    point_indices: Vec<Vec<i32>>,
}

impl TreeBuilder {
    fn new() -> Self {
        TreeBuilder {
            hyperplanes: Vec::new(),
            offsets: Vec::new(),
            children: Vec::new(),
            point_indices: Vec::new(),
        }
    }

    fn add_leaf(&mut self, indices: Vec<i32>) -> usize {
        let idx = self.hyperplanes.len();
        self.hyperplanes.push(vec![-1.0]);
        self.offsets.push(f32::NEG_INFINITY);
        self.children.push((-1, -1));
        self.point_indices.push(indices);
        idx
    }

    fn add_internal(
        &mut self,
        hyperplane: Vec<f32>,
        offset: f32,
        left: usize,
        right: usize,
    ) -> usize {
        let idx = self.hyperplanes.len();
        self.hyperplanes.push(hyperplane);
        self.offsets.push(offset);
        self.children.push((left as i32, right as i32));
        self.point_indices.push(vec![-1]);
        idx
    }
}

/// Angular random projection split: picks 2 random points, computes
/// normalized difference as hyperplane, splits by sign of projection.
fn angular_random_projection_split(
    data: &Array2<f32>,
    indices: &[i32],
    rng: &mut TauRng,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, f32) {
    let dim = data.ncols();
    let n = indices.len();

    let left_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    let mut right_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    if left_idx == right_idx {
        right_idx = (right_idx + 1) % n;
    }

    let left_point = indices[left_idx] as usize;
    let right_point = indices[right_idx] as usize;
    let left_row = data.row(left_point);
    let left_row = left_row.as_slice().unwrap();
    let right_row = data.row(right_point);
    let right_row = right_row.as_slice().unwrap();

    // Compute norms
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for d in 0..dim {
        left_norm += left_row[d] * left_row[d];
        right_norm += right_row[d] * right_row[d];
    }
    left_norm = left_norm.sqrt().max(EPS);
    right_norm = right_norm.sqrt().max(EPS);

    // Hyperplane = normalized left - normalized right
    let inv_left = 1.0 / left_norm;
    let inv_right = 1.0 / right_norm;
    let mut hyperplane = vec![0.0f32; dim];
    for d in 0..dim {
        hyperplane[d] = left_row[d] * inv_left - right_row[d] * inv_right;
    }

    // Normalize hyperplane
    let mut hp_norm = 0.0f32;
    for d in 0..dim {
        hp_norm += hyperplane[d] * hyperplane[d];
    }
    let inv_hp_norm = 1.0 / hp_norm.sqrt().max(EPS);
    for d in 0..dim {
        hyperplane[d] *= inv_hp_norm;
    }

    // Split points by margin (dot product with hyperplane)
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        let row = data.row(idx as usize);
        let row = row.as_slice().unwrap();
        let margin = dot_product(&hyperplane, row);

        if margin.abs() < EPS {
            if rng.tau_rand_int().unsigned_abs() % 2 == 0 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        } else if margin > 0.0 {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // If all went to one side, randomly split
    if left_indices.is_empty() || right_indices.is_empty() {
        left_indices.clear();
        right_indices.clear();
        for &idx in indices {
            if rng.tau_rand_int().unsigned_abs() % 2 == 0 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        // Guarantee both sides have at least one point
        if left_indices.is_empty() {
            left_indices.push(right_indices.pop().unwrap());
        }
        if right_indices.is_empty() {
            right_indices.push(left_indices.pop().unwrap());
        }
    }

    (left_indices, right_indices, hyperplane, 0.0)
}

/// Euclidean random projection split: picks 2 random points, computes
/// their difference as hyperplane and midpoint offset.
fn euclidean_random_projection_split(
    data: &Array2<f32>,
    indices: &[i32],
    rng: &mut TauRng,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, f32) {
    let dim = data.ncols();
    let n = indices.len();

    let left_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    let mut right_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    if left_idx == right_idx {
        right_idx = (right_idx + 1) % n;
    }

    let left_point = indices[left_idx] as usize;
    let right_point = indices[right_idx] as usize;
    let left_row = data.row(left_point);
    let left_row = left_row.as_slice().unwrap();
    let right_row = data.row(right_point);
    let right_row = right_row.as_slice().unwrap();

    // Hyperplane = left - right
    let mut hyperplane = vec![0.0f32; dim];
    let mut offset = 0.0f32;
    for d in 0..dim {
        hyperplane[d] = left_row[d] - right_row[d];
        offset -= hyperplane[d] * (left_row[d] + right_row[d]) * 0.5;
    }

    // Split points by margin
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        let row = data.row(idx as usize);
        let row = row.as_slice().unwrap();
        let margin = dot_product(&hyperplane, row) + offset;

        if margin.abs() < EPS {
            if rng.tau_rand_int().unsigned_abs() % 2 == 0 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        } else if margin > 0.0 {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // If all went to one side, randomly split
    if left_indices.is_empty() || right_indices.is_empty() {
        left_indices.clear();
        right_indices.clear();
        for &idx in indices {
            if rng.tau_rand_int().unsigned_abs() % 2 == 0 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        if left_indices.is_empty() {
            left_indices.push(right_indices.pop().unwrap());
        }
        if right_indices.is_empty() {
            right_indices.push(left_indices.pop().unwrap());
        }
    }

    (left_indices, right_indices, hyperplane, offset)
}

/// Recursively build an angular RP tree (post-order node storage).
fn make_angular_tree(
    data: &Array2<f32>,
    indices: Vec<i32>,
    rng: &mut TauRng,
    leaf_size: usize,
    max_depth: usize,
    builder: &mut TreeBuilder,
) -> usize {
    if indices.len() <= leaf_size || max_depth == 0 {
        return builder.add_leaf(indices);
    }

    let (left_indices, right_indices, hyperplane, offset) =
        angular_random_projection_split(data, &indices, rng);

    let left_node = make_angular_tree(data, left_indices, rng, leaf_size, max_depth - 1, builder);
    let right_node = make_angular_tree(data, right_indices, rng, leaf_size, max_depth - 1, builder);

    builder.add_internal(hyperplane, offset, left_node, right_node)
}

/// Recursively build a euclidean RP tree.
fn make_euclidean_tree(
    data: &Array2<f32>,
    indices: Vec<i32>,
    rng: &mut TauRng,
    leaf_size: usize,
    max_depth: usize,
    builder: &mut TreeBuilder,
) -> usize {
    if indices.len() <= leaf_size || max_depth == 0 {
        return builder.add_leaf(indices);
    }

    let (left_indices, right_indices, hyperplane, offset) =
        euclidean_random_projection_split(data, &indices, rng);

    let left_node = make_euclidean_tree(data, left_indices, rng, leaf_size, max_depth - 1, builder);
    let right_node =
        make_euclidean_tree(data, right_indices, rng, leaf_size, max_depth - 1, builder);

    builder.add_internal(hyperplane, offset, left_node, right_node)
}

/// Build a single dense RP tree.
pub fn make_dense_tree(
    data: &Array2<f32>,
    rng: &mut TauRng,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
) -> FlatTree {
    let n = data.nrows();
    let indices: Vec<i32> = (0..n as i32).collect();

    let mut builder = TreeBuilder::new();

    let root = if angular {
        make_angular_tree(data, indices, rng, leaf_size, max_depth, &mut builder)
    } else {
        make_euclidean_tree(data, indices, rng, leaf_size, max_depth, &mut builder)
    };

    convert_builder_to_flat_tree(&builder, root, data.nrows(), data.ncols())
}

/// Convert the builder's post-order node list to a FlatTree with pre-order layout.
fn convert_builder_to_flat_tree(
    builder: &TreeBuilder,
    root: usize,
    data_size: usize,
    data_dim: usize,
) -> FlatTree {
    let n_nodes = builder.hyperplanes.len();

    let mut hyperplanes = Array2::zeros((n_nodes, data_dim));
    let mut offsets = vec![0.0f32; n_nodes];
    let mut children = Array2::from_elem((n_nodes, 2), -1i32);
    let mut flat_indices = vec![-1i32; data_size];

    let mut node_num = 0i32;
    let mut leaf_start = 0i32;
    let mut max_leaf_size = 0usize;

    fn recurse(
        builder: &TreeBuilder,
        tree_node: usize,
        node_num: &mut i32,
        leaf_start: &mut i32,
        hyperplanes: &mut Array2<f32>,
        offsets: &mut Vec<f32>,
        children: &mut Array2<i32>,
        flat_indices: &mut Vec<i32>,
        max_leaf_size: &mut usize,
        data_dim: usize,
    ) {
        let current = *node_num as usize;

        if builder.children[tree_node].0 < 0 {
            // Leaf node
            let pts = &builder.point_indices[tree_node];
            let start = *leaf_start;
            let end = start + pts.len() as i32;
            children[[current, 0]] = -start;
            children[[current, 1]] = -end;
            for (k, &p) in pts.iter().enumerate() {
                if (start as usize + k) < flat_indices.len() {
                    flat_indices[start as usize + k] = p;
                }
            }
            *max_leaf_size = (*max_leaf_size).max(pts.len());
            *leaf_start = end;
        } else {
            // Internal node
            let hp = &builder.hyperplanes[tree_node];
            for d in 0..hp.len().min(data_dim) {
                hyperplanes[[current, d]] = hp[d];
            }
            offsets[current] = builder.offsets[tree_node];

            // Left child
            let left_tree_node = builder.children[tree_node].0 as usize;
            *node_num += 1;
            let left_node = *node_num;
            children[[current, 0]] = left_node;
            recurse(
                builder,
                left_tree_node,
                node_num,
                leaf_start,
                hyperplanes,
                offsets,
                children,
                flat_indices,
                max_leaf_size,
                data_dim,
            );

            // Right child
            let right_tree_node = builder.children[tree_node].1 as usize;
            *node_num += 1;
            let right_node = *node_num;
            children[[current, 1]] = right_node;
            recurse(
                builder,
                right_tree_node,
                node_num,
                leaf_start,
                hyperplanes,
                offsets,
                children,
                flat_indices,
                max_leaf_size,
                data_dim,
            );
        }
    }

    recurse(
        builder,
        root,
        &mut node_num,
        &mut leaf_start,
        &mut hyperplanes,
        &mut offsets,
        &mut children,
        &mut flat_indices,
        &mut max_leaf_size,
        data_dim,
    );

    FlatTree {
        hyperplanes,
        offsets,
        children,
        indices: flat_indices,
        leaf_size: max_leaf_size,
    }
}

/// Build an RP forest of n_trees trees.
pub fn make_forest(
    data: &Array2<f32>,
    n_neighbors: usize,
    n_trees: usize,
    leaf_size: Option<usize>,
    rng_state: &[i64; 3],
    angular: bool,
    max_depth: usize,
) -> Vec<FlatTree> {
    let leaf_size = leaf_size.unwrap_or_else(|| (5 * n_neighbors).max(60).min(256));

    // Generate independent RNG states for each tree
    let mut master_rng = TauRng::from_state(*rng_state);
    let rng_states: Vec<[i64; 3]> = (0..n_trees)
        .map(|_| {
            let s0 = master_rng.tau_rand_int() as i64;
            let s1 = master_rng.tau_rand_int() as i64;
            let s2 = master_rng.tau_rand_int() as i64;
            [
                s0.wrapping_add(0xFFFF),
                s1.wrapping_add(0xFFFF),
                s2.wrapping_add(0xFFFF),
            ]
        })
        .collect();

    rng_states
        .into_par_iter()
        .map(|state| {
            let mut rng = TauRng::from_state(state);
            make_dense_tree(data, &mut rng, leaf_size, angular, max_depth)
        })
        .collect()
}

/// Extract all leaf arrays from a forest as a 2D array.
/// Each row is a leaf's point indices, padded with -1.
pub fn rptree_leaf_array(forest: &[FlatTree]) -> Array2<i32> {
    let mut all_leaves: Vec<Vec<i32>> = Vec::new();
    let mut max_leaf_size = 0usize;

    for tree in forest {
        let n_nodes = tree.children.nrows();
        for i in 0..n_nodes {
            let left = tree.children[[i, 0]];
            let right = tree.children[[i, 1]];
            // Leaf nodes have children <= 0 (negated indices into flat array)
            if left <= 0 && right <= 0 && (left != 0 || right != 0) {
                let start = (-left) as usize;
                let end = (-right) as usize;
                if end > start && start < tree.indices.len() {
                    let end = end.min(tree.indices.len());
                    let leaf: Vec<i32> = tree.indices[start..end].to_vec();
                    max_leaf_size = max_leaf_size.max(leaf.len());
                    all_leaves.push(leaf);
                }
            }
        }
    }

    if all_leaves.is_empty() {
        return Array2::from_elem((1, 1), -1i32);
    }

    let n_leaves = all_leaves.len();
    let mut result = Array2::from_elem((n_leaves, max_leaf_size), -1i32);
    for (i, leaf) in all_leaves.iter().enumerate() {
        for (j, &idx) in leaf.iter().enumerate() {
            result[[i, j]] = idx;
        }
    }

    result
}

/// Select which side of a hyperplane a point falls on (for query-time tree traversal).
/// Returns 0 for left, 1 for right.
pub fn select_side(hyperplane: &[f32], offset: f32, point: &[f32], rng: &mut TauRng) -> usize {
    let margin = dot_product(hyperplane, &point[..hyperplane.len()]) + offset;

    if margin.abs() < EPS {
        (rng.tau_rand_int().unsigned_abs() % 2) as usize
    } else if margin > 0.0 {
        0
    } else {
        1
    }
}

// ===== GPU-accelerated RP tree construction (BFS with batched projections) =====

/// Minimum total points across all nodes to use batched GPU projection.
#[cfg(feature = "gpu")]
const GPU_BATCH_THRESHOLD: usize = 4096;

/// Compute a hyperplane for an angular split between two random points.
#[cfg(feature = "gpu")]
fn compute_angular_hyperplane(
    data: &Array2<f32>,
    indices: &[i32],
    rng: &mut TauRng,
) -> (Vec<f32>, f32) {
    let dim = data.ncols();
    let n = indices.len();

    let left_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    let mut right_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    if left_idx == right_idx {
        right_idx = (right_idx + 1) % n;
    }

    let left_row = data.row(indices[left_idx] as usize);
    let left_row = left_row.as_slice().unwrap();
    let right_row = data.row(indices[right_idx] as usize);
    let right_row = right_row.as_slice().unwrap();

    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for d in 0..dim {
        left_norm += left_row[d] * left_row[d];
        right_norm += right_row[d] * right_row[d];
    }
    left_norm = left_norm.sqrt().max(EPS);
    right_norm = right_norm.sqrt().max(EPS);

    let inv_left = 1.0 / left_norm;
    let inv_right = 1.0 / right_norm;
    let mut hyperplane = vec![0.0f32; dim];
    for d in 0..dim {
        hyperplane[d] = left_row[d] * inv_left - right_row[d] * inv_right;
    }

    let mut hp_norm = 0.0f32;
    for d in 0..dim {
        hp_norm += hyperplane[d] * hyperplane[d];
    }
    let inv_hp_norm = 1.0 / hp_norm.sqrt().max(EPS);
    for d in 0..dim {
        hyperplane[d] *= inv_hp_norm;
    }

    (hyperplane, 0.0)
}

/// Compute a hyperplane for a euclidean split between two random points.
#[cfg(feature = "gpu")]
fn compute_euclidean_hyperplane(
    data: &Array2<f32>,
    indices: &[i32],
    rng: &mut TauRng,
) -> (Vec<f32>, f32) {
    let dim = data.ncols();
    let n = indices.len();

    let left_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    let mut right_idx = (rng.tau_rand_int().unsigned_abs() as usize) % n;
    if left_idx == right_idx {
        right_idx = (right_idx + 1) % n;
    }

    let left_row = data.row(indices[left_idx] as usize);
    let left_row = left_row.as_slice().unwrap();
    let right_row = data.row(indices[right_idx] as usize);
    let right_row = right_row.as_slice().unwrap();

    let mut hyperplane = vec![0.0f32; dim];
    let mut offset = 0.0f32;
    for d in 0..dim {
        hyperplane[d] = left_row[d] - right_row[d];
        offset -= hyperplane[d] * (left_row[d] + right_row[d]) * 0.5;
    }

    (hyperplane, offset)
}

/// Partition points by margins, handling edge cases.
#[cfg(feature = "gpu")]
fn partition_by_margins(
    indices: &[i32],
    margins: &[f32],
    rng: &mut TauRng,
) -> (Vec<i32>, Vec<i32>) {
    let mut left = Vec::new();
    let mut right = Vec::new();

    for (k, &idx) in indices.iter().enumerate() {
        let m = margins[k];
        if m.abs() < EPS {
            if rng.tau_rand_int().unsigned_abs() % 2 == 0 {
                left.push(idx);
            } else {
                right.push(idx);
            }
        } else if m > 0.0 {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    if left.is_empty() || right.is_empty() {
        left.clear();
        right.clear();
        for &idx in indices {
            if rng.tau_rand_int().unsigned_abs() % 2 == 0 {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }
        if left.is_empty() {
            left.push(right.pop().unwrap());
        }
        if right.is_empty() {
            right.push(left.pop().unwrap());
        }
    }

    (left, right)
}

/// Build a single RP tree using BFS with batched GPU projections.
///
/// At each BFS level, all pending nodes compute their hyperplanes on CPU,
/// then batch all dot-product projections into a single GPU dispatch.
#[cfg(feature = "gpu")]
pub async fn make_dense_tree_gpu(
    data: &Array2<f32>,
    rng: &mut TauRng,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
    gpu: &crate::gpu::GpuContext,
) -> FlatTree {
    let n = data.nrows();
    let dim = data.ncols();

    // BFS tree storage indexed by node ID
    let mut hyperplanes_out: Vec<Vec<f32>> = Vec::new();
    let mut offsets_out: Vec<f32> = Vec::new();
    let mut children_out: Vec<(i32, i32)> = Vec::new();
    let mut point_indices_out: Vec<Vec<i32>> = Vec::new();

    // Allocate root
    hyperplanes_out.push(vec![0.0; dim]);
    offsets_out.push(0.0);
    children_out.push((-1, -1));
    point_indices_out.push(vec![-1]);

    let hyperplane_fn = if angular {
        compute_angular_hyperplane
    } else {
        compute_euclidean_hyperplane
    };

    // BFS queue: (node_id, indices, depth)
    let mut current_level: Vec<(usize, Vec<i32>, usize)> = vec![(0, (0..n as i32).collect(), 0)];

    while !current_level.is_empty() {
        // Separate leaves from nodes to split
        let mut to_split: Vec<(usize, Vec<i32>, usize)> = Vec::new();
        for (node_id, indices, depth) in current_level.drain(..) {
            if indices.len() <= leaf_size || depth >= max_depth {
                hyperplanes_out[node_id] = vec![-1.0];
                offsets_out[node_id] = f32::NEG_INFINITY;
                children_out[node_id] = (-1, -1);
                point_indices_out[node_id] = indices;
            } else {
                to_split.push((node_id, indices, depth));
            }
        }

        if to_split.is_empty() {
            break;
        }

        // Compute hyperplanes for all nodes (CPU — O(n_nodes × dim), fast)
        let mut node_hyperplanes: Vec<(Vec<f32>, f32)> = Vec::with_capacity(to_split.len());
        for (_, indices, _) in &to_split {
            node_hyperplanes.push(hyperplane_fn(data, indices, rng));
        }

        // Total points across all nodes at this level
        let total_points: usize = to_split.iter().map(|(_, idx, _)| idx.len()).sum();

        // Compute margins: batch GPU or CPU based on total work
        let margins_flat: Vec<f32> = if total_points >= GPU_BATCH_THRESHOLD {
            let batch: Vec<(&[f32], f32, Vec<u32>)> = to_split
                .iter()
                .zip(node_hyperplanes.iter())
                .map(|((_, indices, _), (hp, offset))| {
                    let gpu_indices: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                    (hp.as_slice(), *offset, gpu_indices)
                })
                .collect();

            let batch_refs: Vec<(&[f32], f32, &[u32])> = batch
                .iter()
                .map(|(hp, off, idx)| (*hp, *off, idx.as_slice()))
                .collect();

            gpu.compute_projections_batch(&batch_refs).await
        } else {
            let mut margins = Vec::with_capacity(total_points);
            for ((_, indices, _), (hp, offset)) in to_split.iter().zip(node_hyperplanes.iter()) {
                for &idx in indices.iter() {
                    let row = data.row(idx as usize);
                    let row = row.as_slice().unwrap();
                    margins.push(dot_product(hp, row) + offset);
                }
            }
            margins
        };

        // Partition each node and create children
        let mut margin_offset = 0usize;
        let mut next_level: Vec<(usize, Vec<i32>, usize)> = Vec::new();

        for ((node_id, indices, depth), (hp, offset)) in
            to_split.into_iter().zip(node_hyperplanes.into_iter())
        {
            let n_pts = indices.len();
            let node_margins = &margins_flat[margin_offset..margin_offset + n_pts];
            margin_offset += n_pts;

            let (left_indices, right_indices) = partition_by_margins(&indices, node_margins, rng);

            // Allocate child node IDs
            let left_id = hyperplanes_out.len();
            hyperplanes_out.push(vec![0.0; dim]);
            offsets_out.push(0.0);
            children_out.push((-1, -1));
            point_indices_out.push(vec![-1]);

            let right_id = hyperplanes_out.len();
            hyperplanes_out.push(vec![0.0; dim]);
            offsets_out.push(0.0);
            children_out.push((-1, -1));
            point_indices_out.push(vec![-1]);

            hyperplanes_out[node_id] = hp;
            offsets_out[node_id] = offset;
            children_out[node_id] = (left_id as i32, right_id as i32);
            point_indices_out[node_id] = vec![-1];

            next_level.push((left_id, left_indices, depth + 1));
            next_level.push((right_id, right_indices, depth + 1));
        }

        current_level = next_level;
    }

    // Convert BFS tree to FlatTree via DFS traversal into TreeBuilder
    let mut builder = TreeBuilder::new();
    let mut bfs_to_builder: Vec<usize> = vec![0; hyperplanes_out.len()];

    fn dfs_convert(
        node: usize,
        hyperplanes: &[Vec<f32>],
        offsets: &[f32],
        children: &[(i32, i32)],
        point_indices: &[Vec<i32>],
        builder: &mut TreeBuilder,
        mapping: &mut [usize],
    ) -> usize {
        let (left, right) = children[node];
        if left < 0 {
            let id = builder.add_leaf(point_indices[node].clone());
            mapping[node] = id;
            id
        } else {
            let left_id = dfs_convert(
                left as usize,
                hyperplanes,
                offsets,
                children,
                point_indices,
                builder,
                mapping,
            );
            let right_id = dfs_convert(
                right as usize,
                hyperplanes,
                offsets,
                children,
                point_indices,
                builder,
                mapping,
            );
            let id =
                builder.add_internal(hyperplanes[node].clone(), offsets[node], left_id, right_id);
            mapping[node] = id;
            id
        }
    }

    let root = dfs_convert(
        0,
        &hyperplanes_out,
        &offsets_out,
        &children_out,
        &point_indices_out,
        &mut builder,
        &mut bfs_to_builder,
    );

    convert_builder_to_flat_tree(&builder, root, n, dim)
}

/// Build an RP forest with GPU acceleration using batched projections.
#[cfg(feature = "gpu")]
pub async fn make_forest_gpu(
    data: &Array2<f32>,
    n_neighbors: usize,
    n_trees: usize,
    leaf_size: Option<usize>,
    rng_state: &[i64; 3],
    angular: bool,
    max_depth: usize,
    gpu: &crate::gpu::GpuContext,
) -> Vec<FlatTree> {
    let leaf_size = leaf_size.unwrap_or_else(|| (5 * n_neighbors).max(60).min(256));

    let mut master_rng = TauRng::from_state(*rng_state);
    let rng_states: Vec<[i64; 3]> = (0..n_trees)
        .map(|_| {
            let s0 = master_rng.tau_rand_int() as i64;
            let s1 = master_rng.tau_rand_int() as i64;
            let s2 = master_rng.tau_rand_int() as i64;
            [
                s0.wrapping_add(0xFFFF),
                s1.wrapping_add(0xFFFF),
                s2.wrapping_add(0xFFFF),
            ]
        })
        .collect();

    let mut trees = Vec::with_capacity(n_trees);
    for state in rng_states {
        let mut rng = TauRng::from_state(state);
        trees.push(make_dense_tree_gpu(data, &mut rng, leaf_size, angular, max_depth, gpu).await);
    }
    trees
}

/// Search a flat tree to find the leaf containing a query point.
/// Returns (start, end) indices into tree.indices.
pub fn search_flat_tree(tree: &FlatTree, point: &[f32], rng: &mut TauRng) -> (usize, usize) {
    let mut node = 0usize;

    loop {
        let left = tree.children[[node, 0]];
        let right = tree.children[[node, 1]];

        if left <= 0 {
            // Leaf node
            return ((-left) as usize, (-right) as usize);
        }

        let hp = tree.hyperplanes.row(node);
        let offset = tree.offsets[node];
        let side = select_side(hp.as_slice().unwrap(), offset, point, rng);

        if side == 0 {
            node = left as usize;
        } else {
            node = right as usize;
        }
    }
}

// ===== Hub tree (graph-informed RP tree for search optimization) =====

/// Minimum split balance threshold. If the best split has balance below this,
/// create a leaf instead of a poor split. 0.1 means a 10/90 split.
const MIN_SPLIT_BALANCE: f32 = 0.1;

/// Compute global in-degree for all points: how many times each point appears
/// as a neighbor of other points in the KNN graph.
fn compute_global_degrees(neighbor_indices: &Array2<i32>) -> Vec<i32> {
    let n_points = neighbor_indices.nrows();
    let mut degrees = vec![0i32; n_points];
    for row in neighbor_indices.rows() {
        for &neighbor in row.iter() {
            if neighbor >= 0 && (neighbor as usize) < n_points {
                degrees[neighbor as usize] += 1;
            }
        }
    }
    degrees
}

/// Return the actual point indices of the top-k highest-degree points from a subset.
fn get_top_k_hub_indices(indices: &[i32], global_degrees: &[i32], k: usize) -> Vec<i32> {
    let actual_k = k.min(indices.len());
    // (degree, point_index), sorted descending by degree
    let mut top: Vec<(i32, i32)> = Vec::with_capacity(actual_k);

    for &idx in indices {
        let deg = global_degrees[idx as usize];
        if top.len() < actual_k {
            // Insert in sorted position
            let pos = top.iter().position(|&(d, _)| deg > d).unwrap_or(top.len());
            top.insert(pos, (deg, idx));
        } else if deg > top[actual_k - 1].0 {
            let pos = top
                .iter()
                .position(|&(d, _)| deg > d)
                .unwrap_or(actual_k - 1);
            top.pop();
            top.insert(pos, (deg, idx));
        }
    }

    top.iter().map(|&(_, idx)| idx).collect()
}

/// Euclidean hub-based split: uses top-3 highest-degree points to generate
/// hyperplanes between all pairs, picks the most balanced split.
fn euclidean_hub_split(
    data: &Array2<f32>,
    indices: &[i32],
    global_degrees: &[i32],
    rng: &mut TauRng,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, f32, f32) {
    let dim = data.ncols();
    let n_points = indices.len();

    let top_hubs = get_top_k_hub_indices(indices, global_degrees, 3);
    let n_hubs = top_hubs.len();

    let mut best_balance: f32 = 0.0;
    let mut best_hyperplane = vec![0.0f32; dim];
    let mut best_offset: f32 = 0.0;
    let mut best_side = vec![0i8; n_points];
    let mut best_n_left: u32 = 0;
    let mut best_n_right: u32 = 0;
    let mut side = vec![0i8; n_points];

    for hi in 0..n_hubs {
        for hj in (hi + 1)..n_hubs {
            let left_pt = top_hubs[hi] as usize;
            let right_pt = top_hubs[hj] as usize;
            let left_row = data.row(left_pt);
            let left_row = left_row.as_slice().unwrap();
            let right_row = data.row(right_pt);
            let right_row = right_row.as_slice().unwrap();

            // Hyperplane = left - right, offset = midpoint projection
            let mut hyperplane = vec![0.0f32; dim];
            let mut offset = 0.0f32;
            for d in 0..dim {
                hyperplane[d] = left_row[d] - right_row[d];
                offset -= hyperplane[d] * (left_row[d] + right_row[d]) * 0.5;
            }

            // Project all points
            let mut n_left: u32 = 0;
            let mut n_right: u32 = 0;
            for (i, &idx) in indices.iter().enumerate() {
                let row = data.row(idx as usize);
                let row = row.as_slice().unwrap();
                let margin = dot_product(&hyperplane, row) + offset;

                if margin > EPS {
                    side[i] = 0;
                    n_left += 1;
                } else if margin < -EPS {
                    side[i] = 1;
                    n_right += 1;
                } else {
                    side[i] = (i % 2) as i8;
                    if side[i] == 0 {
                        n_left += 1;
                    } else {
                        n_right += 1;
                    }
                }
            }

            if n_left == 0 || n_right == 0 {
                continue;
            }

            let balance = (n_left.min(n_right) as f32) / (n_points as f32);
            if balance > best_balance {
                best_balance = balance;
                best_n_left = n_left;
                best_n_right = n_right;
                best_offset = offset;
                best_hyperplane.copy_from_slice(&hyperplane);
                best_side.copy_from_slice(&side);
            }
        }
    }

    // Fallback to random assignment if no valid split found
    if best_n_left == 0 || best_n_right == 0 {
        best_n_left = 0;
        best_n_right = 0;
        for i in 0..n_points {
            best_side[i] = (rng.tau_rand_int().unsigned_abs() % 2) as i8;
            if best_side[i] == 0 {
                best_n_left += 1;
            } else {
                best_n_right += 1;
            }
        }
    }

    let mut left_indices = Vec::with_capacity(best_n_left as usize);
    let mut right_indices = Vec::with_capacity(best_n_right as usize);
    for (i, &idx) in indices.iter().enumerate() {
        if best_side[i] == 0 {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    (
        left_indices,
        right_indices,
        best_hyperplane,
        best_offset,
        best_balance,
    )
}

/// Angular hub-based split: uses top-3 highest-degree points to generate
/// normalized hyperplanes between all pairs, picks the most balanced split.
fn angular_hub_split(
    data: &Array2<f32>,
    indices: &[i32],
    global_degrees: &[i32],
    rng: &mut TauRng,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, f32, f32) {
    let dim = data.ncols();
    let n_points = indices.len();

    let top_hubs = get_top_k_hub_indices(indices, global_degrees, 3);
    let n_hubs = top_hubs.len();

    let mut best_balance: f32 = 0.0;
    let mut best_hyperplane = vec![0.0f32; dim];
    let mut best_side = vec![0i8; n_points];
    let mut best_n_left: u32 = 0;
    let mut best_n_right: u32 = 0;
    let mut side = vec![0i8; n_points];

    for hi in 0..n_hubs {
        for hj in (hi + 1)..n_hubs {
            let left_pt = top_hubs[hi] as usize;
            let right_pt = top_hubs[hj] as usize;
            let left_row = data.row(left_pt);
            let left_row = left_row.as_slice().unwrap();
            let right_row = data.row(right_pt);
            let right_row = right_row.as_slice().unwrap();

            // Compute norms
            let mut left_norm = 0.0f32;
            let mut right_norm = 0.0f32;
            for d in 0..dim {
                left_norm += left_row[d] * left_row[d];
                right_norm += right_row[d] * right_row[d];
            }
            left_norm = left_norm.sqrt().max(EPS);
            right_norm = right_norm.sqrt().max(EPS);

            // Normalized hyperplane
            let inv_left = 1.0 / left_norm;
            let inv_right = 1.0 / right_norm;
            let mut hyperplane = vec![0.0f32; dim];
            for d in 0..dim {
                hyperplane[d] = left_row[d] * inv_left - right_row[d] * inv_right;
            }

            // Normalize hyperplane vector
            let mut hp_norm = 0.0f32;
            for d in 0..dim {
                hp_norm += hyperplane[d] * hyperplane[d];
            }
            let inv_hp_norm = 1.0 / hp_norm.sqrt().max(EPS);
            for d in 0..dim {
                hyperplane[d] *= inv_hp_norm;
            }

            // Project all points
            let mut n_left: u32 = 0;
            let mut n_right: u32 = 0;
            for (i, &idx) in indices.iter().enumerate() {
                let row = data.row(idx as usize);
                let row = row.as_slice().unwrap();
                let margin = dot_product(&hyperplane, row);

                if margin > EPS {
                    side[i] = 0;
                    n_left += 1;
                } else if margin < -EPS {
                    side[i] = 1;
                    n_right += 1;
                } else {
                    side[i] = (i % 2) as i8;
                    if side[i] == 0 {
                        n_left += 1;
                    } else {
                        n_right += 1;
                    }
                }
            }

            if n_left == 0 || n_right == 0 {
                continue;
            }

            let balance = (n_left.min(n_right) as f32) / (n_points as f32);
            if balance > best_balance {
                best_balance = balance;
                best_n_left = n_left;
                best_n_right = n_right;
                best_hyperplane.copy_from_slice(&hyperplane);
                best_side.copy_from_slice(&side);
            }
        }
    }

    // Fallback to random assignment if no valid split found
    if best_n_left == 0 || best_n_right == 0 {
        best_n_left = 0;
        best_n_right = 0;
        for i in 0..n_points {
            best_side[i] = (rng.tau_rand_int().unsigned_abs() % 2) as i8;
            if best_side[i] == 0 {
                best_n_left += 1;
            } else {
                best_n_right += 1;
            }
        }
    }

    let mut left_indices = Vec::with_capacity(best_n_left as usize);
    let mut right_indices = Vec::with_capacity(best_n_right as usize);
    for (i, &idx) in indices.iter().enumerate() {
        if best_side[i] == 0 {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    (
        left_indices,
        right_indices,
        best_hyperplane,
        0.0,
        best_balance,
    )
}

/// Recursively build a hub-based euclidean tree.
fn make_hub_euclidean_tree(
    data: &Array2<f32>,
    indices: Vec<i32>,
    global_degrees: &[i32],
    rng: &mut TauRng,
    leaf_size: usize,
    max_depth: usize,
    builder: &mut TreeBuilder,
) -> usize {
    if indices.len() <= leaf_size || max_depth == 0 {
        return builder.add_leaf(indices);
    }

    let (left_indices, right_indices, hyperplane, offset, balance) =
        euclidean_hub_split(data, &indices, global_degrees, rng);

    // If split is too unbalanced, make a leaf instead
    if balance < MIN_SPLIT_BALANCE {
        return builder.add_leaf(indices);
    }

    let left_node = make_hub_euclidean_tree(
        data,
        left_indices,
        global_degrees,
        rng,
        leaf_size,
        max_depth - 1,
        builder,
    );
    let right_node = make_hub_euclidean_tree(
        data,
        right_indices,
        global_degrees,
        rng,
        leaf_size,
        max_depth - 1,
        builder,
    );

    builder.add_internal(hyperplane, offset, left_node, right_node)
}

/// Recursively build a hub-based angular tree.
fn make_hub_angular_tree(
    data: &Array2<f32>,
    indices: Vec<i32>,
    global_degrees: &[i32],
    rng: &mut TauRng,
    leaf_size: usize,
    max_depth: usize,
    builder: &mut TreeBuilder,
) -> usize {
    if indices.len() <= leaf_size || max_depth == 0 {
        return builder.add_leaf(indices);
    }

    let (left_indices, right_indices, hyperplane, offset, balance) =
        angular_hub_split(data, &indices, global_degrees, rng);

    // If split is too unbalanced, make a leaf instead
    if balance < MIN_SPLIT_BALANCE {
        return builder.add_leaf(indices);
    }

    let left_node = make_hub_angular_tree(
        data,
        left_indices,
        global_degrees,
        rng,
        leaf_size,
        max_depth - 1,
        builder,
    );
    let right_node = make_hub_angular_tree(
        data,
        right_indices,
        global_degrees,
        rng,
        leaf_size,
        max_depth - 1,
        builder,
    );

    builder.add_internal(hyperplane, offset, left_node, right_node)
}

/// Build a hub tree that uses neighbor graph information for better splits.
pub fn make_hub_tree(
    data: &Array2<f32>,
    neighbor_indices: &Array2<i32>,
    rng: &mut TauRng,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
) -> FlatTree {
    let n = data.nrows();
    let indices: Vec<i32> = (0..n as i32).collect();
    let global_degrees = compute_global_degrees(neighbor_indices);

    let mut builder = TreeBuilder::new();

    let root = if angular {
        make_hub_angular_tree(
            data,
            indices,
            &global_degrees,
            rng,
            leaf_size,
            max_depth,
            &mut builder,
        )
    } else {
        make_hub_euclidean_tree(
            data,
            indices,
            &global_degrees,
            rng,
            leaf_size,
            max_depth,
            &mut builder,
        )
    };

    convert_builder_to_flat_tree(&builder, root, data.nrows(), data.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_data(n: usize, dim: usize, seed: i64) -> Array2<f32> {
        let mut rng = TauRng::new(seed);
        let mut data = Array2::zeros((n, dim));
        for i in 0..n {
            for j in 0..dim {
                data[[i, j]] = rng.tau_rand();
            }
        }
        data
    }

    #[test]
    fn test_make_forest() {
        let data = make_test_data(100, 5, 42);
        let rng_state = [12345i64, 67890, 11111];
        let forest = make_forest(&data, 10, 3, None, &rng_state, false, 200);
        assert_eq!(forest.len(), 3);
    }

    #[test]
    fn test_leaf_array() {
        let data = make_test_data(100, 5, 42);
        let rng_state = [12345i64, 67890, 11111];
        let forest = make_forest(&data, 10, 2, None, &rng_state, false, 200);
        let leaves = rptree_leaf_array(&forest);
        assert!(leaves.nrows() > 0);
        // All data points should appear in at least one leaf
    }

    #[test]
    fn test_tree_search() {
        let data = make_test_data(100, 5, 42);
        let mut rng = TauRng::new(42);
        let tree = make_dense_tree(&data, &mut rng, 10, false, 200);

        let mut search_rng = TauRng::new(99);
        let point = data.row(0).to_vec();
        let (start, end) = search_flat_tree(&tree, &point, &mut search_rng);
        assert!(end > start);
        assert!(end - start <= tree.leaf_size + 1);
    }
}
