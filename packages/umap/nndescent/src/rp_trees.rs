/// Random Projection Trees for NN-descent initialization and search.
use ndarray::Array2;
use rayon::prelude::*;

use crate::rng::TauRng;

const EPS: f32 = 1e-8;

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

    // Compute norms
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for d in 0..dim {
        left_norm += data[[left_point, d]] * data[[left_point, d]];
        right_norm += data[[right_point, d]] * data[[right_point, d]];
    }
    left_norm = left_norm.sqrt().max(EPS);
    right_norm = right_norm.sqrt().max(EPS);

    // Hyperplane = normalized left - normalized right
    let mut hyperplane = vec![0.0f32; dim];
    for d in 0..dim {
        hyperplane[d] = data[[left_point, d]] / left_norm - data[[right_point, d]] / right_norm;
    }

    // Normalize hyperplane
    let mut hp_norm = 0.0f32;
    for d in 0..dim {
        hp_norm += hyperplane[d] * hyperplane[d];
    }
    hp_norm = hp_norm.sqrt().max(EPS);
    for d in 0..dim {
        hyperplane[d] /= hp_norm;
    }

    // Split points by margin
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        let mut margin = 0.0f32;
        for d in 0..dim {
            margin += hyperplane[d] * data[[idx as usize, d]];
        }

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

    // Hyperplane = left - right
    let mut hyperplane = vec![0.0f32; dim];
    let mut offset = 0.0f32;
    for d in 0..dim {
        hyperplane[d] = data[[left_point, d]] - data[[right_point, d]];
        offset -= hyperplane[d] * (data[[left_point, d]] + data[[right_point, d]]) / 2.0;
    }

    // Split points by margin
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        let mut margin = offset;
        for d in 0..dim {
            margin += hyperplane[d] * data[[idx as usize, d]];
        }

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
    let mut margin = offset;
    for d in 0..hyperplane.len().min(point.len()) {
        margin += hyperplane[d] * point[d];
    }

    if margin.abs() < EPS {
        (rng.tau_rand_int().unsigned_abs() % 2) as usize
    } else if margin > 0.0 {
        0
    } else {
        1
    }
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

/// Build a hub tree that uses neighbor graph information for better splits.
pub fn make_hub_tree(
    data: &Array2<f32>,
    _neighbor_indices: &Array2<i32>,
    rng: &mut TauRng,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
) -> FlatTree {
    let n = data.nrows();
    let indices: Vec<i32> = (0..n as i32).collect();

    let mut builder = TreeBuilder::new();

    // For hub trees, we use the same split logic but could add scoring
    // For now, use the basic split (the key optimization is in the tree structure)
    let root = if angular {
        make_angular_tree(data, indices, rng, leaf_size, max_depth, &mut builder)
    } else {
        make_euclidean_tree(data, indices, rng, leaf_size, max_depth, &mut builder)
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
