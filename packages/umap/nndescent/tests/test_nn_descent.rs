/// NN-descent accuracy tests matching test_nndescent_.py.
/// Tests build accuracy (≥98%), query accuracy (≥95%), determinism, etc.
use ndarray::Array2;
use nndescent::distance;
use nndescent::NNDescent;

/// Generate test data matching conftest.py nn_data fixture.
/// Shape: (1002, 5) - 1000 random rows + 2 zero rows.
/// Uses our TauRng for deterministic generation (seed 189212).
fn make_nn_data() -> Array2<f32> {
    use nndescent::rng::TauRng;
    let mut rng = TauRng::new(189212);
    let mut data = Array2::zeros((1002, 5));
    for i in 0..1000 {
        for j in 0..5 {
            data[[i, j]] = rng.tau_rand();
        }
    }
    // rows 1000, 1001 are zeros (corner case)
    data
}

/// Brute-force k-nearest neighbor computation for reference.
/// Includes self-neighbor (distance 0) at position 0, consistent with pynndescent.
fn brute_force_knn(data: &Array2<f32>, k: usize, dist_fn: distance::DistanceFunc) -> Array2<i32> {
    let n = data.nrows();
    let mut result = Array2::from_elem((n, k), -1i32);

    for i in 0..n {
        let mut dists: Vec<(f32, i32)> = (0..n)
            .map(|j| {
                let d = dist_fn(
                    data.row(i).as_slice().unwrap(),
                    data.row(j).as_slice().unwrap(),
                );
                (d, j as i32)
            })
            .collect();
        // Sort by distance, breaking ties by putting self first
        dists.sort_by(|a, b| {
            let cmp = a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal);
            if cmp == std::cmp::Ordering::Equal {
                // Self (index == i) should come first among equal distances
                let a_is_self = a.1 == i as i32;
                let b_is_self = b.1 == i as i32;
                if a_is_self && !b_is_self {
                    std::cmp::Ordering::Less
                } else if !a_is_self && b_is_self {
                    std::cmp::Ordering::Greater
                } else {
                    a.1.cmp(&b.1)
                }
            } else {
                cmp
            }
        });
        for j in 0..k.min(dists.len()) {
            result[[i, j]] = dists[j].1;
        }
    }
    result
}

/// Brute-force k-nearest neighbor for query points against a data set.
fn brute_force_knn_query(
    query: &Array2<f32>,
    data: &Array2<f32>,
    k: usize,
    dist_fn: distance::DistanceFunc,
) -> Array2<i32> {
    let n_queries = query.nrows();
    let n_data = data.nrows();
    let mut result = Array2::from_elem((n_queries, k), -1i32);

    for i in 0..n_queries {
        let mut dists: Vec<(f32, i32)> = (0..n_data)
            .map(|j| {
                let d = dist_fn(
                    query.row(i).as_slice().unwrap(),
                    data.row(j).as_slice().unwrap(),
                );
                (d, j as i32)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for j in 0..k.min(dists.len()) {
            result[[i, j]] = dists[j].1;
        }
    }
    result
}

/// Compute recall: fraction of true neighbors found.
fn compute_recall(true_indices: &Array2<i32>, approx_indices: &Array2<i32>) -> f32 {
    let n = true_indices.nrows();
    let k = true_indices.ncols();
    let mut total_correct = 0;
    let total = n * k;

    for i in 0..n {
        let true_set: std::collections::HashSet<i32> = (0..k)
            .map(|j| true_indices[[i, j]])
            .filter(|&v| v >= 0)
            .collect();
        for j in 0..approx_indices.ncols().min(k) {
            let idx = approx_indices[[i, j]];
            if idx >= 0 && true_set.contains(&idx) {
                total_correct += 1;
            }
        }
    }

    total_correct as f32 / total as f32
}

/// Test NN-descent build accuracy with euclidean metric.
/// Should achieve ≥ 98% recall against brute force.
#[test]
fn test_nn_descent_neighbor_accuracy() {
    let data = make_nn_data();
    let k = 10;
    let nnd = NNDescent::builder(data.clone(), "euclidean", k)
        .random_state(42)
        .build()
        .unwrap();
    let (approx_indices, _approx_distances) = nnd.neighbor_graph().unwrap();

    let true_indices = brute_force_knn(&data, k, distance::euclidean);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(
        recall >= 0.98,
        "Euclidean build accuracy: {:.4} < 0.98",
        recall
    );
}

/// Test NN-descent build accuracy with cosine metric.
/// Should achieve ≥ 98% recall against brute force.
#[test]
fn test_angular_nn_descent_neighbor_accuracy() {
    let data = make_nn_data();
    let k = 10;
    let nnd = NNDescent::builder(data.clone(), "cosine", k)
        .random_state(42)
        .build()
        .unwrap();
    let (approx_indices, _approx_distances) = nnd.neighbor_graph().unwrap();

    let true_indices = brute_force_knn(&data, k, distance::cosine);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(
        recall >= 0.98,
        "Cosine build accuracy: {:.4} < 0.98",
        recall
    );
}

/// Test NN-descent query accuracy with euclidean metric.
/// Build on data[200:], query with data[:200], should achieve ≥ 95%.
#[test]
fn test_nn_descent_query_accuracy() {
    let data = make_nn_data();
    let k = 10;

    // Split data: first 200 rows for query, rest for building
    let train_data = data.slice(ndarray::s![200.., ..]).to_owned();
    let query_data = data.slice(ndarray::s![..200, ..]).to_owned();

    let mut nnd = NNDescent::builder(train_data.clone(), "euclidean", k)
        .random_state(42)
        .build()
        .unwrap();
    let (approx_indices, _approx_distances) = nnd.query(&query_data, k, 0.2);

    let true_indices = brute_force_knn_query(&query_data, &train_data, k, distance::euclidean);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(
        recall >= 0.95,
        "Euclidean query accuracy: {:.4} < 0.95",
        recall
    );
}

/// Test NN-descent query accuracy with cosine metric.
#[test]
fn test_nn_descent_query_accuracy_angular() {
    let data = make_nn_data();
    let k = 10;

    let train_data = data.slice(ndarray::s![200.., ..]).to_owned();
    let query_data = data.slice(ndarray::s![..200, ..]).to_owned();

    let mut nnd = NNDescent::builder(train_data.clone(), "cosine", 30)
        .random_state(42)
        .build()
        .unwrap();
    let (approx_indices, _approx_distances) = nnd.query(&query_data, k, 0.32);

    let true_indices = brute_force_knn_query(&query_data, &train_data, k, distance::cosine);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(
        recall >= 0.95,
        "Cosine query accuracy: {:.4} < 0.95",
        recall
    );
}

/// Test determinism: same seed produces identical results.
#[test]
fn test_deterministic() {
    let data = make_nn_data();

    let nnd1 = NNDescent::builder(data.clone(), "euclidean", 10)
        .random_state(42)
        .build()
        .unwrap();
    let nnd2 = NNDescent::builder(data.clone(), "euclidean", 10)
        .random_state(42)
        .build()
        .unwrap();

    let (idx1, dist1) = nnd1.neighbor_graph().unwrap();
    let (idx2, dist2) = nnd2.neighbor_graph().unwrap();

    assert_eq!(idx1, idx2, "Indices should be deterministic with same seed");
    // Check distances are close (exact equality may differ due to floating point)
    for i in 0..idx1.nrows() {
        for j in 0..idx1.ncols() {
            let d1 = dist1[[i, j]];
            let d2 = dist2[[i, j]];
            assert!(
                (d1 - d2).abs() < 1e-6,
                "Distances should match: {} vs {} at ({},{})",
                d1,
                d2,
                i,
                j
            );
        }
    }
}

/// Test that tree_init=false still works (with potentially lower accuracy).
#[test]
fn test_tree_init_false() {
    let data = make_nn_data();
    let k = 10;

    let train_data = data.slice(ndarray::s![200.., ..]).to_owned();
    let query_data = data.slice(ndarray::s![..200, ..]).to_owned();

    let mut nnd = NNDescent::builder(train_data.clone(), "euclidean", k)
        .random_state(42)
        .tree_init(false)
        .build()
        .unwrap();
    let (approx_indices, _approx_distances) = nnd.query(&query_data, k, 0.2);

    let true_indices = brute_force_knn_query(&query_data, &train_data, k, distance::euclidean);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(
        recall >= 0.95,
        "tree_init=false query accuracy: {:.4} < 0.95",
        recall
    );
}

/// Test NN-descent with manhattan metric.
#[test]
fn test_nn_descent_manhattan() {
    let data = make_nn_data();
    let k = 10;
    let nnd = NNDescent::builder(data.clone(), "manhattan", k)
        .random_state(42)
        .build()
        .unwrap();
    let (approx_indices, _approx_distances) = nnd.neighbor_graph().unwrap();

    let true_indices = brute_force_knn(&data, k, distance::manhattan);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(
        recall >= 0.95,
        "Manhattan build accuracy: {:.4} < 0.95",
        recall
    );
}

/// Test that distances in the neighbor graph are sorted ascending.
#[test]
fn test_distances_sorted() {
    let data = make_nn_data();
    let k = 10;
    let nnd = NNDescent::builder(data, "euclidean", k)
        .random_state(42)
        .build()
        .unwrap();
    let (_indices, distances) = nnd.neighbor_graph().unwrap();

    for i in 0..distances.nrows() {
        for j in 1..k {
            assert!(
                distances[[i, j]] >= distances[[i, j - 1]],
                "Row {} not sorted at col {}: {} < {}",
                i,
                j,
                distances[[i, j]],
                distances[[i, j - 1]]
            );
        }
    }
}

/// Test that neighbor indices are valid (non-negative and within bounds).
#[test]
fn test_valid_indices() {
    let data = make_nn_data();
    let n = data.nrows();
    let k = 10;
    let nnd = NNDescent::builder(data, "euclidean", k)
        .random_state(42)
        .build()
        .unwrap();
    let (indices, _distances) = nnd.neighbor_graph().unwrap();

    for i in 0..indices.nrows() {
        for j in 0..k {
            let idx = indices[[i, j]];
            assert!(
                idx >= 0 && (idx as usize) < n,
                "Invalid index {} at ({},{})",
                idx,
                i,
                j
            );
        }
    }
}

/// Test with small dataset (n < n_neighbors edge case).
#[test]
fn test_small_dataset() {
    use nndescent::rng::TauRng;
    let mut rng = TauRng::new(42);
    let mut data = Array2::zeros((20, 5));
    for i in 0..20 {
        for j in 0..5 {
            data[[i, j]] = rng.tau_rand();
        }
    }
    // n_neighbors = 5 (smaller than n=20)
    let nnd = NNDescent::builder(data, "euclidean", 5)
        .random_state(42)
        .build()
        .unwrap();
    let (indices, distances) = nnd.neighbor_graph().unwrap();
    assert_eq!(indices.nrows(), 20);
    assert_eq!(indices.ncols(), 5);
    assert_eq!(distances.nrows(), 20);
    assert_eq!(distances.ncols(), 5);
}

/// Test one-dimensional data (edge case).
#[test]
fn test_one_dimensional_data() {
    let full_data = make_nn_data();
    let data = full_data.slice(ndarray::s![.., ..1]).to_owned();
    let k = 10;

    let train = data.slice(ndarray::s![200.., ..]).to_owned();
    let query = data.slice(ndarray::s![..200, ..]).to_owned();

    let mut nnd = NNDescent::builder(train.clone(), "euclidean", 20)
        .random_state(42)
        .n_trees(5)
        .build()
        .unwrap();
    let (approx_indices, _) = nnd.query(&query, k, 0.2);

    let true_indices = brute_force_knn_query(&query, &train, k, distance::euclidean);
    let recall = compute_recall(&true_indices, &approx_indices);

    assert!(recall >= 0.95, "1D query accuracy: {:.4} < 0.95", recall);
}

/// Test vertex order: query the training data itself and verify each point
/// finds itself as the nearest neighbor (returned in original numbering).
#[test]
fn test_vertex_order_self_query() {
    let data = make_nn_data();
    let train_data = data.slice(ndarray::s![..500, ..]).to_owned();
    let k = 5;

    let mut nnd = NNDescent::builder(train_data.clone(), "euclidean", 15)
        .random_state(42)
        .build()
        .unwrap();

    // Query the training data itself
    let (indices, distances) = nnd.query(&train_data, k, 0.1);

    // Each point should find itself as the nearest neighbor (distance ~0)
    let mut self_found = 0;
    for i in 0..train_data.nrows() {
        // Check if point i appears in its own result set
        for j in 0..k {
            if indices[[i, j]] == i as i32 {
                self_found += 1;
                // The distance to self should be very close to zero
                assert!(
                    distances[[i, j]] < 1e-5,
                    "Point {} distance to self: {}",
                    i,
                    distances[[i, j]]
                );
                break;
            }
        }
    }

    let self_recall = self_found as f32 / train_data.nrows() as f32;
    assert!(
        self_recall >= 0.95,
        "Self-query recall: {:.4} < 0.95 ({}/{} points found themselves)",
        self_recall,
        self_found,
        train_data.nrows()
    );
}

/// Test that query results use original (pre-reorder) indices.
/// Verify by checking that returned indices index into the original data
/// and produce correct distances.
#[test]
fn test_vertex_order_original_indices() {
    let data = make_nn_data();
    let train_data = data.slice(ndarray::s![200.., ..]).to_owned();
    let query_data = data.slice(ndarray::s![..50, ..]).to_owned();
    let k = 5;

    let mut nnd = NNDescent::builder(train_data.clone(), "euclidean", 15)
        .random_state(42)
        .build()
        .unwrap();

    let (indices, distances) = nnd.query(&query_data, k, 0.2);

    // Verify returned indices are valid and distances are correct
    for i in 0..query_data.nrows() {
        for j in 0..k {
            let idx = indices[[i, j]];
            assert!(
                idx >= 0 && (idx as usize) < train_data.nrows(),
                "Invalid index {} for query {} neighbor {}",
                idx,
                i,
                j
            );

            // Verify the distance matches the actual euclidean distance
            // between the query point and the indexed training point
            let actual_dist = distance::euclidean(
                query_data.row(i).as_slice().unwrap(),
                train_data.row(idx as usize).as_slice().unwrap(),
            );
            assert!(
                (distances[[i, j]] - actual_dist).abs() < 1e-4,
                "Distance mismatch for query {} neighbor {}: returned {} vs actual {}",
                i,
                j,
                distances[[i, j]],
                actual_dist
            );
        }
    }
}

/// Test hub tree: verify that hub tree splits produce a valid tree structure.
#[test]
fn test_hub_tree_structure() {
    use nndescent::rng::TauRng;
    use nndescent::rp_trees;

    let data = make_nn_data();
    let train_data = data.slice(ndarray::s![..500, ..]).to_owned();
    let k = 15;

    // Build KNN graph first
    let nnd = NNDescent::builder(train_data.clone(), "euclidean", k)
        .random_state(42)
        .build()
        .unwrap();
    let (neighbor_indices, _) = nnd.raw_neighbor_graph().unwrap();

    // Build hub tree
    let mut rng = TauRng::new(42);
    let tree = rp_trees::make_hub_tree(&train_data, neighbor_indices, &mut rng, 30, false, 200);

    // Verify tree covers all points
    let n = train_data.nrows();
    let mut seen = vec![false; n];
    for &idx in &tree.indices {
        if idx >= 0 && (idx as usize) < n {
            seen[idx as usize] = true;
        }
    }
    let covered = seen.iter().filter(|&&s| s).count();
    assert_eq!(
        covered, n,
        "Hub tree should cover all {} points, but only covered {}",
        n, covered
    );

    // Verify search still works
    let point = train_data.row(0).to_vec();
    let mut search_rng = TauRng::new(99);
    let (start, end) = rp_trees::search_flat_tree(&tree, &point, &mut search_rng);
    assert!(end > start, "Leaf should be non-empty");
    assert!(
        end - start <= tree.leaf_size + 1,
        "Leaf should not exceed leaf_size"
    );
}
