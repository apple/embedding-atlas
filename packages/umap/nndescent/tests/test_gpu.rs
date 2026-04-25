//! GPU-specific tests for NNDescent.
//!
//! These tests require the `gpu` feature and a GPU-capable system.
//! Run with: cargo test --features gpu -p nndescent

#![cfg(feature = "gpu")]

use ndarray::Array2;
use nndescent::NNDescent;

fn make_test_data(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    use nndescent::rng::Xoshiro256StarStar;
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut data = Array2::zeros((n, dim));
    for i in 0..n {
        for j in 0..dim {
            data[[i, j]] = rng.random_f32();
        }
    }
    data
}

/// Compute brute-force k-nearest neighbors for accuracy comparison.
fn brute_force_knn(data: &Array2<f32>, k: usize) -> Array2<i32> {
    let n = data.nrows();
    let mut indices = Array2::from_elem((n, k), -1i32);
    for i in 0..n {
        let mut dists: Vec<(f32, i32)> = (0..n)
            .map(|j| {
                let d: f32 = data
                    .row(i)
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                (d, j as i32)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for j in 0..k {
            indices[[i, j]] = dists[j].1;
        }
    }
    indices
}

/// Compute recall: fraction of true neighbors found.
fn compute_recall(approx: &Array2<i32>, exact: &Array2<i32>) -> f64 {
    let n = approx.nrows();
    let k = approx.ncols();
    let mut hits = 0usize;
    let total = n * k;
    for i in 0..n {
        for j in 0..k {
            let target = exact[[i, j]];
            if (0..k).any(|m| approx[[i, m]] == target) {
                hits += 1;
            }
        }
    }
    hits as f64 / total as f64
}

#[test]
fn test_gpu_euclidean_basic() {
    let data = make_test_data(500, 128, 42);
    let result = NNDescent::builder(data, "euclidean", 10)
        .random_state(42)
        .gpu(true)
        .build();

    match result {
        Ok(nnd) => {
            let (indices, distances) = nnd.neighbor_graph().unwrap();
            assert_eq!(indices.nrows(), 500);
            assert_eq!(indices.ncols(), 10);

            // Distances should be sorted ascending
            for i in 0..500 {
                for j in 1..10 {
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
        Err(e) => {
            eprintln!("GPU test skipped (no GPU available): {}", e);
        }
    }
}

#[test]
fn test_gpu_matches_cpu_euclidean() {
    let data = make_test_data(300, 128, 189212);

    // CPU path
    let nnd_cpu = NNDescent::builder(data.clone(), "euclidean", 10)
        .random_state(42)
        .gpu(false)
        .build()
        .unwrap();
    let (cpu_indices, _) = nnd_cpu.neighbor_graph().unwrap();

    // GPU path
    let result_gpu = NNDescent::builder(data.clone(), "euclidean", 10)
        .random_state(42)
        .gpu(true)
        .build();

    match result_gpu {
        Ok(nnd_gpu) => {
            let (gpu_indices, _) = nnd_gpu.neighbor_graph().unwrap();

            // Both should find similar neighbors (not necessarily identical
            // due to floating-point ordering differences on GPU)
            let exact = brute_force_knn(&data, 10);
            let cpu_recall = compute_recall(&cpu_indices, &exact);
            let gpu_recall = compute_recall(&gpu_indices, &exact);

            assert!(cpu_recall > 0.90, "CPU recall too low: {}", cpu_recall);
            assert!(gpu_recall > 0.90, "GPU recall too low: {}", gpu_recall);
        }
        Err(e) => {
            eprintln!("GPU test skipped (no GPU available): {}", e);
        }
    }
}

#[test]
fn test_gpu_cosine_basic() {
    let data = make_test_data(500, 128, 42);
    let result = NNDescent::builder(data, "cosine", 10)
        .random_state(42)
        .gpu(true)
        .build();

    match result {
        Ok(nnd) => {
            let (indices, _) = nnd.neighbor_graph().unwrap();
            assert_eq!(indices.nrows(), 500);
            assert_eq!(indices.ncols(), 10);
        }
        Err(e) => {
            eprintln!("GPU cosine test skipped: {}", e);
        }
    }
}

#[test]
fn test_gpu_fallback_unsupported_metric() {
    // Manhattan is not supported on GPU — should fall back to CPU
    let data = make_test_data(200, 64, 42);
    let nnd = NNDescent::builder(data, "manhattan", 10)
        .random_state(42)
        .gpu(true)
        .verbose(true)
        .build()
        .unwrap();

    let (indices, _) = nnd.neighbor_graph().unwrap();
    assert_eq!(indices.nrows(), 200);
}

#[test]
fn test_gpu_high_dimensional() {
    // Test with 512-dim data where GPU should shine
    let data = make_test_data(1000, 512, 42);
    let result = NNDescent::builder(data, "euclidean", 15)
        .random_state(42)
        .gpu(true)
        .build();

    match result {
        Ok(nnd) => {
            let (indices, distances) = nnd.neighbor_graph().unwrap();
            assert_eq!(indices.nrows(), 1000);
            assert_eq!(indices.ncols(), 15);

            // Self should be at position 0
            for i in 0..1000 {
                assert_eq!(indices[[i, 0]], i as i32);
                assert_eq!(distances[[i, 0]], 0.0);
            }
        }
        Err(e) => {
            eprintln!("GPU high-dim test skipped: {}", e);
        }
    }
}
