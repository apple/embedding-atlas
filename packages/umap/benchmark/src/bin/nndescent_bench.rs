//! Benchmark binary for Rust NNDescent implementation.
//!
//! Usage: benchmark <data_dir> <metric> [--gpu] [--output result.json]
//!
//! Reads a data directory containing meta.json, data.bin, and optionally
//! truth_{metric}.bin. Builds the NNDescent index on ALL data, then computes
//! recall against ground truth if available. Writes a single JSON result file.
//!
//! With --gpu, enables GPU-accelerated distance computation (requires the
//! `gpu` feature: cargo run --features gpu --bin nndescent-bench -- <args>).

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;

fn read_meta(dir: &Path) -> (usize, usize, usize) {
    let meta_str = fs::read_to_string(dir.join("meta.json")).expect("Failed to read meta.json");
    let n = extract_json_int(&meta_str, "n_points");
    let dim = extract_json_int(&meta_str, "dim");
    let k = extract_json_int(&meta_str, "k");
    (n, dim, k)
}

fn extract_json_int(json: &str, key: &str) -> usize {
    let pattern = format!("\"{}\":", key);
    let start = json
        .find(&pattern)
        .unwrap_or_else(|| panic!("Key '{}' not found", key));
    let after_colon = &json[start + pattern.len()..];
    let trimmed = after_colon.trim_start();
    let end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    trimmed[..end]
        .parse()
        .unwrap_or_else(|_| panic!("Failed to parse {}", key))
}

fn read_f32_bin(path: &Path, n: usize, dim: usize) -> Array2<f32> {
    let bytes = fs::read(path).unwrap_or_else(|_| panic!("Failed to read {:?}", path));
    let expected = n * dim * 4;
    assert_eq!(
        bytes.len(),
        expected,
        "data.bin size mismatch: got {} expected {}",
        bytes.len(),
        expected
    );
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Array2::from_shape_vec((n, dim), floats).unwrap()
}

fn read_i32_bin(path: &Path, n: usize, k: usize) -> Option<Array2<i32>> {
    let bytes = fs::read(path).ok()?;
    if bytes.is_empty() {
        return None;
    }
    let expected = n * k * 4;
    if bytes.len() != expected {
        return None;
    }
    let ints: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(Array2::from_shape_vec((n, k), ints).unwrap())
}

fn compute_recall(true_indices: &Array2<i32>, approx_indices: &Array2<i32>) -> f32 {
    let n = true_indices.nrows();
    let k = true_indices.ncols();
    let mut total_correct = 0usize;

    for i in 0..n {
        let true_set: HashSet<i32> = (0..k)
            .map(|j| true_indices[[i, j]])
            .filter(|&v| v >= 0)
            .collect();
        let approx_k = approx_indices.ncols().min(k);
        for j in 0..approx_k {
            let idx = approx_indices[[i, j]];
            if idx >= 0 && true_set.contains(&idx) {
                total_correct += 1;
            }
        }
    }

    total_correct as f32 / (n * k) as f32
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: benchmark <data_dir> <metric> [--gpu] [--output results.csv]");
        std::process::exit(1);
    }

    let data_dir = Path::new(&args[1]);
    let metric = &args[2];

    // Parse optional flags
    let use_gpu = args.iter().any(|a| a == "--gpu");
    let output_path = args
        .windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "result.json".to_string());

    // Read metadata and data
    let (n, dim, k) = read_meta(data_dir);
    eprintln!(
        "Rust benchmark: n={}, dim={}, k={}, metric={}, gpu={}",
        n, dim, k, metric, use_gpu
    );

    eprintln!("  Loading data...");
    let data = read_f32_bin(&data_dir.join("data.bin"), n, dim);

    // Load ground truth (pre-computed by generate_data.py)
    let truth_path = data_dir.join(format!("truth_{}.bin", metric));
    let truth = read_i32_bin(&truth_path, n, k);

    // Build index on ALL data
    eprintln!(
        "  Building index on {} points{}...",
        n,
        if use_gpu { " (GPU)" } else { "" }
    );
    let build_start = Instant::now();
    let nnd = nndescent::NNDescent::builder(data, metric, k)
        .random_state(42)
        .gpu(use_gpu)
        .build()
        .expect("NNDescent build failed");
    let build_time = build_start.elapsed().as_secs_f64();
    eprintln!("  Build time: {:.3}s", build_time);

    // Compute recall against ground truth
    let recall_str = if let Some(ref true_indices) = truth {
        let (approx_indices, _) = nnd.neighbor_graph().unwrap();
        let recall = compute_recall(true_indices, &approx_indices);
        eprintln!("  Recall: {:.4}", recall);
        format!("{:.4}", recall)
    } else {
        eprintln!("  No ground truth, skipping recall");
        "N/A".to_string()
    };

    // Write JSON result
    let json = format!(
        concat!(
            "{{\n",
            "  \"implementation\": \"rust\",\n",
            "  \"n_points\": {},\n",
            "  \"dim\": {},\n",
            "  \"metric\": \"{}\",\n",
            "  \"gpu\": {},\n",
            "  \"build_time_s\": {:.3},\n",
            "  \"recall\": \"{}\"\n",
            "}}\n"
        ),
        n, dim, metric, use_gpu, build_time, recall_str
    );
    fs::write(&output_path, &json).unwrap_or_else(|_| panic!("Failed to write {}", output_path));

    eprintln!("  Result written to {}", output_path);
}
