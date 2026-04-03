//! UMAP benchmark binary.
//!
//! Usage: umap-bench <data_dir> <metric> [--output results.csv] [--seed N] [--embedding-path path]
//!
//! Reads meta.json + data.bin, runs UMAP (2D), saves embedding and timing.

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;

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

fn write_f32_bin(path: &Path, data: &Array2<f32>) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data.iter() {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    fs::write(path, &bytes).unwrap_or_else(|_| panic!("Failed to write {:?}", path));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: umap-bench <data_dir> <metric> [--output results.csv]");
        std::process::exit(1);
    }

    let data_dir = Path::new(&args[1]);
    let metric = &args[2];

    // Parse optional flags from remaining args
    let mut output_path: Option<String> = None;
    let mut seed: Option<u64> = None;
    let mut embedding_path: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--seed" => {
                seed = Some(args[i + 1].parse().expect("Invalid seed"));
                i += 2;
            }
            "--embedding-path" => {
                embedding_path = Some(args[i + 1].clone());
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    let output_path =
        output_path.unwrap_or_else(|| data_dir.join("results.csv").to_string_lossy().to_string());

    // Read metadata
    let meta_str =
        fs::read_to_string(data_dir.join("meta.json")).expect("Failed to read meta.json");
    let n = extract_json_int(&meta_str, "n_points");
    let dim = extract_json_int(&meta_str, "dim");

    eprintln!(
        "Rust UMAP benchmark: n={}, dim={}, metric={}",
        n, dim, metric
    );

    // Load data
    eprintln!("  Loading data...");
    let data = read_f32_bin(&data_dir.join("data.bin"), n, dim);

    // Configure and run UMAP
    eprintln!("  Running UMAP...");
    let start = Instant::now();
    let mut builder = umap::Umap::builder(&data)
        .n_neighbors(15)
        .n_components(2)
        .min_dist(0.1)
        .metric(metric)
        .verbose(true);
    if let Some(s) = seed {
        builder = builder.random_state(s);
    }
    let result = builder.build().expect("UMAP build failed");
    let elapsed = start.elapsed().as_secs_f64();

    eprintln!("  UMAP time: {:.3}s", elapsed);

    // Save embedding
    let emb_path = match &embedding_path {
        Some(p) => std::path::PathBuf::from(p),
        None => data_dir.join(format!("rust_{}_embedding.bin", metric)),
    };
    write_f32_bin(&emb_path, &result.embedding);
    eprintln!("  Saved embedding to {:?}", emb_path);

    // Append to results CSV
    let output = Path::new(&output_path);
    let write_header =
        !output.exists() || fs::metadata(output).map(|m| m.len() == 0).unwrap_or(true);

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output)
        .expect("Failed to open output CSV");

    if write_header {
        writeln!(file, "implementation,n_points,dim,metric,time_s").unwrap();
    }
    writeln!(file, "rust,{},{},{},{:.3}", n, dim, metric, elapsed).unwrap();

    eprintln!("  Result appended to {}", output_path);
}
