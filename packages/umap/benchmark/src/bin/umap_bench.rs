//! UMAP benchmark binary.
//!
//! Usage: umap-bench <data_dir> <metric> [--gpu] [--compatibility-layout true|false]
//! [--optimizer sgd|momentum|adam] [--init auto|spectral|tswspectral|random|pca|recursive]
//! [--n-neighbors N] [--min-dist F] [--spread F] [--n-components N] [--n-epochs N]
//! [--learning-rate F] [--repulsion-strength F] [--negative-sample-rate N]
//! [--local-connectivity F] [--set-op-mix-ratio F]
//! [--output result.json] [--seed N] [--embedding-path path]
//!
//! Reads meta.json + data.bin, runs UMAP (2D), saves embedding and timing.

use std::cell::RefCell;
use std::env;
use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use ndarray::Array2;
use umap::{Init, Optimizer};

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
    let mut use_gpu = false;
    let mut compatibility_layout = true;
    let mut optimizer: Option<Optimizer> = None;
    let mut init: Option<Init> = None;
    let mut n_neighbors = 15usize;
    let mut min_dist = 0.1f32;
    let mut spread: Option<f32> = None;
    let mut n_components = 2usize;
    let mut n_epochs: Option<usize> = None;
    let mut learning_rate: Option<f32> = None;
    let mut repulsion_strength: Option<f32> = None;
    let mut negative_sample_rate: Option<usize> = None;
    let mut local_connectivity: Option<f32> = None;
    let mut set_op_mix_ratio: Option<f32> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--n-neighbors" => {
                n_neighbors = args[i + 1].parse().expect("Invalid --n-neighbors");
                i += 2;
            }
            "--min-dist" => {
                min_dist = args[i + 1].parse().expect("Invalid --min-dist");
                i += 2;
            }
            "--spread" => {
                spread = Some(args[i + 1].parse().expect("Invalid --spread"));
                i += 2;
            }
            "--n-components" => {
                n_components = args[i + 1].parse().expect("Invalid --n-components");
                i += 2;
            }
            "--n-epochs" => {
                n_epochs = Some(args[i + 1].parse().expect("Invalid --n-epochs"));
                i += 2;
            }
            "--learning-rate" => {
                learning_rate = Some(args[i + 1].parse().expect("Invalid --learning-rate"));
                i += 2;
            }
            "--repulsion-strength" => {
                repulsion_strength =
                    Some(args[i + 1].parse().expect("Invalid --repulsion-strength"));
                i += 2;
            }
            "--negative-sample-rate" => {
                negative_sample_rate =
                    Some(args[i + 1].parse().expect("Invalid --negative-sample-rate"));
                i += 2;
            }
            "--local-connectivity" => {
                local_connectivity =
                    Some(args[i + 1].parse().expect("Invalid --local-connectivity"));
                i += 2;
            }
            "--set-op-mix-ratio" => {
                set_op_mix_ratio = Some(args[i + 1].parse().expect("Invalid --set-op-mix-ratio"));
                i += 2;
            }
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
            "--compatibility-layout" => {
                compatibility_layout = args[i + 1]
                    .parse()
                    .expect("--compatibility-layout expects true or false");
                i += 2;
            }
            "--optimizer" => {
                optimizer = Some(args[i + 1].parse().unwrap_or_else(|e| panic!("{e}")));
                i += 2;
            }
            "--init" => {
                init = Some(args[i + 1].parse().unwrap_or_else(|e| panic!("{e}")));
                i += 2;
            }
            "--gpu" => {
                use_gpu = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    let output_path =
        output_path.unwrap_or_else(|| data_dir.join("result.json").to_string_lossy().to_string());

    // Read metadata
    let meta_str =
        fs::read_to_string(data_dir.join("meta.json")).expect("Failed to read meta.json");
    let n = extract_json_int(&meta_str, "n_points");
    let dim = extract_json_int(&meta_str, "dim");

    eprintln!(
        "Rust UMAP benchmark: n={}, dim={}, metric={}, gpu={}, compatibility_layout={}",
        n, dim, metric, use_gpu, compatibility_layout
    );

    // Load data
    eprintln!("  Loading data...");
    let data = read_f32_bin(&data_dir.join("data.bin"), n, dim);

    // Configure and run UMAP
    eprintln!("  Running UMAP...");
    let start = Instant::now();
    let stage_starts: Rc<RefCell<Vec<(String, f64)>>> = Rc::new(RefCell::new(Vec::new()));
    let recorder = Rc::clone(&stage_starts);
    let mut builder = umap::Umap::builder(&data)
        .n_neighbors(n_neighbors)
        .n_components(n_components)
        .min_dist(min_dist)
        .metric(metric)
        .progress(Box::new(move |_, stage| {
            let mut starts = recorder.borrow_mut();
            if !stage.starts_with("NNDescent") && !starts.iter().any(|(s, _)| s == stage) {
                starts.push((stage.to_string(), start.elapsed().as_secs_f64()));
            }
        }))
        .gpu(use_gpu)
        .compatibility_layout(compatibility_layout)
        .verbose(true);
    if let Some(v) = spread {
        builder = builder.spread(v);
    }
    if let Some(v) = n_epochs {
        builder = builder.n_epochs(v);
    }
    if let Some(v) = learning_rate {
        builder = builder.learning_rate(v);
    }
    if let Some(v) = repulsion_strength {
        builder = builder.repulsion_strength(v);
    }
    if let Some(v) = negative_sample_rate {
        builder = builder.negative_sample_rate(v);
    }
    if let Some(v) = local_connectivity {
        builder = builder.local_connectivity(v);
    }
    if let Some(v) = set_op_mix_ratio {
        builder = builder.set_op_mix_ratio(v);
    }
    if let Some(o) = optimizer {
        builder = builder.optimizer(o);
    }
    if let Some(i) = init {
        builder = builder.init_method(i);
    }
    if let Some(s) = seed {
        builder = builder.random_state(s);
    }
    let result = builder.build().expect("UMAP build failed");
    let elapsed = start.elapsed().as_secs_f64();
    let stage_time = |name: &str| {
        stage_starts
            .borrow()
            .iter()
            .find(|(s, _)| s == name)
            .map(|(_, t)| *t)
    };
    let knn_s = stage_time(umap::STAGE_GRAPH).unwrap_or(0.0);
    let layout_s = stage_time(umap::STAGE_OPTIMIZATION).map_or(0.0, |t| elapsed - t);

    eprintln!("  UMAP time: {:.3}s", elapsed);

    // Save embedding
    let emb_path = match &embedding_path {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let prefix = if use_gpu { "rust_gpu" } else { "rust" };
            data_dir.join(format!("{}_{}_embedding.bin", prefix, metric))
        }
    };
    write_f32_bin(&emb_path, &result.embedding);
    eprintln!("  Saved embedding to {:?}", emb_path);

    // Write JSON result
    let json = format!(
        concat!(
            "{{\n",
            "  \"implementation\": \"rust\",\n",
            "  \"n_points\": {},\n",
            "  \"dim\": {},\n",
            "  \"metric\": \"{}\",\n",
            "  \"gpu\": {},\n",
            "  \"layout\": \"{}\",\n",
            "  \"time_s\": {:.3},\n",
            "  \"knn_s\": {:.3},\n",
            "  \"layout_s\": {:.3}\n",
            "}}\n"
        ),
        n,
        dim,
        metric,
        use_gpu,
        if compatibility_layout { "0.5" } else { "0.6" },
        elapsed,
        knn_s,
        layout_s
    );
    fs::write(&output_path, &json).unwrap_or_else(|_| panic!("Failed to write {}", output_path));

    eprintln!("  Result written to {}", output_path);
}
