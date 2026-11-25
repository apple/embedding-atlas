//! Times the fuzzy-graph pipeline on a precomputed kNN graph and prints a
//! checksum of the result, so implementations can be compared for speed and
//! exact output parity.
//!
//!   cargo run --release --example graph_bench -- IDX.bin DIST.bin N K
//!
//! IDX.bin: little-endian i32 (N, K); DIST.bin: little-endian f32 (N, K).

use std::time::Instant;

use ndarray::Array2;
use umap::graph::{compute_membership_strengths, normalize_knn_self_column, smooth_knn_dist};

fn read_bin<T: Copy>(path: &str, len: usize, parse: fn([u8; 4]) -> T) -> Vec<T> {
    let bytes = std::fs::read(path).expect("read input");
    assert_eq!(bytes.len(), len * 4, "unexpected size for {path}");
    bytes.as_chunks::<4>().0.iter().map(|&c| parse(c)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (n, k): (usize, usize) = (args[3].parse().unwrap(), args[4].parse().unwrap());
    let idx =
        Array2::from_shape_vec((n, k), read_bin(&args[1], n * k, i32::from_le_bytes)).unwrap();
    let dist =
        Array2::from_shape_vec((n, k), read_bin(&args[2], n * k, f32::from_le_bytes)).unwrap();

    let t = Instant::now();
    let (idx, dist) = normalize_knn_self_column(&idx, &dist);
    let t_norm = t.elapsed();

    let t = Instant::now();
    let (sigmas, rhos) = smooth_knn_dist(&dist, idx.ncols() as f32, 64, 1.0, 1.0);
    let t_smooth = t.elapsed();

    let t = Instant::now();
    let graph = compute_membership_strengths(&idx, &dist, &sigmas, &rhos);
    let t_member = t.elapsed();

    let t = Instant::now();
    let sym = graph.symmetrize(1.0);
    let t_sym = t.elapsed();

    let mut h: u64 = 1469598103934665603;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(1099511628211);
    };
    for &p in &sym.indptr {
        mix(p as u64);
    }
    for &c in &sym.indices {
        mix(c as u64);
    }
    for &v in &sym.data {
        mix(v.to_bits() as u64);
    }
    for &s in &sigmas {
        mix(s.to_bits() as u64);
    }

    println!(
        "n={n} k={k} nnz={} normalize={:.3}s smooth={:.3}s membership={:.3}s symmetrize={:.3}s total={:.3}s checksum={h:016x}",
        sym.nnz(),
        t_norm.as_secs_f64(),
        t_smooth.as_secs_f64(),
        t_member.as_secs_f64(),
        t_sym.as_secs_f64(),
        (t_norm + t_smooth + t_member + t_sym).as_secs_f64(),
    );
}
