//! Peak host heap use and time of the GPU kNN build. Ignored by default because of its size:
//! `GPU_PEAK_ROWS=200000 GPU_PEAK_COLS=384 cargo test -p nndescent --release --features gpu --test gpu_peak_memory -- --ignored --nocapture`

#![cfg(feature = "gpu")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;
use nndescent::NNDescent;

struct CountingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            record_growth(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
            record_growth(new_size);
        }
        new_ptr
    }
}

fn record_growth(size: usize) {
    let current = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(current, Ordering::Relaxed);
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn gpu_knn_peak_memory() {
    let rows = env_usize("GPU_PEAK_ROWS", 20_000);
    let cols = env_usize("GPU_PEAK_COLS", 384);
    let max_candidates: Vec<usize> = std::env::var("GPU_PEAK_CANDIDATES")
        .unwrap_or_else(|_| "22,60".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let mut rng = Xoshiro256StarStar::seed_from_u64(1);
    let data = Array2::from_shape_fn((rows, cols), |_| rng.random_f32());

    for max_candidates in max_candidates {
        let input = data.clone();
        let baseline = CURRENT.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        let start = Instant::now();
        let result = NNDescent::builder(input, "euclidean", 15)
            .max_candidates(max_candidates)
            .random_state(1)
            .gpu(true)
            .build();
        let elapsed = start.elapsed();
        let peak = PEAK.load(Ordering::Relaxed) - baseline;
        println!(
            "{rows} x {cols}, max_candidates {max_candidates}: peak {:.0} MB, {:.2} s, ok={}",
            peak as f64 / 1e6,
            elapsed.as_secs_f64(),
            result.is_ok()
        );
    }
}
