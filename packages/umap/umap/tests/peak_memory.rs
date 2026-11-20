//! Peak heap use of the build path. Ignored by default because of its size:
//! `UMAP_PEAK_ROWS=200000 UMAP_PEAK_COLS=384 cargo test -p umap --release --test peak_memory -- --ignored --nocapture`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Array2;
use nndescent::rng::Xoshiro256StarStar;
use umap::{Init, Umap, UmapBuilder};

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

fn peak_above_baseline(run: impl FnOnce()) -> usize {
    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    run();
    PEAK.load(Ordering::Relaxed) - baseline
}

fn configure(builder: UmapBuilder<'_>) -> UmapBuilder<'_> {
    builder
        .n_epochs(1)
        .init_method(Init::Random)
        .random_state(1)
}

#[test]
#[ignore]
fn shared_data_needs_less_peak_memory_than_borrowed_data() {
    let rows = env_usize("UMAP_PEAK_ROWS", 20_000);
    let cols = env_usize("UMAP_PEAK_COLS", 384);
    let mut rng = Xoshiro256StarStar::seed_from_u64(1);
    let data = Array2::from_shape_fn((rows, cols), |_| rng.random_f32());
    let data_mb = (rows * cols * 4) as f64 / 1e6;

    let borrowed = peak_above_baseline(|| {
        configure(Umap::builder(&data)).build().unwrap();
    });

    let shared_data = Arc::new(data);
    let shared = peak_above_baseline(|| {
        configure(Umap::builder_shared(Arc::clone(&shared_data)))
            .build()
            .unwrap();
    });

    println!(
        "{rows} x {cols} ({data_mb:.0} MB input): extra peak borrowed {:.0} MB, shared {:.0} MB",
        borrowed as f64 / 1e6,
        shared as f64 / 1e6
    );
    assert!(shared + (data_mb * 0.9) as usize * 1_000_000 <= borrowed);
}
