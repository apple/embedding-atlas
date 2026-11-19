# UMAP (Rust)

A pure Rust implementation of UMAP (Uniform Manifold Approximation and Projection) and NNDescent (approximate nearest neighbor search), with WebAssembly bindings for browser use.

This implementation is based on the original Python libraries:

- **[umap-learn](https://github.com/lmcinnes/umap)** — UMAP algorithm by Leland McInnes
- **[pynndescent](https://github.com/lmcinnes/pynndescent)** — Nearest neighbor descent by Leland McInnes

## Structure

This directory contains four crates that are members of the top-level Cargo workspace:

### `nndescent/`

Approximate k-nearest neighbor graph construction, ported from PyNNDescent.

- `distance.rs` — Distance metrics (euclidean, cosine, manhattan, chebyshev, minkowski) with SIMD acceleration via `wide`
- `nn_descent.rs` — Core NN-descent algorithm for iterative neighbor graph refinement
- `rp_trees.rs` — Random projection trees for initialization and search
- `heap.rs` — Max-heap for efficient k-NN tracking
- `search.rs` — Query functionality over constructed indices
- `rng.rs` — Tau RNG matching the Python implementation for reproducibility

### `umap/`

UMAP dimensionality reduction, ported from umap-learn.

- `lib.rs` — Public API with `UmapBuilder` pattern. Orchestrates the full pipeline: neighbor finding, graph construction, initialization, and optimization
- `graph.rs` — Sparse matrix operations, fuzzy simplicial set construction, and symmetrization
- `spectral.rs` — Spectral initialization via LOBPCG eigendecomposition, with Jacobi and Cholesky solvers
- `optimize.rs` — Layout optimization using SGD with Hogwild parallelism and negative sampling. Also hosts the stateful `UmapOptimizer` that selects between SGD, momentum, and Adam
- `init.rs` — Helpers for initial layouts: quantiles and the isotropic rescale of the initial embedding
- `layout_modern.rs` — The umap-learn 0.6 Adam and momentum layout optimizers with deterministic hashed negative sampling and Jacobi updates
- `gpu_layout_modern.rs` and `shaders/layout_modern.wgsl` — WebGPU implementation of the 0.6 Adam and momentum optimizers
- `recursive_init.rs` — The umap-learn 0.6 recursive initialization: label-propagation coarsening of the fuzzy graph, anchored on a PCA of the data

#### Compatibility switch

`UmapBuilder::compatibility_layout(true)` is the default. It reproduces the umap-learn 0.5 pipeline: spectral initialization and the SGD optimizer.
`compatibility_layout(false)` selects the umap-learn 0.6 pipeline: recursive initialization, the Adam optimizer, an isotropic rescale of the initial layout (`init_scale_factor`), and updated spectral and NNDescent parameters.
`init_method` and `optimizer` override the defaults of either pipeline. `UmapSetup` returns an `InitPlan` so that the embedding can be re-initialized with another method after setup. The caller supplies the data again if the method needs it.

### `umap-wasm/`

WebAssembly bindings via `wasm-bindgen`, exposing both UMAP and NNDescent to JavaScript/TypeScript.

### `benchmark/`

Native benchmark binaries (`umap-bench`, `nndescent-bench`) for performance testing.

Run the benchmarks with `uv run python -m benchmark umap`.

#### Browser benchmark

`benchmark/browser/benchmark.html` times the pipeline variants (0.5 SGD, 0.6 Adam, 0.6 Momentum, on the GPU and the CPU) in the browser. It runs in a web worker, so the page stays responsive. Serve `packages/umap` and open <http://localhost:8000/benchmark/browser/benchmark.html>:

```bash
cd packages/umap && python3 -m http.server 8000
```

Pick synthetic datasets or the sweep datasets (create them with `python -m benchmark datasets`). Row counts such as `1000,5000,10000` run each dataset on its first N rows. The table shows the time of each stage, and you can copy it as Markdown or download it as JSON.
