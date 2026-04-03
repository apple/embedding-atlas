# UMAP (Rust)

A pure Rust implementation of UMAP (Uniform Manifold Approximation and Projection) and NNDescent (approximate nearest neighbor search), with WebAssembly bindings for browser use.

This implementation is based on the original Python libraries:

- **[umap-learn](https://github.com/lmcinnes/umap)** — UMAP algorithm by Leland McInnes
- **[pynndescent](https://github.com/lmcinnes/pynndescent)** — Nearest neighbor descent by Leland McInnes

## Structure

This workspace contains three crates:

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
- `optimize.rs` — Layout optimization using SGD with Hogwild parallelism and negative sampling

### `umap-wasm/`

WebAssembly bindings via `wasm-bindgen`, exposing both UMAP and NNDescent to JavaScript/TypeScript.
