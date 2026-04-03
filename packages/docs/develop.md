# Development Instructions

[View the code on GitHub](https://github.com/apple/embedding-atlas).

This repository is organized as a monorepo with the following packages:

**Frontend:**

- `packages/component`: The `EmbeddingView` and `EmbeddingViewMosaic` components.
- `packages/viewer`: The frontend application for visualizing embeddings and other columns. It also provides the `EmbeddingAtlas` component that can be embedded in other applications.
- `packages/utils`: Shared utilities.
- `packages/embedding-atlas`: The published `embedding-atlas` package. It re-exports the above packages as a single unified API.
- `packages/examples`: Examples showing how to use the `embedding-atlas` package.

**Rust / WebAssembly:**

- `packages/density-clustering`: A density clustering algorithm, written in Rust and compiled to WebAssembly.
- `packages/umap`: A Rust implementation of the UMAP and NNDescent algorithms, compiled to WebAssembly.

**Python:**

- `packages/backend`: A Python package named `embedding-atlas` that provides the `embedding-atlas` command-line tool.

**Documentation:**

- `packages/docs`: The documentation website.

## Prerequisites

- [Node.js](https://nodejs.org/) and npm
- [uv](https://docs.astral.sh/uv/) package manager
- [Rust](https://www.rust-lang.org/)
- WebAssembly target: `rustup target add wasm32-unknown-unknown`
- wasm-bindgen CLI: `cargo install -f wasm-bindgen-cli --version 0.2.114`

## Install and Build

Install dependencies:

```bash
npm install
```

Build all packages:

```bash
npm run build
```

This builds all packages, including the WASM packages (`umap-wasm` and `density-clustering`).

## Development

Launch the command-line tool with a demo dataset:

```bash
cd packages/backend
./start.sh
```

Start the development server for the `viewer` package:

```bash
cd packages/viewer
npm run dev
```

The `viewer` package is the main Embedding Atlas UI. Once `npm run dev` is running,
it serves the UI at `http://localhost:5173`. The UI requires a
backend server at `http://localhost:5055` to provide data. You can start one via
`./start.sh` as described above. Without a backend server, you can still visit
`http://localhost:5173/#/test` for a test dataset, or `http://localhost:5173/#/file` to use the file loader.

Start the development server for the `component` package:

```bash
cd packages/component
npm run dev
```

Start the development server for the `examples` package:

```bash
cd packages/examples
npm run dev
```

## Unit Tests

To run tests for individual packages:

```bash
# JavaScript tests
cd packages/utils
npm run test

# Python tests
cd packages/backend
uv run pytest

# Rust tests
cd packages/density-clustering
cargo test
```

To run all JavaScript, Python, and Rust tests at once:

```bash
npm run test
```

## Deployment

Packages and the documentation website are deployed via [GitHub Actions](https://github.com/apple/embedding-atlas/blob/main/.github/workflows/ci.yml).
Deployment is triggered when a release is published with a tag matching `vX.Y.Z`.

The documentation website can be deployed separately by manually running the workflow with "Publish Documentation Website" enabled.
