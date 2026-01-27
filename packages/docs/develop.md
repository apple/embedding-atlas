# Development Instructions

[View the code on GitHub](https://github.com/apple/embedding-atlas).

This repo contains multiple sub-packages:

Frontend:

- `packages/component`: The `EmbeddingView` and `EmbeddingViewMosaic` components.

- `packages/table`: The `Table` component.

- `packages/viewer`: The frontend application for visualizing embedding and other columns. It also provides the `EmbeddingAtlas` component that can be embedded in other applications.

- `packages/density-clustering`: The density clustering algorithm, written in Rust.

- `packages/umap-wasm`: An implementation of UMAP algorithm in WebAssembly (with the [umappp](https://github.com/libscran/umappp) C++ library).

- `packages/embedding-atlas`: The `embedding-atlas` package that get published. It imports all of the above and exposes their API in a single package.

Python:

- `packages/backend`: A Python package named `embedding-atlas` that provides the `embedding-atlas` command line tool.

Documentation:

- `packages/docs`: The documentation website.

## Prerequisites

Basic development:

- Ensure Node.js and npm are installed.
- Install the [uv package manager](https://docs.astral.sh/uv/).

Additional requirements for WASM packages:

- Install [Emscripten](https://emscripten.org/) to develop the `umap-wasm` package.
- Install [Rust](https://www.rust-lang.org/) and [wasm-pack](https://rustwasm.github.io/wasm-pack/) to develop the `density-clustering` package.

## Install and Build

Install the development setup by running:

```bash
npm install
```

You can also build all packages by running:

```bash
npm run build
```

This command builds all packages except the WASM packages (`umap-wasm` and `density-clustering`).
Pre-built WASM outputs (included in the repository to ease development) are used.

You may launch the command-line tool with a demo dataset:

```bash
cd packages/backend
./start.sh
```

To start development server for the frontend packages (`component`, `table`, `viewer`):

```bash
cd packages/viewer # or packages/component, packages/table.
npm run dev
```

Check the individual package READMEs for specific instructions.

## Deployment

The packages and website for the project are deployed with [GitHub Actions](https://github.com/apple/embedding-atlas/blob/main/.github/workflows/ci.yml).
Deployment triggers when a released is published with a tag `vX.Y.Z`.

The documentation website can be deployed by manually running the workflow with "Publish Documentation Website" enabled.
