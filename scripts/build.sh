#!/bin/bash

set -euo pipefail

# Check build dependencies
missing=0

if ! command -v npm &> /dev/null; then
  echo "Error: npm is not installed. Install Node.js and npm from https://nodejs.org/"
  missing=1
fi

if ! command -v cargo &> /dev/null; then
  echo "Error: cargo is not installed. Install Rust from https://www.rust-lang.org/"
  missing=1
fi

if ! command -v wasm-bindgen &> /dev/null; then
  echo "Error: wasm-bindgen is not installed. Install it with: cargo install -f wasm-bindgen-cli --version 0.2.114"
  missing=1
fi

if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
  missing=1
fi

if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  echo "Error: wasm32-unknown-unknown target is not installed. Add it with: rustup target add wasm32-unknown-unknown"
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  exit 1
fi

# Build WASM modules
pushd packages/density-clustering
npm run build
popd

pushd packages/umap/umap-wasm
npm run build
popd

# Build all JavaScript and Python packages

pushd packages/utils
npm run package
popd

pushd packages/component
npm run package
popd

pushd packages/viewer
npm run package
popd

pushd packages/embedding-atlas
npm run package
popd

pushd packages/examples
npm run build
popd

pushd packages/backend
./build.sh
popd

pushd packages/docs
npm run build
popd
