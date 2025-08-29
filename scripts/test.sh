#!/bin/bash

# Build WASM modules

set -euxo pipefail

pushd packages/density-clustering
npm run test
popd

pushd packages/umap-wasm
npm run test
popd