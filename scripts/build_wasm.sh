#!/bin/bash

# Build WASM modules

set -euxo pipefail

pushd packages/density-clustering
npm run build
popd

pushd packages/umap-wasm

pushd third_party
./download_dependencies.sh
popd

make

popd