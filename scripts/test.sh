#!/bin/bash

# Run tests

set -euxo pipefail

pushd packages/density-clustering
npm run test
popd

pushd packages/umap-wasm
npm run test
popd
