#!/bin/bash

# Build all JavaScript and Python packages

set -euxo pipefail

pushd packages/component
npm run package
popd

pushd packages/table
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
