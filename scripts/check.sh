#!/bin/bash

# Type check all TypeScript packages

# Currently type checks don't cause a CI failure
# set -euxo pipefail

pushd packages/component
npm run check
popd

pushd packages/table
npm run check
popd

pushd packages/viewer
npm run check
popd

pushd packages/embedding-atlas
npm run check
popd

pushd packages/examples
npm run check
popd
