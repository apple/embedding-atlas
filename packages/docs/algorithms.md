# Algorithms

The `embedding-atlas` package contains some useful algorithms for computing embeddings and clustering.

## UMAP

This package provides a WebAssembly implementation of [UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction)](https://umap-learn.readthedocs.io/en/latest/) and approximate nearest neighbor search.

The implementation is based on the original Python libraries [umap-learn](https://github.com/lmcinnes/umap) and [pynndescent](https://github.com/lmcinnes/pynndescent) by Leland McInnes, ported to Rust and compiled to WebAssembly.

To initialize the UMAP algorithm, use `createUMAP`:

```js
import { createUMAP } from "embedding-atlas";

let count = 2000;
let inputDim = 100;
let outputDim = 2;

// The data must be a Float32Array with count * inputDim elements.
let data = new Float32Array(count * inputDim);
// ... fill in the data

let options = {
  metric: "cosine",
};

// Use `createUMAP` to initialize the algorithm.
let umap = await createUMAP(count, inputDim, outputDim, data, options);
```

After initialization, use the `run` method to update the embedding coordinates:

```js
// Run the algorithm to completion.
umap.run();
```

At any time, you can get the current embedding by calling the `embedding` method.

```js
// The result is a Float32Array with count * outputDim elements.
let embedding = umap.embedding();
```

After you are done with the instance, use the `destroy` method to release resources.

```js
umap.destroy();
```

## NNDescent

You can use the `createNNDescent` function to perform approximate nearest neighbor search:

```js
import { createNNDescent } from "embedding-atlas";

let count = 2000;
let inputDim = 100;

// The data must be a Float32Array with count * inputDim elements.
let data = new Float32Array(count * inputDim);
// ... fill in the data

let options = {
  metric: "cosine",
};

// Create the NNDescent index
let index = await createNNDescent(count, inputDim, data, options);

// Perform queries
let query = new Float32Array(inputDim);
index.queryByVector(query, k);

// Destroy the instance
index.destroy();
```

## Density-based Clustering

This package provides a WebAssembly implementation of a density map clustering algorithm.
To run the algorithm, use `findClusters`.

```js
import { findClusters } from "embedding-atlas";

// A density map of width * height floating point numbers.
let densityMap: Float32Array;

let clusters = await findClusters(densityMap, width, height);
```

`findClusters` returns an array of clusters, as described below:

<!-- @doc(ts,no-required): Cluster -->
