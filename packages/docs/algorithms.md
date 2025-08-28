# Algorithms

The `embedding-atlas` package contains some useful algorithms for computing embeddings and clustering.

## UMAP

This package provides a WebAssembly implementation of [UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction)](https://umap-learn.readthedocs.io/en/latest/) and approximate nearest neighbor search.

The UMAP implementation is [umappp](https://github.com/libscran/umappp/) by Aaron Lun.

The approximate nearest neighbor search is based on [knncolle](https://github.com/knncolle/knncolle), with two algorithms, including HNSW ([hnswlib](https://github.com/nmslib/hnswlib)) and [nndescent](https://github.com/brj0/nndescent) by Jon Brugger.

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

// Alternatively, you can run up to a given number of epochs.
// This can be useful for animation effects.
for (let i = 0; i < 100; i++) {
  // Run to the i-th epoch.
  umap.run(i);
}
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

## KNN

In addition, you can also use the `createKNN` function to perform approximate nearest neighbor search (with hnswlib or nndescent):

```js
import { createKNN } from "embedding-atlas";

let count = 2000;
let inputDim = 100;

// The data must be a Float32Array with count * inputDim elements.
let data = new Float32Array(count * inputDim);
// ... fill in the data

let options = {
  metric: "cosine",
};

// Create the KNN instance
let knn = await createKNN(count, inputDim, data, options);

// Perform queries
let query = new Float32Array(inputDim);
knn.queryByVector(query, k);

// Destroy the instance
knn.destroy();
```

## Density-based Clustering

This package provides a WebAssembly implementation of a density map clustering algorithm.
To run the algorithm, use `findClusters`.

```js
import { findClusters } from "embedding-atlas";

// A density map of width * height floating point numbers.
let densityMap: Float32Array;

clusters = await findClusters(densityMap, width, height);
```
