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
await umap.run();
```

At any time, you can get the current embedding by reading the `embedding` property.

```js
// The result is a Float32Array with count * outputDim elements.
let embedding = umap.embedding;
```

After you are done with the instance, use the `destroy` method to release resources.

```js
umap.destroy();
```

### Real-time updates

`run` runs the whole schedule of the optimizer. To show the layout while it changes, call `step` instead.
`step(n)` runs `n` epochs at the current learning rate and holds that rate constant, so you control the decay with `setParameters`:

```js
// Anneal the learning rate from 1 to 0.001 over 200 epochs and draw after every epoch.
while (umap.epoch < 200) {
  umap.setParameters({ learningRate: Math.pow(0.001, umap.epoch / 200) });
  await umap.step(1);
  draw(umap.embedding); // `embedding` is a reused buffer: copy it if you keep it.
}
```

`setParameters` also changes `minDist`, `spread`, `repulsionStrength`, and `negativeSampleRate` while the algorithm runs.
After the layout has converged, start the learning rate again to make a parameter change visible.
`setOptimizer` switches the optimizer, and `reset` re-initializes the layout (see below).

### UMAP options

The viewer computes projections in the browser with the umap-learn 0.6 pipeline.
In the settings of the viewer, the Optimizer option selects Adam (default), Momentum, or SGD.
The Repulsion option sets `repulsionStrength` (default 1). Values from 4 to 16 can resolve finer structure in large datasets.

`createUMAP` follows umap-learn: with no options it uses the umap-learn 0.5 pipeline (spectral initialization, SGD).
Set `optimizer` to `"adam"` or `"momentum"`, or set `compatibilityLayout: false`, to use the 0.6 pipeline instead (recursive initialization and updated spectral and nearest-neighbor parameters).
`compatibilityLayout: true` together with `"adam"` or `"momentum"` fails with an error.

```js
let umap = await createUMAP(count, inputDim, outputDim, data, {
  optimizer: "adam",
});
```

You can override the parts of the pipeline separately:

- `initializeMethod`: `"auto"`, `"spectral"`, `"tswspectral"`, `"random"`, `"recursive"`, or `"pca"`. The default `"auto"` is `"spectral"`, or `"recursive"` when `compatibilityLayout` is `false`. `"pca"` needs the input data, so `createUMAPFromKNN` rejects it.
- `optimizer`: `"sgd"`, `"momentum"` (the umap-learn 0.6 heavy-ball momentum), or `"adam"`. The default is `"sgd"`, or `"adam"` when `compatibilityLayout` is `false`.
- `initScaleFactor`: A multiplier for the rescale of the initial layout. It only applies when `compatibilityLayout` is `false`. The default is `1`.
- `initRescale`: How the initial layout is scaled when `compatibilityLayout` is `false`: `"isotropic"` (default), `"perAxis"`, or `"graphEdge"`.
- `a`, `b`: Parameters of the low-dimensional curve. Set both, or leave both unset to fit them from `minDist` and `spread`.
- `disconnectionDistance`: Neighbors at this distance or farther are dropped from the fuzzy graph. The default depends on the metric.
- `negativeSelectionRange`: The window of nodes that negative samples come from in the momentum and Adam optimizers. The default is `200000`.
- `negativeSampleScale`: A fixed scale for the repulsive force. The default is derived from `negativeSelectionRange`.
- `excludeGraphNeighbors`: Skip negative samples that are graph neighbors of the source. The default is `false`.
- `negativeSampleScaleAdaptationSamples`: The number of sources sampled to adapt the negative sample scale of Adam. `0` disables the adaptation. The default is `128`.
- `recursiveInit`: An object with options of the recursive initialization, for example `{ baseInitThreshold: 1024, coarseningRatio: 4, hubSelection: "degree" }`. Missing keys keep their defaults. See `RecursiveInitOptions` in the type definitions for all keys.

See `UMAPOptions` in the type definitions for all other options, such as `nNeighbors`, `minDist`, `spread`, `nEpochs`, `learningRate`, `repulsionStrength`, and `gpu`.

If you already have the nearest neighbors, use `createUMAPFromKNN(count, outputDim, knnIndices, knnDistances, options)`. It does not need the input vectors, and it accepts the same options.
With `gpu: true`, the Adam and momentum optimizers also run on the GPU.

The `reset` method re-initializes the embedding with any of the initialization methods and restarts from epoch 0.
The instance does not keep the input data. The `"pca"` method needs it as the second argument, and `"recursive"` uses it for its PCA anchor (and a random anchor without it):

```js
umap.reset("recursive", data);
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
