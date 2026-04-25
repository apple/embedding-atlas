import { UMAPBuilder, NNDescentBuilder, default as init } from "./pkg/umap_wasm.js";

/**
 * Initialize a UMAP instance.
 * @param {number} count the number of data points
 * @param {number} inputDim the input dimension
 * @param {number} outputDim the output dimension
 * @param {Float32Array} data the data array (count * inputDim elements)
 * @param {object} [options]
 * @returns {Promise<UMAP>}
 */
export async function createUMAP(count, inputDim, outputDim, data, options = {}) {
  await init();

  let builder = new UMAPBuilder(data, count, inputDim, outputDim);

  if (options.metric != null) builder = builder.metric(options.metric);
  if (options.nNeighbors != null) builder = builder.nNeighbors(options.nNeighbors);
  if (options.minDist != null) builder = builder.minDist(options.minDist);
  if (options.spread != null) builder = builder.spread(options.spread);
  if (options.nEpochs != null) builder = builder.nEpochs(options.nEpochs);
  if (options.learningRate != null) builder = builder.learningRate(options.learningRate);
  if (options.negativeSampleRate != null) builder = builder.negativeSampleRate(options.negativeSampleRate);
  if (options.repulsionStrength != null) builder = builder.repulsionStrength(options.repulsionStrength);
  if (options.localConnectivity != null) builder = builder.localConnectivity(options.localConnectivity);
  if (options.mixRatio != null) builder = builder.mixRatio(options.mixRatio);
  if (options.initializeMethod != null) builder = builder.initMethod(options.initializeMethod);
  if (options.seed != null) builder = builder.randomState(BigInt(options.seed));
  if (options.progress != null) builder = builder.progress(options.progress);
  if (options.gpu) builder = builder.gpu(true);

  let result = null;

  return {
    get inputDim() {
      return inputDim;
    },
    get outputDim() {
      return outputDim;
    },
    get embedding() {
      return result?.embedding ?? new Float32Array(0);
    },
    get knnIndices() {
      return result?.knnIndices ?? new Int32Array(0);
    },
    get knnDistances() {
      return result?.knnDistances ?? new Float32Array(0);
    },
    async run() {
      if (result) return;
      result = await builder.build();
    },
    destroy() {
      if (result) {
        result.free();
        result = null;
      }
    },
  };
}

/**
 * Create an NNDescent nearest neighbor index.
 * @param {number} count the number of data points
 * @param {number} inputDim the input dimension
 * @param {Float32Array} data the data array (count * inputDim elements)
 * @param {object} [options]
 * @returns {Promise<NNDescentResult>}
 */
export async function createNNDescent(count, inputDim, data, options = {}) {
  await init();

  const metric = options.metric ?? "euclidean";
  const epsilon = options.epsilon ?? 0.1;
  const buildK = options.nNeighbors ?? 15;

  let builder = new NNDescentBuilder(data, count, inputDim, metric, buildK);
  if (options.nTrees != null) builder = builder.nTrees(options.nTrees);
  if (options.nIters != null) builder = builder.nIters(options.nIters);
  if (options.delta != null) builder = builder.delta(options.delta);
  if (options.treeInit != null) builder = builder.treeInit(options.treeInit);
  if (options.seed != null) builder = builder.randomState(BigInt(options.seed));
  if (options.progress != null) builder = builder.progress(options.progress);
  if (options.gpu) builder = builder.gpu(true);

  let index = await builder.build();

  // Extract the pre-computed neighbor graph for fast queryByIndex lookups.
  const graph = index.neighborGraph();
  const graphIndices = new Int32Array(graph.indices);
  const graphDistances = new Float32Array(graph.distances);
  graph.free();

  // Prepare the search graph for queryByVector.
  index.prepare();

  let destroyed = false;

  return {
    queryByIndex(idx, k) {
      if (idx < 0 || idx >= count) {
        throw new RangeError(`Index ${idx} out of bounds for ${count} points`);
      }
      if (k <= buildK) {
        // Slice directly from the pre-computed neighbor graph.
        const offset = idx * buildK;
        return {
          indices: graphIndices.slice(offset, offset + k),
          distances: graphDistances.slice(offset, offset + k),
        };
      }
      // Fall back to search for k larger than what was pre-computed.
      const vector = data.subarray(idx * inputDim, (idx + 1) * inputDim);
      return this.queryByVector(vector, k);
    },
    queryByVector(vector, k) {
      const result = index.query(vector, 1, inputDim, k, epsilon);
      const indices = new Int32Array(result.indices);
      const distances = new Float32Array(result.distances);
      result.free();
      return { indices, distances };
    },
    destroy() {
      if (!destroyed) {
        index.free();
        destroyed = true;
      }
    },
  };
}
