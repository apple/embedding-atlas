import { NNDescentBuilder, UMAPBuilder, UMAPFromKnnBuilder, default as init } from "./pkg/umap_wasm.js";

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
  if (options.optimizer != null) builder = builder.optimizer(options.optimizer);
  if (options.seed != null) builder = builder.randomState(BigInt(options.seed));
  if (options.progress != null) builder = builder.progress(options.progress);
  if (options.gpu) builder = builder.gpu(true);

  // Eager setup: builds the graph + initial embedding (so `embedding` is valid
  // immediately at epoch 0). `run`/`step` advance the layout from there.
  const umap = await builder.build();

  return wrapUmap(umap, count, inputDim, outputDim);
}

/**
 * Build a UMAP instance from a precomputed kNN graph, skipping the internal
 * nearest-neighbor search. No high-dimensional data is needed.
 *
 * `knnIndices`/`knnDistances` are row-major (`count * k`); row `i` lists point
 * `i`'s neighbors sorted by ascending distance. Caller's contract (the graph is
 * not validated beyond an index-range check, and bad input degrades silently
 * rather than throwing): every index is valid (`0 <= idx < count`); point `i`
 * appears in its own row at most once (self may be column 0 with distance 0, or
 * omitted); distances are finite (a `NaN` propagates to a `NaN` embedding); and the
 * neighbor count is uniform across rows (the effective `k` is the per-row minimum,
 * so one short row trims every row).
 *
 * @param {number} count the number of data points
 * @param {number} outputDim the output dimension
 * @param {Int32Array} knnIndices row-major neighbor indices (count * k)
 * @param {Float32Array} knnDistances row-major neighbor distances (count * k)
 * @param {object} [options] layout options (UMAPOptions minus metric/nNeighbors)
 * @returns {Promise<UMAP>}
 */
export async function createUMAPFromKNN(count, outputDim, knnIndices, knnDistances, options = {}) {
  await init();

  let builder = new UMAPFromKnnBuilder(knnIndices, knnDistances, count, outputDim);

  if (options.minDist != null) builder = builder.minDist(options.minDist);
  if (options.spread != null) builder = builder.spread(options.spread);
  if (options.nEpochs != null) builder = builder.nEpochs(options.nEpochs);
  if (options.learningRate != null) builder = builder.learningRate(options.learningRate);
  if (options.negativeSampleRate != null) builder = builder.negativeSampleRate(options.negativeSampleRate);
  if (options.repulsionStrength != null) builder = builder.repulsionStrength(options.repulsionStrength);
  if (options.localConnectivity != null) builder = builder.localConnectivity(options.localConnectivity);
  if (options.mixRatio != null) builder = builder.mixRatio(options.mixRatio);
  if (options.initializeMethod != null) builder = builder.initMethod(options.initializeMethod);
  if (options.optimizer != null) builder = builder.optimizer(options.optimizer);
  if (options.seed != null) builder = builder.randomState(BigInt(options.seed));
  if (options.progress != null) builder = builder.progress(options.progress);
  if (options.gpu) builder = builder.gpu(true);

  const umap = await builder.build();

  // No high-dim data was supplied, so the input dimension is unknown (reported as 0).
  return wrapUmap(umap, count, 0, outputDim);
}

/**
 * A tiny async sequencer that serializes access to a resource which can only be
 * touched while no async section holds it.
 *
 * `step`/`run` are async wasm methods: wasm-bindgen keeps a mutable borrow on the
 * instance across their GPU-readback await, so touching the instance while one is
 * in flight throws ("recursive use of an object detected", or for free()
 * "attempted to take ownership ... while borrowed").
 *
 * - `enqueue(fn)` runs `fn` as an exclusive section, serialized in submission
 *   order so overlapping `step`/`run` calls queue rather than overlap the wasm
 *   borrow. The chain survives a rejected section, so a later queued one — notably
 *   `destroy()`'s `free()` — still runs.
 * - `runOrQueue(label, action)` is for synchronous wasm ops (setParameters/reset/
 *   free): it runs `action` immediately when nothing is queued or running — so the
 *   effect is observable at once — otherwise queues it behind the in-flight section
 *   (never mid-borrow). Fire-and-forget, so a queued rejection is logged.
 * - `running` is true only while a section is actually executing (the borrow is
 *   held); getters consult it to serve a snapshot instead of reading the borrowed
 *   instance.
 */
function createSequencer() {
  let running = false;
  let pending = 0; // queued-or-running sections
  let tail = Promise.resolve(); // serial chain; sections run in submission order

  function enqueue(fn) {
    pending++;
    const result = tail.then(async () => {
      running = true;
      try {
        return await fn();
      } finally {
        running = false;
        pending--;
      }
    });
    // Keep the chain alive past a rejection so later sections (e.g. free) run.
    tail = result.then(
      () => {},
      () => {},
    );
    return result;
  }

  function runOrQueue(label, action) {
    if (pending === 0) {
      action();
    } else {
      enqueue(action).catch((e) => console.error(`UMAP: ${label} failed`, e));
    }
  }

  return {
    get running() {
      return running;
    },
    enqueue,
    runOrQueue,
  };
}

/**
 * Wrap a built wasm `Umap` instance in the stable JS UMAP interface. A
 * {@link createSequencer} mediates all wasm access so callers can safely step in
 * a requestAnimationFrame loop while tuning parameters, reading the embedding,
 * or tearing the instance down. Shared by `createUMAP` and `createUMAPFromKNN`.
 * @param {import("./pkg/umap_wasm.js").Umap} umap
 * @param {number} count
 * @param {number} inputDim
 * @param {number} outputDim
 * @returns {UMAP}
 */
function wrapUmap(umap, count, inputDim, outputDim) {
  const seq = createSequencer();
  let destroyed = false;

  // JS-side snapshots served while a step/run holds the wasm borrow. All are
  // primed now, at construction, while idle. epoch and embedding change over
  // time, so they are also refreshed on later idle reads; the kNN graph is
  // immutable, so capturing it once here is enough (and means the getters can't
  // return an empty array when first read during an in-flight step).
  const outBuf = new Float32Array(count * outputDim);
  umap.copyEmbeddingInto(outBuf);
  let lastEpoch = umap.epoch;
  const knnIndicesSnapshot = umap.knnIndices;
  const knnDistancesSnapshot = umap.knnDistances;

  // Merged target for setParameters (supports partial updates; applied via the
  // sequencer so it lands immediately when idle, else once the in-flight/queued
  // step/run settles).
  let desiredParams = null;
  function applyDesiredParams() {
    const p = desiredParams;
    umap.setParameters(p.learningRate, p.repulsionStrength, p.negativeSampleRate, p.minDist, p.spread);
  }

  return {
    get inputDim() {
      return inputDim;
    },
    get outputDim() {
      return outputDim;
    },
    get epoch() {
      if (destroyed) return 0;
      if (!seq.running) lastEpoch = umap.epoch;
      return lastEpoch;
    },
    get embedding() {
      if (destroyed) return new Float32Array(0);
      if (!seq.running) umap.copyEmbeddingInto(outBuf);
      return outBuf;
    },
    get knnIndices() {
      return destroyed ? new Int32Array(0) : knnIndicesSnapshot;
    },
    get knnDistances() {
      return destroyed ? new Float32Array(0) : knnDistancesSnapshot;
    },
    async run() {
      if (destroyed) return;
      await seq.enqueue(() => umap.run());
    },
    async step(nEpochs = 1) {
      if (destroyed) return;
      await seq.enqueue(() => umap.step(nEpochs));
    },
    setParameters(params = {}) {
      if (destroyed) return;
      desiredParams = { ...(desiredParams ?? {}), ...params };
      seq.runOrQueue("setParameters", applyDesiredParams);
    },
    setOptimizer(optimizer) {
      if (destroyed) return;
      seq.runOrQueue("setOptimizer", () => umap.setOptimizer(optimizer));
    },
    reset(method = "spectral") {
      if (destroyed) return;
      seq.runOrQueue("reset", () => umap.reset(method));
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      // Frees now if idle, else after any queued sections settle (never on a
      // borrowed instance).
      seq.runOrQueue("free", () => umap.free());
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
