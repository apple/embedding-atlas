/** NNDescent options */
export interface NNDescentOptions {
  /** The distance metric. Default: "euclidean". */
  metric?: "euclidean" | "cosine";

  /** Number of neighbors to find. Default: 15. */
  nNeighbors?: number;

  /** Number of random projection trees. Default: auto. */
  nTrees?: number;

  /** Number of NN-descent iterations. Default: auto. */
  nIters?: number;

  /** Early stopping threshold. Default: 0.001. */
  delta?: number;

  /** Whether to use RP tree initialization. Default: true. */
  treeInit?: boolean;

  /** Accuracy/speed tradeoff for queryByVector (higher = more accurate, slower). Default: 0.1. */
  epsilon?: number;

  /** Random seed for reproducibility. */
  seed?: number;

  /** Progress callback: `(progress: number, stage: string) => void`. Progress is in [0, 1]. */
  progress?: (progress: number, stage: string) => void;

  /** Whether to use GPU acceleration (WebGPU). Default: false. */
  gpu?: boolean;
}

export interface NNDescentQueryResult {
  indices: Int32Array;
  distances: Float32Array;
}

export interface NNDescentResult {
  queryByIndex(index: number, k: number): NNDescentQueryResult;
  queryByVector(data: Float32Array, k: number): NNDescentQueryResult;
  destroy(): void;
}

/**
 * Create an NNDescent nearest neighbor index.
 * @param count the number of data points
 * @param inputDim the input dimension
 * @param data the data array. Must be a Float32Array with count * inputDim elements.
 * @param options options
 */
export function createNNDescent(
  count: number,
  inputDim: number,
  data: Float32Array,
  options?: NNDescentOptions,
): Promise<NNDescentResult>;

/** UMAP options */
export interface UMAPOptions {
  /** The input distance metric */
  metric?: "euclidean" | "cosine";

  /** The initialization method. Default: "spectral" (momentum forces "random"). */
  initializeMethod?: "spectral" | "random";

  /** The optimizer. Default: "sgd". "momentum" gives better global structure. */
  optimizer?: "sgd" | "momentum";

  localConnectivity?: number;
  mixRatio?: number;
  spread?: number;
  minDist?: number;
  repulsionStrength?: number;
  nEpochs?: number;
  learningRate?: number;
  negativeSampleRate?: number;
  nNeighbors?: number;

  /** The random seed. */
  seed?: number;

  /** Progress callback: `(progress: number, stage: string) => void`. Progress is in [0, 1]. */
  progress?: (progress: number, stage: string) => void;

  /** Whether to use GPU acceleration (WebGPU). Default: false. */
  gpu?: boolean;
}

/** A stateful, resumable UMAP instance. */
export interface UMAP {
  /** The input dimension */
  readonly inputDim: number;

  /** The output dimension */
  readonly outputDim: number;

  /** The number of epochs run so far. During an in-flight `step`/`run` this
   *  returns the last settled value (the count is refreshed when the op settles). */
  readonly epoch: number;

  /**
   * Get the current embedding. Valid immediately after `createUMAP` (eager
   * setup, epoch 0) and refreshed by `run`/`step`.
   *
   * Safe to read during an in-flight `step`/`run`: it returns the last settled
   * snapshot rather than reading the (borrowed) wasm instance, so it never
   * throws even when polled from a requestAnimationFrame loop.
   *
   * Note: returns a reused buffer — copy it (e.g. `new Float32Array(embedding)`)
   * if you need a stable snapshot across subsequent reads.
   */
  readonly embedding: Float32Array;

  /** The KNN indices from the neighbor graph. Valid immediately (eager setup);
   *  the graph is immutable, so it is snapshotted at construction and is safe to
   *  access at any time, including during an in-flight `step`/`run`. */
  readonly knnIndices: Int32Array;

  /** The KNN distances from the neighbor graph. Valid immediately (eager setup);
   *  snapshotted at construction and safe to access during an in-flight `step`/`run`. */
  readonly knnDistances: Float32Array;

  /**
   * Anneal the layout to completion: linear learning-rate decay from the current
   * `learningRate` down to 0 over the default horizon, continuing from the current
   * epoch. The peak is the live `learningRate` (from `createUMAP` options or the
   * last `setParameters`).
   */
  run(): Promise<void>;

  /**
   * Advance the layout by `nEpochs` at the current `learningRate`, held flat (no
   * decay). Intended for interactive/real-time stepping. Default: 1 epoch.
   */
  step(nEpochs?: number): Promise<void>;

  /**
   * Update live parameters. `learningRate` is used flat by `step` and as the
   * decay peak by `run`. `minDist`/`spread` re-fit the output curve. Safe to call
   * during an in-flight `step`/`run` — the update is queued and applied in order,
   * once any in-flight/queued sections settle.
   */
  setParameters(
    params: Pick<UMAPOptions, "learningRate" | "repulsionStrength" | "negativeSampleRate" | "minDist" | "spread">,
  ): void;

  /** Switch optimizer. Resets the velocity buffer. Queued; runs in order after any in-flight `step`/`run`. */
  setOptimizer(optimizer: "sgd" | "momentum"): void;

  /** Re-initialize the embedding and restart from epoch 0. Default: "spectral". Queued; runs in order after any in-flight `step`/`run`. */
  reset(method?: "spectral" | "random"): void;

  /** Destroy the instance and release resources */
  destroy(): void;
}

/**
 * Initialize a UMAP instance.
 * @param count the number of data points
 * @param inputDim the input dimension
 * @param outputDim the output dimension
 * @param data the data array. Must be a Float32Array with count * inputDim elements.
 * @param options options
 */
export function createUMAP(
  count: number,
  inputDim: number,
  outputDim: number,
  data: Float32Array,
  options?: UMAPOptions,
): Promise<UMAP>;

/**
 * Options for {@link createUMAPFromKNN}. The kNN-computation knobs (`metric`,
 * `nNeighbors`) do not apply when the graph is supplied directly.
 */
export type UMAPFromKNNOptions = Omit<UMAPOptions, "metric" | "nNeighbors">;

/**
 * Build a UMAP instance from a precomputed kNN graph, skipping the internal
 * nearest-neighbor search. No high-dimensional data is needed — once the graph
 * exists, UMAP uses only it. The returned instance is the same {@link UMAP} as
 * {@link createUMAP} (its `inputDim` is reported as 0 since no vectors were given).
 *
 * `knnIndices` and `knnDistances` are row-major with `count * k` elements; row `i`
 * lists the neighbors of point `i` sorted by ascending distance. The point itself
 * may appear in column 0 (distance 0) or be omitted — this is detected and
 * normalized per row.
 *
 * Caller's contract (the graph is not validated beyond an index-range check, and
 * bad input degrades silently rather than throwing): every index must be valid
 * (`0 <= idx < count`); point `i` may appear in its own row at most once; distances
 * must be finite (a `NaN` propagates to a `NaN` embedding); and the neighbor count
 * must be uniform across rows — the effective `k` is the per-row minimum, so a
 * single short row trims every row down to its length.
 *
 * @param count the number of data points
 * @param outputDim the output dimension
 * @param knnIndices row-major neighbor indices, length count * k
 * @param knnDistances row-major neighbor distances, length count * k (same shape)
 * @param options layout options
 */
export function createUMAPFromKNN(
  count: number,
  outputDim: number,
  knnIndices: Int32Array,
  knnDistances: Float32Array,
  options?: UMAPFromKNNOptions,
): Promise<UMAP>;
