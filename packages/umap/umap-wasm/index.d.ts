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

/** Options of the recursive initialization (`initializeMethod: "recursive"`). */
export interface RecursiveInitOptions {
  /** Graphs with at most this many nodes are placed from the anchor directly. Default: 1024. */
  baseInitThreshold?: number;
  /** Epochs of the finest coarse layout. Default: 50. */
  nEmbeddingEpochs?: number;
  /** Number of parts at the first level. Default: derived from the data size. */
  approxNParts?: number;
  /** Approximate reduction in size per level: 2, 3, or 4. Default: 4. */
  coarseningRatio?: number;
  hubSelection?: "degree" | "diverse" | "random";
  /** Number of rows used for the anchor PCA. Default: 16384. */
  anchorSampleSize?: number;
  repulsionStrength?: number;
  negativeSampleRate?: number;
  negativeSelectionRangeScale?: number;
  labelPropagationShare?: number;
  coarseLayoutWeights?: "transition" | "fuzzyUnion";
  coarsenOnLayoutWeights?: boolean;
  maxPropagationSweeps?: number;
  curveSchedule?:
    | "strongToOne"
    | "original"
    | { initialA: number; initialB: number; recursiveA: number; recursiveB: number };
  goodInitializationPolicy?: "heuristic" | "available" | "always" | "never";
  removeCoarseDiagonal?: boolean;
  anchorMethod?: "sampledPca" | "projectedPca" | "fullPca";
  negativeSelectionRangeMode?: "coarseN" | "fixed200k" | "scaledCoarse";
}

/** UMAP options */
export interface UMAPOptions {
  /** The input distance metric */
  metric?: "euclidean" | "cosine";

  /**
   * Use the umap-learn 0.5 layout pipeline (`true`) or the 0.6 pipeline (`false`).
   * Default: unset, which selects the 0.6 pipeline only when `optimizer` is "adam" or
   * "momentum". Those optimizers throw with `true`.
   */
  compatibilityLayout?: boolean;

  /**
   * The initialization method. Default: "auto", which is "spectral", or
   * "recursive" when `compatibilityLayout` is false. "pca" needs the data, so it
   * is rejected by {@link createUMAPFromKNN}.
   */
  initializeMethod?: "auto" | "spectral" | "tswspectral" | "random" | "recursive" | "pca";

  /** How the initial layout is scaled when `compatibilityLayout` is false. Default: "isotropic". */
  initRescale?: "isotropic" | "perAxis" | "graphEdge";

  /** Curve parameter `a`. Used only together with `b`; otherwise both are fitted from `minDist` and `spread`. */
  a?: number;

  /** Curve parameter `b`. Used only together with `a`. */
  b?: number;

  /** Neighbors at this distance or farther are dropped from the fuzzy graph. Default: depends on the metric. */
  disconnectionDistance?: number;

  /** Window of nodes that negative samples are drawn from. Default: 200000. */
  negativeSelectionRange?: number;

  /** Fixed scale of the repulsive force. Default: derived from `negativeSelectionRange`. */
  negativeSampleScale?: number;

  /** Skip negative samples that are graph neighbors of the source. Default: false. */
  excludeGraphNeighbors?: boolean;

  /** Sources sampled to adapt the negative sample scale of Adam; 0 disables the adaptation. Default: 128. */
  negativeSampleScaleAdaptationSamples?: number;

  /** Options of the recursive initialization. Missing keys keep their defaults. */
  recursiveInit?: RecursiveInitOptions;

  /**
   * The optimizer. Default: "sgd", or "adam" when `compatibilityLayout` is false.
   * "momentum" is the umap-learn 0.6 heavy-ball momentum optimizer.
   */
  optimizer?: "sgd" | "momentum" | "adam";

  /** Multiplier of the isotropic init rescale. Only used when `compatibilityLayout` is false. Default: 1. */
  initScaleFactor?: number;

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
   * Anneal the layout to completion over the default horizon, continuing from
   * the current epoch, with the live `learningRate` (from `createUMAP` options
   * or the last `setParameters`) as α0:
   * - "sgd": linear decay from α0 to 0.
   * - "momentum" / "adam": the umap-learn 0.6 warm-up/decay schedules
   *   (Adam ends at 0.05, momentum at 0).
   */
  run(): Promise<void>;

  /**
   * Advance the layout by `nEpochs` at the current `learningRate`, held flat (no
   * decay). Intended for interactive/real-time stepping; anneal by calling
   * `setParameters({ learningRate })` between steps. Default: 1 epoch.
   * - "sgd" and "adam": step size `learningRate`.
   * - "momentum": step size 0.25 · `learningRate`, momentum 0.5.
   * A `learningRate` of 0 stops the layout.
   */
  step(nEpochs?: number): Promise<void>;

  /**
   * Update live parameters; they apply from the next epoch. `learningRate` is
   * used flat by `step` and as the peak by `run`; for "momentum" the current
   * velocity is rescaled to the new rate. `minDist`/`spread` re-fit the output
   * curve. Changing `minDist`, `spread`, or `repulsionStrength` resets the
   * "adam" moments so the layout responds at full speed. Safe to call during
   * an in-flight `step`/`run` — the update is queued and applied in order, once
   * any in-flight/queued sections settle.
   */
  setParameters(
    params: Pick<UMAPOptions, "learningRate" | "repulsionStrength" | "negativeSampleRate" | "minDist" | "spread">,
  ): void;

  /** Switch optimizer. The new optimizer starts with fresh state. Queued; runs in order after any in-flight `step`/`run`. */
  setOptimizer(optimizer: "sgd" | "momentum" | "adam"): void;

  /**
   * Re-initialize the embedding and restart from epoch 0. Default method: "spectral".
   * `data` is the same `count * inputDim` Float32Array that was given to `createUMAP`;
   * the instance does not keep the data. "pca" needs it. "recursive" uses it for its
   * PCA anchor and uses a random anchor without it. The other methods ignore it.
   * Throws if `data` is missing for "pca", has the wrong length, or is given to an
   * instance built by `createUMAPFromKNN`.
   * Queued; runs in order after any in-flight `step`/`run`.
   */
  reset(method?: "auto" | "spectral" | "tswspectral" | "random" | "recursive" | "pca", data?: Float32Array): void;

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
