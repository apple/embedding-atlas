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

  /** The initialization method. By default we use spectral initialization. */
  initializeMethod?: "spectral" | "random";

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
}

export interface UMAP {
  /** The input dimension */
  readonly inputDim: number;

  /** The output dimension */
  readonly outputDim: number;

  /**
   * Get the current embedding.
   * Returns an empty Float32Array before `run()` is called.
   */
  readonly embedding: Float32Array;

  /**
   * Get the KNN indices from the neighbor graph.
   * Returns an empty Int32Array before `run()` is called.
   */
  readonly knnIndices: Int32Array;

  /**
   * Get the KNN distances from the neighbor graph.
   * Returns an empty Float32Array before `run()` is called.
   */
  readonly knnDistances: Float32Array;

  /**
   * Run the UMAP algorithm to completion.
   */
  run(): void;

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
