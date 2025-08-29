/** UMAP options */
export interface UMAPOptions {
  /** The input distance metric */
  metric?: "euclidean" | "cosine";

  /** The nearest neighbor method. By default we use HNSW with its default parameters. */
  knnMethod?: "hnsw" | "nndescent" | "vptree";

  /** The initialization method. By default we use spectral initialization. */
  initializeMethod?: "spectral" | "random" | "none";

  localConnectivity?: number;
  bandwidth?: number;
  mixRatio?: number;
  spread?: number;
  minDist?: number;
  a?: number;
  b?: number;
  repulsionStrength?: number;
  nEpochs?: number;
  learningRate?: number;
  negativeSampleRate?: number;
  nNeighbors?: number;
  /** The random seed. */
  seed?: number;
}

export interface UMAP {
  /** The current epoch number */
  get epoch(): number;

  /** The input dimension */
  get inputDim(): number;

  /** The output dimension */
  get outputDim(): number;

  /**
   * Get the current embedding.
   * The resulting Float32Array points to WASM internal memory.
   * If you need to use the data outside this library or after further
   * interaction with this library, make sure to create a copy
   * of the array, as the underlying memory may change.
   */
  get embedding(): Float32Array;

  /**
   * Run the UMAP algorithm until reaching `epochLimit` epochs,
   * or to completion if `epochLimit` is not specified.
   * @param epochLimit the epoch number to run to
   */
  run(epochLimit?: number): void;

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

/** KNN options */
export interface KNNOptions {
  /** The distance metric */
  metric?: "euclidean" | "cosine";

  /** The nearest neighbor method. By default we use HNSW with its default parameters. */
  method?: "hnsw" | "nndescent" | "vptree";
}

export interface KNNQueryResult {
  indices: Int32Array;
  distances: Float32Array;
}

export interface KNN {
  queryByIndex(index: number, k: number): KNNQueryResult;
  queryByVector(data: Float32Array, k: number): KNNQueryResult;
  destroy(): void;
}

export function createKNN(count: number, inputDim: number, data: Float32Array, options?: KNNOptions): Promise<KNN>;
