/** A resulting cluster from the find clusters function */
export interface Cluster {
  /** Cluster identifier */
  identifier: number;
  /** The total density */
  sumDensity: number;
  /** The mean x location (weighted by density) */
  meanX: number;
  /** The mean y location (weighted by density) */
  meanY: number;
  /** The maximum density */
  maxDensity: number;
  /** The location with the maximum density */
  maxDensityLocation: [number, number];
  /** The number of pixels in the cluster */
  pixelCount: number;
  /** The cluster's boundary represented as a list of polygons */
  boundary?: [number, number][][];
  /** The cluster's boundary approximated with a list of rectangles, each rectangle is given as an array [x1, y1, x2, y2] */
  boundaryRectApproximation?: [number, number, number, number][];
}

/** Options of the find clusters function */
export interface FindClustersOptions {
  /** The threshold for unioning two clusters */
  unionThreshold: number;
}

/**
 * Find clusters from a density map
 * @param density_map the density map, a `Float32Array` with `width * height` elements
 * @param width the width of the density map
 * @param height the height of the density map
 * @param options algorithm options
 * @returns
 */
export function findClusters(
  densityMap: Float32Array,
  width: number,
  height: number,
  options?: Partial<FindClustersOptions>,
): Promise<Cluster[]>;
