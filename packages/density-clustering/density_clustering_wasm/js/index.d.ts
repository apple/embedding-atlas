/** A resulting cluster from the find clusters function */
export interface Cluster {
  /** Cluster identifier */
  identifier: number;
  /** The total density */
  sum_density: number;
  /** The mean x location (weighted by density) */
  mean_x: number;
  /** The mean y location (weighted by density) */
  mean_y: number;
  /** The maximum density */
  max_density: number;
  /** The location with the maximum density */
  max_density_location: [number, number];
  /** The number of pixels in the cluster */
  pixel_count: number;
  /** The cluster's boundary represented as a list of polygons */
  boundary?: [number, number][][];
  /** The cluster's boundary approximated with a list of rectangles */
  boundary_rect_approximation?: [number, number, number, number][];
}

/** Options of the find clusters function */
export interface FindClustersOptions {
  /** The threshold for unioning two clusters */
  union_threshold: number;
}

/**
 * Find clusters from a density map
 * @param density_map the density map, a `Float32Array` with `width * height` elements
 * @param width the width of the density map
 * @param height the height of the density map
 * @param options algorithm options
 * @returns
 */
export async function findClusters(
  density_map: Float32Array,
  width: number,
  height: number,
  options: Partial<FindClustersOptions> = {},
): Promise<Cluster[]>;
