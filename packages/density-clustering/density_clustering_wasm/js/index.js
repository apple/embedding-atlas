// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as cluster from "../pkg/density_clustering_wasm.js";

/**
 * Find clusters from a density map
 * @param density_map the density map, a `Float32Array` with `width * height` elements
 * @param width the width of the density map
 * @param height the height of the density map
 * @param options algorithm options
 * @returns
 */
export async function findClusters(density_map, width, height, options = {}) {
  await cluster.default();
  // console.debug(`find clusters start, size: ${width}x${height}`);
  let t0 = new Date().getTime();
  let input = new cluster.DensityMap(width, height, density_map);
  let result = cluster.find_clusters(input, {
    clustering_options: {
      use_disjoint_set: true,
      truncate_to_max_density: true,
      perform_neighbor_map_grouping: false,
      union_threshold: 10.0,
      density_upperbound_scaler: 0.2,
      density_lowerbound_scaler: 0.2,
      ...options,
    },
    return_boundary_rects: true,
    smooth_boundaries: true,
  });
  input.free();
  let clusters = [];
  for (let [id, summary] of result.summaries) {
    clusters.push({
      identifier: id,
      sum_density: summary.sum_density,
      mean_x: summary.sum_x_density / summary.sum_density,
      mean_y: summary.sum_y_density / summary.sum_density,
      max_density: summary.max_density,
      max_density_location: summary.max_density_location,
      pixel_count: summary.num_pixels,
      boundary: result.boundaries.get(id),
      boundary_rect_approximation: result.boundary_rects.get(id),
    });
  }
  clusters = clusters.filter((x) => x.boundary != null);
  let t1 = new Date().getTime();
  // console.debug(`find clusters complete, time: ${(t1 - t0).toFixed(0)}ms, count: ${result.summaries.size}`);
  return clusters;
}
