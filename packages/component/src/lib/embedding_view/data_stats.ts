// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ViewportState } from "../utils.js";
import { approximateDensity2D, approximateDensity2DByCategory, median, stdev } from "./statistics.js";

/** Properties of a set of points, used for the default viewport and the zoom-dependent opacity and point size. */
export interface DataStats {
  /** The number of points. */
  count: number;
  /** The number of categories (one more than the largest category index), 1 without categories. */
  categoryCount: number;
  /** The approximate max density at the default viewport (points per square unit). */
  maxDensity: number;
  /** The viewport that fits the points. */
  defaultViewportState: ViewportState;
}

// Stats are cached by the arrays: arrays are treated as immutable (new data comes in new arrays).
const noCategory = {};
const statsCache = new WeakMap<Float32Array, WeakMap<Float32Array, WeakMap<object, DataStats>>>();
const categoryCountCache = new WeakMap<Uint8Array, number>();

/** The stats of points (cached by their arrays). With categories, the max density is per category,
 *  matching how categorical density is drawn. */
export function dataStats(x: Float32Array, y: Float32Array, category: Uint8Array | null | undefined): DataStats {
  let byY = statsCache.get(x);
  if (byY == null) {
    byY = new WeakMap();
    statsCache.set(x, byY);
  }
  let byCategory = byY.get(y);
  if (byCategory == null) {
    byCategory = new WeakMap();
    byY.set(y, byCategory);
  }
  let stats = byCategory.get(category ?? noCategory);
  if (stats == null) {
    let xCenter = median(x);
    let yCenter = median(y);
    let scaler = 1.0 / (Math.max(stdev(x), stdev(y), 1e-3) * 3);
    let binWidth = 0.1 / scaler;
    let categoryCount = categoryCountOf(category);
    stats = {
      count: x.length,
      categoryCount: categoryCount,
      maxDensity:
        category != null && categoryCount > 1
          ? approximateDensity2DByCategory(x, y, category, categoryCount, binWidth, xCenter, yCenter)
          : approximateDensity2D(x, y, binWidth, xCenter, yCenter),
      defaultViewportState: { x: xCenter, y: yCenter, scale: scaler * 0.95 },
    };
    byCategory.set(category ?? noCategory, stats);
  }
  return stats;
}

/** The number of categories (one more than the largest category index), 1 without categories. */
export function categoryCountOf(category: Uint8Array | null | undefined): number {
  if (category == null) {
    return 1;
  }
  let cached = categoryCountCache.get(category);
  if (cached == null) {
    cached = category.reduce((a, b) => Math.max(a, b), 0) + 1;
    categoryCountCache.set(category, cached);
  }
  return cached;
}
