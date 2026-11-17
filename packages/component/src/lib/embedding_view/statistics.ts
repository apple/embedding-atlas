// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import quickselect from "quickselect";

// Reusable scratch buffer for `median`, so repeated/streaming calls don't
// allocate (and GC) a full copy of the input on every call.
let medianScratch = new Float32Array(0);

export function median(values: Float32Array): number {
  let n = values.length;
  if (n === 0) {
    return 0;
  }
  if (medianScratch.length < n) {
    medianScratch = new Float32Array(n);
  }
  medianScratch.set(values);
  let middleIndex = Math.floor(n / 2);
  quickselect(medianScratch as any, middleIndex, 0, n - 1);
  return medianScratch[middleIndex];
}

export function mean(values: Float32Array): number {
  let n = values.length;
  if (n === 0) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += values[i];
  }
  return sum / n;
}

export function stdev(values: Float32Array): number {
  let n = values.length;
  if (n === 0) {
    return 0;
  }
  // Single pass over the data: accumulate sum and sum-of-squares of values
  // shifted by `values[0]` (the shift avoids catastrophic cancellation for
  // large-magnitude coordinates while keeping it to one loop).
  let shift = values[0];
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    let d = values[i] - shift;
    sum += d;
    sumSq += d * d;
  }
  let m = sum / n;
  let variance = sumSq / n - m * m;
  return Math.sqrt(variance > 0 ? variance : 0);
}

// Reusable bin grid for `approximateDensity2D`, allocated lazily on first use so
// merely importing this module costs no memory. Bins are centered on
// `xBinStart`/`yBinStart`; points that fall outside the grid are sparse outliers
// and are ignored (rather than clamped to the edge, which would manufacture a
// spurious high-density edge).
const DENSITY_GRID = 256;
let densityGrid: Int32Array | null = null;

export function approximateDensity2D(
  x: Float32Array,
  y: Float32Array,
  binWidth: number,
  xBinStart: number = 0,
  yBinStart: number = 0,
): number {
  const g = DENSITY_GRID;
  const h = g >> 1;
  let grid = densityGrid;
  if (grid == null) {
    grid = new Int32Array(g * g); // freshly zeroed
    densityGrid = grid;
  } else {
    grid.fill(0);
  }
  let maxValue = 0;
  for (let i = 0; i < x.length; i++) {
    let bx = Math.floor((x[i] - xBinStart) / binWidth) + h;
    let by = Math.floor((y[i] - yBinStart) / binWidth) + h;
    if (bx < 0 || bx >= g || by < 0 || by >= g) {
      continue; // outside the grid — ignore
    }
    let v = ++grid[by * g + bx];
    if (v > maxValue) {
      maxValue = v;
    }
  }
  return maxValue / (binWidth * binWidth);
}

/** Like `approximateDensity2D`, but counts the points of each category separately (the max over cells and
 *  categories), matching how categorical density is drawn. The points are grouped by category (a counting sort of
 *  their cells), then counted one category at a time in one reused grid: O(n) time and 4 bytes per point. */
export function approximateDensity2DByCategory(
  x: Float32Array,
  y: Float32Array,
  category: Uint8Array,
  categoryCount: number,
  binWidth: number,
  xBinStart: number = 0,
  yBinStart: number = 0,
): number {
  const g = DENSITY_GRID;
  const h = g >> 1;
  let cellOf = (i: number): number => {
    let bx = Math.floor((x[i] - xBinStart) / binWidth) + h;
    let by = Math.floor((y[i] - yBinStart) / binWidth) + h;
    return bx < 0 || bx >= g || by < 0 || by >= g ? -1 : by * g + bx;
  };
  // Count the points (in the grid) of each category, then place their cells, grouped by category.
  let starts = new Uint32Array(categoryCount + 1);
  for (let i = 0; i < x.length; i++) {
    if (cellOf(i) >= 0) {
      starts[category[i] + 1] += 1;
    }
  }
  for (let c = 0; c < categoryCount; c++) {
    starts[c + 1] += starts[c];
  }
  let cells = new Int32Array(starts[categoryCount]);
  let offsets = starts.slice(0, categoryCount);
  for (let i = 0; i < x.length; i++) {
    let cell = cellOf(i);
    if (cell >= 0) {
      cells[offsets[category[i]]++] = cell;
    }
  }
  let grid = densityGrid;
  if (grid == null) {
    grid = new Int32Array(g * g);
    densityGrid = grid;
  } else {
    grid.fill(0);
  }
  let maxValue = 0;
  for (let c = 0; c < categoryCount; c++) {
    let end = starts[c + 1];
    for (let j = starts[c]; j < end; j++) {
      let v = ++grid[cells[j]];
      if (v > maxValue) {
        maxValue = v;
      }
    }
    // Reset the cells of this category for the next one.
    for (let j = starts[c]; j < end; j++) {
      grid[cells[j]] = 0;
    }
  }
  return maxValue / (binWidth * binWidth);
}
