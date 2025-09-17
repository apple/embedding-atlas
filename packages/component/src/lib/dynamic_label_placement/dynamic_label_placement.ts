// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// The algorithm is based on the following research paper:
// Been, Ken, Eli Daiches, and Chee Yap. "Dynamic Map Labeling." IEEE Transactions on Visualization and Computer Graphics 12, no. 5 (2006): 773–80. https://doi.org/10.1109/TVCG.2006.136.

import type { Point, Rectangle } from "../utils.js";
import { PriorityQueue } from "./priority_queue.js";

/** Description of a label for placement. */
export interface Label {
  /** The bounds of the label at scale = 1. */
  bounds: Rectangle;
  /** The location of the label at scale = 0. Usually you can set this to the center of `bounds`. */
  locationAtZero: Point;
  /** The minimum scale this label can appear in. */
  minScale?: number;
  /** The maximum scale this label can appear in. */
  maxScale?: number;
  /** The label's priority. */
  priority?: number;
}

/** The placement of a label. */
export interface Placement {
  /** The minimum scale. */
  minScale: number;
  /** The maximum scale. */
  maxScale: number;
}

export interface Options {
  /** The global max scale for all labels. */
  globalMaxScale: number;
}

export function dynamicLabelPlacement(labels: Label[], options: Partial<Options> = {}): (Placement | null)[] {
  let globalMaxScale = options.globalMaxScale ?? 1;
  let n = labels.length;

  let edgeLists: [number, number][][] = [];
  for (let i = 0; i < n; i++) {
    edgeLists.push([]);
  }
  let allLevels = new Set<number>();
  allLevels.add(0);
  // For each pair of label, compute the scale at which they will touch (transition between overlap and no overlap).
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let { x: xi, y: yi } = labels[i].locationAtZero;
      let { x: xj, y: yj } = labels[j].locationAtZero;
      let bi = labels[i].bounds;
      let bj = labels[j].bounds;
      let k1 = xi < xj ? (xi - xj) / (bj.xMin - bi.xMax + xi - xj) : (xi - xj) / (bj.xMax - bi.xMin + xi - xj);
      let k2 = yi < yj ? (yi - yj) / (bj.yMin - bi.yMax + yi - yj) : (yi - yj) / (bj.yMax - bi.yMin + yi - yj);
      let scale = Math.max(k1, k2);
      if (xi == xj && yi == yj) {
        if (bi.xMin < bj.xMax && bi.xMax > bj.xMin && bi.yMin < bj.yMax && bi.yMax > bj.yMin) {
          scale = 0;
        } else {
          scale = Infinity;
        }
      } else if (scale <= 0) {
        scale = Infinity;
      } else {
        // Discretize the scale levels, so we don't end up with too many levels.
        scale = Math.exp(Math.floor(Math.log(scale) * 100) / 100);
      }
      edgeLists[i].push([j, scale]);
      edgeLists[j].push([i, scale]);
      allLevels.add(scale);
    }
  }
  let offsets: number[] = [];
  let ranges: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    edgeLists[i].sort((a, b) => a[1] - b[1]);
    offsets.push(edgeLists[i].length - 1);
    let range: [number, number] = [
      Math.min(globalMaxScale, labels[i].minScale ?? 0),
      Math.min(globalMaxScale, labels[i].maxScale ?? globalMaxScale),
    ];
    ranges.push(range);
    allLevels.add(range[0]);
    allLevels.add(range[1]);
  }
  let placements: (Placement | null)[] = labels.map(() => null);
  let sortedLevels = Array.from(allLevels).sort((a, b) => b - a);

  let queue = new PriorityQueue();
  let numConflicts: Set<number>[] = [];
  for (let i = 0; i < n; i++) {
    numConflicts.push(new Set());
  }
  let inQueue = new Set<number>();
  let isVisible = new Set<number>();

  for (let level of sortedLevels) {
    for (let i = 0; i < n; i++) {
      while (offsets[i] >= 0 && edgeLists[i][offsets[i]][1] >= level) {
        let j = edgeLists[i][offsets[i]][0];
        numConflicts[i].delete(j);
        offsets[i] -= 1;
      }
    }
    for (let i = 0; i < n; i++) {
      if (ranges[i][0] < level && level <= ranges[i][1] && numConflicts[i].size == 0 && !inQueue.has(i)) {
        inQueue.add(i);
        queue.insert(i, labels[i].priority ?? 0);
      }
      if (isVisible.has(i) && placements[i]!.minScale >= level) {
        isVisible.delete(i);
        for (let j = 0; j < n; j++) {
          numConflicts[j].delete(i);
        }
      }
    }
    while (true) {
      let i = queue.popMax();
      if (i == null) {
        break;
      }
      placements[i] = { minScale: ranges[i][0], maxScale: level };
      isVisible.add(i);
      for (let c = 0; c <= offsets[i]; c++) {
        let j = edgeLists[i][c][0];
        numConflicts[j].add(i);
        if (inQueue.has(j)) {
          queue.delete(j);
          inQueue.delete(j);
        }
      }
    }
  }
  return placements;
}
