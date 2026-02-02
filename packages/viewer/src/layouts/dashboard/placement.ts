// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { OccupancyMap } from "./occupancy_map.js";
import type { Placement } from "./types.js";

export class Grid {
  readonly containerWidth: number;
  readonly containerHeight: number;
  readonly numColumns: number;
  readonly numRows: number;
  readonly gap: number;

  constructor(containerWidth: number, containerHeight: number, numColumns: number, numRows: number, gap: number) {
    this.containerWidth = containerWidth;
    this.containerHeight = containerHeight;
    this.numColumns = numColumns;
    this.numRows = numRows;
    this.gap = gap;
  }

  resolvePlacement(placement: Placement): Placement {
    let unitSizeX = (this.containerWidth - this.gap * (this.numColumns - 1)) / this.numColumns;
    let unitSizeY = (this.containerHeight - this.gap * (this.numRows - 1)) / this.numRows;
    return {
      x: placement.x * (unitSizeX + this.gap),
      y: placement.y * (unitSizeY + this.gap),
      width: placement.width * unitSizeX + (placement.width - 1) * this.gap,
      height: placement.height * unitSizeY + (placement.height - 1) * this.gap,
    };
  }

  get xScaler() {
    return (this.containerWidth + this.gap) / this.numColumns;
  }

  get yScaler() {
    return (this.containerHeight + this.gap) / this.numRows;
  }
}

export function computePlacements(
  charts: Record<string, any>,
  placements: Record<string, any>,
  numColumns: number,
): Record<string, Placement> {
  let result: Record<string, Placement> = {};
  let map = new OccupancyMap(numColumns);
  for (let id in charts) {
    let pl = placements[id];
    if (pl != undefined) {
      map.fill(pl.x, pl.y, pl.width, pl.height);
      result[id] = pl;
    }
  }
  for (let id in charts) {
    if (result[id] != undefined) {
      continue;
    }
    let w = 8;
    let h = 5;
    if (charts[id].type == "embedding") {
      w = 16;
      h = 12;
    }
    if (charts[id].type == "table") {
      w = numColumns;
      h = 6;
    }
    if (w > numColumns) {
      w = numColumns;
    }
    let { x, y } = map.find(w, h);
    map.fill(x, y, w, h);
    result[id] = { x, y, width: w, height: h };
  }
  return result;
}
