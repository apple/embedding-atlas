// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { OccupancyMap } from "./occupancy_map.js";
import type { Placement } from "./types.js";

function roundAndClamp(x: number, min: number, max: number) {
  x = Math.round(x);
  return x < min ? min : x > max ? max : x;
}

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

  resolvePlacement(placement: Placement): { x: number; y: number; width: number; height: number } {
    let unitSizeX = (this.containerWidth - this.gap * (this.numColumns - 1)) / this.numColumns;
    let unitSizeY = (this.containerHeight - this.gap * (this.numRows - 1)) / this.numRows;
    let x1 = placement.x * (unitSizeX + this.gap);
    let y1 = placement.y * (unitSizeY + this.gap);
    let x2 = (placement.x + placement.width) * (unitSizeX + this.gap) - this.gap;
    let y2 = (placement.y + placement.height) * (unitSizeY + this.gap) - this.gap;
    x1 = roundAndClamp(x1, 0, this.containerWidth);
    x2 = roundAndClamp(x2, 0, this.containerWidth);
    y1 = roundAndClamp(y1, 0, Infinity);
    y2 = roundAndClamp(y2, 0, Infinity);
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
  }

  totalHeight(maxY: number) {
    if (maxY <= this.numRows) {
      return this.containerHeight;
    } else {
      let p = this.resolvePlacement({ x: 0, y: 0, width: this.numColumns, height: maxY });
      return p.y + p.height;
    }
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
    let h = 6;
    if (charts[id].type == "embedding") {
      w = 16;
      h = 12;
    }
    if (charts[id].type == "table" || charts[id].type == "instances") {
      w = 16;
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

export function overlaps(a: Placement, b: Placement) {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}
