// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { XYBinning, type Rectangle } from "../utils.js";

/** The representative point picked for a region. */
export interface ImageSummarizerResult {
  /** Identifier of the highest-importance point that falls inside the region. */
  id: any;
  /** The x coordinate of that point. */
  x: number;
  /** The y coordinate of that point. */
  y: number;
}

/** Picks a representative point for each region: the point with the highest
 * importance that falls inside the region.
 *
 * Region assignment uses the same {@link XYBinning} grid as the text
 * summarizer, turning an O(points x rectangles) point-in-rectangle test into an
 * O(points) grid lookup. Points are added incrementally via {@link add} so they
 * can be streamed from the database in chunks. */
export class ImageSummarizer {
  private binning: XYBinning;
  private key2RegionIndices: Map<number, number[]>;
  private bestImportance: Float64Array;
  private bestId: any[];
  private bestX: Float64Array;
  private bestY: Float64Array;

  /** Create a new ImageSummarizer for the given regions. */
  constructor(options: { regions: Rectangle[][] }) {
    this.binning = XYBinning.inferFromRegions(options.regions);

    let count = options.regions.length;
    this.bestImportance = new Float64Array(count).fill(Number.NEGATIVE_INFINITY);
    this.bestId = new Array(count).fill(null);
    this.bestX = new Float64Array(count);
    this.bestY = new Float64Array(count);

    // Map from xy key to the regions covering it (regions may overlap).
    this.key2RegionIndices = new Map();
    for (let i = 0; i < count; i++) {
      for (let k of this.binning.keys(options.regions[i])) {
        let v = this.key2RegionIndices.get(k);
        if (v != null) {
          v.push(i);
        } else {
          this.key2RegionIndices.set(k, [i]);
        }
      }
    }
  }

  /** Add a chunk of points to the summarizer. */
  add(data: { x: ArrayLike<number>; y: ArrayLike<number>; importance: ArrayLike<number>; id: ArrayLike<any> }) {
    for (let i = 0; i < data.x.length; i++) {
      let indices = this.key2RegionIndices.get(this.binning.key(data.x[i], data.y[i]));
      if (indices == null) {
        continue;
      }
      let importance = data.importance[i];
      for (let idx of indices) {
        if (importance > this.bestImportance[idx]) {
          this.bestImportance[idx] = importance;
          this.bestId[idx] = data.id[i];
          this.bestX[idx] = data.x[i];
          this.bestY[idx] = data.y[i];
        }
      }
    }
  }

  /** Return the representative point per region, or null for regions with no points. */
  summarize(): (ImageSummarizerResult | null)[] {
    return this.bestId.map((id, i) => (id == null ? null : { id, x: this.bestX[i], y: this.bestY[i] }));
  }
}
