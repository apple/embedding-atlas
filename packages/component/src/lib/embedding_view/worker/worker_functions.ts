// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { findClusters } from "@embedding-atlas/density-clustering";
import { dynamicLabelPlacement } from "../../dynamic_label_placement/dynamic_label_placement.js";
import { TFIDFSummarizer } from "../../text_summarizer/text_summarizer.js";
import type { Rectangle } from "../../utils.js";

export { dynamicLabelPlacement, findClusters };

let textSummarizers = new Map<string, TFIDFSummarizer>();

export function textSummarizerCreate(options: {
  binning: { xMin: number; xStep: number; yMin: number; yStep: number };
  regions: Rectangle[][];
  stopWords?: string[];
}) {
  let key = new Date().getTime() + "-" + Math.random();
  textSummarizers.set(key, new TFIDFSummarizer(options));
  return key;
}

export function textSummarizerDestroy(key: string) {
  return textSummarizers.delete(key);
}

export function textSummarizerAdd(
  key: string,
  data: { x: ArrayLike<number>; y: ArrayLike<number>; text: ArrayLike<string> },
) {
  textSummarizers.get(key)?.add(data);
}

export function textSummarizerSummarize(key: string) {
  return textSummarizers.get(key)?.summarize() ?? [];
}
