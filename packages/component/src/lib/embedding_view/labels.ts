// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { measureText } from "../measure_text.js";
import { type Point, type Rectangle } from "../utils.js";
import type { Label } from "./types.js";
import { dynamicLabelPlacement } from "./worker/index.js";

export interface LabelWithPlacement {
  text: string;
  fontSize: number;
  bounds: Rectangle;
  locationAtZero: Point;
  coordinate: Point;
  placement: { minScale: number; maxScale: number } | null;
}

export async function layoutLabels(
  normalScale: number,
  labels: Label[],
  fontFamily: string,
): Promise<LabelWithPlacement[]> {
  let minLevel = labels.reduce((a, b) => Math.min(a, b.level ?? 0), 0);
  let maxLevel = labels.reduce((a, b) => Math.max(a, b.level ?? 0), 0);
  let globalMaxScale = normalScale * 0.5;
  let threshold = normalScale * 2;

  let result: LabelWithPlacement[] = labels.map((cluster) => {
    let location = { x: cluster.x, y: cluster.y };
    let level = cluster.level ?? 0;
    let fontSize = level == 0 ? 14 : 12;
    let size = measureText({
      text: cluster.text,
      fontSize: fontSize,
      fontFamily: fontFamily,
    });
    size.width += 4;
    size.height += 4;
    return {
      text: cluster.text,
      fontSize: fontSize,
      bounds: {
        xMin: location.x - size.width / 2,
        xMax: location.x + size.width / 2,
        yMin: location.y - size.height / 2,
        yMax: location.y + size.height / 2,
      },
      locationAtZero: location,
      priority: cluster.priority,
      minScale: cluster.level == maxLevel ? null : 1 / (threshold * Math.pow(2, level) * 1.2),
      maxScale: cluster.level == minLevel ? null : 1 / (threshold * Math.pow(2, level - 1)),
      coordinate: { x: cluster.x, y: cluster.y },
      placement: null,
    };
  });
  let placements = await dynamicLabelPlacement(result, { globalMaxScale: 1 / globalMaxScale });
  for (let i = 0; i < placements.length; i++) {
    let placement = placements[i];
    if (placement != null) {
      let maxScale = 1 / placement.minScale;
      let minScale = 1 / placement.maxScale;
      result[i].placement = { minScale, maxScale };
    }
  }
  return result;
}
