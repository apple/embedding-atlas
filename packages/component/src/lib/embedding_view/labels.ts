// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { measureText } from "../measure_text.js";
import { type Point, type Rectangle, type ViewportState } from "../utils.js";
import { Viewport } from "../viewport_utils.js";
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
  width: number,
  height: number,
  viewport: ViewportState,
  labels: Label[],
  fontFamily: string,
): Promise<LabelWithPlacement[]> {
  let vp = new Viewport(viewport, width, height);
  let minLevel = labels.reduce((a, b) => Math.min(a, b.level ?? 0), 0);
  let maxLevel = labels.reduce((a, b) => Math.max(a, b.level ?? 0), 0);
  let currentScale = viewport.scale;
  let maxScale = viewport.scale / 2;
  let scaleThreshold = maxScale * 4;
  let threshold = currentScale / scaleThreshold;

  let result: LabelWithPlacement[] = labels.map((cluster) => {
    let p = vp.pixelLocation(cluster.x, cluster.y);
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
        xMin: p.x - size.width / 2,
        xMax: p.x + size.width / 2,
        yMin: p.y - size.height / 2,
        yMax: p.y + size.height / 2,
      },
      locationAtZero: p,
      priority: cluster.priority,
      minScale: cluster.level == maxLevel ? null : (threshold * Math.pow(2, -level)) / 1.2,
      maxScale: cluster.level == minLevel ? null : threshold * Math.pow(2, -level + 1),
      coordinate: { x: cluster.x, y: cluster.y },
      placement: null,
    };
  });

  let placements = await dynamicLabelPlacement(result, { globalMaxScale: currentScale / maxScale });
  for (let i = 0; i < placements.length; i++) {
    let placement = placements[i];
    if (placement != null) {
      let maxScale = currentScale / placement.minScale;
      let minScale = currentScale / placement.maxScale;
      result[i].placement = { minScale, maxScale };
    }
  }
  return result;
}
