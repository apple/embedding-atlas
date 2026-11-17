// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { measureText } from "../measure_text.js";
import { type Point, type Rectangle } from "../utils.js";
import type { Label, LabelContent } from "./types.js";
import { dynamicLabelPlacement } from "./worker/index.js";

/** Maximum size of image labels in pixels. */
export const IMAGE_LABEL_SIZE = 48;

export interface LabelWithPlacement {
  content: LabelContent;
  fontSize: number;
  bounds: Rectangle;
  locationAtZero: Point;
  coordinate: Point;
  placement: { minScale: number; maxScale: number } | null;
}

/** Labels laid out together (e.g., the labels of one labels layer). */
export interface LabelGroup {
  labels: Label[];
  /** The viewport scale that the labels' levels are relative to (e.g., the scale they were generated at).
   *  Each label's zoom range is derived from its level, this scale, and the group's minimum and maximum level. */
  normalScale: number;
}

/** Lay out labels from one or more groups. Each group's zoom ranges (including how far out its labels can be
 *  shown) are derived from its own levels and scale, so groups don't affect each other's zoom behavior;
 *  the placement (avoiding overlaps) considers all labels. */
export async function layoutLabels(groups: LabelGroup[], fontFamily: string): Promise<LabelWithPlacement[]> {
  // The placement works with inverse scales (1 / scale). Labels can't be shown when zoomed out beyond half of
  // their group's scale: the cap of each label's range, and the global cap is the largest of them.
  let maxInverseScale = (normalScale: number) => 1 / (normalScale * 0.5);
  let globalMaxScale = groups.reduce((a, g) => Math.max(a, maxInverseScale(g.normalScale)), 0);

  let result: LabelWithPlacement[] = groups.flatMap(({ labels, normalScale }) => {
    let minLevel = labels.reduce((a, b) => Math.min(a, b.level ?? 0), 0);
    let maxLevel = labels.reduce((a, b) => Math.max(a, b.level ?? 0), 0);
    let threshold = normalScale * 2;
    let groupMaxScale = maxInverseScale(normalScale);
    return labels.map((cluster) => {
      let location = { x: cluster.x, y: cluster.y };
      let level = cluster.level ?? 0;
      let fontSize = level == 0 ? 14 : 12;
      let size;
      if (typeof cluster.content !== "string") {
        // Use the pre-computed display dimensions for collision detection
        size = { width: cluster.content.width + 4, height: cluster.content.height + 4 };
      } else {
        size = measureText({
          text: cluster.content,
          fontSize: fontSize,
          fontFamily: fontFamily,
        });
        size.width += 4;
        size.height += 4;
      }
      return {
        content: cluster.content,
        fontSize: fontSize,
        bounds: {
          xMin: location.x - size.width / 2,
          xMax: location.x + size.width / 2,
          yMin: location.y - size.height / 2,
          yMax: location.y + size.height / 2,
        },
        locationAtZero: location,
        priority: cluster.priority,
        minScale: level == maxLevel ? null : 1 / (threshold * Math.pow(2, level) * 1.2),
        maxScale: level == minLevel ? groupMaxScale : Math.min(groupMaxScale, 1 / (threshold * Math.pow(2, level - 1))),
        coordinate: { x: cluster.x, y: cluster.y },
        placement: null,
      };
    });
  });
  let placements = await dynamicLabelPlacement(result, { globalMaxScale: globalMaxScale });
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
