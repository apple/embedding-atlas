// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { defaultLabelsLayer, type DefaultLabelsOptions, type LayerSpec } from "./layer_data.js";

export interface LayerPreset {
  label: string;
  layers: LayerSpec[];
}

/** The color of context points (all data, regardless of the filter). */
const contextColor = "#888888";

/** Layer presets for the embedding view. Points and density layers on the view's table follow the filter
 *  by default, and labels layers don't (see `resolveLayerData`), so the labels (if any) are from all data.
 *  The presets use the default labels layer (see `defaultLabelsLayer`). */
export function layerPresets(options: DefaultLabelsOptions): LayerPreset[] {
  let labelsLayer = defaultLabelsLayer(options);
  let labels: LayerSpec[] = labelsLayer != null ? [labelsLayer] : [];
  return [
    // The default: filtered points.
    {
      label: "Points",
      layers: [{ type: "points" }, ...labels],
    },
    // Filtered points over faint gray points that show all data.
    {
      label: "Points with Context",
      layers: [{ type: "points", filter: null, color: contextColor, opacity: 0.15 }, { type: "points" }, ...labels],
    },
    // Density contours and points of the filtered data.
    {
      label: "Density",
      layers: [{ type: "density" }, { type: "points" }, ...labels],
    },
  ];
}
