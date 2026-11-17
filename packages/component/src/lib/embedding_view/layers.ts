// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// Internal types for layers after they are resolved by EmbeddingView / EmbeddingViewMosaic
// (data loaded, defaults applied). These are consumed by EmbeddingViewImpl.

import type { Rectangle, ViewportState } from "../utils.js";
import type { EmbeddingViewConfig } from "./embedding_view_config.js";
import type { Label, LabelContent } from "./types.js";

/** Maximum number of categories for density layers. Density layers with more categories are drawn without categories.
 *  Technically we can support 256 categories, but it's limited by speed and memory usage.
 *  32 is chosen here so that the total memory usage of a 2048 x 2048 x categories x f16 buffer is 256MB. */
export const MAX_DENSITY_CATEGORIES = 32;

/** A data source shared by one or more layers. */
export interface ResolvedSource {
  /** A unique key for the source. Layers with the same data share the same source key. */
  key: string;
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  category: Uint8Array<ArrayBuffer> | null;
  categoryCount: number;
  /** Total number of points (before filtering), used to estimate density. */
  totalCount: number | null;
  /** The approximate max density at the default viewport. */
  maxDensity: number | null;
}

/** The parts of a points or density layer determined by EmbeddingView / EmbeddingViewMosaic (keys and data). */
interface ResolvedLayerBase {
  /** A unique internal key based on the layer's content (see `layerContentKeys`). */
  key: string;
  /** The layer's id for reporting (e.g., in `onLayerValues`): the user-specified id, or the layer's index. */
  id: string;
  source: ResolvedSource;
  /** Whether to color by the source's categories. If false, the layer uses the first category color. */
  useCategory: boolean;
}

export interface ResolvedPointsLayer extends ResolvedLayerBase {
  type: "points";
  categoryColors: string[] | null;
  opacity: number | "auto";
  pointSize: number | null;
  minimumDensity: number;
  downsampleMaxPoints: number | null;
  downsampleDensityWeight: number;
}

export interface ResolvedDensityLayer extends ResolvedLayerBase {
  type: "density";
  categoryColors: string[] | null;
  opacity: number | "auto";
  minimumDensity: number;
  fill: boolean;
  contours: boolean;
}

/** The data that automatic labels are generated from. Its arrays are never modified, so a label generation
 *  that holds a snapshot always works on consistent data. */
export interface LabelSnapshot {
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  /** The viewport the labels are generated over (it fits the layer's data, e.g., its unfiltered data, so that it
   *  doesn't change with the filter). The labels' levels are relative to its scale. */
  viewport: ViewportState;
}

/** How a labels layer's automatic labels are generated (see `LabelLayers` and EmbeddingViewImpl). */
export interface AutoLabels {
  /** Identifies the labels in this session: the labels are regenerated when it changes. */
  requestKey: string;
  /** Identifies the label query, without the data revision (e.g., the filter): when only the data revision
   *  changes, the previous labels are shown until the new ones are generated (debounced). Otherwise, they're
   *  cleared and the new labels are generated right away. */
  queryKey: string;
  /** Generations with the same merge key (running at the same time) are shared. */
  mergeKey: string;
  /** The key of the labels in the persistent cache (stable across sessions), or null to not persist them. */
  persistKey: any | null;
  densityThreshold: number | null;
  /** Whether the inputs are known (e.g., the filter mask is loaded). If not, the layer waits. */
  ready: boolean;
  /** Load the data (only called if the labels are not in the persistent cache). */
  loadSnapshot: () => Promise<LabelSnapshot>;
  /** The labels of the clusters (rectangles in data coordinates) of a snapshot. */
  queryClusterLabels: (clusters: Rectangle[][], snapshot: LabelSnapshot) => Promise<(LabelContent | null)[]>;
}

export interface ResolvedLabelsLayer {
  type: "labels";
  /** A unique internal key based on the layer's content (see `layerContentKeys`). */
  key: string;
  /** The layer's id for reporting (e.g., in `onLayerValues`): the user-specified id, or the layer's index. */
  id: string;
  opacity: number | null;
  /** Static labels. */
  labels: Label[] | null;
  /** Automatic labels, if `labels` is null. Null if they can't be generated (e.g., no text column). */
  auto: AutoLabels | null;
}

export type ResolvedLayer = ResolvedPointsLayer | ResolvedDensityLayer | ResolvedLabelsLayer;

/** The fields of a points layer's spec (the same in EmbeddingView and EmbeddingViewMosaic). */
interface PointsLayerSpec {
  categoryColors?: string[] | null;
  opacity?: number | "auto";
  pointSize?: number | null;
  downsampleMaxPoints?: number | null;
  downsampleDensityWeight?: number | null;
}

/** The fields of a density layer's spec (the same in EmbeddingView and EmbeddingViewMosaic). */
interface DensityLayerSpec {
  categoryColors?: string[] | null;
  opacity?: number | "auto";
  fill?: boolean | null;
  contours?: boolean | null;
}

/** Resolve a points layer, applying the defaults from the component's category colors and config. */
export function resolvePointsLayer(
  spec: PointsLayerSpec,
  base: ResolvedLayerBase,
  categoryColors: string[] | null,
  config: EmbeddingViewConfig | null | undefined,
): ResolvedPointsLayer {
  return {
    type: "points",
    ...base,
    categoryColors: spec.categoryColors ?? categoryColors,
    opacity: spec.opacity ?? "auto",
    pointSize: spec.pointSize ?? config?.pointSize ?? null,
    minimumDensity: config?.minimumDensity ?? 1 / 16,
    // null disables downsampling, undefined uses the config.
    downsampleMaxPoints:
      spec.downsampleMaxPoints !== undefined ? spec.downsampleMaxPoints : (config?.downsampleMaxPoints ?? 4000000),
    downsampleDensityWeight: spec.downsampleDensityWeight ?? config?.downsampleDensityWeight ?? 5,
  };
}

/** Resolve a density layer, applying the defaults from the component's category colors and config. */
export function resolveDensityLayer(
  spec: DensityLayerSpec,
  base: ResolvedLayerBase,
  categoryColors: string[] | null,
  config: EmbeddingViewConfig | null | undefined,
): ResolvedDensityLayer {
  return {
    type: "density",
    ...base,
    categoryColors: spec.categoryColors ?? categoryColors,
    opacity: spec.opacity ?? "auto",
    minimumDensity: config?.minimumDensity ?? 1 / 16,
    fill: spec.fill ?? true,
    contours: spec.contours ?? true,
  };
}

/** Assigns small integer ids to objects by identity (e.g., to tell functions or data arrays apart in keys). */
export function objectIds(): (object: object) => number {
  let ids = new WeakMap<object, number>();
  let nextId = 1;
  return (object) => {
    let id = ids.get(object);
    if (id == null) {
      id = nextId++;
      ids.set(object, id);
    }
    return id;
  };
}

/** The effective values of a layer, reported with `onLayerValues`. */
export type LayerValues =
  | { type: "points"; id: string; opacity: number; pointSize: number }
  | { type: "density"; id: string; opacity: number }
  | { type: "labels"; id: string; opacity: number; labels: Label[] | null };

/** Content-based keys for layers: the type, the data source, what else determines the layer's content
 *  (e.g., the columns that automatic labels are generated from), and a counter for repeated layers.
 *  Layers keep their keys (and GPU resources, generated labels) when other layers are added, removed, hidden,
 *  or reordered, unless those layers have the same content (then they're interchangeable), and keys never
 *  collide (unlike user-specified ids). `source` is null for static labels. */
export function layerContentKeys(layers: { type: string; source: string | null; content?: string }[]): string[] {
  let counts = new Map<string, number>();
  return layers.map((layer) => {
    let base = `${layer.type}|${layer.source ?? "static"}${layer.content != null ? `|${layer.content}` : ""}`;
    let n = counts.get(base) ?? 0;
    counts.set(base, n + 1);
    return `${base}|${n}`;
  });
}

export type DefaultLayerSpec = { type: "points" } | { type: "density" } | { type: "labels"; labels: Label[] | null };

/** The layers to use when `layers` is not specified: points (with density in density mode), and labels
 *  (the given static labels, or automatic labels unless they're disabled). */
export function defaultLayerSpecs(
  config: EmbeddingViewConfig | null | undefined,
  labels: Label[] | null,
): DefaultLayerSpec[] {
  let result: DefaultLayerSpec[] =
    config?.mode == "density" ? [{ type: "density" }, { type: "points" }] : [{ type: "points" }];
  if (labels != null || config?.autoLabelEnabled !== false) {
    result.push({ type: "labels", labels: labels });
  }
  return result;
}
