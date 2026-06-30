<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { cameraFromBounds3D, pointsBounds3D, type Bounds3D, type Camera3DState } from "../camera3d.js";
  import { type EmbeddingViewProps } from "./embedding_view_api.js";
  import { approximateDensity2D, median, stdev } from "./statistics.js";
  import type { DataPoint } from "./types.js";

  let {
    data,
    tooltip = null,
    selection = null,
    rangeSelection = null,
    categoryColors = null,
    width = null,
    height = null,
    pixelRatio = null,
    theme = null,
    config = null,
    viewportState = null,
    camera3DState = null,
    labels = null,
    customTooltip = null,
    customOverlay = null,
    querySelection = null,
    queryByIndex = null,
    queryClusterLabels = null,
    onViewportState = null,
    onCamera3DState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    cache = null,
  }: EmbeddingViewProps = $props();

  let derivedProperties = $derived(computeDerivedProperties(data));

  // Split the O(N) bounding-sphere scan from the camera so it runs only when the 3D
  // point data changes, NOT on every resize: resizing recomputes just the O(1)
  // aspect-dependent distance from the cached bounds.
  let bounds3D = $derived<Bounds3D | null>(data.z != null ? pointsBounds3D(data.x, data.y, data.z) : null);
  let defaultCamera3DState = $derived<Camera3DState | null>(
    bounds3D != null ? cameraFromBounds3D(bounds3D, undefined, (width ?? 800) / (height ?? 800)) : null,
  );

  // For the array-backed view, resolve a point by its index directly from the arrays.
  let resolvedQueryByIndex = $derived<((index: number) => Promise<DataPoint | null>) | null>(
    queryByIndex ??
      (data.z != null
        ? async (index: number) => {
            if (index < 0 || index >= data.x.length) {
              return null;
            }
            let point: DataPoint = { x: data.x[index], y: data.y[index], z: data.z![index] };
            if (data.category != null) {
              point.category = data.category[index];
            }
            return point;
          }
        : null),
  );

  function computeDerivedProperties(data: EmbeddingViewProps["data"]): {
    count: number;
    categoryCount: number;
    maxDensity: number;
    defaultViewportState: { x: number; y: number; scale: number };
  } {
    let categoryCount = 1;
    if (data.category != null) {
      categoryCount = data.category.reduce((a, b) => Math.max(a, b), 0) + 1;
    }
    let xCenter = median(data.x);
    let yCenter = median(data.y);
    let xStd = stdev(data.x);
    let yStd = stdev(data.y);
    let scaler = 1.0 / (Math.max(xStd, yStd, 1e-3) * 3);
    let binWidth = 0.1 / scaler;
    let maxDensity = approximateDensity2D(data.x, data.y, binWidth, xCenter, yCenter);
    return {
      count: data.x.length,
      categoryCount: categoryCount,
      maxDensity: maxDensity,
      defaultViewportState: { x: xCenter, y: yCenter, scale: scaler * 0.95 },
    };
  }
</script>

<EmbeddingViewImpl
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  theme={theme}
  config={config}
  data={{ x: data.x, y: data.y, z: data.z ?? null, category: data.category ?? null }}
  totalCount={derivedProperties.count}
  maxDensity={derivedProperties.maxDensity}
  categoryCount={derivedProperties.categoryCount}
  categoryColors={categoryColors}
  defaultViewportState={derivedProperties.defaultViewportState}
  defaultCamera3DState={defaultCamera3DState}
  camera3DState={camera3DState}
  onCamera3DState={onCamera3DState}
  querySelection={querySelection}
  queryByIndex={resolvedQueryByIndex}
  queryClusterLabels={queryClusterLabels}
  labels={labels}
  customTooltip={customTooltip}
  customOverlay={customOverlay}
  tooltip={tooltip}
  onTooltip={onTooltip}
  selection={selection}
  onSelection={onSelection}
  viewportState={viewportState}
  onViewportState={onViewportState}
  rangeSelection={rangeSelection}
  onRangeSelection={onRangeSelection}
  cache={cache}
/>
