<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { type EmbeddingViewProps } from "./embedding_view_api.js";
  import { approximateDensity2D, median, stdev } from "./statistics.js";

  let {
    data,
    tooltip = null,
    selection = null,
    rangeSelection = null,
    categoryColors = null,
    width = null,
    height = null,
    pixelRatio = null,
    colorScheme = "light",
    theme = null,
    viewportState = null,
    automaticLabels = false,
    mode = "density",
    minimumDensity = 1 / 16,
    customTooltip = null,
    customOverlay = null,
    querySelection = null,
    queryClusterLabels = null,
    onViewportState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
  }: EmbeddingViewProps = $props();

  let derivedProperties = $derived(computeDerivedProperties(data));

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
  mode={mode ?? "points"}
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  colorScheme={colorScheme ?? "light"}
  theme={theme}
  data={{ x: data.x, y: data.y, category: data.category ?? null }}
  totalCount={derivedProperties.count}
  maxDensity={derivedProperties.maxDensity}
  categoryCount={derivedProperties.categoryCount}
  categoryColors={categoryColors}
  defaultViewportState={derivedProperties.defaultViewportState}
  querySelection={querySelection}
  queryClusterLabels={queryClusterLabels}
  automaticLabels={automaticLabels ?? false}
  minimumDensity={minimumDensity ?? 1 / 16}
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
/>
