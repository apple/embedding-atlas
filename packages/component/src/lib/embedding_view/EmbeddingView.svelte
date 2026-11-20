<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { type EmbeddingViewProps } from "./embedding_view_api.js";
  import {
    defaultLayerSpecs,
    layerContentKeys,
    objectIds,
    resolveDensityLayer,
    resolvePointsLayer,
    type AutoLabels,
    type ResolvedLayer,
    type ResolvedSource,
  } from "./layers.js";
  import { categoryCountOf, dataStats, type DataStats } from "./data_stats.js";

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
    labels = null,
    layers = null,
    customTooltip = null,
    customOverlay = null,
    querySelection = null,
    queryClusterLabels = null,
    onViewportState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    onLayerValues = null,
    cache = null,
  }: EmbeddingViewProps = $props();

  type LayerSpec = NonNullable<EmbeddingViewProps["layers"]>[number];
  type SourceData = {
    x: Float32Array<ArrayBuffer>;
    y: Float32Array<ArrayBuffer>;
    category?: Uint8Array<ArrayBuffer> | null;
  };

  let mainSourceData = $derived({ x: data.x, y: data.y, category: data.category ?? null });
  let derivedProperties = $derived(computeDataStats(mainSourceData));

  // The plain view estimates the density without categories. Stats are cached by the arrays, so a streaming `data`
  // (new object every frame, same category array) doesn't rescan the categories.
  function computeDataStats(data: SourceData): DataStats {
    return { ...dataStats(data.x, data.y, null), categoryCount: categoryCountOf(data.category) };
  }

  // Ids for data arrays (a revision of a layer's data, to detect when it changes).
  const objectId = objectIds();

  // Derived properties of layer-specific data, by source key.
  let layerSourceCache = new Map<string, { data: SourceData; properties: DataStats }>();
  // The source key last used by each layer id, so a layer keeps its source key when its data changes.
  let layerSourceKeys = new Map<string, string>();
  let nextLayerSourceKey = 0;

  function makeSource(key: string, data: SourceData, properties: DataStats): ResolvedSource {
    return {
      key: key,
      x: data.x,
      y: data.y,
      category: data.category ?? null,
      categoryCount: properties.categoryCount,
      totalCount: properties.count,
      maxDensity: properties.maxDensity,
    };
  }

  /** Whether `data` can be drawn from `source`: the same x and y arrays, and the same categories if
   *  `needsCategory` is set (layers that don't use categories can use any source with the same x and y). */
  function sameData(source: SourceData, data: SourceData, needsCategory: boolean = true) {
    return (
      source.x === data.x &&
      source.y === data.y &&
      (!needsCategory || (source.category ?? null) === (data.category ?? null))
    );
  }

  /** The automatic labels of a labels layer (null if there's no `queryClusterLabels`). */
  function autoLabels(layer: Extract<LayerSpec, { type: "labels" }>, id: string): AutoLabels | null {
    let fn = layer.queryClusterLabels ?? queryClusterLabels;
    if (fn == null) {
      return null;
    }
    let ownData = layer.data != null && !sameData(mainSourceData, layer.data, false);
    let data = ownData ? layer.data! : mainSourceData;
    let ownFunction = fn !== queryClusterLabels;
    let stopWords = config?.autoLabelStopWords ?? null;
    let cacheKey = layer.cacheKey ?? null;
    // The query: the data (the main data, or the layer's own data, identified by the layer so that it's stable while
    // the data changes), and the function (a layer's own function is identified by the layer, not by the function,
    // so an inline function that changes on every render doesn't regenerate the labels). Labels layers don't upload
    // their data to the renderer: label generation uploads it temporarily.
    let query = {
      data: ownData ? `layer:${id}` : "main",
      fn: ownFunction ? `layer:${id}` : null,
      cacheKey,
      stopWords,
      densityThreshold: layer.densityThreshold ?? null,
    };
    let queryKey = JSON.stringify(query);
    // Labels of the main data are generated once (streaming data doesn't regenerate them); labels of a layer's own
    // data are regenerated when its data changes.
    let revision = ownData ? `${objectId(data.x)},${objectId(data.y)}` : null;
    let requestKey = JSON.stringify([queryKey, revision]);
    let x = data.x;
    let y = data.y;
    return {
      requestKey,
      queryKey,
      mergeKey: requestKey,
      // The cache is specific to the component's data (see `cache`): its main labels are identified by the stop
      // words; other labels need a `cacheKey`.
      persistKey:
        cacheKey != null ? { cacheKey, stopWords } : !ownData && !ownFunction ? { data: "main", stopWords } : null,
      densityThreshold: layer.densityThreshold ?? null,
      // Wait for data (e.g., streaming data that starts empty): labels of the main data are generated once.
      ready: data.x.length > 0,
      loadSnapshot: async () => ({ x, y, viewport: dataStats(x, y, null).defaultViewportState }),
      queryClusterLabels: (clusters) => fn(clusters),
    };
  }

  function resolveLayers(layers: LayerSpec[] | null): ResolvedLayer[] {
    let mainSource = makeSource("main", mainSourceData, derivedProperties);

    let specs: LayerSpec[] = layers ?? defaultLayerSpecs(config, labels);

    // Layers with their own data get a source keyed by the data: the key previously used for the same
    // arrays, or else the key previously used by the layer (so the key is stable while the layer's data
    // changes). Keys don't depend on layer positions, so they survive adding, removing, or reordering
    // other layers. Layers sharing data share a source.
    let sources: ResolvedSource[] = [];
    let usedKeys = new Set<string>();
    let newLayerSourceKeys = new Map<string, string>();
    function sourceFor(layerId: string, data: SourceData | null | undefined, needsCategory: boolean): ResolvedSource {
      if (data == null || sameData(mainSourceData, data, needsCategory)) {
        return mainSource;
      }
      let existing = sources.find((s) => sameData(s, data, needsCategory));
      if (existing != null) {
        newLayerSourceKeys.set(layerId, existing.key);
        return existing;
      }
      let key: string | undefined = undefined;
      for (let [k, v] of layerSourceCache) {
        if (!usedKeys.has(k) && sameData(v.data, data)) {
          key = k;
          break;
        }
      }
      if (key == null) {
        let previous = layerSourceKeys.get(layerId);
        key = previous != null && !usedKeys.has(previous) ? previous : "layer:" + nextLayerSourceKey++;
      }
      usedKeys.add(key);
      newLayerSourceKeys.set(layerId, key);
      let cached = layerSourceCache.get(key);
      if (cached == null || !sameData(cached.data, data)) {
        cached = { data: data, properties: computeDataStats(data) };
        layerSourceCache.set(key, cached);
      }
      let source = makeSource(key, data, cached.properties);
      sources.push(source);
      return source;
    }

    let result: ResolvedLayer[] = [];
    specs.forEach((layer, index) => {
      let id = layer.id ?? String(index);
      // The key is assigned below, once all layers are resolved.
      let key = "";
      if (layer.visible === false) {
        return;
      }
      switch (layer.type) {
        case "points":
        case "density": {
          let base = {
            key: key,
            id: id,
            source: sourceFor(id, layer.data, layer.data?.category != null),
            // Layers with their own data without categories don't use the categories of a shared source.
            useCategory: layer.data == null || layer.data.category != null,
          };
          // The component's colors are for the main categories. Layers with their own categories default to
          // colors for their own number of categories (null: see the renderer).
          let ownCategories = layer.data?.category != null && layer.data.category !== mainSourceData.category;
          let colors = ownCategories ? null : categoryColors;
          result.push(
            layer.type == "points"
              ? resolvePointsLayer(layer, base, colors, config)
              : resolveDensityLayer(layer, base, colors, config),
          );
          break;
        }
        case "labels": {
          result.push({
            type: "labels",
            key: key,
            id: id,
            opacity: layer.opacity ?? null,
            labels: layer.labels ?? null,
            auto: layer.labels == null ? autoLabels(layer, id) : null,
          });
          break;
        }
      }
    });

    // Labels layers are keyed by their label query (they don't have render sources).
    let keys = layerContentKeys(
      result.map((l) =>
        l.type == "labels"
          ? { type: l.type, source: null, content: l.auto?.queryKey ?? "static" }
          : { type: l.type, source: l.source.key },
      ),
    );
    result.forEach((layer, i) => {
      layer.key = keys[i];
    });

    for (let key of layerSourceCache.keys()) {
      if (!usedKeys.has(key)) {
        layerSourceCache.delete(key);
      }
    }
    layerSourceKeys = newLayerSourceKeys;

    return result;
  }

  let resolvedLayers = $derived(resolveLayers(layers));
</script>

<EmbeddingViewImpl
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  theme={theme}
  config={config}
  layers={resolvedLayers}
  pointCount={derivedProperties.count}
  categoryCount={derivedProperties.categoryCount}
  categoryColors={categoryColors}
  defaultViewportState={derivedProperties.defaultViewportState}
  querySelection={querySelection}
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
  onLayerValues={onLayerValues}
  cache={cache}
/>
