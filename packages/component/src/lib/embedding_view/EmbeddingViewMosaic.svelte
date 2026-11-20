<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { deepEquals, imageToDataUrl } from "@embedding-atlas/utils";
  import {
    coordinator as defaultCoordinator,
    isSelection,
    type MosaicClient,
    type Selection,
  } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { onDestroy, untrack } from "svelte";

  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { defaultCategoryColors } from "../colors.js";
  import { ImageSummarizer } from "../image_summarizer/image_summarizer.js";
  import { type Point, type Rectangle, type ViewportState } from "../utils.js";
  import {
    categoryIndexExpr,
    fieldExpr,
    fieldKey,
    scaleCategoryCount,
    scaleColors,
    type ColorScale,
    type SQLField,
  } from "./data/fields.js";
  import { bitsetAny, normalizeKey, type KeyIndex } from "./data/key_index.js";
  import {
    chunkQuery,
    MaskedArrays,
    maskPositions,
    TableStore,
    type ColumnSpec,
    type FilterMask,
  } from "./data/table_store.js";
  import { dataStats } from "./data_stats.js";
  import type { EmbeddingViewMosaicProps } from "./embedding_view_mosaic_api.js";
  import { IMAGE_LABEL_SIZE } from "./labels.js";
  import {
    defaultLayerSpecs,
    layerContentKeys,
    objectIds,
    resolveDensityLayer,
    resolvePointsLayer,
    type AutoLabels,
    type LabelSnapshot,
    type ResolvedLayer,
    type ResolvedSource,
  } from "./layers.js";
  import { DataPointQuery, predicateForDataPoints, predicateForRangeSelection } from "./mosaic_client.js";
  import type { DataPoint, DataPointID, LabelContent } from "./types.js";
  import {
    textSummarizerAdd,
    textSummarizerCreate,
    textSummarizerDestroy,
    textSummarizerSummarize,
  } from "./worker/index.js";

  let {
    coordinator = defaultCoordinator(),
    table,
    x,
    y,
    category = null,
    colorScale = null,
    color = null,
    text = null,
    image = null,
    importance = null,
    identifier = null,
    filter = null,
    categoryColors = null,
    tooltip = null,
    additionalFields = null,
    selection = null,
    rangeSelection = null,
    rangeSelectionValue = null,
    width = null,
    height = null,
    pixelRatio = null,
    config = null,
    theme = null,
    viewportState = null,
    labels = null,
    layers = null,
    customTooltip = null,
    customOverlay = null,
    onViewportState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    onLayerValues = null,
    cache = null,
  }: EmbeddingViewMosaicProps = $props();

  type LayerSpec = NonNullable<EmbeddingViewMosaicProps["layers"]>[number];

  let effectiveTooltip: DataPoint | null = $state.raw(null);
  let effectiveSelection: DataPoint[] | null = $state.raw(null);
  let effectiveRangeSelection: Rectangle | Point[] | null = $state.raw(null);

  // ---------------------------------------------------------------------------------------------------------------
  // Data
  //
  // The columns of each table are loaded once by the TableStore, for all layers, ordered by the table's key column
  // (its identifier column, or DuckDB's rowid), so the columns of a table are aligned. A filter is a mask (one query
  // of the keys of the matching rows), applied to the loaded columns. Arrays and masks are never modified (a change
  // gives new ones), so automatic labels are generated from consistent snapshots of the same data.

  let store = $derived(new TableStore(coordinator));
  $effect.pre(() => {
    let current = store;
    return () => current.destroy();
  });

  /** Report a failed query (e.g., a layer specifies a column or table that doesn't exist). */
  function logError(what: any, error: unknown) {
    console.error(`Embedding view: failed to load ${JSON.stringify(what)}`, error);
  }

  // Ids for Selection objects (part of source and mask keys).
  const objectId = objectIds();
  function selectionId(selection: Selection | null): number | null {
    return selection != null ? objectId(selection) : null;
  }

  interface LayerColumns {
    table: string;
    /** The table's key column: its identifier column, or null for DuckDB's rowid. */
    key: string | null;
    /** The x and y fields (null if a layer on another table doesn't specify them). */
    x: SQLField | null;
    y: SQLField | null;
    category: SQLField | null;
    text: SQLField | null;
    image: SQLField | null;
    importance: SQLField | null;
  }

  /** A category field, and the scale that maps its values to category indices. */
  interface CategorySpec {
    field: SQLField;
    scale: ColorScale | null;
  }

  /** The data of a points or density layer: a table's fields, filtered by a selection. */
  interface SourceDescriptor {
    table: string;
    key: string | null;
    x: SQLField;
    y: SQLField;
    category: CategorySpec | null;
    filter: Selection | null;
  }

  /** A key of the category indices of a category spec (the scale's colors don't change them). */
  function categoryKey(c: CategorySpec | null): string | null {
    if (c == null) {
      return null;
    }
    let s = c.scale;
    let indices =
      scaleCategoryCount(s) != null ? [s!.domain?.map(String) ?? null, s!.thresholds ?? null, s!.type ?? null] : null;
    return JSON.stringify([fieldKey(c.field), indices]);
  }

  /** The SQL expression of a category spec's indices. */
  function categoryExpr(c: CategorySpec): SQL.ExprNode {
    return categoryIndexExpr(fieldExpr(c.field), c.scale);
  }

  /** The colors of category indices with a scale (null: default colors). */
  /** The colors of category indices: a scale's colors, otherwise the fallback (`categoryColors`, which a scale
   *  replaces even without a range). Null: default colors. */
  function colorsOf(scale: ColorScale | null, fallback: string[] | null): string[] | null {
    return scale != null ? (scaleColors(scale) ?? scale.range ?? null) : fallback;
  }

  interface PlannedLayer {
    spec: LayerSpec;
    /** The content-based key (see layerContentKeys). */
    key: string;
    /** The id for reporting: the user-specified id, or the layer's index. */
    id: string;
    columns: LayerColumns;
    filter: Selection | null;
    /** The source key of a points or density layer (layers with the same data share a source). */
    source: string | null;
    descriptor: SourceDescriptor | null;
    /** Whether the layer colors by category. */
    useCategory: boolean;
  }

  let mainCategory: CategorySpec | null = $derived(category != null ? { field: category, scale: colorScale } : null);
  let mainDescriptor: SourceDescriptor = $derived({ table, key: identifier, x, y, category: mainCategory, filter });
  /** The category index of the main data (tooltips and selections report it as the point's category). */
  // Fields used by the selection clauses and the point query, which only change when their SQL changes (so that,
  // e.g., an equal `{ sql }` object or color scale passed again doesn't publish the clauses again).
  let xKey = $derived(fieldKey(x));
  let yKey = $derived(fieldKey(y));
  let textKey = $derived(fieldKey(text));
  let mainCategoryKey = $derived(categoryKey(mainCategory));
  let stableX: SQLField = $derived.by(() => {
    xKey;
    return untrack(() => x);
  });
  let stableY: SQLField = $derived.by(() => {
    yKey;
    return untrack(() => y);
  });
  let stableText: SQLField | null = $derived.by(() => {
    textKey;
    return untrack(() => text);
  });
  let mainCategoryIndex: SQL.ExprNode | null = $derived.by(() => {
    mainCategoryKey;
    return untrack(() => (mainCategory != null ? categoryExpr(mainCategory) : null));
  });
  /** The colors of the main data: by category, or the fixed color. */
  let mainColors: string[] | null = $derived(
    category != null ? colorsOf(colorScale, categoryColors) : color != null ? [color] : categoryColors,
  );

  /** A key of a source's data: its columns (table, key, x, y, category) and filter. Optionally without the category
   *  (the positions that layers without categories can share) or without the filter (the columns to load). */
  function descriptorKey(d: SourceDescriptor, options: { category?: boolean; filter?: boolean } = {}): string {
    return JSON.stringify([
      d.table,
      d.key,
      fieldKey(d.x),
      fieldKey(d.y),
      options.category === false ? null : categoryKey(d.category),
      options.filter === false ? null : selectionId(d.filter),
    ]);
  }

  /** The source key ("main" for the main data). Sources with the same key share their data and GPU buffers. */
  function sourceKey(descriptor: SourceDescriptor): string {
    let key = descriptorKey(descriptor);
    return key == descriptorKey(mainDescriptor) ? "main" : key;
  }

  /** The columns to load for a source (aligned by the table's key). */
  function descriptorColumns(d: SourceDescriptor): ColumnSpec[] {
    return [
      ...positionColumns(d.x, d.y),
      ...(d.category != null ? [{ expr: categoryExpr(d.category), type: "uint8" as const }] : []),
    ];
  }

  function positionColumns(x: SQLField, y: SQLField): ColumnSpec[] {
    return [
      { expr: fieldExpr(x), type: "float32" },
      { expr: fieldExpr(y), type: "float32" },
    ];
  }

  function columnSetKey(d: SourceDescriptor) {
    return descriptorKey(d, { filter: false });
  }

  function maskKey(table: string, key: string | null, filter: Selection): string {
    return JSON.stringify([table, key, selectionId(filter)]);
  }

  function layerColumns(spec: LayerSpec): LayerColumns {
    let data: {
      table?: string;
      identifier?: string | null;
      x?: SQLField;
      y?: SQLField;
      category?: SQLField | null;
      text?: SQLField | null;
      image?: SQLField | null;
      importance?: SQLField | null;
    } = spec.data ?? {};
    // Fields are inherited from the component only when the layer uses the same table.
    let inherit = data.table == null || data.table == table;
    let get = <T,>(value: T | null | undefined, fallback: T | null): T | null =>
      value !== undefined ? value : inherit ? fallback : null;
    // A layer with a fixed color doesn't inherit the component's category.
    let inheritedCategory = spec.type != "labels" && spec.color != null ? null : category;
    return {
      table: data.table ?? table,
      key: get(data.identifier, identifier),
      x: get(data.x, x),
      y: get(data.y, y),
      category: spec.type != "labels" ? get(data.category, inheritedCategory) : null,
      text: spec.type == "labels" ? get(data.text, text) : null,
      image: spec.type == "labels" ? get(data.image, image) : null,
      importance: spec.type == "labels" ? get(data.importance, importance) : null,
    };
  }

  /** Whether a layer's category is the component's (the same table and field). */
  function isMainCategory(columns: LayerColumns): boolean {
    return columns.table == table && category != null && fieldKey(columns.category) == fieldKey(category);
  }

  /** Whether automatic labels can be generated from the columns: a text column, or image and importance columns. */
  function hasLabelColumns(columns: LayerColumns): boolean {
    return columns.text != null || (columns.image != null && columns.importance != null);
  }

  // Warn once per layer about layers that can't be drawn.
  let warnedLayers = new Set<string>();

  function planLayers(layers: LayerSpec[] | null): PlannedLayer[] {
    let specs: LayerSpec[] = layers ?? defaultLayerSpecs(config, labels);
    let result: PlannedLayer[] = [];
    specs.forEach((spec, index) => {
      if (spec.visible === false) {
        return;
      }
      let id = spec.id ?? String(index);
      let columns = layerColumns(spec);
      // Static labels don't need data, so they don't need x and y either.
      let isStaticLabels = spec.type == "labels" && spec.labels != null;
      if ((columns.x == null || columns.y == null) && !isStaticLabels) {
        let warning = `Embedding view layer "${id}" uses a different table, x and y must be specified.`;
        if (!warnedLayers.has(warning)) {
          warnedLayers.add(warning);
          console.warn(warning);
        }
        return;
      }
      // Points and density layers on the component's table default to the component's filter.
      // Layers on other tables default to no filter (the component's filter is for its own table).
      // Labels layers default to no filter, so automatic labels are generated once from all data.
      let filteredByDefault = spec.type != "labels" && columns.table == table;
      let layerFilter = spec.filter !== undefined ? spec.filter : filteredByDefault ? filter : null;
      let useCategory = spec.type != "labels" && columns.category != null;
      let descriptor: SourceDescriptor | null = null;
      let source: string | null = null;
      if (useCategory) {
        // The layer's scale, or the component's for the component's category.
        let scale =
          spec.type != "labels" && spec.colorScale !== undefined
            ? spec.colorScale
            : isMainCategory(columns)
              ? colorScale
              : null;
        let { table, key, x, y } = columns;
        descriptor = { table, key, x: x!, y: y!, category: { field: columns.category!, scale }, filter: layerFilter };
        source = sourceKey(descriptor);
      }
      result.push({ spec, key: "", id, columns, filter: layerFilter, source, descriptor, useCategory });
    });
    // Points and density layers without categories share the source of the main data, or of another layer with the
    // same table, x, y, and filter (regardless of its category). The columns are loaded once either way; sharing a
    // source also shares its GPU buffers.
    for (let layer of result) {
      if (layer.useCategory || layer.spec.type == "labels") {
        continue;
      }
      let c = layer.columns;
      let own: SourceDescriptor = {
        table: c.table,
        key: c.key,
        x: c.x!,
        y: c.y!,
        category: null,
        filter: layer.filter,
      };
      let positions = descriptorKey(own, { category: false });
      let matches = (d: SourceDescriptor) => descriptorKey(d, { category: false }) == positions;
      if (matches(mainDescriptor)) {
        layer.descriptor = mainDescriptor;
      } else {
        let other = result.find((l) => l.useCategory && matches(l.descriptor!));
        layer.descriptor = other?.descriptor ?? own;
      }
      layer.source = sourceKey(layer.descriptor);
    }
    // Content-based keys (see layerContentKeys). Labels layers are keyed by their label columns.
    let keys = layerContentKeys(
      result.map((l) => ({
        type: l.spec.type,
        source: l.source,
        content:
          l.spec.type == "labels"
            ? JSON.stringify([
                l.columns.table,
                fieldKey(l.columns.x),
                fieldKey(l.columns.y),
                fieldKey(l.columns.text),
                fieldKey(l.columns.image),
                fieldKey(l.columns.importance),
              ])
            : undefined,
      })),
    );
    result.forEach((layer, i) => {
      layer.key = keys[i];
    });
    return result;
  }

  let plannedLayers = $derived(planLayers(layers));

  // Loaded columns (x, y, and category) of the main data and of the points and density layers, by column set key.
  interface ColumnSet {
    x: Float32Array<ArrayBuffer>;
    y: Float32Array<ArrayBuffer>;
    category: Uint8Array<ArrayBuffer> | null;
  }
  let columnSets: Record<string, ColumnSet> = $state.raw({});
  // Filter masks, by mask key.
  let masks: Record<string, FilterMask> = $state.raw({});
  // The store that `columnSets`, `masks`, and the mask clients belong to.
  let dataStore: TableStore | null = null;
  let requestedColumnSets = new Set<string>();
  let maskClients = new Map<string, MosaicClient>();
  // The mask clients by mask key (a snapshot of `maskClients`, for `selfClients`).
  let activeMaskClients: Map<string, MosaicClient> = $state.raw(new Map());

  let neededDescriptors: SourceDescriptor[] = $derived([
    mainDescriptor,
    ...plannedLayers.filter((l) => l.descriptor != null).map((l) => l.descriptor!),
  ]);

  // Masks needed by the main data and the layers that are filtered.
  let neededMasks: Map<string, { table: string; key: string | null; filter: Selection }> = $derived.by(() => {
    let result = new Map<string, { table: string; key: string | null; filter: Selection }>();
    let add = (table: string, key: string | null, filter: Selection | null) => {
      if (filter != null) {
        result.set(maskKey(table, key, filter), { table, key, filter });
      }
    };
    add(table, identifier, filter);
    for (let layer of plannedLayers) {
      add(layer.columns.table, layer.columns.key, layer.filter);
    }
    return result;
  });

  /** Release the columns that no layer needs anymore (so switching columns doesn't keep every column in memory).
   *  Column sets are released once all needed sets are loaded, so a layer keeps its data until its new data
   *  arrives (e.g., when its category changes). */
  function releaseColumns(currentStore: TableStore) {
    let descriptors = untrack(() => neededDescriptors);
    let needed = new Set(descriptors.map(columnSetKey));
    for (let k of [...requestedColumnSets]) {
      if (!needed.has(k)) {
        requestedColumnSets.delete(k);
      }
    }
    let current = untrack(() => columnSets);
    if ([...needed].every((k) => k in current) && Object.keys(current).some((k) => !needed.has(k))) {
      columnSets = Object.fromEntries(Object.entries(current).filter(([k]) => needed.has(k)));
    }
    // The store keeps the columns of the needed column sets and of the labels layers.
    let labelsColumns = untrack(() => plannedLayers)
      .filter((l) => l.spec.type == "labels" && l.spec.labels == null && hasLabelColumns(l.columns))
      .map((l) => ({
        table: l.columns.table,
        key: l.columns.key,
        columns: [
          ...positionColumns(l.columns.x!, l.columns.y!),
          ...(l.columns.image != null && l.columns.importance != null
            ? [{ expr: fieldExpr(l.columns.importance), type: "float64" as const }]
            : []),
        ],
      }));
    currentStore.retain([
      ...descriptors.map((d) => ({ table: d.table, key: d.key, columns: descriptorColumns(d) })),
      ...labelsColumns,
    ]);
  }

  function destroyMaskClients() {
    for (let client of maskClients.values()) {
      client.destroy();
    }
    maskClients.clear();
    activeMaskClients = new Map();
  }

  // Load the needed column sets, and keep the mask clients in sync with the needed masks.
  $effect.pre(() => {
    let currentStore = store;
    let descriptors = neededDescriptors;
    let needed = neededMasks;
    untrack(() => {
      if (dataStore !== currentStore) {
        destroyMaskClients();
        dataStore = currentStore;
        requestedColumnSets.clear();
        columnSets = {};
        masks = {};
      }
      for (let d of descriptors) {
        let k = columnSetKey(d);
        if (requestedColumnSets.has(k)) {
          continue;
        }
        requestedColumnSets.add(k);
        currentStore
          .columns(d.table, d.key, descriptorColumns(d))
          .then(([cx, cy, cc]) => {
            if (dataStore === currentStore) {
              columnSets = {
                ...columnSets,
                [k]: {
                  x: cx as Float32Array<ArrayBuffer>,
                  y: cy as Float32Array<ArrayBuffer>,
                  category: (cc as Uint8Array<ArrayBuffer> | undefined) ?? null,
                },
              };
              releaseColumns(currentStore);
            }
          })
          .catch((error) => {
            requestedColumnSets.delete(k);
            logError({ table: d.table, x: d.x, y: d.y, category: d.category?.field ?? null }, error);
          });
      }
      releaseColumns(currentStore);
      for (let [k, client] of maskClients) {
        if (!needed.has(k)) {
          client.destroy();
          maskClients.delete(k);
          let { [k]: _, ...rest } = masks;
          masks = rest;
        }
      }
      for (let [k, m] of needed) {
        if (maskClients.has(k)) {
          continue;
        }
        let client = currentStore.filterMask(m.table, m.key, m.filter, {
          onMask: (mask) => {
            if (maskClients.get(k) !== client) {
              return;
            }
            masks = { ...masks, [k]: mask };
          },
          onError: (error) => logError({ table: m.table, filter: "mask" }, error),
        });
        maskClients.set(k, client);
      }
      if (
        activeMaskClients.size != maskClients.size ||
        [...maskClients].some(([k, client]) => activeMaskClients.get(k) !== client)
      ) {
        activeMaskClients = new Map(maskClients);
      }
    });
  });

  onDestroy(destroyMaskClients);

  // When the main data's filter changes, the tooltip and selection are cleared (they may be filtered out).
  let mainMask: FilterMask | null = $derived(
    filter != null ? (masks[maskKey(table, identifier, filter)] ?? null) : null,
  );
  $effect.pre(() => {
    mainMask;
    untrack(() => {
      updateTooltip(null);
      updateSelection(null);
    });
  });

  const maskedArrays = new MaskedArrays();
  const emptyFloat32Array = new Float32Array();

  /** The data of a source: its columns with its filter applied (empty until they're loaded). */
  function resolvedSource(key: string, d: SourceDescriptor): ResolvedSource {
    let columns = columnSets[columnSetKey(d)];
    let mask = d.filter != null ? masks[maskKey(d.table, d.key, d.filter)] : null;
    if (columns == null || mask === undefined) {
      return {
        key,
        x: emptyFloat32Array,
        y: emptyFloat32Array,
        category: null,
        categoryCount: 1,
        totalCount: null,
        maxDensity: null,
      };
    }
    // Stats of the unfiltered data (so the zoom-dependent opacity and point size don't change with the filter).
    let stats = dataStats(columns.x, columns.y, columns.category);
    return {
      key,
      x: maskedArrays.get(columns.x, mask),
      y: maskedArrays.get(columns.y, mask),
      category: columns.category != null ? maskedArrays.get(columns.category, mask) : null,
      categoryCount: stats.categoryCount,
      totalCount: stats.count,
      maxDensity: stats.maxDensity,
    };
  }

  let mainColumns = $derived(columnSets[columnSetKey(mainDescriptor)] ?? null);
  let mainSource = $derived(resolvedSource("main", mainDescriptor));
  let defaultViewportState: ViewportState | null = $derived(
    mainColumns != null ? dataStats(mainColumns.x, mainColumns.y, mainColumns.category).defaultViewportState : null,
  );

  // The source of the view's selection clauses (the tooltip, the selection, and the range selection). Mosaic calls
  // `reset` on clause sources when a selection is reset (e.g., filters are cleared).
  const interactionSource = { reset: () => reset() };

  // The clients that show the view's own table, and the interaction source. These are excluded from this view's own
  // selections, so that, e.g., brushing in the view doesn't filter the view's own layers on the main table.
  // (Layers on other tables are filtered by their own selections, which may be derived from this view's
  // selections, e.g., through a cross-table relation: those layers then follow this view's brush.)
  // The set is only replaced when its members change, since updating a clause re-runs other clients' queries.
  let previousSelfClients = new Set<any>();
  let selfClients: Set<any> = $derived.by(() => {
    let result = new Set<any>([interactionSource]);
    for (let [k, client] of activeMaskClients) {
      if (neededMasks.get(k)?.table == table) {
        result.add(client);
      }
    }
    if (result.size == previousSelfClients.size && [...result].every((c) => previousSelfClients.has(c))) {
      return previousSelfClients;
    }
    previousSelfClients = result;
    return result;
  });

  let resolvedLayers: ResolvedLayer[] = $derived.by(() => {
    let result: ResolvedLayer[] = [];
    for (let layer of plannedLayers) {
      let spec = layer.spec;
      switch (spec.type) {
        case "points":
        case "density": {
          let base = {
            key: layer.key,
            id: layer.id,
            source: layer.source == "main" ? mainSource : resolvedSource(layer.source!, layer.descriptor!),
            useCategory: layer.useCategory,
          };
          let colors: string[] | null;
          if (layer.useCategory) {
            // The layer's scale colors, or the component's colors for the component's category. Layers with other
            // categories default to colors for their own number of categories (null: see the renderer).
            let scale = spec.colorScale;
            colors =
              scale !== undefined
                ? colorsOf(scale, null)
                : isMainCategory(layer.columns)
                  ? colorsOf(colorScale, categoryColors)
                  : null;
          } else {
            let fixed = spec.color ?? color ?? categoryColors?.[0];
            colors = [fixed ?? defaultCategoryColors(1)[0]];
          }
          result.push(
            spec.type == "points"
              ? resolvePointsLayer(spec, base, colors, config)
              : resolveDensityLayer(spec, base, colors, config),
          );
          break;
        }
        case "labels": {
          result.push({
            type: "labels",
            key: layer.key,
            id: layer.id,
            opacity: spec.opacity ?? null,
            labels: spec.labels ?? null,
            auto: spec.labels == null ? autoLabels(layer, spec) : null,
          });
          break;
        }
      }
    }
    return result;
  });

  /** The automatic labels of a labels layer (null if there are no columns to generate them from). */
  function autoLabels(layer: PlannedLayer, spec: Extract<LayerSpec, { type: "labels" }>): AutoLabels | null {
    let c = layer.columns;
    if (!hasLabelColumns(c)) {
      return null;
    }
    let stopWords = spec.stopWords ?? config?.autoLabelStopWords ?? null;
    let densityThreshold = spec.densityThreshold ?? null;
    let x = c.x!;
    let y = c.y!;
    let query = {
      table: c.table,
      key: c.key,
      x,
      y,
      text: c.text,
      image: c.image,
      importance: c.importance,
      stopWords,
      densityThreshold,
      filter: selectionId(layer.filter),
    };
    let queryKey = JSON.stringify(query);
    // Labels that follow a filter wait for its mask, and are regenerated when it changes.
    let mask: FilterMask | null = layer.filter != null ? (masks[maskKey(c.table, c.key, layer.filter)] ?? null) : null;
    let ready = layer.filter == null || mask != null;
    let requestKey = JSON.stringify([queryKey, mask?.predicateString ?? null]);
    let currentStore = store;
    return {
      requestKey,
      queryKey,
      mergeKey: requestKey,
      // Only unfiltered labels are persisted (filtered labels change with every filter state).
      persistKey:
        layer.filter == null
          ? {
              data: { table: c.table, x, y },
              query: { text: c.text, image: c.image, importance: c.importance, identifier: c.key, stopWords },
            }
          : null,
      densityThreshold,
      ready,
      loadSnapshot: async () => {
        let [cx, cy] = await currentStore.columns(c.table, c.key, positionColumns(x, y));
        let xs = cx as Float32Array<ArrayBuffer>;
        let ys = cy as Float32Array<ArrayBuffer>;
        // The viewport fits the unfiltered data, so it doesn't change with the filter.
        let viewport = dataStats(xs, ys, null).defaultViewportState;
        return { x: maskedArrays.get(xs, mask), y: maskedArrays.get(ys, mask), viewport };
      },
      queryClusterLabels: (clusters, snapshot) =>
        c.image != null && c.importance != null
          ? queryClusterImageLabels(currentStore, c, mask, clusters)
          : queryClusterTextLabels(currentStore, c, mask, stopWords, clusters, snapshot),
    };
  }

  /** Bind a value of the view (the tooltip or the selection) to a Mosaic Selection, while running in an effect:
   *  the value follows the selection's value for this view, and the selection gets this view's clause
   *  (with the view's clients excluded) when the value changes. Returns the cleanup, which clears the clause. */
  function bindSelection<T>(
    captured: Selection,
    client: any,
    getValue: () => T | null,
    setValue: (value: T | null) => void,
    predicate: (value: T) => any,
  ): () => void {
    setValue((captured.valueFor(client) ?? null) as any);
    let listener = () => {
      setValue((captured.valueFor(client) ?? null) as any);
    };

    let clauseClients = untrack(() => selfClients);
    $effect(() => {
      let value = getValue();
      clauseClients = selfClients;
      captured.update({
        source: client,
        clients: clauseClients,
        predicate: value != null ? predicate(value) : null,
        value: value,
      });
    });

    captured.addEventListener("value", listener);
    return () => {
      captured.removeEventListener("value", listener);
      captured.update({
        source: client,
        clients: clauseClients,
        value: null,
        predicate: null,
      });
    };
  }

  // Tooltip
  $effect(() => {
    if (isSelection(tooltip)) {
      let client = interactionSource;
      return bindSelection(
        tooltip,
        client,
        () => effectiveTooltip,
        (v) => (effectiveTooltip = v),
        (value) => predicateForDataPoints({ x: stableX, y: stableY, category: mainCategoryIndex, identifier }, [value]),
      );
    } else if (tooltip == null || typeof tooltip == "object") {
      effectiveTooltip = tooltip;
    } else {
      if (effectiveTooltip?.identifier == tooltip) {
        return;
      }
      let obsolete = false;
      queryPoints([tooltip]).then((value) => {
        if (obsolete) {
          return;
        }
        if (value.length > 0) {
          effectiveTooltip = value[0];
        } else {
          effectiveTooltip = null;
        }
      });
      return () => {
        obsolete = true;
      };
    }
  });

  function updateTooltip(value: DataPoint | null) {
    if (deepEquals(tooltip, value)) {
      return;
    }
    effectiveTooltip = value;
    onTooltip?.(value);
  }

  // Selection
  $effect(() => {
    if (isSelection(selection)) {
      let client = interactionSource;
      return bindSelection(
        selection,
        client,
        () => effectiveSelection,
        (v) => (effectiveSelection = v),
        (value) => predicateForDataPoints({ x: stableX, y: stableY, category: mainCategoryIndex, identifier }, value),
      );
    } else if (selection == null) {
      effectiveSelection = null;
    } else if (selection.length == 0) {
      effectiveSelection = [];
    } else {
      if (selection.every((x) => typeof x == "object")) {
        effectiveSelection = selection;
      } else {
        let obsolete = false;
        queryPoints(selection).then((value) => {
          if (obsolete) {
            return;
          }
          effectiveSelection = value;
        });
        return () => {
          obsolete = true;
        };
      }
    }
  });

  function updateSelection(value: DataPoint[] | null) {
    if (deepEquals(selection, value)) {
      return;
    }
    effectiveSelection = value;
    onSelection?.(value);
  }

  // Range Selection
  $effect(() => {
    let client = interactionSource;
    let captured = rangeSelection;
    if (captured == null) {
      return;
    }

    let clauseClients = untrack(() => selfClients);
    $effect(() => {
      let value = effectiveRangeSelection;
      let source = { x: stableX, y: stableY };
      clauseClients = selfClients;
      let clause = {
        source: client,
        clients: clauseClients,
        predicate: value != null ? predicateForRangeSelection(source, value) : null,
        value: value,
      };
      captured.update(clause);
      captured.activate(clause);
    });

    return () => {
      captured.update({
        source: client,
        clients: clauseClients,
        value: null,
        predicate: null,
      });
    };
  });

  $effect(() => {
    if (
      !deepEquals(
        untrack(() => effectiveRangeSelection),
        rangeSelectionValue,
      )
    ) {
      effectiveRangeSelection = rangeSelectionValue;
    }
  });

  // Reset tooltip, selection, and range selection.
  function reset() {
    updateSelection(null);
    updateTooltip(null);
    onRangeSelection?.(null);
    effectiveRangeSelection = null;
  }

  // Point query
  let pointQuery = $derived(
    new DataPointQuery(coordinator, {
      table,
      x: stableX,
      y: stableY,
      category: mainCategoryIndex,
      text: stableText,
      identifier,
      additionalFields,
    }),
  );

  async function querySelection(px: number, py: number, unitDistance: number): Promise<DataPoint | null> {
    return await pointQuery.queryClosestPoint(filter?.predicate?.(interactionSource as any), px, py, unitDistance);
  }

  async function queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    return await pointQuery.queryPoints(identifiers);
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Cluster labels

  /** A key as a SQL literal (BIGINT keys beyond the safe integer range are cast from strings). */
  function keyLiteral(value: any): SQL.ExprNode {
    value = normalizeKey(value);
    if (typeof value == "bigint") {
      return SQL.cast(SQL.literal(String(value)), "HUGEINT");
    }
    return SQL.literal(value);
  }

  /** Whether any position in [start, end) is set in a mask (always true without a mask). */
  function maskHasAny(mask: FilterMask | null, start: number, end: number): boolean {
    return mask == null || mask.bits == null || bitsetAny(mask.bits, start, end);
  }

  /** Text labels: the text of the rows (matching the mask) is queried in chunks of keys, and positioned with the
   *  loaded x and y columns (so x and y aren't queried again). */
  async function queryClusterTextLabels(
    currentStore: TableStore,
    c: LayerColumns,
    mask: FilterMask | null,
    stopWords: string[] | null,
    clusters: Rectangle[][],
    snapshot: LabelSnapshot,
  ): Promise<(LabelContent | null)[]> {
    if (c.text == null) {
      return clusters.map(() => null);
    }
    let index: KeyIndex = await currentStore.keyIndex(c.table, c.key);
    let [cx, cy] = await currentStore.columns(c.table, c.key, positionColumns(c.x!, c.y!));
    let keyExpr = await currentStore.keyExpressionOf(c.table, c.key);
    let summarizer = await textSummarizerCreate({ regions: clusters, stopWords: stopWords });
    let lastAdd: Promise<unknown> | null = null;
    try {
      for (let chunk of index.chunks(50000)) {
        if (!maskHasAny(mask, chunk.start, chunk.end)) {
          continue;
        }
        let r: any = await coordinator.query(
          chunkQuery(
            c.table,
            keyExpr,
            { text: fieldExpr(c.text) },
            keyLiteral(chunk.low),
            keyLiteral(chunk.high),
            mask?.predicate,
          ),
        );
        let keys = r.getChild("k").toArray();
        let texts = r.getChild("text").toArray();
        let xs = new Float32Array(keys.length);
        let ys = new Float32Array(keys.length);
        for (let i = 0; i < keys.length; i++) {
          let p = index.positionOf(keys[i]);
          xs[i] = p >= 0 ? cx[p] : NaN;
          ys[i] = p >= 0 ? cy[p] : NaN;
        }
        if (lastAdd != null) {
          await lastAdd;
        }
        lastAdd = textSummarizerAdd(summarizer, { x: xs, y: ys, text: texts });
      }
      if (lastAdd != null) {
        await lastAdd;
      }
      let summarizeResult = await textSummarizerSummarize(summarizer);
      return summarizeResult.map((words) => {
        if (words.length == 0) {
          return null;
        } else if (words.length > 2) {
          return words.slice(0, 2).join("-") + "-\n" + words.slice(2).join("-");
        } else {
          return words.join("-");
        }
      });
    } finally {
      await textSummarizerDestroy(summarizer);
    }
  }

  /** Image labels: a representative row of each cluster by importance (from the loaded columns), then the images
   *  of those rows. */
  async function queryClusterImageLabels(
    currentStore: TableStore,
    c: LayerColumns,
    mask: FilterMask | null,
    clusters: Rectangle[][],
  ): Promise<(LabelContent | null)[]> {
    let { image, importance } = c;
    if (image == null || importance == null) {
      return clusters.map(() => null);
    }
    let index: KeyIndex = await currentStore.keyIndex(c.table, c.key);
    let [cx, cy, ci] = await currentStore.columns(c.table, c.key, [
      ...positionColumns(c.x!, c.y!),
      { expr: fieldExpr(importance), type: "float64" },
    ]);
    let keyExpr = await currentStore.keyExpressionOf(c.table, c.key);

    let output: ({ image: string; width: number; height: number; x: number; y: number } | null)[] = clusters.map(
      () => null,
    );
    let summarizer = new ImageSummarizer({ regions: clusters });
    let positions = maskPositions(mask);
    if (positions == null) {
      summarizer.add({ x: cx, y: cy, importance: ci, id: index.keys });
    } else {
      let pick = <T extends ArrayLike<any>>(values: T) => Array.from(positions!, (p) => values[p]);
      summarizer.add({
        x: pick(cx),
        y: pick(cy),
        importance: pick(ci),
        id: pick(index.keys),
      });
    }
    let winners = summarizer.summarize();
    let winnerKeys = Array.from(new Set(winners.filter((w) => w != null).map((w) => normalizeKey(w!.id))));
    let keyToImage = new Map<any, any>();
    if (winnerKeys.length > 0) {
      let r: any = await coordinator.query(
        SQL.Query.from(c.table)
          .select({ k: keyExpr, image: fieldExpr(image) })
          .where(SQL.isIn(keyExpr, winnerKeys.map(keyLiteral))),
      );
      for (let row of r) {
        keyToImage.set(normalizeKey(row.k), row.image);
      }
    }
    for (let regionId = 0; regionId < winners.length; regionId++) {
      let winner = winners[regionId];
      if (winner == null) continue;
      let bestImage = keyToImage.get(normalizeKey(winner.id));
      if (bestImage == null) continue;
      let dataUrl = imageToDataUrl(bestImage);
      if (dataUrl == null) continue;
      output[regionId] = { image: dataUrl, width: 0, height: 0, x: winner.x, y: winner.y };
    }
    await Promise.all(
      output.map(async (item) => {
        if (item == null) {
          return;
        }
        let { width, height } = await measureImageSize(item.image);
        // Fit to IMAGE_LABEL_SIZE while maintaining aspect ratio
        let scale = Math.min(IMAGE_LABEL_SIZE / width, IMAGE_LABEL_SIZE / height);
        item.width = width * scale;
        item.height = height * scale;
      }),
    );
    return output;
  }

  function measureImageSize(src: string): Promise<{ width: number; height: number }> {
    return new Promise((resolve) => {
      let img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve({ width: IMAGE_LABEL_SIZE, height: IMAGE_LABEL_SIZE });
      img.src = src;
    });
  }
</script>

<EmbeddingViewImpl
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  theme={theme}
  config={config}
  layers={resolvedLayers}
  pointCount={mainSource.x.length}
  categoryCount={mainSource.categoryCount}
  categoryColors={mainColors}
  defaultViewportState={defaultViewportState}
  querySelection={querySelection}
  customTooltip={customTooltip}
  customOverlay={customOverlay}
  tooltip={effectiveTooltip}
  onTooltip={updateTooltip}
  selection={effectiveSelection}
  onSelection={updateSelection}
  viewportState={viewportState}
  onViewportState={onViewportState}
  rangeSelection={effectiveRangeSelection}
  onRangeSelection={(v) => {
    effectiveRangeSelection = v;
    onRangeSelection?.(v);
  }}
  onLayerValues={onLayerValues}
  cache={cache}
/>
