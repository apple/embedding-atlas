<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script module lang="ts">
  import { fieldExpr, maxDensityModeCategories, type DataPoint, type ViewportState } from "@embedding-atlas/component";
  import { type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import Overlay from "./Overlay.svelte";
  import Tooltip from "./Tooltip.svelte";

  import { resolveSQLTemplate } from "../../utils/database.js";
  import { createCustomComponentClass } from "./custom_components.js";
  import type { EmbeddingField } from "./types.js";

  /** A field with its SQL template variables (e.g., `$table`) resolved, for the embedding view. */
  function resolveField<T extends EmbeddingField | null | undefined>(field: T, table: string): T {
    if (field == null || typeof field == "string") {
      return field;
    }
    return { sql: resolveSQLTemplate(field.sql, { table, filter: "(true)" }) } as T;
  }

  /** The SQL expression of a field (evaluated for the given table). */
  function fieldSQL(field: EmbeddingField, table: string): SQL.ExprNode {
    return fieldExpr(resolveField(field, table));
  }

  async function defaultViewportScale(
    coordinator: Coordinator,
    table: string,
    x: SQL.ExprNode,
    y: SQL.ExprNode,
  ): Promise<number> {
    let { stdX, stdY } = (
      await coordinator.query(
        SQL.Query.from(table).select({
          stdX: SQL.sql`STDDEV(${x})::FLOAT`,
          stdY: SQL.sql`STDDEV(${y})::FLOAT`,
        }),
      )
    ).get(0);
    let scale = 1.0 / (Math.max(stdX, stdY, 1e-3) * 3);
    return scale;
  }

  const CustomTooltip = createCustomComponentClass(Tooltip);
  const CustomOverlay = createCustomComponentClass(Overlay);
</script>

<script lang="ts">
  import { type EmbeddingViewMosaicProps } from "@embedding-atlas/component";
  import { EmbeddingViewMosaic } from "@embedding-atlas/component/svelte";
  import { deepEquals } from "@embedding-atlas/utils";
  import { untrack } from "svelte";
  import { cubicOut } from "svelte/easing";

  import PopupButton from "../../widgets/PopupButton.svelte";
  import SegmentedControl from "../../widgets/SegmentedControl.svelte";
  import Select from "../../widgets/Select.svelte";
  import ConfirmButton from "../builder/ConfirmButton.svelte";
  import SpecEditor from "../builder/SpecEditor.svelte";
  import LayerControls from "./LayerControls.svelte";
  import Legend from "./Legend.svelte";
  import SliderRow from "./SliderRow.svelte";

  import { IconSettings } from "../../assets/icons.js";
  import { latestAsync } from "../../utils/latest_async.js";
  import { FocusRequestTracker, resolveEffectiveOverlay } from "../../utils/overlay.js";
  import { isolatedWritable } from "../../utils/store.js";
  import type { ChartViewProps, Overlay as OverlayData, RowID } from "../chart.js";
  import { resolveChartTheme } from "../common/theme.js";
  import { computeLegendData, resolveLegend, type EmbeddingLegend, type LegendData } from "./color_scale.js";
  import {
    defaultLabelsOptions,
    defaultLayers,
    fieldKey,
    resolveLayerData,
    type LayerData,
    type LayerSpec,
    type LayerValues,
  } from "./layer_data.js";
  import { layerPresets } from "./layer_presets.js";
  import type { EmbeddingColorScale, EmbeddingSpec, EmbeddingState } from "./types.js";
  import { interpolateViewport } from "./viewport_animation.js";

  // Shared, so the labels stay identical (and aren't laid out again) when the layers are recomputed.
  const noLabels: never[] = [];

  const maxCategories = Math.min(20, maxDensityModeCategories());
  /** The value of field selects for a SQL expression field. */
  const customField = "$sql";
  const defaultMinimumDensity = 1 / 16;
  const defaultDownsampleMaxPoints = 4000000;
  const minDownsampleMaxPoints = 50000;

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onStateChange,
    onSpecChange,
  }: ChartViewProps<EmbeddingSpec, EmbeddingState> = $props();

  // svelte-ignore state_referenced_locally
  let { colorScheme, columnStyles, searcher, theme: themeConfig } = context;

  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  let dataTable = $derived(spec.data.table ?? context.table);
  let rowId = $derived(spec.data.id ?? context.id);
  let filter = $derived(context.filterFor(spec.data.table));
  let isDefaultTable = $derived(dataTable === context.table);

  // svelte-ignore state_referenced_locally
  let highlightStore = isolatedWritable(context.highlight);

  let categoryField = $derived(spec.data.category);

  let totalPointCount: number | null = $state.raw(null);

  // Query total point count for render limit slider
  $effect.pre(() => {
    context.coordinator
      .query(SQL.Query.from(dataTable).select({ count: SQL.sql`COUNT(*)::INT` }))
      .then((result: any) => {
        totalPointCount = result.get(0).count;
      });
  });

  let tooltip = $state.raw<DataPoint | null>(null);
  let selection = $state.raw<DataPoint[] | null>(null);
  let overlayProps = $state.raw<{ nodes?: DataPoint[]; edges?: { start: DataPoint; end: DataPoint }[] } | null>(null);

  type ComponentLayer = NonNullable<EmbeddingViewMosaicProps["layers"]>[number];

  /** A category field with its table and color scale (the legends of the main data and of layers). */
  interface LegendSpec {
    table: string;
    field: EmbeddingField;
    colorScale: EmbeddingColorScale | null;
  }

  function legendKey(l: LegendSpec): string {
    return JSON.stringify([l.table, fieldKey(l.field), l.colorScale ?? null]);
  }

  /** The legend spec without colors: what the categories (and counts) depend on. */
  function legendDataSpec(l: LegendSpec): LegendSpec {
    let { type, domain, bin } = l.colorScale ?? {};
    let colorScale = type != null || domain != null || bin != null ? { type, domain, bin } : null;
    return { table: l.table, field: l.field, colorScale };
  }

  /** The key of a legend's categories (changing the colors doesn't change them). */
  function legendDataKey(l: LegendSpec): string {
    return legendKey(legendDataSpec(l));
  }

  // The categories (and counts) of the legends, by legend data key. Colors are added with the theme (`legendOf`).
  let legendData: Record<string, LegendData | null> = $state.raw({});

  // Resolved legends, memoized by key (for the same data and theme), so the view gets the same scales.
  let resolvedLegends = new Map<string, { data: LegendData; theme: typeof theme; legend: EmbeddingLegend }>();

  /** The legend of a category field (null until its categories are loaded). */
  function legendOf(l: LegendSpec): EmbeddingLegend | null {
    let key = legendKey(l);
    let data = legendData[legendDataKey(l)];
    if (data == null) {
      return null;
    }
    let cached = resolvedLegends.get(key);
    if (cached != null && cached.data === data && cached.theme === theme) {
      return cached.legend;
    }
    let legend = resolveLegend(data, fieldSQL(l.field, l.table), l.colorScale, theme);
    resolvedLegends.set(key, { data, theme, legend });
    return legend;
  }

  /** The legend spec of a layer's category (null if the layer has none). Layers with the same table, category, and
   *  color scale (e.g., layers inheriting `data.category`) have the same legend. */
  function layerLegendSpec(d: LayerData): LegendSpec | null {
    return d.category != null ? { table: d.table, field: d.category, colorScale: d.colorScale } : null;
  }

  /** The legend of a layer's category (null if it has none, or until its categories are loaded). */
  function layerLegend(d: LayerData): EmbeddingLegend | null {
    let l = layerLegendSpec(d);
    return l != null ? legendOf(l) : null;
  }

  function resolveLayer(layer: LayerSpec, d: LayerData): ComponentLayer {
    let common = {
      id: layer.id,
      visible: layer.visible,
      // "$filter" refers to the filter of the layer's table.
      filter: d.filtered ? context.filterFor(d.table) : null,
    };
    let data = {
      table: d.table,
      // The table's row id column: the component loads and filters the table's data by it.
      identifier: d.id ?? (d.isDataTable ? rowId : (context.tables[d.table]?.id ?? null)),
      ...(d.x != null ? { x: resolveField(d.x, d.table) } : {}),
      ...(d.y != null ? { y: resolveField(d.y, d.table) } : {}),
    };
    switch (layer.type) {
      case "points":
      case "density": {
        let legend = layerLegend(d);
        // Until the legend is loaded, the layer is drawn with the default color.
        let colorProps = {
          data: { ...data, category: legend != null ? resolveField(d.category, d.table) : null },
          colorScale: legend?.scale ?? null,
          color: d.color ?? theme.embeddingColor,
        };
        if (layer.type == "points") {
          return { type: "points", ...common, ...colorProps, opacity: layer.opacity, pointSize: layer.pointSize };
        } else {
          return {
            type: "density",
            ...common,
            ...colorProps,
            opacity: layer.opacity,
            fill: layer.fill,
            contours: layer.contours,
          };
        }
      }
      case "labels": {
        return {
          type: "labels",
          ...common,
          opacity: layer.opacity,
          // "$labels" refers to the labels provided to the viewer (e.g., with the CLI's --labels option).
          labels: layer.labels == "$labels" ? (context.embeddingViewLabels ?? noLabels) : (layer.labels ?? null),
          data: {
            ...data,
            text: resolveField(d.text, d.table),
            image: resolveField(d.image, d.table),
            importance: resolveField(d.importance, d.table),
          },
        };
      }
    }
  }

  // Incremented to reset the preset select after a preset is applied.
  let presetSelectKey = $state(0);

  /** The viewer's options for the labels of views without `layers` (e.g., the CLI's --labels option). */
  let labelsOptions = $derived(defaultLabelsOptions(context, isDefaultTable));

  let presets = $derived(layerPresets(labelsOptions));

  function applyLayerPreset(index: number) {
    let preset = presets[index];
    if (preset == null) {
      return;
    }
    onSpecChange((draft) => {
      draft.layers = structuredClone(preset.layers);
      // The layers replace `mode` and `pointSize`. (`minimumDensity` applies to all layers, so it's kept.)
      delete draft.mode;
      delete draft.pointSize;
    });
  }

  // The Options popup can edit the spec with the UI controls or as code.
  let optionsVisible = $state(false);
  let optionsMode: "ui" | "code" = $state("ui");
  // The spec loaded into the code editor, and the edited spec (undefined if it doesn't parse).
  let specEditorInitial: any = $state.raw(undefined);
  let editingSpec: any | undefined = $state.raw(undefined);
  // Incremented to reload the code editor with `specEditorInitial`.
  let specEditorKey = $state(0);
  // Whether the code editor has edits that are not confirmed yet.
  let specEditorDirty = false;

  function reloadSpecEditor() {
    let current = $state.snapshot(spec);
    if (specEditorInitial !== undefined && !specEditorDirty && deepEquals(current, specEditorInitial)) {
      return;
    }
    specEditorInitial = current;
    editingSpec = current;
    specEditorDirty = false;
    specEditorKey += 1;
  }

  // Reload the code editor when the popup opens or the Spec tab is selected (dropping unconfirmed edits),
  // and when the spec changes elsewhere while there are no unconfirmed edits.
  $effect(() => {
    if (optionsVisible && optionsMode == "code") {
      untrack(() => {
        specEditorDirty = false;
        reloadSpecEditor();
      });
    }
  });
  $effect(() => {
    $state.snapshot(spec);
    untrack(() => {
      if (optionsVisible && optionsMode == "code" && !specEditorDirty) {
        reloadSpecEditor();
      }
    });
  });

  /** Effective values of the layers (including auto values), matched to layers by id (or index).
   *  Only shown in the Options popup: they change on every zoom frame (auto opacity and point size), so the state
   *  is only updated while the popup is open (the popup's content stays mounted while it's closed). */
  let layerValues: LayerValues[] = $state.raw([]);
  let latestLayerValues: LayerValues[] = [];

  function onLayerValues(values: LayerValues[]) {
    latestLayerValues = values;
    if (optionsVisible) {
      layerValues = values;
    }
  }

  $effect.pre(() => {
    if (optionsVisible) {
      layerValues = latestLayerValues;
    }
  });

  let showMaxPoints = $derived(totalPointCount != null && totalPointCount > minDownsampleMaxPoints);

  // Specs without `layers` (default charts, the builder, and older specs) get the default layers, so they follow
  // `mode`, `pointSize`, and the viewer's labels options (provided, automatic, or disabled labels).
  let layers: LayerSpec[] = $derived(spec.layers ?? defaultLayers(spec, labelsOptions));

  let hasDensityLayer = $derived(layers.some((l) => l.type == "density" && l.visible !== false));

  // Each layer with its table, columns, and filter (resolved once, and shared below).
  let resolvedLayers: { layer: LayerSpec; data: LayerData }[] = $derived(
    layers.map((layer) => ({ layer, data: resolveLayerData(spec, layer, dataTable) })),
  );

  let componentLayers: ComponentLayer[] = $derived(resolvedLayers.map(({ layer, data }) => resolveLayer(layer, data)));

  // The legends to load: the legends of the layers. Keyed by a string, so editing other layer options (e.g.,
  // dragging a slider) doesn't run this again.
  let neededLegendsKey = $derived.by(() => {
    let specs = new Map<string, LegendSpec>();
    for (let l of resolvedLayers.map(({ data }) => layerLegendSpec(data))) {
      if (l != null) {
        specs.set(legendDataKey(l), legendDataSpec(l));
      }
    }
    return JSON.stringify([...specs.values()]);
  });

  $effect.pre(() => {
    let specs: LegendSpec[] = JSON.parse(neededLegendsKey);
    let coordinator = context.coordinator;
    // The legends of density layers (to switch to points if they have too many categories).
    let densityKeys = untrack(
      () =>
        new Set(
          resolvedLayers
            .filter(({ layer }) => layer.type == "density")
            .map(({ data }) => layerLegendSpec(data))
            .filter((l) => l != null)
            .map((l) => legendDataKey(l)),
        ),
    );
    for (let l of specs) {
      let key = legendDataKey(l);
      context.cache
        .value(`embedding/legend/${key}`, () =>
          computeLegendData(coordinator, l.table, fieldSQL(l.field, l.table), l.colorScale),
        )
        .then((v) => {
          if (legendData[key] !== v) {
            legendData = { ...legendData, [key]: v };
          }
          // Density mode can't show many categories: switch to points (for views with the default layers).
          let k = v?.labels.length ?? 0;
          let count = v != null ? k + (v.counts[k] > 0 ? 1 : 0) + (v.counts[k + 1] > 0 ? 1 : 0) : 0;
          if (densityKeys.has(key) && count > maxCategories && spec.layers == null && spec.mode == "density") {
            onSpecChange((draft) => {
              draft.mode = "points";
            });
          }
        })
        .catch((error) => {
          console.error("Embedding view: failed to compute the legend of", l, error);
        });
    }
  });

  // The legends to display: the legends of the visible layers, in layer order (layers can share a legend).
  let legends = $derived.by(() => {
    let result: { key: string; table: string; legend: EmbeddingLegend }[] = [];
    for (let { layer, data } of resolvedLayers) {
      if (layer.visible === false) {
        continue;
      }
      let l = layerLegendSpec(data);
      if (l == null) {
        continue;
      }
      let key = legendKey(l);
      let legend = legendOf(l);
      if (legend != null && !result.some((x) => x.key == key)) {
        result.push({ key, table: l.table, legend });
      }
    }
    return result;
  });

  // The category the component reports for points in tooltips and selections: the category of the first visible
  // points or density layer on the view's table.
  let pointCategory = $derived.by(() => {
    for (let { layer, data } of resolvedLayers) {
      if (layer.visible === false || layer.type == "labels" || !data.isDataTable || data.category == null) {
        continue;
      }
      let legend = layerLegend(data);
      return legend != null ? { field: resolveField(data.category, dataTable), scale: legend.scale } : null;
    }
    return null;
  });

  $effect.pre(() => {
    let isOnMount = true;
    let previousValue: RowID[] | null = null;
    return highlightStore.subscribe((v) => {
      if (!isDefaultTable) {
        return;
      }

      selection = v;

      // Don't animate immediately on mount.
      if (isOnMount) {
        isOnMount = false;
        previousValue = v;
        return;
      }
      // Animate when a single new point is added.
      let newIDs = v ?? [];
      let oldIDs = previousValue ?? [];
      let enteringIDs = newIDs.filter((x) => oldIDs.indexOf(x) < 0);
      if (enteringIDs.length == 1) {
        animateToPoint(enteringIDs[0]);
      }
      if (tooltip != null && newIDs.indexOf(tooltip) < 0) {
        tooltip = null;
      }
      previousValue = v;
    });
  });

  // The overlay to render is resolved from two sources: the internal overlay
  // (context.overlay, e.g. produced by search) and the external overlay
  // (context.externalOverlay, from the viewer's `overlay` prop), which takes
  // precedence whenever it is set. See resolveEffectiveOverlay for the rules.
  let internalOverlay: OverlayData | null = null;
  let externalOverlay: OverlayData | null | undefined = undefined;

  // Resolve an overlay (row IDs) to renderable points (coordinates).
  // Wrapped with latestAsync so that when overlays change in quick succession,
  // a slower coordinate query for an older overlay cannot overwrite the newer one.
  const applyOverlay = latestAsync(
    async (overlay: OverlayData | null) => {
      if (overlay == null) {
        return null;
      }
      // Collect all ids
      let ids: RowID[] = [
        ...(overlay.nodes ?? []), // all nodes
        ...(overlay.edges?.flatMap((e) => [e.start, e.end]) ?? []), // all points in edges
      ];
      if (ids.length == 0) {
        return { nodes: [], edges: [] };
      }
      // Query for coordinates from ids
      let queryResult = Array.from(
        await context.coordinator.query(
          SQL.Query.from(dataTable)
            .select({
              id: SQL.column(rowId),
              x: fieldSQL(spec.data.x, dataTable),
              y: fieldSQL(spec.data.y, dataTable),
            })
            .where(
              SQL.isIn(
                SQL.column(rowId),
                ids.map((x) => SQL.literal(x)),
              ),
            ),
        ),
      );
      // Arrow returns BIGINT IDs as bigint even when the caller supplied a number.
      const idKey = (id: RowID) => (Number.isSafeInteger(id) ? BigInt(id) : id);
      let mapper = new Map(queryResult.map((p) => [idKey(p.id), { identifier: p.id, x: p.x, y: p.y }]));
      return {
        nodes: overlay.nodes?.map((n) => mapper.get(idKey(n))).filter((x) => x != undefined),
        edges: overlay.edges
          ?.map((n) => {
            let start = mapper.get(idKey(n.start));
            let end = mapper.get(idKey(n.end));
            if (start == undefined || end == undefined) {
              return undefined;
            }
            return { start, end };
          })
          .filter((x) => x != undefined),
      };
    },
    (props) => {
      overlayProps = props;
    },
  );

  $effect.pre(() => {
    let update = () => applyOverlay(resolveEffectiveOverlay(externalOverlay, internalOverlay));
    let unsubscribeInternal = context.overlay.subscribe((v) => {
      internalOverlay = v;
      update();
    });
    let unsubscribeExternal = context.externalOverlay.subscribe((v) => {
      externalOverlay = v;
      update();
    });
    return () => {
      unsubscribeInternal();
      unsubscribeExternal();
    };
  });

  // Each embedding view of the default table handles the shared request independently.
  const focusTracker = new FocusRequestTracker();
  $effect.pre(() => {
    if (!isDefaultTable) {
      return;
    }
    let active = true;
    let initial = true;
    let unsubscribe = context.focusRequest.subscribe((request) => {
      // The store immediately emits its current value. Only subsequent requests move this view.
      if (initial) {
        initial = false;
        focusTracker.markHandled(request);
        return;
      }
      let seq = focusTracker.accept(request);
      if (seq == null || request == null) {
        return;
      }
      void (async () => {
        let scale = await pointViewportScale();
        let location = await queryPointLocation(request.id);
        if (location == null || !active || !focusTracker.isLatest(seq)) {
          return;
        }
        startViewportAnimation({ x: location.x, y: location.y, scale: scale });
        tooltip = request.id;
      })();
    });
    return () => {
      active = false;
      unsubscribe();
    };
  });

  async function pointViewportScale(): Promise<number> {
    let x = fieldSQL(spec.data.x, dataTable);
    let y = fieldSQL(spec.data.y, dataTable);
    let defaultScale = await context.cache.value(
      `embedding/default-viewport-scale/${JSON.stringify([dataTable, fieldKey(spec.data.x), fieldKey(spec.data.y)])}`,
      () => defaultViewportScale(context.coordinator, dataTable, x, y),
    );
    return defaultScale * 2;
  }

  // Query the x, y location of a point. Returns null if the ID does not exist.
  async function queryPointLocation(identifier: RowID): Promise<{ x: number; y: number } | null> {
    let x = fieldSQL(spec.data.x, dataTable);
    let y = fieldSQL(spec.data.y, dataTable);
    let result = await context.coordinator.query(
      SQL.Query.from(dataTable)
        .select({ x, y })
        .where(SQL.eq(SQL.column(rowId), SQL.literal(identifier))),
    );
    return (result.get(0) as { x: number; y: number } | null) ?? null;
  }

  async function animateToPoint(identifier: RowID): Promise<void> {
    let scale = await pointViewportScale();
    let location = await queryPointLocation(identifier);
    if (location == null) {
      return;
    }
    // Start animation and show tooltip.
    startViewportAnimation({ x: location.x, y: location.y, scale: scale });
    tooltip = identifier;
  }

  let currentViewportAnimation: number | null;
  let animatingViewport = $state.raw<ViewportState | undefined>(undefined);
  function startViewportAnimation(newState: ViewportState) {
    tooltip = null;
    let start = animatingViewport ?? chartState.viewport;
    if (start == null) {
      onStateChange((draft) => {
        draft.viewport = newState;
      });
      return;
    }
    animatingViewport = start;
    let duration = 800;
    let t0 = new Date().getTime();
    let callback = () => {
      let t = (new Date().getTime() - t0) / duration;
      if (t > 1) {
        t = 1;
      }
      animatingViewport = interpolateViewport(start, newState, cubicOut(t));
      if (t < 1) {
        currentViewportAnimation = requestAnimationFrame(callback);
      } else {
        onStateChange((draft) => {
          draft.viewport = animatingViewport;
        });
      }
    };
    if (currentViewportAnimation) {
      cancelAnimationFrame(currentViewportAnimation);
    }
    currentViewportAnimation = requestAnimationFrame(callback);
  }

  async function nearestNeighbors(id: any): Promise<{ id: any; distance: number }[]> {
    if (spec.data.neighbors == undefined) {
      return [];
    }
    let q = SQL.Query.from(dataTable)
      .select({ knn: fieldSQL(spec.data.neighbors, dataTable) })
      .where(SQL.eq(SQL.column(rowId), SQL.literal(id)));
    let result = await context.coordinator.query(q);
    let items: any[] = Array.from(result);
    if (items.length != 1) {
      return [];
    }
    let { distances, ids } = items[0].knn;
    let r = Array.from(ids)
      .map((nid, i) => {
        return { id: nid, distance: distances[i] };
      })
      .filter((x) => x.id != id);
    return r;
  }
</script>

<div class="relative bg-white dark:bg-black">
  <EmbeddingViewMosaic
    width={width}
    height={height}
    coordinator={context.coordinator}
    table={dataTable}
    filter={filter}
    rangeSelection={filter}
    identifier={rowId}
    x={resolveField(spec.data.x, dataTable)}
    y={resolveField(spec.data.y, dataTable)}
    text={resolveField(spec.data.text, dataTable)}
    image={resolveField(spec.data.image, dataTable)}
    importance={resolveField(spec.data.importance, dataTable)}
    category={pointCategory?.field ?? null}
    colorScale={pointCategory?.scale ?? null}
    color={spec.color ?? theme.embeddingColor}
    theme={{ brandingLink: null }}
    config={{
      colorScheme: $colorScheme,
      ...context.embeddingViewConfig,
      ...(spec.minimumDensity != null ? { minimumDensity: spec.minimumDensity } : {}),
      ...(spec.pointSize != null ? { pointSize: spec.pointSize } : {}),
      downsampleMaxPoints: spec.downsampleMaxPoints ?? defaultDownsampleMaxPoints,
    }}
    layers={componentLayers}
    cache={context.persistentCache}
    additionalFields={isDefaultTable
      ? Object.fromEntries((context.tables[context.table]?.columns ?? []).map((c) => [c.name, c.name]))
      : {}}
    customTooltip={isDefaultTable
      ? {
          class: CustomTooltip,
          props: {
            context: context,
            darkMode: $colorScheme,
            columnStyles: $columnStyles,
            onNearestNeighborSearch: spec.data.neighbors
              ? async (id: any) => {
                  let neighbors = await nearestNeighbors(id);
                  let nids = neighbors.map((x) => x.id);
                  searcher.search(
                    {
                      label: "Neighbors of #" + id,
                      items: neighbors,
                      overlay: {
                        nodes: [id, ...nids],
                        edges: nids.map((ni) => ({ start: id, end: ni })),
                      },
                    },
                    "raw",
                  );
                }
              : undefined,
          },
        }
      : undefined}
    customOverlay={{
      class: CustomOverlay,
      props: { ...(overlayProps ?? { nodes: [], edges: [] }) },
    }}
    viewportState={animatingViewport ?? chartState.viewport}
    onLayerValues={onLayerValues}
    onViewportState={(v) =>
      onStateChange((draft) => {
        draft.viewport = v;
      })}
    rangeSelectionValue={chartState.brush}
    onRangeSelection={(v) =>
      onStateChange((draft) => {
        if (v) {
          draft.brush = v;
        } else {
          delete draft.brush;
        }
      })}
    tooltip={tooltip}
    onTooltip={(v) => {
      tooltip = v;
    }}
    selection={selection}
    onSelection={(points) => {
      selection = points;
      if (isDefaultTable) {
        highlightStore.set(points?.map((p) => p.identifier) ?? null);
      }
    }}
  />
  <div class="absolute top-0 left-0 right-0 flex flex-wrap justify-between items-start pointer-events-none">
    <!-- The legends, flowing to the right. -->
    <div class="flex-1 flex flex-wrap justify-end items-start gap-2 p-2 order-3">
      {#each legends as item (item.key)}
        <div class="flex-none p-2 rounded-md bg-slate-100/75 dark:bg-slate-800/75 backdrop-blur-sm pointer-events-auto">
          <Legend
            context={context}
            filter={context.filterFor(item.table)}
            spec={{ items: item.legend.items }}
            state={chartState.legends?.[item.key] ?? {}}
            mode="view"
            onSpecChange={() => {}}
            onStateChange={(update) => {
              onStateChange((draft) => {
                draft.legends ??= {};
                if (typeof update == "function") {
                  update((draft.legends[item.key] ??= {}));
                } else if (update != null) {
                  draft.legends[item.key] = update;
                } else {
                  delete draft.legends[item.key];
                }
              });
            }}
          />
        </div>
      {/each}
    </div>
    <div
      class="flex-none p-2 rounded-ee-md bg-white/75 dark:bg-black/75 backdrop-blur-sm flex items-center gap-2 pointer-events-auto order-1"
    >
      <Select
        class="max-w-64"
        label="Color"
        value={typeof categoryField == "string" ? categoryField : categoryField != null ? customField : undefined}
        onChange={(v) => {
          if (v == customField) {
            return;
          }
          onSpecChange((draft) => {
            draft.data.category = v;
            // The color scale is for the previous field.
            delete draft.colorScale;
          });
        }}
        options={[
          { value: undefined, label: "--" },
          ...(categoryField != null && typeof categoryField != "string"
            ? [{ value: customField, label: "Custom (SQL)" }]
            : []),
          ...(context.tables[context.table]?.columns ?? [])
            .filter((c) => c.jsType == "string" || c.jsType == "number" || c.jsType == "Date")
            .map((c) => ({ value: c.name, label: `${c.name} (${c.type})` })),
        ]}
      />
      <PopupButton icon={IconSettings} title="Options" bind:visible={optionsVisible}>
        <div class="flex flex-col gap-2 w-[448px]">
          <div class="flex items-center justify-between">
            <SegmentedControl
              value={optionsMode}
              onChange={(v) => {
                optionsMode = v as "ui" | "code";
              }}
              options={[
                { value: "ui", label: "Editor" },
                { value: "code", label: "Spec" },
              ]}
            />
            {#key presetSelectKey}
              <Select
                value={undefined}
                placeholder="Load Preset…"
                title="Replace the layers with a preset"
                onChange={(v) => {
                  applyLayerPreset(v);
                  // Reset the select so it shows the placeholder again.
                  presetSelectKey += 1;
                }}
                options={presets.map((preset, index) => ({ value: index, label: preset.label }))}
              />
            {/key}
          </div>
          {#if optionsMode == "code"}
            {#key specEditorKey}
              <SpecEditor
                class="w-full h-96"
                initialValue={specEditorInitial}
                colorScheme={$colorScheme}
                onChange={(s) => {
                  editingSpec = s;
                  specEditorDirty = true;
                }}
              />
            {/key}
            <div class="flex items-center gap-2">
              <ConfirmButton
                label="Confirm"
                disabled={editingSpec == undefined || (editingSpec as any).type != "embedding"}
                onClick={() => {
                  if (editingSpec != undefined && (editingSpec as any).type == "embedding") {
                    // The editor is reloaded with the new spec by the effect above.
                    specEditorDirty = false;
                    onSpecChange(editingSpec as EmbeddingSpec);
                  }
                }}
              />
              {#if editingSpec == undefined}
                <span class="text-sm text-red-500 select-none">Invalid spec</span>
              {:else if (editingSpec as any).type != "embedding"}
                <span class="text-sm text-red-500 select-none">The type must be "embedding"</span>
              {/if}
            </div>
          {:else}
            <LayerControls
              context={context}
              spec={spec}
              dataTable={dataTable}
              resolvedLayers={resolvedLayers}
              labelsOptions={labelsOptions}
              layerValues={layerValues}
              defaultColor={theme.embeddingColor}
              layerLegend={layerLegend}
              onSpecChange={onSpecChange}
            />
            {#if hasDensityLayer || showMaxPoints}
              <div
                class="bg-white dark:bg-slate-900 p-2 rounded-md border border-slate-200 dark:border-slate-700 grid grid-cols-2 gap-x-4 gap-y-2"
              >
                {#if hasDensityLayer}
                  <div class="flex flex-col gap-1">
                    <SliderRow
                      label="Crossfade"
                      value={Math.log((spec.minimumDensity ?? defaultMinimumDensity) / defaultMinimumDensity)}
                      min={-4}
                      max={4}
                      step={0.05}
                      toggle={{
                        label: "Default",
                        active: spec.minimumDensity == null,
                        title: "Use the default threshold",
                        onClick: () =>
                          onSpecChange((draft) => {
                            delete draft.minimumDensity;
                          }),
                      }}
                      onChange={(v) =>
                        onSpecChange((draft) => {
                          draft.minimumDensity = defaultMinimumDensity * Math.exp(v);
                        })}
                    />
                    <div class="text-xs text-slate-500 dark:text-slate-400 select-none">
                      Where density fades out and points fade in as you zoom (for layers with auto opacity). Move right
                      to switch to points sooner.
                    </div>
                  </div>
                {/if}
                {#if showMaxPoints && totalPointCount != null}
                  {@const effectiveLimit =
                    spec.downsampleMaxPoints ?? Math.min(defaultDownsampleMaxPoints, totalPointCount)}
                  {@const isMaxed = effectiveLimit >= totalPointCount}
                  {@const formatCount = (n: number) =>
                    n >= 1000000 ? (n / 1000000).toFixed(1) + "M" : (n / 1000).toFixed(0) + "K"}
                  <div class="flex flex-col gap-1">
                    <SliderRow
                      label="Max Points"
                      value={effectiveLimit}
                      min={minDownsampleMaxPoints}
                      max={totalPointCount}
                      step={Math.max(10000, Math.floor(totalPointCount / 100 / 10000) * 10000)}
                      toggle={{
                        label: "All",
                        active: isMaxed,
                        title: "Render all points",
                        onClick: () =>
                          onSpecChange((draft) => {
                            draft.downsampleMaxPoints = totalPointCount;
                          }),
                      }}
                      onChange={(v) =>
                        onSpecChange((draft) => {
                          draft.downsampleMaxPoints = v;
                        })}
                    />
                    <div class="text-xs text-slate-500 dark:text-slate-400 select-none">
                      <span class="font-medium text-slate-700 dark:text-slate-300">
                        {isMaxed
                          ? `All ${formatCount(totalPointCount)} points`
                          : `${formatCount(effectiveLimit)} of ${formatCount(totalPointCount)} points`}
                      </span>
                      — when more points are in view, a subset is drawn (fewer in dense areas) to keep rendering fast.
                    </div>
                  </div>
                {/if}
              </div>
            {/if}
          {/if}
        </div>
      </PopupButton>
    </div>
  </div>
</div>
