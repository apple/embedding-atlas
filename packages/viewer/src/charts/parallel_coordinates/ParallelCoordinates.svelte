<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import {
    ParallelCoordinatesView,
    type ParallelCoordinatesViewProps,
    type ParallelCoordinatesViewThemeConfig,
  } from "@embedding-atlas/component";
  import { makeClient, type MosaicClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { onDestroy, untrack } from "svelte";

  import Container from "../common/Container.svelte";
  import ColorLegend from "../spec/ColorLegend.svelte";

  import { IconClose } from "../../assets/icons.js";

  import { predicateToString, resolveSQLTemplate } from "../../utils/database.js";
  import type { ChartViewProps } from "../chart.js";
  import { computeFieldStats } from "../common/aggregate.js";
  import { resolveChartTheme } from "../common/theme.js";
  import type { Scale, SQLField, SQLTable } from "../spec/spec.js";
  import { resolveAxis, type ResolvedAxis } from "./axes.js";
  import { buildColorScale, buildPalette, resolveColorData, type ResolvedColorData } from "./color.js";
  import { applyCategoryJitter, hashIndex } from "./jitter.js";
  import type { ParallelCoordinatesSpec, ParallelCoordinatesState } from "./types.js";

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onStateChange,
  }: ChartViewProps<ParallelCoordinatesSpec, ParallelCoordinatesState> = $props();

  // svelte-ignore state_referenced_locally
  let { colorScheme, theme: themeConfig } = context;

  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  let pcTheme = $derived<ParallelCoordinatesViewThemeConfig>({
    fontFamily: theme.labelFontFamily,
    axisLineColor: theme.ruleColor,
    tickColor: theme.ruleColor,
    labelColor: theme.ruleColor,
  });

  // ---- Resolved encodings + queried data ----
  // Resolved axes, in field order; fields that can't be resolved are skipped. `fieldKey` identifies the
  // axis's field, whose position in `spec.data.fields` is the index of its entry in `chartState.brush`.
  let axes = $state.raw<(ResolvedAxis & { fieldKey: string })[]>([]);
  let colorData = $state.raw<ResolvedColorData | null>(null);
  let rows = $state.raw<{ values: Float32Array[]; colorValue: Float32Array | Uint8Array } | null>(null);

  let palette = $derived(
    colorData != null ? buildPalette(colorData, spec.scales?.color?.range, theme) : [theme.embeddingColor],
  );

  // Concrete value→color scale for the legend; null when there's no color encoding (single color).
  let colorScale = $derived(colorData != null ? buildColorScale(colorData, palette, theme) : null);

  let plotWidth = $state(0);
  let plotHeight = $state(0);

  let rowCount = $derived(rows?.colorValue.length ?? 0);
  let fieldKeys = $derived(spec.data.fields.map((f) => JSON.stringify(f)));
  // Per-axis brush (the view's indexing), picked from the per-field brush state. Looked up by field key so
  // it stays right while the axes of a previous field list are still shown.
  let axisBrush = $derived(axes.map((a) => chartState.brush?.[fieldKeys.indexOf(a.fieldKey)] ?? null));
  // Number of axes with an active brush.
  let filterCount = $derived(axisBrush.filter((b) => b != null).length);
  let statusText = $derived(
    filterCount > 0
      ? `${rowCount.toLocaleString()} points · ${filterCount} ${filterCount == 1 ? "axis" : "axes"} filtered`
      : `${rowCount.toLocaleString()} points`,
  );

  function clearAllFilters() {
    onStateChange((draft) => {
      draft.brush = undefined;
    });
  }

  // ---- SQL helpers (mirror the spec runtime's BuildContext) ----
  function fieldExpr(field: SQLField): SQL.ExprNode {
    if (typeof field == "string") {
      return SQL.column(field);
    }
    return SQL.sql`${resolveSQLTemplate(field.sql, { table: context.table, filter: "(true)" })}`;
  }
  function fromExpr(table: SQLTable, predicate?: string | null): SQL.FromExpr {
    if (typeof table == "string") {
      return new SQL.TableRefNode(table);
    }
    return SQL.sql`(${resolveSQLTemplate(table.sql, { table: context.table, filter: predicate ?? "(true)" })})`;
  }
  function fieldTitle(field: SQLField): string {
    return typeof field == "string" ? field : field.sql;
  }
  function statsKey(from: SQLTable, field: SQLField): string {
    return JSON.stringify(["pc", typeof from == "string" ? from : from.sql, field]);
  }
  function toFloat32(vector: any): Float32Array {
    let arr = vector.toArray();
    // Nulls are placed by the axis SQL; guard anyway so a null never becomes 0 (= top of the axis).
    return arr instanceof Float32Array ? arr : Float32Array.from(arr, (v: any) => (v == null ? NaN : Number(v)));
  }
  // Fallback ordering hash when there's no row-id column (not globally consistent, but deterministic per query).
  function buildIndexOrder(n: number): Uint32Array {
    let o = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      o[i] = hashIndex(i);
    }
    return o;
  }

  // ---- Setup: compute stats, resolve axes + color, (re)build the Mosaic client ----
  let client: MosaicClient | null = null;
  // Field keys of the previous setup run, to carry brushes along when fields are reordered / removed.
  let previousFieldKeys: string[] | null = null;
  function destroyClient() {
    if (client != null) {
      try {
        client.destroy();
      } catch (_) {}
      client = null;
    }
  }

  $effect(() => {
    // Only rebuild when query-affecting inputs change (not on opacity / title / brush edits).
    let querySig = JSON.stringify({
      fields: spec.data.fields,
      color: spec.data.color ?? null,
      from: spec.data.from ?? null,
      filter: spec.data.filter ?? null,
      colorType: spec.scales?.color?.type ?? null,
      colorDomain: spec.scales?.color?.domain ?? null,
      colorConstant: spec.scales?.color?.constant ?? null,
    });
    querySig; // track only the signature

    // Read the actual inputs without tracking them individually.
    let { fields, colorField, scaleType, scaleDomain, scaleConstant, from, useFilter } = untrack(() => ({
      fields: spec.data.fields,
      colorField: spec.data.color ?? null,
      scaleType: spec.scales?.color?.type,
      scaleDomain: spec.scales?.color?.domain,
      scaleConstant: spec.scales?.color?.constant,
      from: spec.data.from ?? context.table,
      useFilter: spec.data.filter == "$filter",
    }));

    // The brush is indexed by field position: when the fields change, move each brush to its field's new
    // position (dropping brushes of removed fields) so it keeps filtering the same field.
    let keys = fields.map((f) => JSON.stringify(f));
    let oldKeys = previousFieldKeys;
    previousFieldKeys = keys;
    if (oldKeys != null && JSON.stringify(oldKeys) != JSON.stringify(keys)) {
      let oldBrush = untrack(() => chartState.brush);
      if (oldBrush != null) {
        let newBrush = keys.map((k) => {
          let i = oldKeys.indexOf(k);
          return i >= 0 ? (oldBrush[i] ?? null) : null;
        });
        onStateChange((draft) => {
          draft.brush = newBrush.some((b) => b != null) ? newBrush : undefined;
        });
      }
    }

    let cancelled = false;
    rows = null;
    destroyClient();

    (async () => {
      let fromNode = fromExpr(from);
      let axisExprs = fields.map((f) => fieldExpr(f));
      let axisTitles = fields.map((f) => fieldTitle(f));
      let axisStats = await Promise.all(
        fields.map((f, i) =>
          context.cache.value(statsKey(from, f), () => computeFieldStats(context.coordinator, fromNode, axisExprs[i])),
        ),
      );
      let colorExpr = colorField != null ? fieldExpr(colorField) : null;
      let colorStats =
        colorField != null
          ? await context.cache.value(statsKey(from, colorField), () =>
              computeFieldStats(context.coordinator, fromNode, colorExpr!),
            )
          : undefined;
      if (cancelled) {
        return;
      }

      let resolvedAxes = fields.flatMap((_, i) => {
        let axis = resolveAxis(axisStats[i], axisExprs[i], axisTitles[i]);
        return axis != null ? [{ ...axis, fieldKey: keys[i] }] : [];
      });
      let resolvedColor = resolveColorData(
        colorStats,
        { type: scaleType, domain: scaleDomain, constant: scaleConstant } as Scale,
        colorExpr,
      );

      axes = resolvedAxes;
      colorData = resolvedColor;

      if (resolvedAxes.length == 0) {
        rows = { values: [], colorValue: new Uint8Array(0) };
        return;
      }

      let select: Record<string, SQL.ExprNode> = {};
      resolvedAxes.forEach((a, i) => (select["f" + i] = a.selectExpr));
      select["c"] = resolvedColor.selectExpr;
      // For nominal axes we jitter in JS; order points by a stable hash of the row id (globally consistent).
      let hasNominal = resolvedAxes.some((a) => a.kind == "nominal");
      if (hasNominal && context.id != null) {
        select["o"] = SQL.sql`(hash(${SQL.column(context.id)}) % 4294967296)::UINTEGER`;
      }

      let c = makeClient({
        coordinator: context.coordinator,
        selection: useFilter ? context.filter : undefined,
        query: (predicate: any) =>
          SQL.Query.from(fromExpr(from, predicateToString(predicate)))
            .select(select)
            .where(useFilter ? predicate : []),
        queryResult: (data: any) => {
          // The view reserves `labelFontSize` (11px) margins top & bottom; the [0,1] axis spans the rest
          // of the plot height (the full height minus the status line).
          let plotHeightPx = Math.max(1, (height ?? 500) - 2 * 11);
          let order: Uint32Array | undefined;
          if (hasNominal) {
            let oc = data.getChild("o");
            order = oc != null ? (oc.toArray() as Uint32Array) : buildIndexOrder(data.numRows);
          }
          let values = resolvedAxes.map((axis, i) => {
            let col = data.getChild("f" + i);
            if (axis.kind == "nominal") {
              // Seed the jitter per axis so each ranks its points independently; a shared order key would
              // make category→category bundles form coherent non-crossing threads (a stripe/moiré artifact).
              return applyCategoryJitter(col.toArray(), axis.categoryCount ?? 1, plotHeightPx, order!, hashIndex(i));
            }
            return toFloat32(col);
          });
          let colorValue = resolvedColor.toColorValue(data.getChild("c"), data.numRows);
          rows = { values, colorValue };
        },
      });
      if (cancelled) {
        try {
          c.destroy();
        } catch (_) {}
        return;
      }
      client = c;
    })();

    return () => {
      cancelled = true;
      destroyClient();
    };
  });

  // ---- Brush → cross-filter (self-filter: clients new Set([]) filters this view too) ----
  // `reset` is invoked by Mosaic when the shared filter is cleared (e.g. the global "reset
  // filters" action), so the brush state must be dropped to match.
  const source = { reset: () => clearAllFilters() };
  $effect(() => {
    let brush = axisBrush;
    let resolvedAxes = axes;
    let preds: SQL.ExprNode[] = [];
    brush.forEach((b, i) => {
      if (b != null) {
        let p = resolvedAxes[i].brushToPredicate(b[0], b[1]);
        if (p != null) {
          preds.push(p);
        }
      }
    });
    let anyBrush = preds.length > 0;
    context.filter.update({
      source,
      clients: new Set([]),
      value: anyBrush ? chartState.brush : null,
      predicate: anyBrush ? SQL.and(...preds) : null,
    } as any);
  });

  // The view reports brushes per axis; store them per field.
  function handleBrushChange(b: ([number, number] | null)[]) {
    let brush: ([number, number] | null)[] = fieldKeys.map(() => null);
    axes.forEach((a, i) => {
      let fi = fieldKeys.indexOf(a.fieldKey);
      if (fi >= 0) {
        brush[fi] = b[i] ?? null;
      }
    });
    onStateChange((draft) => {
      draft.brush = brush.some((x) => x != null) ? brush : undefined;
    });
  }

  // ---- Render: own the ParallelCoordinatesView instance lifecycle ----
  let container: HTMLDivElement | null = $state(null);
  let view: ParallelCoordinatesView | null = null;

  // Derived so they keep their identity across unrelated updates (brush drags, resizes, theme changes):
  // the view diffs props by reference, and a new `data` object re-uploads every row to the GPU.
  let viewData = $derived({ values: rows?.values ?? [], colorValue: rows?.colorValue ?? null });
  let axisLabels = $derived(axes.map((a) => a.ticks));
  let axisTitles = $derived(axes.map((a) => a.title));

  function buildProps(): ParallelCoordinatesViewProps {
    return {
      data: viewData,
      axisLabels: axisLabels,
      axisTitles: axisTitles,
      categoryColors: palette,
      opacity: spec.opacity ?? null,
      width: plotWidth,
      height: plotHeight,
      pixelRatio: typeof window != "undefined" ? window.devicePixelRatio : 2,
      colorScheme: $colorScheme,
      theme: pcTheme,
      brush: axisBrush,
      onBrushChange: handleBrushChange,
    };
  }

  $effect(() => {
    let props = buildProps();
    if (container == null) {
      return;
    }
    if (view == null) {
      view = new ParallelCoordinatesView(container, props);
    } else {
      view.update(props);
    }
  });

  onDestroy(() => {
    view?.destroy();
    view = null;
    destroyClient();
    context.filter.update({ source, clients: new Set([]), value: null, predicate: null } as any);
  });
</script>

<Container width={spec.width ?? width} height={spec.height ?? height} class="flex flex-col" defaultHeight={240}>
  <div class="flex-1 relative" bind:clientWidth={plotWidth} bind:clientHeight={plotHeight}>
    <div class="absolute" bind:this={container}></div>
  </div>
  {#if colorScale != null}
    <div class="flex-none mt-1" class:max-w-72={colorScale.type != "band"}>
      <ColorLegend scale={colorScale} theme={theme} />
    </div>
  {/if}
  <div class="flex justify-end text-sm items-center text-slate-400 dark:text-slate-500">
    {#if rows != null}
      <span class="whitespace-nowrap overflow-hidden text-ellipsis">{statusText}</span>
      {#if filterCount > 0}
        <button
          class="ml-2 shrink-0 flex items-center gap-0.5 select-none rounded px-1 text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700"
          title="Clear all filters"
          onclick={clearAllFilters}
        >
          <IconClose class="w-3 h-3" />
          Clear
        </button>
      {/if}
    {/if}
  </div>
</Container>
