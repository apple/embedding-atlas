<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script module lang="ts">
  import { maxDensityModeCategories, type DataPoint, type ViewportState } from "@embedding-atlas/component";
  import { type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import Overlay from "./Overlay.svelte";
  import Tooltip from "./Tooltip.svelte";

  import { type EmbeddingLegend } from "../../utils/database.js";
  import { createCustomComponentClass } from "./custom_components.js";

  async function defaultViewportScale(coordinator: Coordinator, table: string, x: string, y: string): Promise<number> {
    let { stdX, stdY } = (
      await coordinator.query(
        SQL.Query.from(table).select({
          stdX: SQL.sql`STDDEV(${SQL.column(x)})::FLOAT`,
          stdY: SQL.sql`STDDEV(${SQL.column(y)})::FLOAT`,
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
  import { EmbeddingViewMosaic } from "@embedding-atlas/component/svelte";
  import { cubicOut } from "svelte/easing";

  import Button from "../../widgets/Button.svelte";
  import PopupButton from "../../widgets/PopupButton.svelte";
  import Select from "../../widgets/Select.svelte";
  import Slider from "../../widgets/Slider.svelte";
  import Legend from "./Legend.svelte";

  import { IconSettings } from "../../assets/icons.js";
  import { isolatedWritable } from "../../utils/store.js";
  import type { ChartViewProps, RowID } from "../chart.js";
  import { resolveChartTheme } from "../common/theme.js";
  import { makeCategoryColumn } from "./category_column.js";
  import type { EmbeddingSpec, EmbeddingState } from "./types.js";
  import { interpolateViewport } from "./viewport_animation.js";

  const maxCategories = Math.min(20, maxDensityModeCategories());
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

  let { colorScheme, columnStyles, searchResult, theme: themeConfig } = context;
  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  let highlightStore = isolatedWritable(context.highlight);

  let categoryColumn = $derived(spec.data.category);

  let categoryLegend: EmbeddingLegend | null = $state.raw(null);
  let totalPointCount: number | null = $state.raw(null);

  // Query total point count for render limit slider
  $effect.pre(() => {
    context.coordinator
      .query(SQL.Query.from(context.table).select({ count: SQL.sql`COUNT(*)::INT` }))
      .then((result: any) => {
        totalPointCount = result.get(0).count;
      });
  });

  let tooltip = $state.raw<DataPoint | null>(null);
  let selection = $state.raw<DataPoint[] | null>(null);
  let overlayProps = $state.raw<{ center: DataPoint | null; points: DataPoint[] } | null>(null);

  // Update the category mapping and legend.
  $effect.pre(() => {
    let promise = context.cache.value(`embedding/category/${categoryColumn}`, () =>
      makeCategoryColumn(context.coordinator, context.table, categoryColumn, theme),
    );
    promise.then((v) => {
      categoryLegend = v;
      if ((categoryLegend?.legend.length ?? 0) > maxCategories) {
        onSpecChange({ mode: "points" });
      }
    });
  });

  $effect.pre(() => {
    let isOnMount = true;
    return highlightStore.subscribe((v) => {
      // Note: don't animate immediately on mount.
      if (v !== null && !isOnMount) {
        animateToPoint(v);
      }
      isOnMount = false;
    });
  });

  $effect.pre(() =>
    searchResult.subscribe(async (result) => {
      if (result == null || result.ids.length == 0) {
        overlayProps = null;
        return;
      }
      let centerId: RowID | null = null;
      if (result.mode == "neighbors") {
        centerId = result.query;
      }
      let r = Array.from(
        await context.coordinator.query(
          SQL.Query.from(context.table)
            .select({ identifier: SQL.column(context.id), x: SQL.column(spec.data.x), y: SQL.column(spec.data.y) })
            .where(
              SQL.isIn(
                context.id,
                result.ids.concat(centerId != null ? [centerId] : []).map((x) => SQL.literal(x)),
              ),
            ),
        ),
      ) as DataPoint[];
      overlayProps = {
        center: r.filter((p) => p.identifier === centerId)[0] ?? null,
        points: r.filter((p) => p.identifier !== centerId),
      };
    }),
  );

  // Subscribe to highlightIds for programmatic multi-point highlighting
  $effect.pre(() => {
    if (context.highlightIds == null) return;
    return context.highlightIds.subscribe(async (ids) => {
      if (ids == null || ids.length == 0) {
        overlayProps = null;
        return;
      }
      let r = Array.from(
        await context.coordinator.query(
          SQL.Query.from(context.table)
            .select({ identifier: SQL.column(context.id), x: SQL.column(spec.data.x), y: SQL.column(spec.data.y) })
            .where(
              SQL.isIn(
                context.id,
                ids.map((x) => SQL.literal(x)),
              ),
            ),
        ),
      ) as DataPoint[];
      overlayProps = {
        center: null,
        points: r,
      };
    });
  });

  async function animateToPoint(identifier: RowID): Promise<void> {
    let defaultScale = await context.cache.value(`embedding/default-viewport-scale/${spec.data.x},${spec.data.y}`, () =>
      defaultViewportScale(context.coordinator, context.table, spec.data.x, spec.data.y),
    );
    let scale = defaultScale * 2;
    // Query the x, y location.
    let result = await context.coordinator.query(
      SQL.Query.from(context.table)
        .select({
          x: SQL.column(spec.data.x),
          y: SQL.column(spec.data.y),
        })
        .where(SQL.eq(SQL.column(context.id), SQL.literal(identifier))),
    );
    let { x, y } = result.get(0) as { x: number; y: number };
    // Start animation and show tooltip.
    startViewportAnimation({ x: x, y: y, scale: scale });
    selection = [identifier];
    tooltip = identifier;
  }

  let currentViewportAnimation: number | null;
  let animatingViewport = $state.raw<ViewportState | null>(null);
  function startViewportAnimation(newState: ViewportState) {
    tooltip = null;
    let start = animatingViewport ?? chartState.viewport;
    if (start == null) {
      onStateChange({ viewport: newState });
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
        onStateChange({ viewport: animatingViewport });
      }
    };
    if (currentViewportAnimation) {
      cancelAnimationFrame(currentViewportAnimation);
    }
    currentViewportAnimation = requestAnimationFrame(callback);
  }
</script>

<div class="relative">
  <EmbeddingViewMosaic
    width={width}
    height={height}
    coordinator={context.coordinator}
    table={context.table}
    filter={context.filter}
    rangeSelection={context.filter}
    identifier={context.id}
    x={spec.data.x}
    y={spec.data.y}
    text={spec.data.text}
    category={categoryLegend?.indexColumn}
    categoryColors={categoryLegend?.legend.map((x) => x.color) ?? [theme.embeddingColor]}
    config={{
      colorScheme: $colorScheme,
      ...context.embeddingViewConfig,
      mode: spec.mode ?? "points",
      ...(spec.minimumDensity != null ? { minimumDensity: spec.minimumDensity } : {}),
      ...(spec.pointSize != null ? { pointSize: spec.pointSize } : {}),
      downsampleMaxPoints: spec.downsampleMaxPoints ?? defaultDownsampleMaxPoints,
    }}
    labels={context.embeddingViewLabels}
    cache={context.persistentCache}
    additionalFields={Object.fromEntries(context.columns.map((c) => [c.name, c.name]))}
    customTooltip={{
      class: CustomTooltip,
      props: {
        darkMode: $colorScheme,
        columnStyles: $columnStyles,
        onNearestNeighborSearch:
          (context.searchModes ?? []).indexOf("neighbors") >= 0 ? (id: any) => context.search?.(id, "neighbors") : null,
      },
    }}
    customOverlay={{
      class: CustomOverlay,
      props: { ...(overlayProps ?? { points: [], center: null }) },
    }}
    viewportState={animatingViewport ?? chartState.viewport}
    onViewportState={(v) => onStateChange({ viewport: v })}
    rangeSelectionValue={chartState.brush}
    onRangeSelection={(v) => onStateChange({ brush: v ?? undefined })}
    tooltip={tooltip}
    onTooltip={(v) => {
      tooltip = v;
    }}
    selection={selection}
    onSelection={(points) => {
      selection = points;
      if (points != null && points.length == 1) {
        highlightStore.set(points[0].identifier);
      }
    }}
  />
  <div class="absolute top-0 left-0 right-0 flex flex-wrap justify-between items-start pointer-events-none">
    {#if categoryLegend != null}
      <div
        class="flex-none m-2 p-2 rounded-md bg-slate-100/75 dark:bg-slate-800/75 backdrop-blur-sm pointer-events-auto order-3"
      >
        <Legend
          context={context}
          spec={{ items: categoryLegend.legend }}
          state={chartState.legend ?? {}}
          mode="view"
          onSpecChange={() => {}}
          onStateChange={(update, mode) => {
            onStateChange({ legend: update });
          }}
        />
      </div>
    {/if}
    <div
      class="flex-none p-2 rounded-ss-md rounded-ee-md bg-white/75 dark:bg-black/75 backdrop-blur-sm flex items-center gap-2 pointer-events-auto order-1"
    >
      <Select
        class="max-w-64"
        label="Color"
        value={categoryColumn}
        onChange={(v) => onSpecChange({ data: { ...spec.data, category: v } })}
        options={[
          { value: undefined, label: "--" },
          ...context.columns
            .filter((c) => c.jsType == "string" || c.jsType == "number")
            .map((c) => ({ value: c.name, label: `${c.name} (${c.type})` })),
        ]}
      />
      <PopupButton icon={IconSettings} title="Options">
        <div class="flex flex-col gap-2 w-64">
          <div class="text-slate-500 dark:text-slate-400 select-none">Display Mode</div>
          <div class="flex gap-2 items-center">
            <Select
              value={spec.mode ?? "points"}
              onChange={(v) => onSpecChange({ mode: v })}
              disabled={categoryLegend != null && categoryLegend.legend.length > maxCategories}
              options={[
                { value: "points", label: "Points" },
                { value: "density", label: "Density" },
              ]}
            />
            {#if (spec.mode ?? "points") == "density"}
              <Slider
                bind:value={
                  () => Math.log((spec.minimumDensity ?? defaultMinimumDensity) / defaultMinimumDensity),
                  (v) => onSpecChange({ minimumDensity: defaultMinimumDensity * Math.exp(v) })
                }
                min={-4}
                max={4}
                step={0.05}
              />
            {/if}
          </div>
          <div class="text-slate-500 dark:text-slate-400 select-none">Point Size</div>
          <div class="flex gap-2 items-center">
            <Slider
              bind:value={() => spec.pointSize ?? 1, (v) => onSpecChange({ pointSize: v })}
              min={1}
              max={10}
              step={0.05}
            />
            <Button label="Auto" onClick={() => onSpecChange({ pointSize: undefined })} />
          </div>
          {#if totalPointCount != null && totalPointCount > minDownsampleMaxPoints}
            {@const effectiveLimit = spec.downsampleMaxPoints ?? Math.min(defaultDownsampleMaxPoints, totalPointCount)}
            {@const isMaxed = effectiveLimit >= totalPointCount}
            <div class="text-slate-500 dark:text-slate-400 select-none">
              Max Points: {isMaxed
                ? "All"
                : effectiveLimit >= 1000000
                  ? (effectiveLimit / 1000000).toFixed(1) + "M"
                  : (effectiveLimit / 1000).toFixed(0) + "K"}
              {#if !isMaxed}
                <span class="text-slate-400 dark:text-slate-500"
                  >/ {totalPointCount >= 1000000
                    ? (totalPointCount / 1000000).toFixed(1) + "M"
                    : (totalPointCount / 1000).toFixed(0) + "K"}</span
                >
              {/if}
            </div>
            <div class="flex gap-2 items-center">
              <Slider
                bind:value={
                  () =>
                    spec.downsampleMaxPoints ??
                    Math.min(defaultDownsampleMaxPoints, totalPointCount ?? defaultDownsampleMaxPoints),
                  (v) => onSpecChange({ downsampleMaxPoints: v })
                }
                min={minDownsampleMaxPoints}
                max={totalPointCount}
                step={Math.max(10000, Math.floor(totalPointCount / 100 / 10000) * 10000)}
              />
            </div>
          {/if}
        </div>
      </PopupButton>
    </div>
  </div>
</div>
