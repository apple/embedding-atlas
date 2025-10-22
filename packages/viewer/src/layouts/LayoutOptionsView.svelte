<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { applyUpdatesForKeyIfNeeded, applyUpdatesIfNeeded } from "@embedding-atlas/utils";

  import type { ChartContext } from "../charts/chart.js";
  import { findLayoutOptionsComponent } from "./layout_types.js";

  interface Props {
    context: ChartContext;
    charts: Record<string, any>;
    chartStates: Record<string, any>;
    layout: string;
    layoutStates: Record<string, any>;
    onChartsChange?: (charts: Record<string, any>) => void;
    onChartStatesChange?: (states: Record<string, any>) => void;
    onLayoutStatesChange?: (states: Record<string, any>) => void;
  }

  let {
    context,
    charts,
    chartStates,
    layout,
    layoutStates,
    onLayoutStatesChange,
    onChartsChange,
    onChartStatesChange,
  }: Props = $props();

  let LayoutClass = $derived(findLayoutOptionsComponent(layout));
  let layoutState = $derived(layoutStates[layout] ?? {});

  function updateCharts(update: any, mode: "merge" | "replace" = "merge") {
    applyUpdatesIfNeeded(charts, update, mode, (r) => onChartsChange?.(r));
  }

  function updateChartStates(update: any, mode: "merge" | "replace" = "merge") {
    applyUpdatesIfNeeded(chartStates, update, mode, (r) => onChartStatesChange?.(r));
  }

  function onLayoutStateChange(layout: string, update: any, mode: "merge" | "replace" = "merge") {
    applyUpdatesForKeyIfNeeded(layoutStates, layout, update, mode, (r) => onLayoutStatesChange?.(r));
  }
</script>

<LayoutClass
  context={context}
  charts={charts}
  state={layoutState}
  onStateChange={onLayoutStateChange.bind(null, layout)}
  onChartsChange={updateCharts}
  onChartStatesChange={updateChartStates}
/>
