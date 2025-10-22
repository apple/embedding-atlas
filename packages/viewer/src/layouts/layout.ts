// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Snippet } from "svelte";

import type { ChartContext } from "../charts/chart.js";

export interface LayoutProps<State = unknown> {
  context: ChartContext;

  /** A dictionary of charts to layout. The key is the chart id, and the value is the chart spec. */
  charts: Record<string, any>;

  /** The state of the layout. */
  state: State;

  /**
   * Callback for when the state changes.
   * The default update mode is "merge", where the new state is recursively merged into the existing state.
   * In "replace" mode, the new state completely replaces the existing state.
   */
  onStateChange: (state: Partial<State>, mode?: "merge" | "replace") => void;

  /**
   * Callback for when charts change.
   * The default update mode is "merge", where the new charts is recursively merged into the existing charts.
   * In "replace" mode, the new charts completely replaces the existing charts.
   */
  onChartsChange: (charts: Record<string, any>, mode?: "merge" | "replace") => void;

  /** Callback for when chart states change. */
  onChartStatesChange: (states: Record<string, any>, mode?: "merge" | "replace") => void;

  /** A snippet that renders a given chart. */
  chartView: Snippet<[{ id: string; width?: number | "container"; height?: number | "container" }]>;
}

export type LayoutOptionsProps<State = unknown> = Omit<LayoutProps<State>, "chartView">;
