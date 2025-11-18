// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { EmbeddingViewConfig, Label } from "@embedding-atlas/component";
import type { CustomCell } from "@embedding-atlas/table";
import type { Coordinator, Selection } from "@uwdata/mosaic-core";
import type { Readable, Writable } from "svelte/store";

import type { ColumnDesc } from "../utils/database.js";
import type { ChartThemeConfig } from "./common/theme.js";

export class ChartContextCache {
  private contents: Map<string, any>;

  constructor() {
    this.contents = new Map();
  }

  get(key: string): any | null {
    return this.contents.get(key) ?? null;
  }

  set(key: string, value: any) {
    this.contents.set(key, value);
  }

  value<T>(key: string, valueFunc: () => T): T {
    if (this.contents.has(key)) {
      return this.contents.get(key) as T;
    }
    const value = valueFunc();
    this.contents.set(key, value);
    return value;
  }
}

export type RowID = any;

export interface ChartContext {
  /** The Mosaic coordinator. */
  coordinator: Coordinator;

  /** The data table. */
  table: string;

  /** The row id column. */
  id: string;

  /** A list of columns the table contains. */
  columns: ColumnDesc[];

  /** The global cross filter selection. */
  filter: Selection;

  /** The current color scheme. */
  colorScheme: Readable<"light" | "dark">;

  /** The chart theme. */
  theme: Readable<ChartThemeConfig | undefined>;

  /** The column styles. */
  columnStyles: Readable<any>;

  /**
   * A cache for shared intermediate results.
   * Values in this cache is kept during the hosting component's lifecycle.
   * You can store any value in this cache (including values that reference the coordinator or filter).
   */
  cache: ChartContextCache;

  /**
   * A persistent cache for intermediate results.
   * Values in this cache is kept by the backend (if available).
   * Values in this cache must be JSON serializable.
   */
  persistentCache: {
    get(key: string): Promise<any | null>;
    set(key: string, value: any): Promise<void>;
  };

  /** Tell the parent to show a search box. */
  search?: (query: any, mode: string) => void;

  /** A list of supported search modes. */
  searchModes?: string[];

  /** Current search result */
  searchResult: Readable<{ query: any; mode: string; ids: RowID[] } | null>;

  /** The current highlight point. When this changes, supported views will highlight the given point. */
  highlight: Writable<RowID | null>;

  /** Configuration for the embedding view. See docs for the EmbeddingView. */
  embeddingViewConfig?: EmbeddingViewConfig | null;

  /** Labels for the embedding view. */
  embeddingViewLabels?: Label[] | null;

  /** Custom cell renderers for the table view. */
  tableCellRenderers?: Record<string, CustomCell | "markdown">;
}

/** Props passed into a chart view. */
export interface ChartViewProps<Spec = unknown, State = unknown> {
  /**
   * The context of the chart. The context is constant during the chart view's lifecycle
   * (i.e., if the coordinator or table changes, the chart view will be re-created)
   */
  context: ChartContext;

  /**
   * The chart width. If specified, the chart shall fit itself with the width.
   * If not specified, the chart can decide its own width.
   */
  width?: number;

  /**
   * The chart height. If specified, the chart shall fit itself with the height.
   * If not specified, the chart can decide its own height.
   */
  height?: number;

  /**
   * A set of properties that defines the chart. The includes things like the data column for x and y,
   * the title, the x and y axis labels, the color scale, etc.
   * The chart can change its own spec, e.g., have a dropdown to change its own X scale type.
   * The spec must be a JSON-serializable object.
   */
  spec: Spec;

  /**
   * The current user interaction state. This includes things like a brush filter's current value, a checkbox's checked state, etc.
   * Sometimes the line between spec and state is blurry (e.g., the X scale type could be considered a state if there's a dropdown to change it.)
   * The functional difference is that when we reset the chart or load it from scratch, the state will be set to `{}` and the spec is unchanged.
   */
  state: State;

  /** The mode of the chart view. The view can decide how to interpret this. */
  mode: "view" | "edit";

  /**
   * Callback for when the state changes.
   * The default update mode is "merge", where the new state is recursively merged into the existing state.
   * In "replace" mode, the new state completely replaces the existing state.
   */
  onStateChange: (state: Partial<State>, mode?: "merge" | "replace") => void;

  /**
   * Callback for when the spec changes.
   * The default update mode is "merge", where the new spec is recursively merged into the existing spec.
   * In "replace" mode, the new spec completely replaces the existing spec.
   */
  onSpecChange: (spec: Partial<Spec>, mode?: "merge" | "replace") => void;
}

export type { ChartBuilderDescription } from "./builder/builder_description.js";
