// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { deepEquals } from "@embedding-atlas/utils";
import { type Coordinator, Selection } from "@uwdata/mosaic-core";
import { type Draft, produce } from "immer";
import { createContext } from "svelte";
import { get, type Readable, writable, type Writable } from "svelte/store";

import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "../api.js";
import { type ChartContext, ChartContextCache, type ChartDelegate } from "../charts/chart.js";
import type { ChartThemeConfig } from "../charts/common/theme.js";
import { defaultCharts } from "../charts/default_charts.js";
import { EMBEDDING_ATLAS_VERSION } from "../constants.js";
import type { ColumnStyle } from "../renderers/types.js";
import { type ColorScheme, makeColorSchemeStore } from "../utils/color_scheme.js";
import { type ColumnDesc, columnDescriptions, predicateToString } from "../utils/database.js";
import { findUnusedId } from "../utils/identifier.js";
import { stableDerived, stableWritable } from "../utils/store.js";
import { HistoryManager } from "./history_manager.js";
import { SearchStore } from "./search_store.js";

/** Internal state is EmbeddingAtlasState but all fields are required. */
type InternalState = Required<EmbeddingAtlasState>;

/** Store instance specific to an EmbeddingAtlas view instance */
export class EmbeddingAtlasStore {
  readonly coordinator: Coordinator;
  readonly crossFilter: Selection;

  readonly chartContext: ChartContext;

  readonly userColorScheme: Writable<ColorScheme | null | undefined>;
  readonly colorScheme: Readable<ColorScheme>;

  readonly activePanel: Writable<string | undefined>;

  readonly state: Writable<InternalState>;

  // Parts of the state
  readonly charts: Readable<Record<string, any>>;
  readonly chartStates: Readable<Record<string, any>>;
  readonly layouts: Readable<Record<string, any>>;
  readonly layoutOrder: Readable<string[]>;
  readonly currentLayout: Readable<string>;
  readonly columnStyles: Readable<Record<string, ColumnStyle>>;

  readonly columns: Writable<ColumnDesc[]>;
  readonly chartTheme: Writable<ChartThemeConfig | undefined>;

  readonly search: SearchStore;
  readonly history: HistoryManager<InternalState>;

  readonly props: EmbeddingAtlasProps;

  readonly chartDelegates: Map<string, Set<ChartDelegate>>;

  readonly canUndo: Writable<boolean>;
  readonly canRedo: Writable<boolean>;

  readonly ready: Promise<void>;

  constructor(options: EmbeddingAtlasProps) {
    this.props = options;
    const { coordinator, data } = options;
    this.coordinator = coordinator;
    this.crossFilter = Selection.crossfilter();

    // Color scheme
    const { colorScheme, userColorScheme } = makeColorSchemeStore();
    this.userColorScheme = userColorScheme;
    this.colorScheme = colorScheme;

    this.activePanel = writable(undefined);

    this.state = stableWritable(resolveInternalState({}));

    this.chartTheme = stableWritable(options.chartTheme ?? undefined);
    this.columns = writable([]);

    this.charts = stableDerived([this.state], ([state]) => state.charts);
    this.chartStates = stableDerived([this.state], ([state]) => state.chartStates);
    this.layouts = stableDerived([this.state], ([state]) => state.layouts);
    this.layoutOrder = stableDerived([this.state], ([state]) => state.layoutOrder);
    this.currentLayout = stableDerived([this.state], ([state]) => state.currentLayout);
    this.columnStyles = stableDerived(
      [stableDerived([this.state], ([state]) => state.columnStyles ?? {}), this.columns],
      ([columnStyles, columns]) => resolveColumnStyles(columns, columnStyles, data.text),
    );

    this.history = new HistoryManager({ debounce: 200 });
    this.canUndo = writable(false);
    this.canRedo = writable(false);
    this.search = new SearchStore();

    this.chartContext = {
      coordinator: coordinator,
      filter: this.crossFilter,
      table: data.table,
      id: data.id,
      columns: [], // Later set by initialize
      colorScheme: this.colorScheme,
      theme: this.chartTheme,
      columnStyles: this.columnStyles,
      cache: new ChartContextCache(),
      persistentCache: options.cache ?? { get: async () => null, set: async (key, value) => {} },
      searcher: {
        search: (query, mode) => {
          this.activePanel.set("search");
          this.search.search(query, mode);
        },
      },
      highlight: writable(null),
      overlay: writable(null),
      embeddingViewConfig: options.embeddingViewConfig,
      embeddingViewLabels: options.embeddingViewLabels,
    };

    this.search.result.subscribe((result) => {
      if (result == undefined) {
        this.chartContext.overlay.set(null);
      } else {
        this.chartContext.overlay.set(result.overlay);
      }
    });

    this.chartDelegates = new Map();

    if (options.onPredicateChange) {
      const onPredicateChange = options.onPredicateChange;
      let lastPredicate: string | null | undefined = undefined;
      this.crossFilter.addEventListener("value", () => {
        const predicate = this.getCurrentPredicate();
        if (predicate !== lastPredicate) {
          lastPredicate = predicate;
          onPredicateChange(predicate);
        }
      });
    }

    const initialize = async () => {
      let descs = await columnDescriptions(this.coordinator, data.table);
      this.columns.set(descs.filter((x) => !x.name.startsWith("__")));
      this.chartContext.columns = get(this.columns);

      let initialState: EmbeddingAtlasState = {};
      if (options.initialState) {
        // Get a deep copy of the provided state, in case it get modified outside
        initialState = JSON.parse(JSON.stringify(options.initialState));
      }

      // Populate default charts if state has no chart
      if (Object.keys(initialState.charts ?? {}).length == 0) {
        try {
          let newCharts = await this._defaultCharts();
          initialState.charts = Object.fromEntries(newCharts.map((spec, i) => [`${i + 1}`, spec]));
        } catch (e) {
          console.error("Failed to create default charts");
          console.trace(e);
        }
      }

      this.importState(initialState);

      this.search.textColumn.set(
        data.text ?? get(this.columns).filter((x) => x.jsType == "string")[0]?.name ?? undefined,
      );
      this.search.config.set({
        coordinator: coordinator,
        table: data.table,
        id: data.id,
        getCurrentPredicate: () => this.getCurrentPredicate(),
        getColumns: () => get(this.columns),
        neighbors: data.neighbors ?? undefined,
        searcher: options.searcher ?? undefined,
      });
    };

    this.ready = initialize();
  }

  private async _defaultCharts() {
    let data = this.props.data;
    return await defaultCharts({
      coordinator: this.coordinator,
      table: data.table,
      id: data.id,
      projection: data.projection
        ? {
            ...data.projection,
            text: data.text ?? undefined,
            image: data.image ?? undefined,
            importance: data.importance ?? undefined,
            neighbors: data.neighbors ?? undefined,
          }
        : undefined,
      config: this.props.defaultChartsConfig ?? undefined,
    });
  }

  getCurrentPredicate(): string | null {
    return predicateToString(this.crossFilter.predicate(null));
  }

  getCurrentState(): EmbeddingAtlasState {
    return get(this.state);
  }

  private _syncHistory() {
    this.canUndo.set(this.history.canUndo);
    this.canRedo.set(this.history.canRedo);
  }

  undo() {
    const state = this.history.undo();
    if (state) {
      this.state.set(state);
      this._syncHistory();
      this.props.onStateChange?.(state);
    }
  }

  redo() {
    const state = this.history.redo();
    if (state) {
      this.state.set(state);
      this._syncHistory();
      this.props.onStateChange?.(state);
    }
  }

  resetFilter() {
    for (let item of this.crossFilter.clauses) {
      let source = item.source;
      source?.reset?.();
      this.crossFilter.update({ ...item, value: null, predicate: null });
    }
  }

  private _updateState(updater: (draft: Draft<InternalState>) => void, options: { cleanup?: boolean } = {}) {
    let cleanup = options.cleanup ?? false;
    if (cleanup) {
      this.state.update((state) => cleanupState(produce(state, updater)));
    } else {
      this.state.update((state) => produce(state, updater));
    }
    const currentState = get(this.state);
    this.history.update(currentState);
    this._syncHistory();
    this.props.onStateChange?.(currentState);
  }

  importState(state: EmbeddingAtlasState) {
    this.state.set(resolveInternalState(state));
    const currentState = get(this.state);
    this.history.clear();
    this.history.update(currentState);
    this._syncHistory();
    this.props.onStateChange?.(currentState);
  }

  setCurrentLayout(layout: string) {
    this._updateState(
      (draft) => {
        draft.currentLayout = layout;
      },
      { cleanup: true },
    );
  }

  setLayoutOrder(order: string[]) {
    this._updateState(
      (draft) => {
        draft.layoutOrder = order;
      },
      { cleanup: true },
    );
  }

  addLayout(type: "list" | "dashboard"): string {
    let layoutId: string = "";
    this._updateState(
      (draft) => {
        layoutId = findUnusedId(draft.layouts);
        const name = `Dashboard ${layoutId}`;
        draft.layouts[layoutId] = { type, name, chartIds: [] };
        draft.currentLayout = layoutId;
      },
      { cleanup: true },
    );
    return layoutId;
  }

  async addLayoutWithDefaultCharts(type: "list" | "dashboard"): Promise<string> {
    let charts = await this._defaultCharts();

    this._updateState(
      (draft) => {
        const id = findUnusedId(draft.layouts);
        const name = `Dashboard ${id}`;
        const chartIds: string[] = [];
        for (let spec of charts) {
          let chartId = findUnusedId(draft.charts);
          draft.charts[chartId] = spec;
          chartIds.push(chartId);
        }
        draft.layouts[id] = { type, name, chartIds };
        draft.currentLayout = id;
      },
      { cleanup: true },
    );

    return get(this.currentLayout);
  }

  removeLayout(id: string) {
    this._updateState(
      (draft) => {
        delete draft.layouts[id];
      },
      { cleanup: true },
    );
  }

  setColumnStyle(column: string, style: ColumnStyle) {
    this._updateState((draft) => {
      draft.columnStyles[column] = style;
    });
  }

  updateChart<T extends object>(id: string, update: T | ((draft: Draft<T>) => void)) {
    this._updateState((draft) => {
      if (typeof update == "function") {
        update(draft.charts[id]);
      } else {
        draft.charts[id] = update;
      }
    });
  }

  updateChartState<T extends object>(id: string, update: T | ((draft: Draft<T>) => void)) {
    this._updateState((draft) => {
      if (typeof update == "function") {
        draft.chartStates[id] ??= {};
        update(draft.chartStates[id]);
      } else {
        draft.chartStates[id] = update;
      }
      // Delete if the state is empty
      if (Object.keys(draft.chartStates[id] ?? {}).length == 0) {
        delete draft.chartStates[id];
      }
    });
  }

  replaceChart<T extends object>(id: string, spec: T) {
    this._updateState((draft) => {
      draft.charts[id] = spec;
      delete draft.chartStates[id];
    });
  }

  updateLayout<T extends object>(id: string, update: T | ((draft: Draft<T>) => void)) {
    this._updateState((draft) => {
      if (typeof update == "function") {
        update(draft.layouts[id]);
      } else {
        draft.layouts[id] = update;
      }
    });
  }

  addChartToLayout(layoutId: string, spec: any = { type: "builder", title: "New" }): string {
    let chartId: string = "";
    this._updateState((draft) => {
      chartId = findUnusedId(draft.charts);
      draft.charts[chartId] = spec;
      let layout = draft.layouts[layoutId];
      layout.chartIds ??= [];
      layout.chartIds.push(chartId);
      if (layout.type === "list") {
        layout.chartsOrder = [chartId, ...(layout.chartsOrder ?? []).filter((x: string) => x !== chartId)];
      }
    });
    return chartId;
  }

  removeChartFromLayout(layoutId: string, chartId: string) {
    this._updateState(
      (draft) => {
        delete draft.charts[chartId];
        let layout = draft.layouts[layoutId];
        layout.chartIds = (layout.chartIds ?? []).filter((id: string) => id !== chartId);
        if (layout.type === "dashboard" && layout.grids) {
          for (let key in layout.grids) {
            delete layout.grids[key].placements?.[chartId];
            if (layout.grids[key].order) {
              layout.grids[key].order = layout.grids[key].order.filter((id: string) => id !== chartId);
            }
          }
        }
      },
      { cleanup: true },
    );
  }

  registerChartDelegate(id: string, delegate: ChartDelegate): () => void {
    if (!this.chartDelegates.has(id)) {
      this.chartDelegates.set(id, new Set());
    }
    this.chartDelegates.get(id)!.add(delegate);
    return () => {
      this.chartDelegates.get(id)?.delete(delegate);
    };
  }
}

function resolveColumnStyles(
  columns: ColumnDesc[],
  styles: Record<string, ColumnStyle>,
  textColumn?: string | null,
): Record<string, ColumnStyle> {
  let result: Record<string, ColumnStyle> = {};
  for (let column of columns) {
    result[column.name] = {
      display: textColumn == column.name ? "full" : "badge",
      ...(styles[column.name] ?? {}),
    };
  }
  return result;
}

/**
 * Turn a user-specified state (from the API) into an internal state.
 * An InternalState is just a EmbeddingAtlasState with all fields non-optional
 * and clean.
 */
function resolveInternalState(state: EmbeddingAtlasState): InternalState {
  let layouts = state.layouts ?? {};
  if (Object.keys(layouts).length == 0 && Object.keys(state.charts ?? {}).length > 0) {
    layouts = {
      1: { type: "list", name: "Default", chartIds: Object.keys(state.charts ?? {}) },
    };
  }
  return cleanupState({
    version: EMBEDDING_ATLAS_VERSION,
    charts: state.charts ?? {},
    chartStates: state.chartStates ?? {},
    layouts: layouts,
    layoutOrder: state.layoutOrder ?? [],
    currentLayout: state.currentLayout ?? "unknown",
    columnStyles: state.columnStyles ?? {},
  });
}

/**
 * Remove orphaned entries from state. Returns the original object unchanged if no cleanup is needed.
 * - Charts not referenced by any layout's chartIds
 * - Chart states with no matching chart
 * - Layout order entries referencing non-existent layouts
 * - currentLayout pointing to a non-existent layout (reset to "default")
 */
function cleanupState(state: InternalState): InternalState {
  return produce(state, (draft) => {
    // Delete charts that are not referenced by any layout
    const referencedChartIds = new Set<string>();
    for (const layout of Object.values(draft.layouts)) {
      if (layout.chartIds) {
        for (const id of layout.chartIds) {
          referencedChartIds.add(id);
        }
      }
    }
    for (const id of Object.keys(draft.charts)) {
      if (!referencedChartIds.has(id)) {
        delete draft.charts[id];
      }
    }

    // Delete chart states that have no matching chart
    for (const id of Object.keys(draft.chartStates)) {
      if (!(id in draft.charts)) {
        delete draft.chartStates[id];
      }
    }

    // Cleanup layout order
    let resolvedOrder = draft.layoutOrder.filter((x) => x in draft.layouts);
    for (let id in draft.layouts) {
      if (resolvedOrder.indexOf(id) < 0) {
        resolvedOrder.push(id);
      }
    }
    if (!deepEquals(resolvedOrder, draft.layoutOrder)) {
      draft.layoutOrder = resolvedOrder;
    }

    // Set currentLayout if not set or set to non-existing layout
    draft.currentLayout =
      draft.currentLayout in draft.layouts ? draft.currentLayout : (Object.keys(draft.layouts)[0] ?? "undefined");
  });
}

export const [getStoreContext, setStoreContext] = createContext<EmbeddingAtlasStore>();
