<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { debounce } from "@embedding-atlas/utils";
  import { Selection } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";
  import { writable } from "svelte/store";
  import { scale } from "svelte/transition";

  import LayoutOptionsView from "./layouts/LayoutOptionsView.svelte";
  import LayoutView from "./layouts/LayoutView.svelte";
  import ColumnStylePicker from "./views/ColumnStylePicker.svelte";
  import FilteredCount from "./views/FilteredCount.svelte";
  import SearchResultList from "./views/SearchResultList.svelte";
  import ActionButton from "./widgets/ActionButton.svelte";
  import Button from "./widgets/Button.svelte";
  import Input from "./widgets/Input.svelte";
  import PopupButton from "./widgets/PopupButton.svelte";
  import SegmentedControl from "./widgets/SegmentedControl.svelte";
  import Select from "./widgets/Select.svelte";
  import Spinner from "./widgets/Spinner.svelte";

  import {
    IconDarkMode,
    IconDashboardLayout,
    IconDownload,
    IconExport,
    IconLightMode,
    IconListLayout,
    IconSettings,
  } from "./assets/icons.js";

  import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "./api.js";
  import { ChartContextCache, type ChartContext, type RowID } from "./charts/chart.js";
  import { type ChartThemeConfig } from "./charts/common/theme.js";
  import { defaultCharts } from "./charts/default_charts.js";
  import { EMBEDDING_ATLAS_VERSION } from "./constants.js";
  import { type ColumnStyle } from "./renderers/index.js";
  import { querySearchResultItems, resolveSearcher, type SearchResultItem } from "./search/search.js";
  import { makeColorSchemeStore } from "./utils/color_scheme.js";
  import { columnDescriptions, predicateToString, type ColumnDesc } from "./utils/database.js";

  const searchLimit = 500;

  let {
    coordinator,
    data,
    initialState,
    searcher: specifiedSearcher,
    defaultChartsConfig,
    embeddingViewConfig = null,
    embeddingViewLabels = null,
    chartTheme,
    colorScheme: colorSchemeProp,
    tableCellRenderers,
    onExportApplication,
    onExportSelection,
    onStateChange,
    cache,
  }: EmbeddingAtlasProps = $props();

  const { colorScheme, userColorScheme } = makeColorSchemeStore();

  $effect.pre(() => {
    $userColorScheme = colorSchemeProp;
  });

  let initialized = $state(false);

  let exportFormat: "json" | "jsonl" | "csv" | "parquet" = $state("parquet");

  const crossFilter = Selection.crossfilter();

  function currentPredicate(): string | null {
    return predicateToString(crossFilter.predicate(null));
  }

  let columns: ColumnDesc[] = $state.raw([]);

  // Column styles
  let columnStyles = $state.raw<Record<string, ColumnStyle>>({});
  let resolvedColumnStyles = writable<Record<string, ColumnStyle>>({});
  $effect.pre(() => {
    let resolved = resolveColumnStyles(columns, columnStyles);
    resolvedColumnStyles.set(resolved);
  });

  function resolveColumnStyles(
    columns: ColumnDesc[],
    styles: Record<string, ColumnStyle>,
  ): Record<string, ColumnStyle> {
    let result: Record<string, ColumnStyle> = {};
    for (let column of columns) {
      let style = styles[column.name];
      if (style == null) {
        // Default display style
        style = { display: data.text == column.name ? "full" : "badge" };
      }
      result[column.name] = style;
    }
    return result;
  }

  // Search

  // Use a default searcher FullTextSearcher when searcher is not specified
  let searcher = resolveSearcher({
    coordinator,
    table: data.table,
    idColumn: data.id,
    textColumn: data.text,
    neighborsColumn: data.neighbors,
    searcher: specifiedSearcher,
  });

  let searchModes = [
    ...(searcher.fullTextSearch != null ? ["full-text"] : []),
    ...(searcher.vectorSearch != null ? ["vector"] : []),
    ...(searcher.nearestNeighbors != null ? ["neighbors"] : []),
  ];

  const searchModeOptions: Record<string, { value: string; label: string }> = {
    "full-text": { value: "full-text", label: "Full Text" },
    vector: { value: "vector", label: "Vector" },
    neighbors: { value: "neighbors", label: "Neighbors" },
  };

  let searchMode = $state<"full-text" | "vector">("full-text");

  let searchQuery = $state("");
  let searcherStatus = $state("");
  let searchResultVisible = $state(false);
  let searchResult: {
    label: string;
    highlight: string;
    items: SearchResultItem[];
  } | null = $state.raw(null);
  let searchResultStore = writable<{ query: any; mode: string; ids: RowID[] } | null>(null);

  async function doSearch(query: any, mode: string) {
    if (searcher == null || searchModes.indexOf(mode) < 0) {
      clearSearch();
      return;
    }

    searchResultVisible = true;
    searcherStatus = "Searching...";

    let predicate = currentPredicate();
    let searcherResult: { id: any }[] = [];
    let highlight: string = "";
    let label = query.toString();

    if (mode == "full-text" && searcher.fullTextSearch != null) {
      query = query.trim();
      searcherResult = await searcher.fullTextSearch(query, {
        limit: searchLimit,
        predicate: predicate,
        onStatus: (status: string) => {
          searcherStatus = status;
        },
      });
      highlight = query;
    } else if (mode == "vector" && searcher.vectorSearch != null) {
      query = query.trim();
      searcherResult = await searcher.vectorSearch(query, {
        limit: searchLimit,
        predicate: predicate,
        onStatus: (status: string) => {
          searcherStatus = status;
        },
      });
      highlight = query;
    } else if (mode == "neighbors" && searcher.nearestNeighbors != null) {
      label = "Neighbors of #" + query.toString();
      searcherResult = await searcher.nearestNeighbors(query, {
        limit: searchLimit,
        predicate: predicate,
        onStatus: (status: string) => {
          searcherStatus = status;
        },
      });
    }

    // Apply predicate in case the searcher does not handle predicate.
    // And convert the search result ids to tuples.
    let result = await querySearchResultItems(
      coordinator,
      data.table,
      { id: data.id, x: data.projection?.x, y: data.projection?.y, text: data.text },
      Object.fromEntries(columns.map((c) => [c.name, c.name])),
      predicate,
      searcherResult,
    );

    searcherStatus = "";
    searchResult = { label: label, highlight: highlight, items: result };
    searchResultStore.set({ query: query, mode: mode, ids: result.map((x) => x.id) });
  }

  const debouncedSearch = debounce(doSearch, 500);

  function clearSearch() {
    searchResult = null;
    searchResultStore.set(null);
    searchResultVisible = false;
  }

  $effect.pre(() => {
    if (searchQuery == "") {
      clearSearch();
    } else {
      debouncedSearch(searchQuery, searchMode);
    }
  });

  // Filter

  function resetFilter() {
    for (let item of crossFilter.clauses) {
      let source = item.source;
      source?.reset?.();
      crossFilter.update({ ...item, value: null, predicate: null });
    }
  }

  function loadState(state: EmbeddingAtlasState) {
    if (typeof state.version != "string") {
      return;
    }
    charts = state.charts ?? {};
    chartStates = state.chartStates ?? {};
    layout = state.layout ?? "list";
    layoutStates = state.layoutStates ?? {};
  }

  // Emit onStateChange event.
  $effect(() => {
    if (!initialized) {
      return;
    }
    let state: EmbeddingAtlasState = {
      version: EMBEDDING_ATLAS_VERSION,
      timestamp: new Date().getTime() / 1000,
      charts: charts,
      chartStates: chartStates,
      layout: layout,
      layoutStates: layoutStates,
      predicate: currentPredicate(),
    };
    onStateChange?.(state);
  });

  onMount(async () => {
    columns = (await columnDescriptions(coordinator, data.table)).filter((x) => !x.name.startsWith("__"));
    chartContext.columns = columns;

    if (initialState) {
      loadState(initialState);
    }
    if (Object.keys(charts).length == 0) {
      let newCharts = await defaultCharts({
        coordinator,
        table: data.table,
        id: data.id,
        projection: data.projection
          ? {
              ...data.projection,
              text: data.text ?? undefined,
            }
          : undefined,
        config: defaultChartsConfig ?? undefined,
      });
      charts = Object.fromEntries(newCharts.map((spec, i) => [`${i + 1}`, spec]));
    }

    initialized = true;
  });

  function onWindowKeydown(e: KeyboardEvent) {
    if (e.key == "Escape") {
      resetFilter();
      e.preventDefault();
      try {
        let active: any = document.activeElement;
        active?.blur?.();
      } catch (e) {}
    }
  }

  let chartThemeStore = writable<ChartThemeConfig | undefined>(chartTheme ?? undefined);
  $effect.pre(() => {
    chartThemeStore.set(chartTheme ?? undefined);
  });

  let chartContext: ChartContext = {
    coordinator: coordinator,
    filter: crossFilter,
    table: data.table,
    id: data.id,
    columns: [],
    colorScheme: colorScheme,
    theme: chartThemeStore,
    columnStyles: resolvedColumnStyles,
    cache: new ChartContextCache(),
    persistentCache: cache ?? { get: async () => null, set: async (key, value) => {} },
    searchModes: searchModes,
    search: doSearch,
    searchResult: searchResultStore,
    highlight: writable(null),
    embeddingViewConfig: embeddingViewConfig,
    embeddingViewLabels: embeddingViewLabels,
    tableCellRenderers: tableCellRenderers,
  };

  let charts = $state.raw<Record<string, any>>({});
  let chartStates = $state.raw<Record<string, any>>({});
  let layout = $state.raw("list");
  let layoutStates = $state.raw<Record<string, any>>({});
</script>

<div class="embedding-atlas-root" style:width="100%" style:height="100%">
  <div
    class="w-full h-full flex flex-col text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
    class:dark={$colorScheme == "dark"}
    style:color-scheme={$colorScheme}
  >
    <!-- Toolbar -->
    <div class="m-2 flex flex-row items-center gap-2 flex-wrap">
      {#if initialized}
        <!-- Left side -->
        <div class="flex flex-row flex-1 justify-between min-w-[180px]">
          {#if searchMode.length > 0}
            <div class="relative w-full">
              <Input type="search" placeholder="Search..." className="w-full max-w-[400px] " bind:value={searchQuery} />
              {#if searchModes.filter((x) => x != "neighbors").length > 1}
                <Select
                  options={searchModes.filter((x) => x != "neighbors").map((x) => searchModeOptions[x])}
                  value={searchMode}
                  onChange={(v) => (searchMode = v)}
                />
              {/if}

              {#if searchResultVisible}
                <div
                  class="absolute w-96 left-0 top-[32px] rounded-md right-0 z-20 border border-slate-300 dark:border-slate-600 overflow-hidden resize shadow-lg bg-white/75 dark:bg-slate-800/75 backdrop-blur-sm"
                  style:height="48em"
                >
                  {#if searchResult != null}
                    <SearchResultList
                      items={searchResult.items}
                      label={searchResult.label}
                      highlight={searchResult.highlight}
                      limit={searchLimit}
                      onClick={async (item) => {
                        chartContext.highlight.set(item.id);
                      }}
                      onClose={clearSearch}
                      columnStyles={$resolvedColumnStyles}
                    />
                  {:else if searcherStatus != null}
                    <div class="p-2">
                      <Spinner status={searcherStatus} />
                    </div>
                  {/if}
                </div>
              {/if}
            </div>
          {:else}
            <div class="text-slate-500 dark:text-slate-400">Embedding Atlas</div>
          {/if}
        </div>
        <!-- Right side -->
        <div class="flex flex-none gap-2 items-center">
          <FilteredCount coordinator={coordinator} filter={crossFilter} table={data.table} />
          <div class="flex flex-row gap-1 items-center">
            <button
              class="flex px-2.5 mr-1 select-none items-center justify-center text-slate-500 dark:text-slate-300 rounded-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 focus-visible:outline-2 outline-blue-600 -outline-offset-1"
              onclick={resetFilter}
              title="Clear filters"
            >
              Clear
            </button>
          </div>
        </div>
        <div class="flex flex-none flex-row gap-0.5">
          <div class="grid grid-cols-1 grid-rows-1 justify-items-end items-center">
            {#key layout}
              <div transition:scale class="col-start-1 row-start-1">
                <LayoutOptionsView
                  context={chartContext}
                  charts={charts}
                  chartStates={chartStates}
                  layout={layout}
                  layoutStates={layoutStates}
                  onChartsChange={(v) => (charts = v)}
                  onChartStatesChange={(v) => (chartStates = v)}
                  onLayoutStatesChange={(v) => (layoutStates = v)}
                />
              </div>
            {/key}
          </div>
          <SegmentedControl
            value={layout}
            onChange={(v) => (layout = v)}
            options={[
              { value: "list", icon: IconListLayout, title: "List layout" },
              { value: "dashboard", icon: IconDashboardLayout, title: "Dashboard layout" },
            ]}
          />
          {#if colorSchemeProp == null}
            <Button
              icon={$colorScheme == "dark" ? IconLightMode : IconDarkMode}
              title="Toggle light / dark mode"
              onClick={() => {
                $userColorScheme = $colorScheme == "light" ? "dark" : "light";
              }}
            />
          {/if}
          <PopupButton icon={IconSettings} title="Options">
            <div class="min-w-[420px] flex flex-col gap-2">
              <!-- Text style settings -->
              {#if columns.length > 0}
                <h4 class="text-slate-500 dark:text-slate-400 select-none">Column Styles</h4>
                <ColumnStylePicker
                  columns={columns}
                  styles={$resolvedColumnStyles}
                  onStylesChange={(value) => {
                    columnStyles = value;
                  }}
                />
              {/if}
              <!-- Export -->
              {#if onExportSelection || onExportApplication}
                <h4 class="text-slate-500 dark:text-slate-400 select-none">Export</h4>
                <div class="flex flex-col gap-2">
                  {#if onExportSelection}
                    <div class="flex flex-row gap-2">
                      <ActionButton
                        icon={IconExport}
                        label="Export Selection"
                        title="Export the selected points"
                        class="w-48"
                        onClick={() => onExportSelection(currentPredicate(), exportFormat)}
                      />
                      <Select
                        label="Format"
                        value={exportFormat}
                        onChange={(v) => (exportFormat = v)}
                        options={[
                          { value: "parquet", label: "Parquet" },
                          { value: "jsonl", label: "JSONL" },
                          { value: "json", label: "JSON" },
                          { value: "csv", label: "CSV" },
                        ]}
                      />
                    </div>
                  {/if}
                  {#if onExportApplication}
                    <ActionButton
                      icon={IconDownload}
                      label="Export Application"
                      title="Download a self-contained static web application"
                      class="w-48"
                      onClick={onExportApplication}
                    />
                  {/if}
                </div>
              {/if}
              <h4 class="text-slate-500 dark:text-slate-400 select-none">About</h4>
              <div>Embedding Atlas, {EMBEDDING_ATLAS_VERSION}</div>
            </div>
          </PopupButton>
        </div>
      {/if}
    </div>
    <!-- Main Content -->
    <div class="flex-1 overflow-hidden h-full ml-2 mr-2 mb-2">
      {#if initialized}
        <LayoutView
          context={chartContext}
          layout={layout}
          layoutStates={layoutStates}
          charts={charts}
          chartStates={chartStates}
          onChartsChange={(v) => (charts = v)}
          onChartStatesChange={(v) => (chartStates = v)}
          onLayoutStatesChange={(v) => (layoutStates = v)}
        />
      {/if}
    </div>
  </div>
</div>
<svelte:window onkeydown={onWindowKeydown} />
