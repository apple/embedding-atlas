<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";
  import { cubicOut } from "svelte/easing";
  import { Tween } from "svelte/motion";
  import { slide } from "svelte/transition";

  import { maxDensityModeCategories } from "@embedding-atlas/component";
  import { Table } from "@embedding-atlas/table/svelte";
  import * as SQL from "@uwdata/mosaic-sql";
  import * as vg from "@uwdata/vgplot";

  import EmbeddingView from "./EmbeddingView.svelte";
  import PlotList from "./PlotList.svelte";
  import FilteredCount from "./plots/FilteredCount.svelte";
  import SearchResultList from "./SearchResultList.svelte";
  import Spinner from "./Spinner.svelte";
  import TextStylePicker from "./TextStylePicker.svelte";
  import ActionButton from "./widgets/ActionButton.svelte";
  import Button from "./widgets/Button.svelte";
  import Input from "./widgets/Input.svelte";
  import PopupButton from "./widgets/PopupButton.svelte";
  import Select from "./widgets/Select.svelte";
  import Slider from "./widgets/Slider.svelte";
  import ToggleButton from "./widgets/ToggleButton.svelte";

  import {
    IconDarkMode,
    IconDownload,
    IconEmbeddingView,
    IconExport,
    IconLightMode,
    IconMenu,
    IconSettings,
    IconTable,
  } from "./icons.js";

  import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "./api.js";
  import { EMBEDDING_ATLAS_VERSION } from "./constants.js";
  import { Context } from "./contexts.js";
  import { CustomOverlay, CustomTooltip } from "./custom_components.js";
  import { makeDarkModeStore } from "./dark_mode_store.js";
  import { TableInfo, type ColumnDesc, type EmbeddingLegend } from "./database_utils.js";
  import type { Plot } from "./plots/plot.js";
  import { PlotStateStoreManager } from "./plots/plot_state_store.js";
  import { makeCustomCellRenderers } from "./renderers/index.js";
  import { querySearchResultItems, resolveSearcher, type SearchResultItem } from "./search.js";
  import { tableTheme } from "./tableTheme";
  import { debounce, getDefaults, setDefaults, startDrag } from "./utils.js";

  const searchLimit = 500;

  const maxCategories = Math.min(20, maxDensityModeCategories());

  const animationDuration = 300;

  let {
    coordinator,
    table,
    idColumn,
    projectionColumns,
    neighborsColumn,
    textColumn,
    onExportApplication,
    onExportSelection,
    onStateChange,
    initialState,
    searcher: specifiedSearcher,
    cache,
    automaticLabels,
    colorScheme,
    tableCellRenderers,
  }: EmbeddingAtlasProps = $props();

  const { darkMode, userDarkMode } = makeDarkModeStore();

  Context.coordinator = coordinator;
  Context.darkMode = darkMode;

  $effect(() => {
    switch (colorScheme) {
      case "light":
        $userDarkMode = false;
        break;
      case "dark":
        $userDarkMode = true;
        break;
      case null:
        $userDarkMode = null;
        break;
    }
  });

  let initialized = $state(false);

  // View mode
  let showEmbedding: boolean = $state(projectionColumns != null);
  let showTable: boolean = $state(!(projectionColumns != null));
  let showSidebar: boolean = $state(true);

  let tableHeight: number = $state(320);
  let panelWidth: number = $state(400);

  const tableInfo = new TableInfo(coordinator, table);

  let embeddingViewMode: "points" | "density" = $state("points");
  let minimumDensityExpFactor: number = $state(0);
  let defaultViewportScale = $derived(
    projectionColumns != null ? tableInfo.defaultViewportScale(projectionColumns.x, projectionColumns.y) : null,
  );
  let automaticLabelsConfig = $derived(
    automaticLabels
      ? cache != null
        ? {
            cache: {
              get: (key: string) => cache.get("labels-" + key),
              set: (key: string, value: any) => cache.set("labels-" + key, value),
            },
          }
        : true
      : false,
  );

  let exportFormat: "json" | "jsonl" | "csv" | "parquet" = $state("parquet");

  const crossFilter = vg.Selection.crossfilter();

  function currentPredicate(): string | null {
    let predicate = crossFilter.predicate(null);
    if (predicate == null || predicate.length == 0) {
      return null;
    }
    if (typeof predicate == "string") {
      return predicate;
    }
    let joined = predicate.map((x: any) => x.toString()).join(" AND ");
    return joined;
  }

  let columns: ColumnDesc[] = $state.raw([]);
  let plots: Plot[] = $state.raw([]);
  let plotStateStores = new PlotStateStoreManager();

  let embeddingView: EmbeddingView | null = $state.raw(null);

  // let selection: any[] | null = $state.raw([]);
  let additionalFields = $derived(makeAdditionalFields(columns));

  let sidebarTween = new Tween(1, { duration: animationDuration, easing: cubicOut });

  $effect.pre(() => {
    sidebarTween.set(showSidebar ? 1 : 0);
  });

  // Text Renderers

  let textRenderers: Record<string, string> = $state(getDefaults("textRenderers", {}));
  let customCellRenderers = $derived(makeCustomCellRenderers(textRenderers, tableCellRenderers));
  $effect(() => {
    // Save the text renderers to a state.
    setDefaults("textRenderers", textRenderers);
  });

  // Search

  // Use a default searcher FullTextSearcher when searcher is not specified
  let searcher = $derived(
    resolveSearcher({
      coordinator,
      table,
      idColumn,
      textColumn,
      neighborsColumn,
      searcher: specifiedSearcher,
    }),
  );

  let allowFullTextSearch = $derived(searcher.fullTextSearch != null);
  let allowVectorSearch = $derived(searcher.vectorSearch != null);
  let allowNearestNeighborSearch = $derived(searcher.nearestNeighbors != null);
  let searchMode = $state<"full-text" | "vector">("full-text");
  let searchModeOptions = $derived([
    ...(allowFullTextSearch ? [{ label: "Full Text", value: "full-text" }] : []),
    ...(allowVectorSearch ? [{ label: "Vector", value: "vector" }] : []),
    ...(allowNearestNeighborSearch ? [{ label: "Neighbors", value: "neighbors" }] : []),
  ]);

  let searchQuery = $state("");
  let searcherStatus = $state("");
  let searchResultVisible = $state(false);
  let searchResult: {
    label: string;
    highlight: string;
    items: SearchResultItem[];
  } | null = $state(null);
  let searchResultHighlight = $state<SearchResultItem | null>(null);

  async function doSearch(query: any, mode: string) {
    if (searcher == null || searchModeOptions.length == 0) {
      clearSearch();
      return;
    }

    if (searchModeOptions.map((x) => x.value).indexOf(searchMode) < 0) {
      mode = searchModeOptions[0].value;
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
      table,
      { id: idColumn, x: projectionColumns?.x, y: projectionColumns?.y, text: textColumn },
      predicate,
      searcherResult,
    );

    searcherStatus = "";
    searchResult = { label: label, highlight: highlight, items: result };
  }

  const debouncedSearch = debounce(doSearch, 500);

  function clearSearch() {
    searchResult = null;
    searchResultVisible = false;
  }

  $effect.pre(() => {
    if (searchQuery == "") {
      clearSearch();
    } else {
      debouncedSearch(searchQuery, searchMode);
    }
  });

  // Category column

  let selectedCategoryColumn: string | null = $state(null);
  let selectedTooltipTextColumn: string | null = $state(null);
  let effectiveTooltipTextColumn = $derived(selectedTooltipTextColumn ?? textColumn);
  let tooltipTextRenderer = $derived(
    effectiveTooltipTextColumn != null ? (textRenderers[effectiveTooltipTextColumn] ?? "plain") : "plain",
  );
  let categoryLegend: EmbeddingLegend | null = $state.raw(null);

  async function setCategoryColumn(column: string | null) {
    if (column == null) {
      categoryLegend = null;
      return;
    }
    let candidate = columns.find((x) => x.name == column);
    if (candidate == null) {
      return;
    }
    let result;
    if (candidate.jsType == "string") {
      result = await tableInfo.makeCategoryColumn(candidate.name, 10);
    } else if (candidate.jsType == "number") {
      if (candidate.distinctCount <= 10) {
        result = await tableInfo.makeCategoryColumn(candidate.name, 10);
      } else {
        result = await tableInfo.makeBinnedNumericColumn(candidate.name);
      }
    } else {
      return;
    }
    categoryLegend = result;
    if (result.legend.length > maxCategories) {
      embeddingViewMode = "points";
    }
  }

  $effect.pre(() => {
    setCategoryColumn(selectedCategoryColumn);
  });

  // Animation

  async function animateEmbeddingViewToPoint(identifier?: any, x?: number, y?: number) {
    if (defaultViewportScale == null) {
      return;
    }
    let scale = (await defaultViewportScale) * 2;
    if (x == null || y == null) {
      if (projectionColumns == null) {
        return;
      }
      let result = await coordinator.query(
        SQL.Query.from(table)
          .select({
            x: SQL.column(projectionColumns.x),
            y: SQL.column(projectionColumns.y),
          })
          .where(SQL.eq(SQL.column(idColumn), SQL.literal(identifier))),
      );
      let item: { x: number; y: number } = result.get(0);
      x = item.x;
      y = item.y;
    }
    embeddingView?.startViewportAnimation({
      x: x,
      y: y,
      scale: scale,
    });
  }

  // Filter

  function resetFilter() {
    for (let item of crossFilter.clauses) {
      let source = item.source;
      source?.reset?.();
      crossFilter.update({ ...item, value: null, predicate: null });
    }
  }

  function makeAdditionalFields(columns: any) {
    let fields: any = {};
    fields.id = idColumn;
    for (let c of columns) {
      if (
        c.name == textColumn ||
        c.name == projectionColumns?.x ||
        c.name == projectionColumns?.y ||
        c.name == idColumn
      ) {
        continue;
      }

      fields[c.name] = c.name;
    }
    return fields;
  }

  let tableScrollTo: any | null = $state(null);
  const scrollTableTo = (identifier: any) => {
    tableScrollTo = identifier;
  };

  function loadState(state: EmbeddingAtlasState) {
    if (typeof state.version != "string") {
      return;
    }
    // Set plot states
    plotStateStores.set(state.plotStates ?? {});

    // Load the spec
    function load(key: string, setter: (value: any) => void) {
      if (state.view && key in state.view) {
        setter(state.view[key]);
      }
    }
    load("showEmbedding", (x) => (showEmbedding = x));
    load("showTable", (x) => (showTable = x));
    load("showSidebar", (x) => (showSidebar = x));
    load("textRenderers", (x) => (textRenderers = x));
    load("selectedCategoryColumn", (x) => (selectedCategoryColumn = x));
    load("embeddingViewMode", (x) => (embeddingViewMode = x));
    load("minimumDensityExpFactor", (x) => (minimumDensityExpFactor = x));
    load("userDarkMode", (x) => ($userDarkMode = x));

    if (state.plots != null) {
      plots = state.plots;
    }
  }

  // Emit onStateChange event.
  $effect(() => {
    if (!initialized) {
      return;
    }

    let state: EmbeddingAtlasState = {
      version: EMBEDDING_ATLAS_VERSION,
      timestamp: new Date().getTime() / 1000,
      view: {
        showEmbedding: showEmbedding,
        showTable: showTable,
        showSidebar: showSidebar,
        textRenderers: textRenderers,
        selectedCategoryColumn: selectedCategoryColumn,
        embeddingViewMode: embeddingViewMode,
        minimumDensityExpFactor: minimumDensityExpFactor,
        userDarkMode: $userDarkMode,
      },
      plots: plots,
      plotStates: $plotStateStores,
      predicate: currentPredicate(),
    };
    onStateChange?.(state);
  });

  // Load initial state.
  if (initialState) {
    loadState(initialState);
  }

  onMount(async () => {
    let ignoreColumns = [idColumn, textColumn, projectionColumns?.x, projectionColumns?.y].filter((x) => x != null);
    columns = (await tableInfo.columnDescriptions()).filter((x) => !x.name.startsWith("__"));
    if (plots.length == 0) {
      plots = await tableInfo.defaultPlots(columns.filter((x) => ignoreColumns.indexOf(x.name) < 0));
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
</script>

<div class="embedding-atlas-root" style:width="100%" style:height="100%">
  <div
    class="w-full h-full flex flex-col text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
    class:dark={$darkMode}
  >
    <div class="m-2 flex flex-row justify-between items-center">
      <div class="flex flex-row flex-1 justify-between">
        <div class="flex flex-row items-center">
          <div class="flex-1">
            {#if searcher}
              <div class="relative">
                <Input type="search" placeholder="Search..." className="w-64" bind:value={searchQuery} />
                {#if searchModeOptions.filter((x) => x.value != "neighbors").length > 1}
                  <Select
                    options={searchModeOptions.filter((x) => x.value != "neighbors")}
                    value={searchMode}
                    onChange={(v) => (searchMode = v)}
                  />
                {/if}

                {#if searchResultVisible}
                  <div
                    class="absolute w-96 left-0 top-[32px] rounded-md right-0 z-20 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 overflow-hidden resize shadow-lg"
                    style:height="48em"
                  >
                    {#if searchResult != null}
                      <SearchResultList
                        items={searchResult.items}
                        label={searchResult.label}
                        highlight={searchResult.highlight}
                        limit={searchLimit}
                        onClick={async (item) => {
                          scrollTableTo(item.id);
                          searchResultHighlight = item;
                          animateEmbeddingViewToPoint(item.id, item.x, item.y);
                        }}
                        onClose={clearSearch}
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
        </div>
        <div class="flex flex-row items-center gap-3">
          {#if showEmbedding}
            <Select
              label="Color"
              value={selectedCategoryColumn}
              onChange={(v) => (selectedCategoryColumn = v)}
              options={[
                { value: null, label: "(none)" },
                ...columns
                  .filter(
                    (c) =>
                      c.distinctCount > 1 &&
                      ((c.jsType == "string" && c.distinctCount <= 10000) || c.jsType == "number"),
                  )
                  .map((c) => ({ value: c.name, label: `${c.name} (${c.type})` })),
              ]}
            />
            <Select
              label="Display"
              value={embeddingViewMode}
              onChange={(v) => (embeddingViewMode = v)}
              disabled={categoryLegend != null && categoryLegend.legend.length > maxCategories}
              options={[
                { value: "points", label: "Points" },
                { value: "density", label: "Density" },
              ]}
            />
            {#if embeddingViewMode == "density"}
              <div class="select-none flex items-center gap-2">
                <span class="text-slate-500 dark:text-slate-400">Threshold</span>
                <Slider bind:value={minimumDensityExpFactor} min={-4} max={4} step={0.1} />
              </div>
            {/if}
          {/if}
        </div>
      </div>
      <div
        class="relative h-full"
        style:--sidebar-tween={sidebarTween.current}
        style:--padded-width="calc({panelWidth}px - 149.5px - {$darkMode ? 0 : 1}px)"
        style:max-width="var(--padded-width)"
        style:flex-basis="calc(var(--padded-width) * var(--sidebar-tween))"
      >
        <div
          class="absolute left-0 right-0 top-0 bottom-0 overflow-hidden transition-opacity"
          style:opacity={showSidebar ? 1 : 0}
        >
          <div class="flex h-full gap-2 items-center justify-end whitespace-nowrap">
            <FilteredCount filter={crossFilter} table={table} />
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
        </div>
      </div>
      <div class="w-3"></div>
      <div class="flex flex-row gap-0.5">
        <PopupButton icon={IconSettings} title="Options">
          <div class="min-w-96">
            <!-- Tooltip settings -->
            <h4 class="text-slate-500 dark:text-slate-400 mb-2 select-none">Tooltip</h4>
            <Select
              class="w-full"
              value={selectedTooltipTextColumn}
              onChange={(v) => (selectedTooltipTextColumn = v)}
              options={[
                { value: null, label: textColumn ?? "(none)" },
                "---",
                ...Object.keys(additionalFields).map((c) => ({ value: c, label: c })),
              ]}
            />
            {#if columns.length > 0}
              <!-- Text style settings -->
              <h4 class="text-slate-500 dark:text-slate-400 my-2 select-none">Text Style</h4>
              <TextStylePicker
                columns={columns}
                styles={textRenderers}
                onStylesChange={(value) => {
                  textRenderers = value;
                }}
              />
            {/if}
            <!-- Export -->
            <h4 class="text-slate-500 dark:text-slate-400 my-2 select-none">Export</h4>
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
            <h4 class="text-slate-500 dark:text-slate-400 my-2 select-none">About</h4>
            <div>Embedding Atlas, {EMBEDDING_ATLAS_VERSION}</div>
          </div>
        </PopupButton>
        {#if colorScheme == null}
          <Button
            icon={$darkMode ? IconLightMode : IconDarkMode}
            title="Toggle dark mode"
            onClick={() => {
              $userDarkMode = !$darkMode;
            }}
          />
        {/if}
        {#if projectionColumns != null}
          <ToggleButton icon={IconEmbeddingView} title="Show / hide embedding" bind:checked={showEmbedding} />
        {/if}
        <ToggleButton icon={IconTable} title="Show / hide table" bind:checked={showTable} />
        <ToggleButton icon={IconMenu} title="Show / hide sidebar" bind:checked={showSidebar} />
      </div>
    </div>
    <div class="flex flex-row overflow-hidden h-full">
      {#if showTable || showEmbedding}
        <div class="flex-1 flex flex-col mt-0 ml-2 mb-2 mr-2 overflow-hidden">
          {#if showEmbedding && projectionColumns != null}
            <div class="flex-1 relative bg-white dark:bg-black rounded-md overflow-hidden">
              <EmbeddingView
                bind:this={embeddingView}
                table={table}
                filter={crossFilter}
                id={idColumn}
                x={projectionColumns.x}
                y={projectionColumns.y}
                text={textColumn}
                additionalFields={additionalFields}
                categoryLegend={categoryLegend}
                mode={embeddingViewMode}
                minimumDensityExpFactor={minimumDensityExpFactor}
                automaticLabels={automaticLabelsConfig}
                customTooltip={{
                  class: CustomTooltip,
                  props: {
                    textRenderer: tooltipTextRenderer,
                    textField: selectedTooltipTextColumn,
                    darkMode: $darkMode,
                    onNearestNeighborSearch: allowNearestNeighborSearch
                      ? async (id: any) => {
                          doSearch(id, "neighbors");
                        }
                      : null,
                  },
                }}
                customOverlay={searchResult
                  ? { class: CustomOverlay, props: { items: searchResult.items, highlightItem: searchResultHighlight } }
                  : null}
                onClickPoint={(p) => scrollTableTo(p.identifier)}
                stateStore={plotStateStores.store("embedding-view")}
              />
            </div>
          {/if}
          {#if showTable}
            {#if showEmbedding}
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="h-2 cursor-row-resize"
                onmousedown={(e1) => {
                  let h0 = tableHeight;
                  startDrag(e1, (_, dy) => (tableHeight = Math.max(60, h0 - dy)));
                }}
              ></div>
            {/if}
            <div
              class="z-10 bg-white dark:bg-slate-900 rounded-md overflow-hidden"
              style:height={showEmbedding ? tableHeight + "px" : null}
              class:h-full={!showEmbedding}
              transition:slide={{ duration: animationDuration }}
              style:--hover-color="var(--color-amber-200)"
            >
              {#if columns.length > 0}
                {#key columns}
                  <Table
                    coordinator={coordinator}
                    table={table}
                    rowKey={idColumn}
                    columns={columns.map((x) => x.name)}
                    filter={crossFilter}
                    scrollTo={tableScrollTo}
                    onRowClick={async (identifier) => {
                      await animateEmbeddingViewToPoint(identifier);
                      embeddingView?.showTooltip(identifier);
                    }}
                    numLines={3}
                    colorScheme={$darkMode ? "dark" : "light"}
                    theme={tableTheme}
                    customCells={customCellRenderers}
                    highlightHoveredRow={true}
                  />
                {/key}
              {/if}
            </div>
          {/if}
        </div>
      {/if}
      {#if showSidebar}
        {@const fullWidth = !(showTable || showEmbedding)}
        {#if !fullWidth}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class="w-2 -ml-2 cursor-col-resize"
            onmousedown={(e) => {
              let w0 = panelWidth;
              startDrag(e, (dx, _) => (panelWidth = Math.max(300, w0 - dx)));
            }}
          ></div>
        {/if}
        <div
          class="flex flex-col mr-2 mb-2 dark:bg-slate-800"
          style:width={fullWidth ? null : `${panelWidth}px`}
          class:ml-2={fullWidth}
          class:flex-none={!fullWidth}
          class:flex-1={fullWidth}
          transition:slide={{ axis: "x", duration: animationDuration }}
        >
          <div
            class="w-full rounded-md overflow-x-hidden overflow-y-scroll"
            style:width={fullWidth ? null : `${panelWidth}px`}
          >
            <PlotList
              bind:plots={plots}
              table={table}
              columns={columns}
              filter={crossFilter}
              layout={fullWidth ? "full" : "sidebar"}
              stateStores={plotStateStores}
            />
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
<svelte:window onkeydown={onWindowKeydown} />
