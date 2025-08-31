<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator, isSelection, type MosaicClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { untrack } from "svelte";

  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { deepEquals, type Point, type Rectangle, type ViewportState } from "../utils.js";
  import type { EmbeddingViewMosaicProps } from "./embedding_view_mosaic_api.js";
  import {
    DataPointQuery,
    predicateForDataPoints,
    predicateForRangeSelection,
    queryApproximateDensity,
  } from "./mosaic_client.js";
  import { makeClient } from "./mosaic_helper.js";
  import type { DataPoint, DataPointID } from "./types.js";
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
    text = null,
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
    colorScheme = "light",
    theme = null,
    viewportState = null,
    automaticLabels = false,
    mode = "density",
    minimumDensity = 1 / 16,
    customTooltip = null,
    customOverlay = null,
    onViewportState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
  }: EmbeddingViewMosaicProps = $props();

  let xData: Float32Array = $state.raw(new Float32Array());
  let yData: Float32Array = $state.raw(new Float32Array());
  let categoryData: Uint8Array | null = $state.raw(null);
  let categoryCount: number = $state.raw(1);
  let totalCount: number = $state.raw(1);
  let maxDensity: number = $state.raw(1);
  let defaultViewportState: ViewportState | null = $state.raw(null);

  let effectiveTooltip: DataPoint | null = $state.raw(null);
  let effectiveSelection: DataPoint[] | null = $state.raw(null);
  let effectiveRangeSelection: Rectangle | Point[] | null = $state.raw(null);

  let clientId: any | null = $state.raw(null);

  $effect(() => {
    // Let Svelte track the dependencies.
    let deps = { coordinator: coordinator, source: { table, x, y, category } };

    let client: { destroy: () => void } | null = null;
    let didDestroy = false;

    async function initClient() {
      let source = deps.source;
      let approxDensity = await queryApproximateDensity(deps.coordinator, source);
      if (didDestroy) {
        return;
      }
      let scaler = approxDensity.scaler * 0.95; // shrink a bit so the point is not exactly on the edge.
      defaultViewportState = { x: approxDensity.centerX, y: approxDensity.centerY, scale: scaler };
      totalCount = approxDensity.totalCount;
      maxDensity = approxDensity.maxDensity;
      categoryCount = approxDensity.categoryCount;

      // A client is a thing that queries data from a selection with user-defined query
      client = makeClient({
        coordinator: deps.coordinator,
        selection: filter,
        query: (predicate) => {
          return SQL.Query.from(source.table)
            .select({
              x: SQL.sql`${SQL.column(source.x)}::FLOAT`,
              y: SQL.sql`${SQL.column(source.y)}::FLOAT`,
              ...(source.category != null ? { c: SQL.sql`${SQL.column(source.category)}::UTINYINT` } : {}),
            })
            .where(predicate);
        },
        queryResult: (data) => {
          let xArray = data.getChild("x").toArray();
          let yArray = data.getChild("y").toArray();
          let categoryArray = data.getChild("c")?.toArray() ?? null;
          // Ensure that the arrays are typed arrays.
          if (xArray != null && !(xArray instanceof Float32Array)) {
            xArray = new Float32Array(xArray);
          }
          if (yArray != null && !(yArray instanceof Float32Array)) {
            yArray = new Float32Array(yArray);
          }
          if (categoryArray != null && !(categoryArray instanceof Uint8Array)) {
            categoryArray = new Uint8Array(categoryArray);
          }
          xData = xArray;
          yData = yArray;
          categoryData = categoryArray;
          updateTooltip(null);
          updateSelection(null);
        },
      });
      (client as any).reset = () => {
        reset();
      };
      clientId = client;
    }

    initClient();

    return () => {
      clientId = null;
      didDestroy = true;
      client?.destroy();
    };
  });

  // Tooltip
  $effect(() => {
    if (isSelection(tooltip)) {
      let client = clientId;
      if (client == null) {
        return;
      }
      let captuerd = tooltip;
      effectiveTooltip = captuerd.value;
      let listener = () => {
        effectiveTooltip = captuerd.value;
      };

      $effect(() => {
        let value = effectiveTooltip;
        let source = { x, y, category, identifier };
        captuerd.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          predicate: value != null ? predicateForDataPoints(source, [value]) : null,
          value: value,
        });
      });

      captuerd.addEventListener("value", listener);
      return () => {
        captuerd.removeEventListener("value", listener);
        captuerd.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          value: null,
          predicate: null,
        });
      };
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
      let client = clientId;
      if (client == null) {
        return;
      }
      let captuerd = selection;
      effectiveSelection = captuerd.value;
      let listener = () => {
        effectiveSelection = captuerd.value;
      };

      $effect(() => {
        let value = effectiveSelection;
        let source = { x, y, category, identifier };
        captuerd.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          predicate: value != null ? predicateForDataPoints(source, value) : null,
          value: value,
        });
      });

      captuerd.addEventListener("value", listener);
      return () => {
        captuerd.removeEventListener("value", listener);
        captuerd.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          value: null,
          predicate: null,
        });
      };
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
    let client = clientId;
    if (client == null) {
      return;
    }
    let captured = rangeSelection;
    if (captured == null) {
      return;
    }

    $effect(() => {
      let value = effectiveRangeSelection;
      let source = { x, y };
      let clause = {
        source: client,
        clients: new Set<MosaicClient>().add(client),
        predicate: value != null ? predicateForRangeSelection(source, value) : null,
        value: value,
      };
      captured.update(clause);
      captured.activate(clause);
    });

    return () => {
      captured.update({
        source: client,
        clients: new Set<MosaicClient>().add(client),
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
    new DataPointQuery(coordinator, { table, x, y, category, text, identifier, additionalFields }),
  );

  async function querySelection(px: number, py: number, unitDistance: number): Promise<DataPoint | null> {
    return await pointQuery.queryClosestPoint(filter?.predicate?.(clientId), px, py, unitDistance);
  }

  async function queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    return await pointQuery.queryPoints(identifiers);
  }

  // Cluster Labels
  async function queryClusterLabels(clusters: Rectangle[][]): Promise<(string | null)[]> {
    if (text == null) {
      return clusters.map(() => null);
    }
    // Create text summarizer (in the worker)
    let summarizer = await textSummarizerCreate({ regions: clusters });
    // Add text data to the summarizer
    let start = 0;
    let chunkSize = 10000;
    let lastAdd: Promise<unknown> | null = null;
    while (true) {
      let r: any = await coordinator.query(
        SQL.Query.from(table)
          .select({ x: SQL.column(x), y: SQL.column(y), text: SQL.column(text) })
          .offset(start)
          .limit(chunkSize),
      );
      let data = {
        x: r.getChild("x").toArray(),
        y: r.getChild("y").toArray(),
        text: r.getChild("text").toArray(),
      };
      if (lastAdd != null) {
        await lastAdd;
      }
      lastAdd = textSummarizerAdd(summarizer, data);
      if (r.getChild("text").length < chunkSize) {
        break;
      }
      start += chunkSize;
    }
    if (lastAdd != null) {
      await lastAdd;
    }
    let summarizeResult = await textSummarizerSummarize(summarizer);
    await textSummarizerDestroy(summarizer);

    return summarizeResult.map((words) => {
      if (words.length == 0) {
        return null;
      } else if (words.length > 2) {
        return words.slice(0, 2).join("-") + "-\n" + words.slice(2).join("-");
      } else {
        return words.join("-");
      }
    });
  }
</script>

<EmbeddingViewImpl
  mode={mode ?? "points"}
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  colorScheme={colorScheme ?? "light"}
  theme={theme}
  data={{ x: xData, y: yData, category: categoryData }}
  totalCount={totalCount}
  maxDensity={maxDensity}
  categoryCount={categoryCount}
  categoryColors={categoryColors}
  defaultViewportState={defaultViewportState}
  querySelection={querySelection}
  queryClusterLabels={queryClusterLabels}
  automaticLabels={text != null ? (automaticLabels ?? false) : false}
  minimumDensity={minimumDensity ?? 1 / 16}
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
/>
