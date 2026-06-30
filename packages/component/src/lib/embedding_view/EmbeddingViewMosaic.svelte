<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { imageToDataUrl } from "@embedding-atlas/utils";
  import { coordinator as defaultCoordinator, isSelection, makeClient, type MosaicClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { untrack } from "svelte";

  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { cameraFromBounds3D, pointsBounds3D, type Bounds3D, type Camera3DState } from "../camera3d.js";
  import { deepEquals, type Point, type Rectangle, type ViewportState } from "../utils.js";
  import type { EmbeddingViewMosaicProps } from "./embedding_view_mosaic_api.js";
  import { IMAGE_LABEL_SIZE } from "./labels.js";
  import {
    DataPointQuery,
    predicateForDataPoints,
    predicateForRangeSelection,
    queryApproximateDensity,
  } from "./mosaic_client.js";
  import type { DataPoint, DataPointID, LabelContent } from "./types.js";
  import {
    textSummarizerAdd,
    textSummarizerCreate,
    textSummarizerDestroy,
    textSummarizerSummarize,
  } from "./worker/index.js";

  /** Returns `array` as an instance of `Ctor`, wrapping (copying) only if needed. */
  function asTypedArray<T>(array: any, Ctor: new (source: any) => T): T | null {
    if (array == null) {
      return null;
    }
    return array instanceof Ctor ? array : new Ctor(array);
  }

  // Row cap for materializing a rowid-less source's identifier column to enable exact
  // 3D picks (see initClient). Bounds browser memory: at this many rows even ~64-byte
  // string/UUID identifiers stay in the tens-of-MB range; larger sources use the
  // coordinate fallback instead. Base tables are unaffected (compact rowid, any size).
  const IDENTIFIER_MATERIALIZE_MAX_ROWS = 500_000;

  let {
    coordinator = defaultCoordinator(),
    table,
    x,
    y,
    z = null,
    category = null,
    text = null,
    image = null,
    importance = null,
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
    config = null,
    theme = null,
    viewportState = null,
    camera3DState = null,
    labels = null,
    customTooltip = null,
    customOverlay = null,
    onViewportState = null,
    onCamera3DState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    cache = null,
  }: EmbeddingViewMosaicProps = $props();

  let xData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let yData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let zData: Float32Array<ArrayBuffer> | null = $state.raw(null);
  let categoryData: Uint8Array<ArrayBuffer> | null = $state.raw(null);
  // Compact stable row identity (DuckDB `rowid`, a BIGINT), row-aligned with the
  // rendered arrays, used to resolve a 3D pick to the EXACT row regardless of the
  // user identifier's type (string/UUID included) — without downloading per-row
  // identifiers. Null when the source has no usable rowid (e.g. a view); those fall
  // back to the configured identifier (below) or the coordinate lookup. See
  // queryByIndex / the bulk query below.
  let rowidData: BigInt64Array | null = $state.raw(null);
  // Fallback exact identity for sources WITHOUT a rowid (views/joins) that DO have a
  // configured identifier column: the per-row identifier values, row-aligned with the
  // rendered arrays, so a 3D pick resolves to the exact row even when several rows
  // share rendered coordinates. Null when rowid is available (rowid is preferred and
  // compact) or no identifier is configured.
  let identifierData: DataPointID[] | null = $state.raw(null);
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
    let deps = { coordinator: coordinator, source: { table, x, y, z, category, identifier } };

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

      // Decide ONCE (per dataset) whether the source exposes a usable `rowid`
      // pseudocolumn (true for base tables, false for most views/joins). When it does,
      // the bulk query carries a compact rowid per point so 3D picks resolve to the
      // exact row for ANY identifier type; otherwise picks fall back to coordinates.
      let rowidAvailable = false;
      if (source.z != null) {
        try {
          await deps.coordinator.query(
            SQL.Query.from(source.table)
              .select({ __probe: SQL.sql`rowid` })
              .limit(1),
          );
          rowidAvailable = true;
        } catch {
          rowidAvailable = false;
        }
        // Check AFTER the probe settles either way: a rejected probe (views/joins
        // without rowid) must honor a teardown that happened during the await too,
        // otherwise makeClient below would build a stale client that is never
        // destroyed and could write obsolete data into the next view.
        if (didDestroy) {
          return;
        }
      }

      // Exact 3D picks for rowid-LESS sources (views/joins) need a per-row identity, so
      // we carry the configured identifier row-aligned with the rendered points. But
      // that materializes the WHOLE identifier column in the browser (potentially long
      // strings/UUIDs), so only do it when the source is small enough to stay bounded;
      // above the cap, picks use the (bounded) coordinate fallback. Base tables are
      // unaffected — they use the compact 8-byte rowid regardless of size. Gated on the
      // unfiltered total (known here, pre-query) so the heavy column is never even
      // requested for large sources.
      let identifierColumn =
        source.z != null &&
        !rowidAvailable &&
        source.identifier != null &&
        approxDensity.totalCount <= IDENTIFIER_MATERIALIZE_MAX_ROWS
          ? source.identifier
          : null;

      // A client is a thing that queries data from a selection with user-defined query
      client = makeClient({
        coordinator: deps.coordinator,
        selection: filter ?? undefined,
        query: (predicate) => {
          return SQL.Query.from(source.table)
            .select({
              x: SQL.sql`${SQL.column(source.x)}::FLOAT`,
              y: SQL.sql`${SQL.column(source.y)}::FLOAT`,
              ...(source.z != null ? { z: SQL.sql`${SQL.column(source.z)}::FLOAT` } : {}),
              ...(source.category != null ? { c: SQL.sql`${SQL.column(source.category)}::UTINYINT` } : {}),
              // 3D pick identity: a COMPACT stable rowid per point (when the source
              // supports it) so picks resolve to the exact row for any identifier type,
              // without downloading per-row (string) identifiers.
              ...(source.z != null && rowidAvailable ? { __rowid: SQL.sql`rowid` } : {}),
              // No rowid (views/joins) but an identifier IS configured and the source is
              // small enough (identifierColumn): carry the identifier per point so a 3D
              // pick still resolves to the EXACT row even when several rows share
              // rendered coordinates.
              ...(identifierColumn != null ? { __identifier: SQL.column(identifierColumn) } : {}),
            })
            .where(predicate);
        },
        queryResult: (data: any) => {
          // Ensure each column is the typed array the renderers expect.
          xData = asTypedArray(data.getChild("x").toArray(), Float32Array)!;
          yData = asTypedArray(data.getChild("y").toArray(), Float32Array)!;
          zData = asTypedArray(data.getChild("z")?.toArray() ?? null, Float32Array);
          categoryData = asTypedArray(data.getChild("c")?.toArray() ?? null, Uint8Array);
          // rowid is row-aligned with x/y/z; keep it only when present and complete
          // (no nulls) so a pick maps cleanly to a row.
          let rowidChild = data.getChild("__rowid");
          rowidData =
            rowidChild != null && rowidChild.length > 0 && rowidChild.nullCount === 0
              ? (rowidChild.toArray() as BigInt64Array)
              : null;
          // Same for the row-aligned identifier fallback (rowid-less sources). Require
          // no nulls so every rendered point maps to an exact row; otherwise drop to
          // the coordinate lookup rather than resolve an ambiguous/partial identity.
          let identifierChild = data.getChild("__identifier");
          identifierData =
            identifierChild != null && identifierChild.length > 0 && identifierChild.nullCount === 0
              ? (Array.from(identifierChild.toArray()) as DataPointID[])
              : null;
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
      let captured = tooltip;
      effectiveTooltip = (captured.valueFor(client) ?? null) as any;
      let listener = () => {
        effectiveTooltip = (captured.valueFor(client) ?? null) as any;
      };

      $effect(() => {
        let value = effectiveTooltip;
        let source = { x, y, z, category, identifier };
        captured.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          predicate: value != null ? predicateForDataPoints(source, [value]) : null,
          value: value,
        });
      });

      captured.addEventListener("value", listener);
      return () => {
        captured.removeEventListener("value", listener);
        captured.update({
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
      let captured = selection;
      effectiveSelection = (captured.valueFor(client) ?? null) as any;
      let listener = () => {
        effectiveSelection = (captured.valueFor(client) ?? null) as any;
      };

      $effect(() => {
        let value = effectiveSelection;
        let source = { x, y, z, category, identifier };
        captured.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          predicate: value != null ? predicateForDataPoints(source, value) : null,
          value: value,
        });
      });

      captured.addEventListener("value", listener);
      return () => {
        captured.removeEventListener("value", listener);
        captured.update({
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
    new DataPointQuery(coordinator, { table, x, y, z, category, text, identifier, additionalFields }),
  );

  // 3D default camera, fitted to the downloaded points. The O(N) bounding-sphere
  // scan is split from the camera so it runs only when the 3D point data changes,
  // NOT on every resize: resizing recomputes just the O(1) aspect-dependent
  // distance from the cached bounds, so large 3D views do not rescan on resize.
  let bounds3D = $derived<Bounds3D | null>(
    z != null && zData != null && xData.length > 0 ? pointsBounds3D(xData, yData, zData) : null,
  );
  let defaultCamera3DState = $derived<Camera3DState | null>(
    bounds3D != null ? cameraFromBounds3D(bounds3D, undefined, (width ?? 800) / (height ?? 800)) : null,
  );

  async function querySelection(px: number, py: number, unitDistance: number): Promise<DataPoint | null> {
    return await pointQuery.queryClosestPoint(filter?.predicate?.(clientId), px, py, unitDistance);
  }

  async function queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    return await pointQuery.queryPoints(identifiers);
  }

  // Resolve a point by its render instance index (3D pick).
  //
  // Exact-identity paths (preferred): a per-row key is row-aligned with the rendered
  // arrays, so the picked index maps to the EXACT row even when points share x/y/z or
  // collide after Float32 rounding. (1) the compact stable rowid (rowidData) for base
  // tables, any identifier type, no per-row identifier download; (2) the configured
  // identifier (identifierData) for rowid-less sources (views/joins) small enough to
  // materialize it (see IDENTIFIER_MATERIALIZE_MAX_ROWS). Either key is AUTHORITATIVE: a
  // null/error resolution is treated as INDETERMINATE (return null -> selection
  // unchanged), never downgraded to coordinates — a coordinate match could pick a
  // co-located twin and then emit an over-selecting predicate for the picked point.
  //
  // Last-resort fallback (no exact identity available — no rowid and either no
  // identifier or a source too large to materialize one): a coordinate lookup matching
  // the rendered Float32 x/y/z exactly (CAST AS FLOAT). This cannot distinguish rows
  // that share coordinates, which is unavoidable without a per-row key.
  async function queryByIndex(index: number): Promise<DataPoint | null> {
    if (index < 0 || index >= xData.length) {
      return null;
    }

    if (rowidData != null && index < rowidData.length) {
      try {
        // Constrain to the active cross-filter so a pick resolving by rowid cannot
        // resurrect a row another chart has filtered out while it resolved.
        return await pointQuery.queryByRowId(rowidData[index], filter?.predicate?.(clientId));
      } catch (e) {
        console.error("queryByIndex by rowid failed", e);
        return null;
      }
    }

    if (identifierData != null && index < identifierData.length) {
      try {
        return await pointQuery.queryByIdentifier(identifierData[index], filter?.predicate?.(clientId));
      } catch (e) {
        console.error("queryByIndex by identifier failed", e);
        return null;
      }
    }

    // No exact identity for this dataset: fall back to a coordinate lookup from the
    // picked point's rendered coordinates.
    let base: DataPoint = { x: xData[index], y: yData[index] };
    if (zData != null) {
      base.z = zData[index];
    }
    if (categoryData != null) {
      base.category = categoryData[index];
    }
    let scale = defaultViewportState?.scale ?? 1;
    let unitDistance = 2 / scale / Math.max(1, Math.min(width ?? 800, height ?? 800));
    try {
      let predicate = filter?.predicate?.(clientId);
      let full =
        base.z != null
          ? await pointQuery.queryClosestPoint3D(predicate, base.x, base.y, base.z, unitDistance)
          : await pointQuery.queryClosestPoint(predicate, base.x, base.y, unitDistance);
      if (full != null) {
        return full;
      }
    } catch (e) {
      console.error("queryByIndex enrichment failed", e);
    }
    // Do NOT return the synthetic base point: its Float32 coordinates are rounded
    // relative to the source DOUBLEs and it has no stable identifier, so it would
    // build a coordinated-selection predicate that matches the wrong rows or none.
    // A null result simply leaves selection/tooltip unchanged for this rare pick.
    return null;
  }

  // Cluster Labels
  async function queryClusterLabels(clusters: Rectangle[][]): Promise<(LabelContent | null)[]> {
    // If we have image + importance columns, query for representative images
    if (image != null && importance != null) {
      return await queryClusterImageLabels(clusters);
    }
    // Otherwise fall back to text summarization
    if (text == null) {
      return clusters.map(() => null);
    }
    // Create text summarizer (in the worker)
    let summarizer = await textSummarizerCreate({
      regions: clusters,
      stopWords: config?.autoLabelStopWords ?? null,
    });
    // Add text data to the summarizer
    let start = 0;
    let chunkSize = 10000;
    let lastAdd: Promise<unknown> | null = null;
    while (true) {
      let r = await coordinator.query(
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

  async function queryClusterImageLabels(clusters: Rectangle[][]): Promise<(LabelContent | null)[]> {
    if (image == null || importance == null) {
      return [];
    }
    // Build a VALUES table of all rectangles with their region index
    let values = clusters
      .flatMap((rects, regionId) =>
        rects.map(
          (r) => SQL.sql`(
            ${SQL.literal(regionId)},
            ${SQL.literal(r.xMin)}, ${SQL.literal(r.xMax)},
            ${SQL.literal(r.yMin)}, ${SQL.literal(r.yMax)}
          )`,
        ),
      )
      .join(", ");
    let sql = `
      WITH rectangles(regionId, xMin, xMax, yMin, yMax) AS (VALUES ${values})
      SELECT
        r.regionId AS regionId,
        arg_max(${SQL.column(image, "t")}, ${SQL.column(importance, "t")}) AS bestImage,
        arg_max(${SQL.column(x, "t")}, ${SQL.column(importance, "t")}) AS bestX,
        arg_max(${SQL.column(y, "t")}, ${SQL.column(importance, "t")}) AS bestY
      FROM rectangles r
      JOIN "${table}" AS t ON
        ${SQL.column(x, "t")} BETWEEN r.xMin AND r.xMax AND
        ${SQL.column(y, "t")} BETWEEN r.yMin AND r.yMax
      GROUP BY r.regionId
      ORDER BY r.regionId
    `;
    let result = await coordinator.query(sql);
    let rows = result.toArray();

    // Map results back by region_id, measuring image dimensions for aspect ratio
    let output: ({
      image: string;
      width: number;
      height: number;
      x: number;
      y: number;
    } | null)[] = clusters.map(() => null);

    for (let i = 0; i < rows.length; i++) {
      let { bestImage, bestX, bestY, regionId } = rows[i];
      if (bestImage == null) continue;
      let dataUrl = imageToDataUrl(bestImage);
      if (dataUrl == null) continue;
      output[regionId] = { image: dataUrl, width: 0, height: 0, x: bestX, y: bestY };
    }

    await Promise.all(
      output.map(async (item) => {
        if (item == null) {
          return;
        }
        let { width, height } = await measureImageSize(item.image);
        // Fit to IMAGE_LABEL_SIZE while maintaining aspect ratio
        let scale = Math.min(IMAGE_LABEL_SIZE / width, IMAGE_LABEL_SIZE / height);
        item.width = width * scale;
        item.height = height * scale;
      }),
    );

    return output;
  }

  function measureImageSize(src: string): Promise<{ width: number; height: number }> {
    return new Promise((resolve) => {
      let img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve({ width: IMAGE_LABEL_SIZE, height: IMAGE_LABEL_SIZE });
      img.src = src;
    });
  }
</script>

<EmbeddingViewImpl
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  theme={theme}
  config={config}
  data={{ x: xData, y: yData, z: zData, category: categoryData }}
  totalCount={totalCount}
  maxDensity={maxDensity}
  categoryCount={categoryCount}
  categoryColors={categoryColors}
  defaultViewportState={defaultViewportState}
  defaultCamera3DState={defaultCamera3DState}
  camera3DState={camera3DState}
  onCamera3DState={onCamera3DState}
  querySelection={querySelection}
  queryByIndex={queryByIndex}
  queryClusterLabels={queryClusterLabels}
  labels={labels}
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
  cache={cache}
/>
