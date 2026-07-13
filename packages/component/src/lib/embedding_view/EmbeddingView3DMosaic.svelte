<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator, makeClient, type MosaicClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import EmbeddingView3D from "./EmbeddingView3D.svelte";

  import { DataPointQuery } from "./mosaic_client.js";
  import type { DataField, DataPoint, DataPointID } from "./types.js";

  import type { Coordinator, Selection } from "@uwdata/mosaic-core";

  interface Props {
    coordinator?: Coordinator;
    table: string;
    x: string;
    y: string;
    z: string;
    category?: string | null;
    identifier?: string | null;
    additionalFields?: Record<string, DataField> | null;
    filter?: Selection | null;
    categoryColors?: string[] | null;
    width?: number | null;
    height?: number | null;
    pointSize?: number | null;
    /** The current single or multiple point selection, as full `DataPoint` objects. */
    selection?: DataPoint[] | null;
    onSelection?: ((value: DataPoint[] | null) => void) | null;
    /** Called with the hovered point (or `null`) as the cursor moves, for tooltips. */
    onTooltip?: ((value: DataPoint | null) => void) | null;
    /** Approximate maximum number of points to render at once; see `EmbeddingView3D`. */
    downsampleMaxPoints?: number | null;
    showStatusBar?: boolean;
    /** A Mosaic `Selection` to update when the user completes a marquee/lasso drag (see
     *  `EmbeddingView3D`'s `onRangeSelect` doc comment - this is a one-shot snapshot, not a
     *  persistent geometric region). The predicate prioritizes the configured `identifier`
     *  column, falling back to coordinate equality when none is configured. */
    rangeSelection?: Selection | null;
  }

  let {
    coordinator = defaultCoordinator(),
    table,
    x,
    y,
    z,
    category = null,
    identifier = null,
    additionalFields = null,
    filter = null,
    categoryColors = null,
    width = null,
    height = null,
    pointSize = null,
    selection = null,
    onSelection = null,
    onTooltip = null,
    downsampleMaxPoints = 500000,
    showStatusBar = true,
    rangeSelection = null,
  }: Props = $props();

  let mosaicClient: MosaicClient | null = $state.raw(null);

  let xData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let yData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let zData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let categoryData: Uint8Array<ArrayBuffer> | null = $state.raw(null);
  let identifierData: DataPointID[] | null = $state.raw(null);

  let pointQuery = $derived(
    new DataPointQuery(coordinator, { table, x, y, category, identifier, additionalFields }),
  );

  // A Mosaic `Selection`'s `.clauses` is a plain mutable array on a class instance, not a Svelte
  // reactive value - `$derived` would only re-run when the `filter`/`rangeSelection` prop
  // *references* change, never when their internal clauses are updated in place. Selections emit
  // a "value" event on every update (the same event `EmbeddingViewMosaic.svelte`'s tooltip/
  // selection effects already listen for), so mirror that pattern here.
  let activeFilterHidden = $state.raw(false);

  $effect(() => {
    function updateActiveFilterHidden() {
      activeFilterHidden =
        (filter != null && filter.clauses.length > 0) ||
        (rangeSelection != null && rangeSelection.clauses.length > 0);
    }
    updateActiveFilterHidden();
    filter?.addEventListener("value", updateActiveFilterHidden);
    rangeSelection?.addEventListener("value", updateActiveFilterHidden);
    return () => {
      filter?.removeEventListener("value", updateActiveFilterHidden);
      rangeSelection?.removeEventListener("value", updateActiveFilterHidden);
    };
  });

  $effect(() => {
    let deps = { coordinator, source: { table, x, y, z, category, identifier } };
    let client: MosaicClient | null = null;

    async function initClient() {
      let source = deps.source;
      client = makeClient({
        coordinator: deps.coordinator,
        selection: filter ?? undefined,
        query: (predicate) => {
          return SQL.Query.from(source.table)
            .select({
              x: SQL.sql`${SQL.column(source.x)}::FLOAT`,
              y: SQL.sql`${SQL.column(source.y)}::FLOAT`,
              z: SQL.sql`${SQL.column(source.z)}::FLOAT`,
              ...(source.category != null ? { c: SQL.sql`${SQL.column(source.category)}::UTINYINT` } : {}),
              ...(source.identifier != null ? { id: SQL.column(source.identifier) } : {}),
            })
            .where(predicate);
        },
        queryResult: (data: any) => {
          let xArray = data.getChild("x").toArray();
          let yArray = data.getChild("y").toArray();
          let zArray = data.getChild("z").toArray();
          let categoryArray = data.getChild("c")?.toArray() ?? null;
          let idColumn = data.getChild("id");
          xData = xArray instanceof Float32Array ? xArray : new Float32Array(xArray);
          yData = yArray instanceof Float32Array ? yArray : new Float32Array(yArray);
          zData = zArray instanceof Float32Array ? zArray : new Float32Array(zArray);
          categoryData = categoryArray == null ? null : categoryArray instanceof Uint8Array ? categoryArray : new Uint8Array(categoryArray);
          identifierData = idColumn != null ? idColumn.toArray() : null;
        },
      });
      mosaicClient = client;
    }

    initClient();

    return () => {
      mosaicClient = null;
      client?.destroy();
    };
  });

  let selectedIndices = $derived.by(() => {
    if (selection == null || selection.length === 0 || identifierData == null) return null;
    let ids = new Set(selection.map((p) => p.identifier));
    let indices: number[] = [];
    for (let i = 0; i < identifierData.length; i++) {
      if (ids.has(identifierData[i])) indices.push(i);
    }
    return indices;
  });

  async function handlePointClick(index: number | null) {
    if (onSelection == null) return;
    if (index == null) {
      onSelection(null);
      return;
    }
    if (identifierData != null) {
      let id = identifierData[index];
      let points = await pointQuery.queryPoints([id]);
      onSelection(points.length > 0 ? points : null);
    } else {
      // No identifier column configured - fall back to the coordinates we already have locally.
      onSelection([{ x: xData[index], y: yData[index], z: zData[index], category: categoryData?.[index], fields: {} }]);
    }
  }

  function handlePointHover(index: number | null) {
    if (onTooltip == null) return;
    if (index == null) {
      onTooltip(null);
      return;
    }
    // Resolved directly from the already-loaded local arrays (no SQL round-trip) so hover stays
    // responsive; unlike click-to-select, a hover tooltip doesn't need `text`/`additionalFields`.
    onTooltip({
      x: xData[index],
      y: yData[index],
      z: zData[index],
      category: categoryData?.[index],
      identifier: identifierData?.[index],
      fields: {},
    });
  }

  const MAX_RANGE_SELECT_ROWS = 2000;

  function handleRangeSelect(indices: number[]) {
    if (rangeSelection == null || mosaicClient == null) return;
    if (indices.length === 0) {
      rangeSelection.update({ source: mosaicClient, clients: new Set([mosaicClient]), predicate: null, value: null });
      return;
    }
    if (indices.length > MAX_RANGE_SELECT_ROWS) {
      console.warn(
        `3D marquee/lasso selection matched ${indices.length} points, more than the ${MAX_RANGE_SELECT_ROWS}-row ` +
          `cap for a single filter predicate - only the first ${MAX_RANGE_SELECT_ROWS} are included.`,
      );
      indices = indices.slice(0, MAX_RANGE_SELECT_ROWS);
    }

    let predicate =
      identifier != null && identifierData != null
        ? SQL.isIn(
            SQL.column(identifier),
            indices.map((i) => SQL.literal(identifierData![i])),
          )
        : SQL.or(
            ...indices.map((i) =>
              SQL.and(
                SQL.eq(SQL.cast(SQL.column(x), "DOUBLE"), SQL.literal(xData[i])),
                SQL.eq(SQL.cast(SQL.column(y), "DOUBLE"), SQL.literal(yData[i])),
                SQL.eq(SQL.cast(SQL.column(z), "DOUBLE"), SQL.literal(zData[i])),
              ),
            ),
          );
    rangeSelection.update({ source: mosaicClient, clients: new Set([mosaicClient]), predicate, value: indices });
  }
</script>

<EmbeddingView3D
  data={{ x: xData, y: yData, z: zData, category: categoryData }}
  categoryColors={categoryColors}
  width={width ?? 800}
  height={height ?? 800}
  pointSize={pointSize}
  selectedIndices={selectedIndices}
  onPointClick={handlePointClick}
  onPointHover={handlePointHover}
  downsampleMaxPoints={downsampleMaxPoints}
  activeFilterHidden={activeFilterHidden}
  showStatusBar={showStatusBar}
  onRangeSelect={rangeSelection != null ? handleRangeSelect : null}
/>
