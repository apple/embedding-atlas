<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator, makeClient } from "@uwdata/mosaic-core";
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
  }: Props = $props();

  let xData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let yData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let zData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let categoryData: Uint8Array<ArrayBuffer> | null = $state.raw(null);
  let identifierData: DataPointID[] | null = $state.raw(null);

  let pointQuery = $derived(
    new DataPointQuery(coordinator, { table, x, y, category, identifier, additionalFields }),
  );

  let activeFilterHidden = $derived(filter != null && filter.clauses.length > 0);

  $effect(() => {
    let deps = { coordinator, source: { table, x, y, z, category, identifier } };
    let client: { destroy: () => void } | null = null;

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
    }

    initClient();

    return () => {
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
/>
