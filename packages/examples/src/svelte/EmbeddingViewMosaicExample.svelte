<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Coordinator, wasmConnector, type DuckDBWASMConnector } from "@uwdata/mosaic-core";
  import * as vg from "@uwdata/vgplot";

  import { EmbeddingViewMosaic } from "embedding-atlas/svelte";

  import { generateSampleDataset } from "../sample_datasets.js";

  const coordinator = new Coordinator();

  let mode: "points" | "density" = $state("density");
  let colorScheme: "light" | "dark" = $state("light");
  let minimumDensity: number = $state(1 / 16);
  let pointCount: number = $state(0);
  let ready = $state(false);
  let loadingStatus = $state("");

  let vgPlotContainer: HTMLDivElement;
  let brush: any = $state.raw(null);

  const tooltip = vg.Selection.single({ empty: true });
  const selection = vg.Selection.single({ empty: true });

  let viewportState: any | null = $state.raw(null);

  let initDatabase = (async () => {
    const wasm = await wasmConnector();
    coordinator.databaseConnector(wasm);
  })();

  async function initialize() {
    loadingStatus = "Loading DuckDB...";
    await initDatabase;
    ready = false;

    loadingStatus = "Loading Data (may take some time depending on network speed and data size)...";

    vgPlotContainer.replaceChildren();

    coordinator.clear();

    await coordinator.exec(`DROP TABLE IF EXISTS data_table`);

    let dataset = generateSampleDataset({ numPoints: 500000, numCategories: 3, numSubClusters: 32 });
    let db = await (coordinator.databaseConnector()! as DuckDBWASMConnector).getDuckDB();
    await db.registerFileText("rows.json", JSON.stringify(dataset));
    await (await db.connect()).insertJSONFromPath("rows.json", { name: "data_table" });

    let r = await coordinator.query("SELECT COUNT(*) AS count FROM data_table");
    pointCount = r.get(0).count;

    brush = vg.Selection.crossfilter();
    vg.coordinator(coordinator);

    let plot = vg.vconcat(
      vg.plot(
        vg.rectX(vg.from("data_table", { filterBy: brush }), {
          y: "category",
          x: vg.count(),
          fill: "steelblue",
          inset: 0.5,
        }),
        vg.toggleY({ as: brush }),
        vg.yDomain(vg.Fixed),
        vg.xGrid(true),
        vg.width(400),
        vg.height(100),
        vg.marginLeft(65),
      ),
      vg.plot(
        vg.rectY(vg.from("data_table", { filterBy: brush }), {
          x: vg.bin("x", { steps: 50 }),
          y: vg.count(),
          fill: "steelblue",
          inset: 0.5,
        }),
        vg.intervalX({ as: brush }),
        vg.xDomain(vg.Fixed),
        vg.yScale("sqrt"),
        vg.yGrid(true),
        vg.width(400),
        vg.height(200),
        vg.marginLeft(65),
      ),
      vg.plot(
        vg.rectY(vg.from("data_table", { filterBy: brush }), {
          x: vg.bin("y", { steps: 50 }),
          y: vg.count(),
          fill: "steelblue",
          inset: 0.5,
        }),
        vg.intervalX({ as: brush }),
        vg.xDomain(vg.Fixed),
        vg.yScale("sqrt"),
        vg.yGrid(true),
        vg.width(400),
        vg.height(200),
        vg.marginLeft(65),
      ),
      vg.table({
        from: "data_table",
        filterBy: tooltip,
      }),
      vg.table({
        from: "data_table",
        filterBy: selection,
      }),
    );
    vgPlotContainer.replaceChildren();
    vgPlotContainer.appendChild(plot);

    ready = true;
  }

  $effect(() => {
    initialize();
  });
</script>

<div>
  <div style="margin-bottom:5px;display:flex;align-items:center;gap:8px">
    <label style="display:flex;align-items:center;gap:4px">
      Mode:
      <select bind:value={mode}>
        <option value="points">Points</option>
        <option value="density">Density</option>
      </select>
    </label>

    <label style="display:flex;align-items:center;gap:4px">
      Color Scheme:
      <select bind:value={colorScheme}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>

    <input type="range" bind:value={minimumDensity} min={0} max={0.2} step={0.0001} />
    {minimumDensity.toFixed(4)}
  </div>
  <div style="display:flex;gap:8px">
    <div>
      {#if ready}
        <div style:border="1px solid black">
          <EmbeddingViewMosaic
            coordinator={coordinator}
            config={{
              mode: mode,
              minimumDensity: minimumDensity,
              colorScheme: colorScheme,
            }}
            width={800}
            height={800}
            table="data_table"
            x="x"
            y="y"
            category="category"
            text="text"
            tooltip={tooltip}
            selection={selection}
            viewportState={viewportState}
            onViewportState={(v) => (viewportState = v)}
            rangeSelection={brush}
            filter={brush}
          />
        </div>
      {/if}
    </div>
    <div>
      <div bind:this={vgPlotContainer}></div>
      {#if ready}
        <div>
          Total: {pointCount} points.<br />
          Viewport:<br />
          <pre>{JSON.stringify(viewportState, null, 2)}</pre>
        </div>
      {/if}
    </div>
  </div>
  {#if !ready}
    <div>{loadingStatus}</div>
  {/if}
</div>
