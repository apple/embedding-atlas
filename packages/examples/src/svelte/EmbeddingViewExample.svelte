<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { DataPoint, Rectangle, ViewportState } from "embedding-atlas/svelte";
  import { EmbeddingView } from "embedding-atlas/svelte";

  import { generateSampleDataset } from "../sample_datasets.js";

  let dataset = generateSampleDataset({ numPoints: 500000, numCategories: 3, numSubClusters: 32 });
  let data = {
    x: new Float32Array(dataset.map((r) => r.x)),
    y: new Float32Array(dataset.map((r) => r.y)),
    category: new Uint8Array(dataset.map((r) => r.category)),
  };

  let tooltip: DataPoint | null = $state.raw(null);
  let selection: DataPoint[] | null = $state.raw([]);
  let rangeSelection: any | null = $state.raw(null);

  let mode: "points" | "density" = $state.raw("density");
  let colorScheme: "light" | "dark" = $state.raw("light");
  let minimumDensity: number = $state.raw(1 / 16);
  let viewportState: ViewportState | null = $state.raw(null);

  async function querySelection(x: number, y: number, unitDistance: number): Promise<DataPoint | null> {
    let minDistance2: number | null = null;
    let minIndex: number | null = null;
    for (let i = 0; i < data.x.length; i++) {
      let d2 = (data.x[i] - x) * (data.x[i] - x) + (data.y[i] - y) * (data.y[i] - y);
      if (minDistance2 == null || d2 < minDistance2) {
        minDistance2 = d2;
        minIndex = i;
      }
    }
    if (minIndex == null || minDistance2 == null || Math.sqrt(minDistance2) > unitDistance * 10) {
      return null;
    }
    return { x: data.x[minIndex], y: data.y[minIndex], text: dataset[minIndex].text, fields: {} };
  }

  async function queryClusterLabels(clusters: Rectangle[][]): Promise<(string | null)[]> {
    return clusters.map(() => "label");
  }
</script>

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
  <div style:border="1px solid black">
    <EmbeddingView
      data={data}
      mode={mode}
      colorScheme={colorScheme}
      minimumDensity={minimumDensity}
      tooltip={tooltip}
      onTooltip={(v) => {
        tooltip = v;
      }}
      selection={selection}
      onSelection={(v) => {
        selection = v;
      }}
      rangeSelection={rangeSelection}
      onRangeSelection={(v) => {
        rangeSelection = v;
      }}
      viewportState={viewportState}
      onViewportState={(v) => {
        viewportState = v;
      }}
      automaticLabels={true}
      queryClusterLabels={queryClusterLabels}
      querySelection={querySelection}
    />
  </div>
  <div>
    {#if tooltip}
      Tooltip:<br />
      <pre>{JSON.stringify(tooltip, null, 2)}</pre>
    {/if}
    {#if selection && selection.length > 0}
      {selection.length} Selected points:<br />
      {#each selection as point}
        <pre>{JSON.stringify(point, null, 2)}</pre>
      {/each}
    {/if}
    {#if rangeSelection}
      <pre>{JSON.stringify(rangeSelection, null, 2)}</pre>
    {/if}
    Viewport:<br />
    <pre>{JSON.stringify(viewportState, null, 2)}</pre>
  </div>
</div>
