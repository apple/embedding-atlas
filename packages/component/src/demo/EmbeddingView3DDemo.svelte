<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Camera3DState, DataPoint } from "../lib/index.js";
  import { generateSampleDataset3D } from "./sample_datasets.js";

  import EmbeddingView from "../lib/embedding_view/EmbeddingView.svelte";

  const numPoints = 10000;

  let dataset = generateSampleDataset3D({ numPoints: numPoints, numCategories: 3 });
  let data = {
    x: new Float32Array(dataset.map((r) => r.x)),
    y: new Float32Array(dataset.map((r) => r.y)),
    z: new Float32Array(dataset.map((r) => r.z)),
    category: new Uint8Array(dataset.map((r) => r.category)),
  };

  let mode: "points" | "points-3d" = $state.raw("points-3d");
  let colorScheme: "light" | "dark" = $state.raw("light");
  let pointSize: number = $state.raw(6);
  let fogDensity: number = $state.raw(0.6);
  let downsampleMaxPoints: number = $state.raw(numPoints);

  let tooltip: DataPoint | null = $state.raw(null);
  let selection: DataPoint[] | null = $state.raw([]);
  let camera3DState: Camera3DState | null = $state.raw(null);

  // A queryByIndex that also carries the demo text so tooltips show content.
  async function queryByIndex(index: number): Promise<DataPoint | null> {
    if (index < 0 || index >= dataset.length) {
      return null;
    }
    let r = dataset[index];
    return { x: r.x, y: r.y, z: r.z, category: r.category, text: r.text, identifier: r.identifier };
  }
</script>

<div style="margin-bottom:5px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
  <label style="display:flex;align-items:center;gap:4px">
    Mode:
    <select bind:value={mode} data-testid="mode-select">
      <option value="points-3d">Points 3D</option>
      <option value="points">Points (2D)</option>
    </select>
  </label>

  <label style="display:flex;align-items:center;gap:4px">
    Color Scheme:
    <select bind:value={colorScheme}>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
  </label>

  <label style="display:flex;align-items:center;gap:4px">
    Point Size:
    <input type="range" bind:value={pointSize} min={1} max={20} step={0.5} />
    {pointSize}
  </label>

  <label style="display:flex;align-items:center;gap:4px">
    Fog:
    <input type="range" bind:value={fogDensity} min={0} max={2} step={0.05} />
    {fogDensity.toFixed(2)}
  </label>

  <label style="display:flex;align-items:center;gap:4px">
    Max Points:
    <input type="range" bind:value={downsampleMaxPoints} min={500} max={numPoints} step={500} />
    {downsampleMaxPoints.toLocaleString()}
  </label>

  <span style="opacity:0.6">Drag to orbit · Shift+drag to pan · Scroll to zoom · Double-click to focus</span>
</div>

<div style="display:flex;gap:8px">
  <div style:border="1px solid black" data-testid="embedding-3d">
    <EmbeddingView
      width={720}
      height={600}
      data={data}
      config={{
        mode: mode,
        colorScheme: colorScheme,
        pointSize: pointSize,
        fogDensity: fogDensity,
        downsampleMaxPoints: downsampleMaxPoints,
      }}
      tooltip={tooltip}
      onTooltip={(v) => {
        tooltip = v;
      }}
      selection={selection}
      onSelection={(v) => {
        selection = v;
      }}
      camera3DState={camera3DState}
      onCamera3DState={(v) => {
        camera3DState = v;
      }}
      queryByIndex={queryByIndex}
    />
  </div>
  <div style="font-size:12px;font-family:monospace">
    <div>{numPoints.toLocaleString()} points · 3 blobs</div>
    {#if tooltip}
      Tooltip:<br />
      <pre data-testid="tooltip">{JSON.stringify(tooltip, null, 2)}</pre>
    {/if}
    {#if selection && selection.length > 0}
      {selection.length} Selected:<br />
      {#each selection as point}
        <pre>{JSON.stringify(point, null, 2)}</pre>
      {/each}
    {/if}
    {#if camera3DState}
      Camera:<br />
      <pre>{JSON.stringify(camera3DState, null, 2)}</pre>
    {/if}
  </div>
</div>
