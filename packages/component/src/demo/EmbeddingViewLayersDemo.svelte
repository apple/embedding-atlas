<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { DataPoint, EmbeddingViewProps, Label, Rectangle, ViewportState } from "../lib/index.js";
  import { generateSampleDataset } from "./sample_datasets.js";

  import EmbeddingView from "../lib/embedding_view/EmbeddingView.svelte";

  // The main dataset.
  let datasetA = generateSampleDataset({ numPoints: 500000, numCategories: 3, numSubClusters: 16 });
  let dataA = {
    x: new Float32Array(datasetA.map((r) => r.x)),
    y: new Float32Array(datasetA.map((r) => r.y)),
    category: new Uint8Array(datasetA.map((r) => r.category)),
  };

  // A second dataset to overlay on top of the first one.
  let datasetB = generateSampleDataset({ numPoints: 20000, numCategories: 1, numSubClusters: 4 });
  let dataB = {
    x: new Float32Array(datasetB.map((r) => r.x * 0.5 + 0.5)),
    y: new Float32Array(datasetB.map((r) => r.y * 0.5 - 0.5)),
  };

  // Static labels at the centers of the second dataset's clusters.
  let staticLabels: Label[] = [
    { x: 0.5, y: -0.5, content: "Dataset B", level: 0 },
    { x: 0.5, y: -0.8, content: "Static label", level: 1 },
  ];

  let tooltip: DataPoint | null = $state.raw(null);
  let viewportState: ViewportState | null = $state.raw(null);
  let colorScheme: "light" | "dark" = $state.raw("light");

  let showDensity = $state(true);
  let showPointsA = $state(true);
  let showPointsB = $state(true);
  let showAutoLabels = $state(true);
  let showStaticLabels = $state(true);
  let pointsBOpacity: number | "auto" = $state.raw(0.9);

  let layers: EmbeddingViewProps["layers"] = $derived([
    { type: "density", id: "density", visible: showDensity, contours: true },
    { type: "points", id: "pointsA", visible: showPointsA },
    {
      type: "points",
      id: "pointsB",
      visible: showPointsB,
      data: dataB,
      categoryColors: ["#e31a1c"],
      opacity: pointsBOpacity,
      pointSize: 2,
    },
    { type: "labels", id: "auto", visible: showAutoLabels },
    { type: "labels", id: "static", visible: showStaticLabels, labels: staticLabels, opacity: 1 },
  ]);

  async function querySelection(x: number, y: number, unitDistance: number): Promise<DataPoint | null> {
    let minDistance2: number | null = null;
    let minIndex: number | null = null;
    for (let i = 0; i < dataA.x.length; i++) {
      let d2 = (dataA.x[i] - x) * (dataA.x[i] - x) + (dataA.y[i] - y) * (dataA.y[i] - y);
      if (minDistance2 == null || d2 < minDistance2) {
        minDistance2 = d2;
        minIndex = i;
      }
    }
    if (minIndex == null || minDistance2 == null || Math.sqrt(minDistance2) > unitDistance * 10) {
      return null;
    }
    return { x: dataA.x[minIndex], y: dataA.y[minIndex], text: datasetA[minIndex].text, fields: {} };
  }

  async function queryClusterLabels(clusters: Rectangle[][]): Promise<(string | null)[]> {
    return clusters.map((_, i) => `cluster ${i}`);
  }
</script>

<div style="margin-bottom:5px;display:flex;align-items:center;gap:8px">
  <label><input type="checkbox" bind:checked={showDensity} /> Density (A)</label>
  <label><input type="checkbox" bind:checked={showPointsA} /> Points (A)</label>
  <label><input type="checkbox" bind:checked={showPointsB} /> Points (B)</label>
  <label><input type="checkbox" bind:checked={showAutoLabels} /> Auto Labels (A)</label>
  <label><input type="checkbox" bind:checked={showStaticLabels} /> Static Labels</label>
  <label style="display:flex;align-items:center;gap:4px">
    Points (B) Opacity:
    <select bind:value={pointsBOpacity}>
      <option value="auto">Auto</option>
      <option value={0.3}>0.3</option>
      <option value={0.9}>0.9</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:4px">
    Color Scheme:
    <select bind:value={colorScheme}>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
  </label>
</div>

<div style:border="1px solid black" style:display="inline-block">
  <EmbeddingView
    data={dataA}
    layers={layers}
    config={{ colorScheme: colorScheme }}
    tooltip={tooltip}
    onTooltip={(v) => {
      tooltip = v;
    }}
    viewportState={viewportState}
    onViewportState={(v) => {
      viewportState = v;
    }}
    queryClusterLabels={queryClusterLabels}
    querySelection={querySelection}
  />
</div>
