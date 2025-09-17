// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as React from "react";
import { useState } from "react";

import { EmbeddingView, type DataPoint, type Rectangle, type ViewportState } from "embedding-atlas/react";

import { generateSampleDataset } from "../sample_datasets.js";

export default function Component() {
  let [{ data, dataset }, _] = useState<any>(() => {
    let dataset = generateSampleDataset({ numPoints: 500000, numCategories: 3, numSubClusters: 32 });
    let data = {
      x: new Float32Array(dataset.map((r) => r.x)),
      y: new Float32Array(dataset.map((r) => r.y)),
      category: new Uint8Array(dataset.map((r) => r.category)),
    };
    return { data, dataset };
  });

  let [tooltip, setTooltip] = useState<DataPoint | null>(null);
  let [selection, setSelection] = useState<DataPoint[] | null>([]);
  let [rangeSelection, setRangeSelection] = useState<any | null>(null);

  let [mode, setMode] = useState<"points" | "density">("density");
  let [colorScheme, setColorScheme] = useState<"light" | "dark">("light");
  let [minimumDensity, setMinimumDensity] = useState<number>(1 / 16);
  let [viewportState, setViewportState] = useState<ViewportState | null>(null);

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

  async function queryClusterLabels(regions: Rectangle[][]): Promise<(string | null)[]> {
    return regions.map((_) => "label");
  }

  return (
    <>
      <div style={{ marginBottom: "5px", display: "flex", alignItems: "center", gap: "8px" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          Mode:
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="points">Points</option>
            <option value="density">Density</option>
          </select>
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          Color Scheme:
          <select value={colorScheme} onChange={(e) => setColorScheme(e.target.value as any)}>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>

        <input
          type="range"
          min={0}
          max={0.2}
          step={0.0001}
          value={minimumDensity}
          onChange={(e) => setMinimumDensity(e.target.value as any)}
        />
        {minimumDensity.toFixed(4)}
      </div>
      <div style={{ display: "flex", gap: "8px" }}>
        <div style={{ border: "1px solid black" }}>
          <EmbeddingView
            data={data}
            config={{
              mode: mode,
              colorScheme: colorScheme,
              minimumDensity: minimumDensity,
            }}
            tooltip={tooltip}
            onTooltip={(v) => {
              setTooltip(v);
            }}
            selection={selection}
            onSelection={(v) => {
              setSelection(v);
            }}
            rangeSelection={rangeSelection}
            onRangeSelection={(v) => {
              setRangeSelection(v);
            }}
            viewportState={viewportState}
            onViewportState={(v) => {
              setViewportState(v);
            }}
            queryClusterLabels={queryClusterLabels}
            querySelection={querySelection}
          />
        </div>
        <div>
          {tooltip && (
            <>
              Tooltip:
              <br />
              <pre>{JSON.stringify(tooltip, null, 2)}</pre>
            </>
          )}
          {selection && selection.length > 0 && (
            <>
              {selection.length} Selected points:
              <br />
              {selection.map((point, index) => (
                <pre key={index}>{JSON.stringify(point, null, 2)}</pre>
              ))}
            </>
          )}
          {rangeSelection && <pre>{JSON.stringify(rangeSelection, null, 2)}</pre>}
          Viewport:
          <br />
          <pre>{JSON.stringify(viewportState, null, 2)}</pre>
        </div>
      </div>
    </>
  );
}
