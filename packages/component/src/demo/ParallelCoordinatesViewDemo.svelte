<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { defaultCategoryColors } from "../lib/index.js";

  import ParallelCoordinatesView from "../lib/parallel_coordinates_view/ParallelCoordinatesView.svelte";

  const numFields = 8;
  const numCategories = 6;
  const continuousPalette = ["#440154", "#3b528b", "#21908c", "#5dc863", "#fde725"];
  const categoryPalette = defaultCategoryColors(numCategories);

  let numRows: number = $state(100000);
  let binCount: "auto" | number = $state("auto");
  let opacity: number = $state(1);
  let autoOpacity: boolean = $state(true);
  let colorScheme: "light" | "dark" = $state("light");
  let colorMode: "categorical" | "continuous" = $state("categorical");
  let enabled: boolean[] = $state(Array.from({ length: numCategories }, () => true));

  let plotWidth: number = $state(1000);
  let plotHeight: number = $state(520);
  let brush: ([number, number] | null)[] = $state([]);

  function mulberry32(a: number) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussianRandom(rng: () => number) {
    let u: number, v: number, s: number;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    return u * Math.sqrt((-2 * Math.log(s)) / s);
  }

  function generateData(numRows: number, numFields: number, numCategories: number) {
    let rng = mulberry32(42);
    let columns: Float32Array[] = Array.from({ length: numFields }, () => new Float32Array(numRows));
    let clusterIds = new Uint8Array(numRows);

    let centers: number[][] = [];
    let spreads: number[][] = [];
    for (let c = 0; c < numCategories; c++) {
      let center: number[] = [];
      let spread: number[] = [];
      for (let d = 0; d < numFields; d++) {
        center.push(rng() * 0.6 + 0.2);
        spread.push(rng() * 0.05 + 0.02);
      }
      centers.push(center);
      spreads.push(spread);
    }

    for (let i = 0; i < numRows; i++) {
      let cluster = Math.floor(rng() * numCategories);
      clusterIds[i] = cluster;
      for (let d = 0; d < numFields; d++) {
        columns[d][i] = centers[cluster][d] + gaussianRandom(rng) * spreads[cluster][d];
      }
    }

    // Normalize each field to [0, 1].
    let mins: number[] = [];
    let maxs: number[] = [];
    for (let d = 0; d < numFields; d++) {
      let mn = Infinity;
      let mx = -Infinity;
      let col = columns[d];
      for (let i = 0; i < numRows; i++) {
        if (col[i] < mn) mn = col[i];
        if (col[i] > mx) mx = col[i];
      }
      let range = mx - mn || 1;
      for (let i = 0; i < numRows; i++) {
        col[i] = (col[i] - mn) / range;
      }
      mins.push(mn);
      maxs.push(mx);
    }

    // Make dimension 3 a categorical axis with 3 discrete values.
    const discreteDim = 3;
    const discreteValues = [0.25, 0.5, 0.75];
    for (let i = 0; i < numRows; i++) {
      columns[discreteDim][i] = discreteValues[i % discreteValues.length];
    }
    mins[discreteDim] = 0;
    maxs[discreteDim] = 1;

    return { columns, clusterIds, mins, maxs };
  }

  let generated = $derived.by(() => generateData(numRows, numFields, numCategories));

  // Keep only the rows whose category is enabled and that fall within every active axis brush.
  let filtered = $derived.by(() => {
    let en = [...enabled];
    let columns = generated.columns;
    let clusterIds = generated.clusterIds;
    let n = clusterIds.length;

    // Active axis brushes, normalized [0, 1] (same space as the columns).
    let active: { axis: number; lo: number; hi: number }[] = [];
    for (let a = 0; a < columns.length; a++) {
      let b = brush?.[a];
      if (b != null) {
        active.push({ axis: a, lo: Math.min(b[0], b[1]), hi: Math.max(b[0], b[1]) });
      }
    }

    let passes = (i: number) => {
      if (!en[clusterIds[i]]) {
        return false;
      }
      for (let k = 0; k < active.length; k++) {
        let v = columns[active[k].axis][i];
        if (v < active[k].lo || v > active[k].hi) {
          return false;
        }
      }
      return true;
    };

    let keepCount = 0;
    for (let i = 0; i < n; i++) {
      if (passes(i)) keepCount++;
    }

    let cols: Float32Array[] = columns.map(() => new Float32Array(keepCount));
    let cv = new Uint8Array(keepCount);
    let j = 0;
    for (let i = 0; i < n; i++) {
      if (!passes(i)) continue;
      for (let d = 0; d < columns.length; d++) {
        cols[d][j] = columns[d][i];
      }
      cv[j] = clusterIds[i];
      j++;
    }
    return { columns: cols, clusterIds: cv };
  });

  let colorValue = $derived<Float32Array | Uint8Array>(
    colorMode == "categorical" ? filtered.clusterIds : filtered.columns[0],
  );
  let categoryColors = $derived(colorMode == "categorical" ? categoryPalette : continuousPalette);

  let axisLabels = $derived(
    generated.mins.map((mn, d) =>
      [0, 0.25, 0.5, 0.75, 1].map((t) => ({ value: t, label: (mn + t * (generated.maxs[d] - mn)).toFixed(2) })),
    ),
  );

  let axisTitles = $derived(Array.from({ length: numFields }, (_, d) => `Dimension ${d}`));

  function toggleCategory(i: number) {
    enabled = enabled.map((v, j) => (j == i ? !v : v));
  }
</script>

<div style="margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
  <label style="display:flex;align-items:center;gap:4px">
    Rows:
    <select bind:value={numRows}>
      <option value={10000}>10K</option>
      <option value={100000}>100K</option>
      <option value={500000}>500K</option>
      <option value={1000000}>1M</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:4px">
    Bins:
    <select bind:value={binCount}>
      <option value="auto">Auto</option>
      <option value={50}>50</option>
      <option value={100}>100</option>
      <option value={200}>200</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:4px">
    Color:
    <select bind:value={colorMode}>
      <option value="categorical">Categorical (Uint8)</option>
      <option value="continuous">Continuous (Float32)</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:4px">
    Scheme:
    <select bind:value={colorScheme}>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:4px">
    Opacity:
    <input type="range" bind:value={opacity} min={0.05} max={2} step={0.05} />
    {opacity.toFixed(2)}
  </label>
  <label style="display:flex;align-items:center;gap:4px">
    <input type="checkbox" bind:checked={autoOpacity} />
    Auto opacity
  </label>
</div>

<div style="margin-bottom:8px;display:flex;align-items:center;gap:6px;flex-wrap:wrap">
  <span style="color:#666">Categories:</span>
  {#each categoryPalette as color, i (i)}
    <button
      onclick={() => toggleCategory(i)}
      style:background={enabled[i] ? color : "transparent"}
      style:border="2px solid {color}"
      style:color={enabled[i] ? "#fff" : color}
      style:opacity={enabled[i] ? "1" : "0.6"}
      style="border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px"
    >
      C{i}
    </button>
  {/each}
  <button onclick={() => (enabled = enabled.map(() => true))} style="margin-left:8px;font-size:12px">All</button>
  <button onclick={() => (enabled = enabled.map(() => false))} style="font-size:12px">None</button>
</div>

<div
  bind:clientWidth={plotWidth}
  bind:clientHeight={plotHeight}
  style:background={colorScheme == "light" ? "#fff" : "#000"}
  style="resize:both;overflow:hidden;width:1000px;height:520px;min-width:240px;min-height:160px;box-sizing:border-box;border:1px solid black;"
>
  <ParallelCoordinatesView
    data={{ values: filtered.columns, colorValue: colorValue }}
    axisLabels={axisLabels}
    axisTitles={axisTitles}
    categoryColors={categoryColors}
    opacity={opacity}
    binCount={binCount === "auto" ? null : binCount}
    colorScheme={colorScheme}
    autoOpacity={autoOpacity}
    brush={brush}
    onBrushChange={(b) => (brush = b)}
    width={plotWidth}
    height={plotHeight}
  />
</div>
