<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import EmbeddingView3D from "../../../component/src/lib/embedding_view/EmbeddingView3D.svelte";

  interface StarData {
    x: number[];
    y: number[];
    z: number[];
    category: number[];
    categoryNames: string[];
    magnitude: number[];
    distancePc: number[];
  }

  let stars: StarData | null = $state.raw(null);
  let error: string | null = $state.raw(null);

  let data = $derived(
    stars != null
      ? {
          x: new Float32Array(stars.x),
          y: new Float32Array(stars.y),
          z: new Float32Array(stars.z),
          category: new Uint8Array(stars.category),
        }
      : null,
  );

  let categoryColors = ["#f2c744", "#7ea8f9", "#8891a3"]; // bright / medium / dim

  fetch("/real_stars.json")
    .then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    })
    .then((json) => {
      stars = json;
    })
    .catch((e) => {
      error = String(e);
    });
</script>

<h1 class="text-xl font-bold mb-2">Real Stars (LSDB / Gaia DR3) — 3D</h1>
<p class="mb-4 text-sm text-slate-600">
  Real stars from a Gaia DR3 cone search via LSDB (see <code>get_real_stars.py</code>). X/Y are the sky-plane RA/Dec
  offset (degrees) from the cone center; Z is distance derived from parallax, rescaled for a navigable depth range.
  Drag to orbit, scroll to zoom.
</p>

<div style="display:flex;gap:12px">
  <div style:border="1px solid black">
    {#if error}
      <div style="width:800px;height:800px;display:flex;align-items:center;justify-content:center;color:red">
        Failed to load star data: {error}
      </div>
    {:else if data == null}
      <div style="width:800px;height:800px;display:flex;align-items:center;justify-content:center">Loading stars…</div>
    {:else}
      <EmbeddingView3D data={data} categoryColors={categoryColors} width={800} height={800} pointSize={4} />
    {/if}
  </div>
  <div style="flex:1;min-width:280px">
    <h2 class="font-bold mb-1">Data</h2>
    {#if stars != null}
      <p>{stars.x.length.toLocaleString()} stars</p>
      <table class="text-xs mb-2">
        <thead>
          <tr>
            <td class="pr-2">Category</td>
            <td class="pr-2">Color</td>
            <td>Count</td>
          </tr>
        </thead>
        <tbody>
          {#each stars.categoryNames as name, i}
            <tr>
              <td class="pr-2">{name}</td>
              <td class="pr-2"><span style:display="inline-block" style:width="10px" style:height="10px" style:background={categoryColors[i]}></span></td>
              <td>{stars.category.filter((c) => c === i).length.toLocaleString()}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <h3 class="font-bold mb-1">First 20 rows</h3>
      <table class="text-xs">
        <thead>
          <tr>
            <td class="pr-2">RA offset</td>
            <td class="pr-2">Dec offset</td>
            <td class="pr-2">Distance (pc)</td>
            <td class="pr-2">Magnitude</td>
            <td>Class</td>
          </tr>
        </thead>
        <tbody>
          {#each stars.x.slice(0, 20) as _, i}
            <tr>
              <td class="pr-2">{stars.x[i].toFixed(3)}</td>
              <td class="pr-2">{stars.y[i].toFixed(3)}</td>
              <td class="pr-2">{stars.distancePc[i].toFixed(1)}</td>
              <td class="pr-2">{stars.magnitude[i].toFixed(2)}</td>
              <td>{stars.categoryNames[stars.category[i]]}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
</div>
