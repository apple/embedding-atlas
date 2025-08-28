<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { findClusters, type Cluster } from "embedding-atlas";
  import { onMount } from "svelte";
  import { RNG, randomUniform } from "../sample_datasets.js";

  const width = 800;
  const height = 600;

  let canvas: HTMLCanvasElement;
  let clusters = $state<Cluster[]>([]);

  function generateDensity(width: number, height: number) {
    let rng = RNG(45);
    // Generate a mixture of gaussians
    let gaussians: { x: number; y: number; s1: number; s2: number; angle: number; amplitude: number }[] = [];
    for (let i = 0; i < 100; i++) {
      gaussians.push({
        x: randomUniform(0, width, rng),
        y: randomUniform(0, height, rng),
        s1: randomUniform(20, 80, rng),
        s2: randomUniform(20, 80, rng),
        angle: randomUniform(0, Math.PI * 2, rng),
        amplitude: randomUniform(0.1, 1, rng),
      });
    }
    let densityMap = new Float32Array(width * height);
    for (let y = 0; y < width; y++) {
      for (let x = 0; x < width; x++) {
        let d = 0;
        for (let g of gaussians) {
          let dx = g.x - x;
          let dy = g.y - y;
          let rx = dx * Math.cos(g.angle) + dy * Math.sin(g.angle);
          let ry = -dx * Math.sin(g.angle) + dy * Math.cos(g.angle);
          rx /= g.s1;
          ry /= g.s2;
          d += Math.exp(-(rx ** 2 + ry ** 2)) * g.amplitude;
        }
        densityMap[y * width + x] = d * 80;
      }
    }
    return densityMap;
  }

  async function run() {
    const densityMap = generateDensity(width, height);
    clusters = await findClusters(densityMap, width, height);

    let ctx = canvas.getContext("2d")!;
    let data = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < width * height; i++) {
      let value = densityMap[i];
      data.data[i * 4 + 0] = value;
      data.data[i * 4 + 1] = value;
      data.data[i * 4 + 2] = value;
      data.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(data, 0, 0);
  }

  onMount(run);
</script>

<div style:position="relative" style:width="{width}px" style:height="{height}px">
  <canvas style:position="absolute" bind:this={canvas} width={width} height={height}></canvas>
  <svg style:position="absolute" width={width} height={height}>
    {#each clusters as c}
      <g>
        {#each c.boundaryRectApproximation ?? [] as [x1, y1, x2, y2]}
          <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} style:stroke="rgba(255,0,0,0.1)" style:fill="none" />
        {/each}
        {#each c.boundary ?? [] as boundary}
          <path
            d={"M" + boundary.map(([x, y]) => `${x},${y}`).join("L") + "Z"}
            style:stroke="rgba(255,127,14,1)"
            style:fill="rgba(255,127,14,0.1)"
          />
        {/each}
        <circle cx={c.meanX} cy={c.meanY} r={2} style:fill="rgba(255,127,14,1)" />
      </g>
    {/each}
  </svg>
</div>

<style>
  div {
    padding: 20px;
    box-sizing: content-box;
    background: black;
  }
  g {
    opacity: 0.2;
  }
  g:hover {
    opacity: 1;
  }
</style>
