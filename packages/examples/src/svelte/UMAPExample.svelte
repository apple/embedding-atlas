<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { createUMAP, defaultCategoryColors, type UMAP } from "embedding-atlas";
  import { EmbeddingView, type ViewportState } from "embedding-atlas/svelte";
  import { onMount } from "svelte";
  import { RNG, randomNormal } from "../sample_datasets.js";

  type Optimizer = "adam" | "momentum" | "sgd";

  const SCHEDULE_EPOCHS = 200;
  const SIZE = 640;
  // The learning-rate ranges (from, to) that worked best for each optimizer.
  const LEARNING_RATES: Record<Optimizer, [number, number]> = {
    adam: [1, 0.001],
    momentum: [1, 0.002],
    sgd: [2, 0.001],
  };

  let umap: UMAP | null = null;
  let dataset = { data: new Float32Array(0), labels: new Uint8Array(0) };
  let annealStart = 0; // the epoch at which the learning rate was last set to its peak

  /** Exponential anneal from the first to the second rate over SCHEDULE_EPOCHS. */
  function learningRateAt(optimizer: Optimizer, epochsSinceStart: number) {
    const [from, to] = LEARNING_RATES[optimizer];
    return from * Math.pow(to / from, Math.min(epochsSinceStart / SCHEDULE_EPOCHS, 1));
  }

  // ---------------------------------------------------------------------
  // The UMAP API
  // ---------------------------------------------------------------------

  // 1. Create the instance once. createUMAP computes the neighbor graph and the
  //    initial layout, so `umap.embedding` is valid at epoch 0. The `progress`
  //    callback reports the setup stages: (fraction in [0, 1], stage name).
  //    `compatibilityLayout: false` selects the 0.6 pipeline, which supports
  //    all three optimizers and switching between them with `setOptimizer`.
  //    `data` is a row-major Float32Array of count x dim values.
  //    `metric` is "euclidean" or "cosine".
  async function createInstance(onProgress: (text: string) => void) {
    const { count, dim, nNeighbors, minDist, repulsionStrength, optimizer, gpu } = settings;
    dataset = makeShapes(count, dim);
    umap = await createUMAP(count, dim, 2, dataset.data, {
      metric: "euclidean",
      nNeighbors,
      minDist,
      repulsionStrength,
      optimizer,
      compatibilityLayout: false,
      gpu,
      seed: 42,
      progress: (fraction, stage) => onProgress(`${stage} ${(fraction * 100).toFixed(0)}%`),
    });
    annealStart = 0;
  }

  // 2. Step in a loop and draw after every step.
  //    `step(n)` runs n epochs and holds the learning rate flat, so re-annealing is
  //    your job: set `setParameters({ learningRate })` between steps. The rate here
  //    follows learningRateAt(optimizer, umap.epoch - annealStart), so setting
  //    `annealStart = umap.epoch` jumps the rate back to its peak: this is how you
  //    restart the learning after a parameter change (see 3).
  //    `umap.embedding` returns the same Float32Array every time. The values change
  //    after each step. Copy it (showFrame does) before you hand it to a component.
  //    (`run()` is different: it follows the full schedule of the optimizer, see 5.)
  async function stepLoop() {
    try {
      while (active && umap && umap.epoch - annealStart < SCHEDULE_EPOCHS) {
        umap.setParameters({ learningRate: learningRateAt(settings.optimizer, umap.epoch - annealStart) });
        await umap.step(1);
        showFrame(umap);
        await new Promise(requestAnimationFrame);
      }
    } catch (error) {
      status = `Error: ${error}`;
    } finally {
      // No await between the loop check and here, so ensureLoop never sees a loop that is about to end.
      loop = null;
      running = false;
    }
  }

  // 3. Live changes. These calls are queued and applied in order after the step that
  //    is currently running, so you can call them at any time. A changed minDist,
  //    repulsionStrength or optimizer re-heats the layout: the learning rate returns to
  //    its peak and decays again over SCHEDULE_EPOCHS, starting from the CURRENT layout.
  function applyLiveSettings() {
    umap?.setParameters({ minDist: settings.minDist, repulsionStrength: settings.repulsionStrength });
    reanneal();
  }
  function applyOptimizer() {
    umap?.setOptimizer(settings.optimizer);
    reanneal();
  }
  function reanneal() {
    if (umap == null) return;
    annealStart = umap.epoch;
    converged = false;
    if (active) ensureLoop(); // the loop stops at convergence, so wake it up
  }

  // 4. Restart from epoch 0 with a fresh layout. `reset` is queued, like the other
  //    calls. Awaiting an empty `step(0)` waits until it is applied and refreshes
  //    `epoch` and `embedding`. The instance does not keep the input data, so pass it
  //    for the recursive initialization ("auto" picks the initialization method).
  async function resetLayout() {
    umap!.reset("auto", dataset.data);
    await umap!.step(0);
    annealStart = umap!.epoch;
  }

  // 5. One-shot alternative: run the whole schedule at once. The live
  //    `learningRate` is the peak, so set it first. `run()` picks the number of epochs
  //    itself (more for fewer points) and applies the optimizer's own schedule.
  async function runToEnd() {
    umap!.setParameters({ learningRate: LEARNING_RATES[settings.optimizer][0] });
    await umap!.run();
    annealStart = umap!.epoch - SCHEDULE_EPOCHS; // converged: nothing left to anneal
  }

  // 6. Release the instance when you are done (wasm memory, GPU buffers).
  function destroyInstance() {
    umap?.destroy();
    umap = null;
  }

  // ---------------------------------------------------------------------
  // The page (not part of the API walkthrough)
  // ---------------------------------------------------------------------

  // `rebuild` controls need a new instance; the others apply to the running one.
  const CONTROLS = [
    { key: "count", label: "Points", min: 1000, max: 20000, step: 1000, rebuild: true },
    { key: "dim", label: "Dimension", min: 2, max: 128, step: 1, rebuild: true },
    { key: "nNeighbors", label: "nNeighbors", min: 5, max: 50, step: 1, rebuild: true },
    { key: "minDist", label: "minDist", min: 0, max: 1, step: 0.01, rebuild: false },
    { key: "repulsionStrength", label: "repulsionStrength", min: 0.1, max: 16, step: 0.1, rebuild: false },
  ] as const;

  let settings = $state({
    count: 5000,
    dim: 32,
    nNeighbors: 15,
    minDist: 0.1,
    repulsionStrength: 1,
    optimizer: "adam" as Optimizer,
    gpu: true,
  });
  let status = $state("Setting up...");
  let running = $state(false); // the step loop is looping
  let converged = $state(false); // the learning rate schedule is complete
  let active = false; // the user wants the layout to optimize: Start was pressed and Pause was not
  let resumeAfterBuild = false;
  let loop: Promise<void> | null = null;
  let generation = 0; // identifies the latest rebuild, so an older one can discard its instance
  let frame = $state.raw<{ x: Float32Array<ArrayBuffer>; y: Float32Array<ArrayBuffer> } | null>(null);
  let viewportState = $state.raw<ViewportState | null>(null);

  let data = $derived(frame && { ...frame, category: dataset.labels as Uint8Array<ArrayBuffer> });

  /** Points of one shape in 3D. Each shape has a different structure that UMAP should keep. */
  const SHAPES: ((rng: () => number) => [number, number, number])[] = [
    (rng) => {
      // swiss roll: a rolled-up sheet
      const t = 1.5 * Math.PI * (1 + 2 * rng());
      return [(t * Math.cos(t)) / 4, 6 * rng() - 3, (t * Math.sin(t)) / 4];
    },
    (rng) => {
      // ring
      const t = 2 * Math.PI * rng();
      return [3 * Math.cos(t), 3 * Math.sin(t), 0];
    },
    (rng) => {
      // helix
      const t = 6 * Math.PI * rng();
      return [Math.cos(t), Math.sin(t), t / 6 - Math.PI];
    },
    (rng) => {
      // sphere surface
      const [x, y, z] = [randomNormal(0, 1, rng), randomNormal(0, 1, rng), randomNormal(0, 1, rng)];
      const r = Math.hypot(x, y, z) / 2.5;
      return [x / r, y / r, z / r];
    },
    (rng) => [randomNormal(0, 0.4, rng), randomNormal(0, 0.4, rng), randomNormal(0, 0.4, rng)], // dense blob
    (rng) => [randomNormal(0, 1.5, rng), randomNormal(0, 1.5, rng), randomNormal(0, 1.5, rng)], // sparse blob
  ];

  const categoryColors = defaultCategoryColors(SHAPES.length);

  /**
   * Six 3D shapes (swiss roll, ring, helix, sphere, dense and sparse blob), each mapped into `dim`
   * dimensions by its own random linear map and offset, plus a little noise.
   */
  function makeShapes(count: number, dim: number) {
    const rng = RNG(1);
    const basis = SHAPES.map(() => Float32Array.from({ length: 3 * dim }, () => randomNormal(0, 1, rng)));
    const offsets = SHAPES.map(() => Float32Array.from({ length: dim }, () => randomNormal(0, 3, rng)));
    const data = new Float32Array(count * dim);
    const labels = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      const shape = Math.floor(rng() * SHAPES.length);
      labels[i] = shape;
      const p = SHAPES[shape](rng);
      for (let d = 0; d < dim; d++) {
        const [b0, b1, b2] = [basis[shape][d], basis[shape][dim + d], basis[shape][2 * dim + d]];
        data[i * dim + d] = offsets[shape][d] + p[0] * b0 + p[1] * b1 + p[2] * b2 + randomNormal(0, 0.05, rng);
      }
    }
    return { data, labels };
  }

  /** Copies `umap.embedding` into one array per axis, which is what EmbeddingView takes. */
  function showFrame(umap: UMAP) {
    const xy = umap.embedding;
    const x = new Float32Array(xy.length / 2);
    const y = new Float32Array(xy.length / 2);
    for (let i = 0; i < x.length; i++) {
      x[i] = xy[2 * i];
      y[i] = xy[2 * i + 1];
    }
    frame = { x, y };
    status = `Epoch ${umap.epoch}`;
    converged = umap.epoch - annealStart >= SCHEDULE_EPOCHS;
  }

  /** Frames the layout once: center on the mean, show 3 standard deviations. */
  function fitViewport({ x, y }: { x: Float32Array; y: Float32Array }): ViewportState {
    const mean = (a: Float32Array) => a.reduce((s, v) => s + v, 0) / a.length;
    const stdev = (a: Float32Array) => Math.sqrt(a.reduce((s, v) => s + (v - mean(a)) ** 2, 0) / a.length);
    return { x: mean(x), y: mean(y), scale: 0.9 / (3 * Math.max(stdev(x), stdev(y), 1e-6)) };
  }

  function ensureLoop() {
    if (loop == null) {
      running = true;
      loop = Promise.resolve().then(stepLoop);
    }
  }

  async function stopLoop() {
    active = false;
    await loop;
  }

  async function pause() {
    resumeAfterBuild = false;
    await stopLoop();
  }

  /** Replaces the instance with a new one, and continues if the layout was optimizing. */
  async function build() {
    resumeAfterBuild ||= active;
    const mine = ++generation;
    await stopLoop();
    destroyInstance();
    status = "Setting up...";
    try {
      await createInstance((text) => mine === generation && (status = text));
    } catch (error) {
      status = `Error: ${error}`;
      return;
    }
    if (mine !== generation) return destroyInstance(); // a newer build has replaced this one
    showFrame(umap!);
    viewportState = fitViewport(frame!);
    active = resumeAfterBuild;
    resumeAfterBuild = false;
    if (active) ensureLoop();
  }

  async function start() {
    active = true;
    if (umap == null) return build();
    if (umap.epoch - annealStart >= SCHEDULE_EPOCHS) {
      await resetLayout();
      showFrame(umap);
    }
    ensureLoop();
  }

  async function reset() {
    await pause();
    if (umap == null) return;
    await resetLayout();
    showFrame(umap);
    viewportState = fitViewport(frame!);
  }

  async function runOnce() {
    if (umap == null) {
      await build();
    }
    await stopLoop();
    status = "Running...";
    await runToEnd();
    showFrame(umap!);
  }

  // Dragging a slider fires many input events; wait until it settles.
  function debounced(callback: () => void, ms = 150) {
    let timer: ReturnType<typeof setTimeout>;
    return () => {
      clearTimeout(timer);
      timer = setTimeout(callback, ms);
    };
  }
  const rebuildLater = debounced(() => umap && build());
  const applyLiveSettingsLater = debounced(applyLiveSettings);

  onMount(() => {
    start();
    return () => {
      generation++;
      active = false;
      destroyInstance();
    };
  });
</script>

<h1 class="font-bold">UMAP</h1>
<div class="flex gap-8 items-start">
  <div class="grid gap-3.5 w-70 flex-none">
    {#each CONTROLS as { key, label, min, max, step, rebuild }}
      <label class="grid gap-1">
        <span class="flex justify-between"><span>{label}</span><output>{settings[key]}</output></span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          bind:value={settings[key]}
          oninput={rebuild ? rebuildLater : applyLiveSettingsLater}
        />
      </label>
    {/each}
    <label class="grid gap-1">
      <span>Optimizer</span>
      <select bind:value={settings.optimizer} onchange={applyOptimizer}>
        <option>adam</option>
        <option>momentum</option>
        <option>sgd</option>
      </select>
    </label>
    <label class="flex items-center gap-2">
      <input type="checkbox" bind:checked={settings.gpu} onchange={() => umap && build()} />
      <span>Use the GPU (WebGPU)</span>
    </label>
    <div class="flex flex-wrap gap-2">
      <button class="px-2.5 border border-slate-400 rounded-sm disabled:opacity-40" onclick={start} disabled={running}
        >Start</button
      >
      <button class="px-2.5 border border-slate-400 rounded-sm disabled:opacity-40" onclick={pause} disabled={!running}
        >Pause</button
      >
      <button class="px-2.5 border border-slate-400 rounded-sm disabled:opacity-40" onclick={reset}>Reset</button>
      <button
        class="px-2.5 border border-slate-400 rounded-sm disabled:opacity-40"
        onclick={runOnce}
        disabled={converged}>Run to the end</button
      >
    </div>
  </div>
  <div>
    <div class="box-content border border-slate-300" style:width="{SIZE}px" style:height="{SIZE}px">
      {#if data}
        <EmbeddingView
          data={data}
          width={SIZE}
          height={SIZE}
          categoryColors={categoryColors}
          viewportState={viewportState}
          config={{ mode: "points", autoLabelEnabled: false }}
        />
      {/if}
    </div>
    <div class="mt-2 text-slate-600 tabular-nums">{status}</div>
  </div>
</div>
