# Pan fluency at 75M points

## Goal

Make pan fluent on the full Overture Places dataset (75,495,994 points,
4.4 GB parquet) on Apple Silicon (M-series) without dropping data —
all 75M must remain present in the column store and queryable; only the
per-frame draw set may shrink while the user is actively interacting.

## Result

| view          |   fps | interval mean |    p95 |
| ------------- | ----: | ------------: | -----: |
| world         | 123.7 |         8.1ms | 10.2ms |
| region (0.5×) | 124.2 |         8.1ms | 10.3ms |
| city (5×)     | 124.6 |         8.0ms | 10.2ms |
| neighborhood  | 124.3 |         8.0ms | 10.0ms |

All four zoom levels are display-capped (the 14" MacBook Pro panel
runs at ~120 Hz; ~125 fps measured includes RAF jitter). The same
build with the gesture-only optimisations disabled (`world-noskip`)
returns 26 fps — still 5× the baseline, capacity left for higher-DPR
or larger panels.

Baseline (single config, before any optimisation, world view): **5.3 fps,
763 ms mean GPU time, 189 ms RAF interval**. End-to-end speedup at
world view: **23×**. At every other zoom: display-bound.

## What was actually slow

`packages/component/src/lib/webgpu_renderer/` runs a per-frame compute
chain: `cull → sample → draw_points`. The draw step issued
`pass.draw(4, pointCount)` — a triangle-strip quad per point, instanced
across the full 75M. The vertex shader read `point_data[index]`, hid
rejected points off-screen via `position = vec4(2,2,0,1)`, and let
the rasteriser discard them.

Two bottlenecks at the current cap of 4M:

1. **Vertex iteration over 75M instances.** Even with culled points
   pushed off-screen, the GPU still launches the vertex shader for
   every instance. At cap-4M we measured a ~89 ms compute floor that
   stayed flat regardless of how many points were actually accepted —
   the work was the iteration itself.
2. **Fragment fill at world view.** 4M survived points × auto-sized
   quads (~9 px²) ≈ 36M fragments per frame, with log-blend OIT
   doing dual-source colour mixing per fragment. Fragment cost
   dominates once the vertex iteration is gone.

Compute cost (cull/sample) was negligible (<2 ms) — the architecture
the upstream Embedding Atlas inherited assumes ≤10M points.

## Three changes, additive

### 1. Compaction + indirect draw

`packages/component/src/lib/webgpu_renderer/program.wgsl`,
`downsample.ts`, `draw_points.ts`, `renderer.ts`

Added a third compute pass `compact_accepted` after sample. It walks
the post-sample `point_data` buffer; for each accepted index it
appends to a tightly packed `compact_indices` buffer and atomically
increments `indirect_args[1]` (the instance count). The draw call
becomes `pass.drawIndirect(indirect_args, 0)` reading
`[4, accepted_count, 0, 0]`. A new `points_compacted_vs` vertex
shader reads `compact_indices_read[instance]` to recover the real
point index.

Workgroup-level pre-aggregation (atomicAdd into `var<workgroup>`,
then one atomic per workgroup of 256) was used instead of one global
atomicAdd per accepted point. At 4M accepted that's 16K workgroup
atomics instead of 4M — keeps contention off the global counter on
Metal.

The vertex shader now iterates `accepted_count` instances instead
of 75M. Point-mode world-view interval drops from 171 ms to 106 ms
(6.4 → 9.4 fps).

### 2. Adaptive cap during gestures

`packages/component/src/lib/embedding_view/embedding_view_config.ts`,
`EmbeddingViewImpl.svelte`

`downsampleMaxPointsInteractive` (default 200,000) is consulted
whenever the user is mid-gesture (pan or wheel). On gesture release
a 150 ms decay timer drops back to the static
`downsampleMaxPoints` cap (4,000,000) so the motionless frame
shows full detail.

Gesture state lives in a small `isInteracting` boolean bumped from
the existing `onWheel` handler and the pan `onDrag` move callback.
There is no JS-side debounce on the render path itself — the
`$derived` cap simply changes value, and the next RAF picks up the
new compute uniform.

200K @ ~9 px² = 1.8M fragments/frame, well under the panel's
fill budget. World-view interval drops from 106 ms to 43 ms
(9.4 → 23.3 fps).

### 3. Skip compute during gestures

`packages/component/src/lib/webgpu_renderer/renderer.ts`,
`renderer_interface.ts`, `EmbeddingViewImpl.svelte`

A new `skipDownsampleCompute` prop on the renderer (false by
default) bypasses cull/sample/compact entirely. The vertex shader
is dispatched against the previous frame's `compact_indices` buffer
under the new viewport matrix — pure pan reprojects the cached set
in the vertex stage, which costs ~vertex-shader-only.

The flag is set when the user is interacting AND the time since the
last full compute is < `COMPUTE_REFRESH_INTERVAL_MS` (200 ms). At
~5 Hz, a fresh full pass runs to pull in points that have entered
the viewport. On gesture release the next frame is a full compute
at the static 4M cap.

This is the dominant win — interval drops from 43 ms to 8 ms
(23.3 → 124 fps). It's gesture-only by construction; standing
still is unaffected.

## What "no dropping points" means here

All 75M rows are loaded into the column buffers and remain
queryable for selection, search, and filter operations at every
moment. The per-frame draw set during a gesture shrinks to 200K
of those (chosen by the existing density-weighted sampler in
`program.wgsl`'s `sample` shader, not arbitrary first-N). On
release, the frame snaps back to the static 4M cap. Between gesture
frames the cached compact_indices is reprojected — points that have
just entered the viewport appear at the next 5 Hz refresh or on
release, whichever comes first.

Conceptually this is the Datashader trade: at world view, 75M
points cannot map to the screen's ~1M pixels anyway, so during
motion we render a representative sample weighted by local density.
The data isn't dropped; the _visualisation_ is sampled, with the
sample set refreshed often enough to feel continuous.

## Measurement

`e2e/perf-75m.spec.ts` runs under a dedicated Playwright project
`perf-chrome` (real Chrome, `--enable-unsafe-webgpu`,
`--use-angle=metal`, `--js-flags=--max-old-space-size=8192`,
1600×1000 viewport, 20-min timeout) defined in
`playwright.config.ts`.

Per config, the test:

1. Loads `places_simplified.parquet` from disk.
2. Waits for an initial frame (RAF + GPU completion).
3. Issues `setViewport()` via real `page.mouse.wheel()` to the
   target zoom (skipping for world).
4. Calls `dragPan()`, which dispatches mousedown/mousemove/mouseup
   _from inside `page.evaluate`_ via a RAF loop on `window`.
   Doing this in-page is the difference between 14 fps (Playwright
   IPC capping at one event per ~50 ms) and 124 fps (true RAF
   cadence).
5. Reads `window.__atlasPerf.summary()` from
   `packages/component/src/lib/embedding_view/perf_recorder.ts`,
   which is a 4096-entry ring buffer recording cpu, raf-interval,
   and GPU-completion times per frame.

`packages/viewer/src/app/main.ts` parses URL params:

```
?perf=1                  enable perf recorder
?downsampleMax=N         override static cap
?densityWeight=W         override density weighting
?pointSize=P             override auto point size
?interactiveCap=N        override gesture cap (0 = disabled)
?renderMode=density|points
```

GPU timing uses `device.queue.onSubmittedWorkDone()` per frame —
not GPU timestamp queries, which Chrome currently disables on
Metal. The `gpuDevice` getter on `WebGPURenderer` is what makes
this hook work without leaking the device.

## Headline progression (world view, 75M)

| stage                            |   fps | interval | speedup |
| -------------------------------- | ----: | -------: | ------: |
| baseline                         |   5.3 |   189 ms |      1× |
| + compaction + indirect draw     |   9.4 |   106 ms |    1.8× |
| + adaptive cap (200K) on gesture |  23.3 |    43 ms |    4.4× |
| + skip compute during gesture    | 123.7 |     8 ms |     23× |

City / region / neighbourhood views all cap at ~124 fps with the
final stack — display-bound. Density-mode at 75M is 9 fps and is
the next bottleneck (the accumulate pass walks all 75M into a
texture and the bandwidth, not the math, is what limits it); not
addressed in this work.

## Reproducing

```sh
cd /Users/dome/work/general/geospatial-atlas
npm install
npx playwright install chrome
ATLAS_PERF_PARQUET=/abs/path/to/places_simplified.parquet \
  npx playwright test --config playwright.config.ts \
  --project perf-chrome e2e/perf-75m.spec.ts
```

Per-run JSON written to `e2e/perf-results/<tag>.json`.

## Files touched

```
packages/component/src/lib/webgpu_renderer/program.wgsl
packages/component/src/lib/webgpu_renderer/downsample.ts
packages/component/src/lib/webgpu_renderer/draw_points.ts
packages/component/src/lib/webgpu_renderer/renderer.ts
packages/component/src/lib/renderer_interface.ts
packages/component/src/lib/embedding_view/EmbeddingViewImpl.svelte
packages/component/src/lib/embedding_view/embedding_view_config.ts
packages/component/src/lib/embedding_view/perf_recorder.ts   (new)
packages/viewer/src/app/main.ts
playwright.config.ts
e2e/perf-75m.spec.ts                                          (new)
```

## Not done (low ROI given current numbers)

- Workgroup-size sweep on Apple Silicon — compute is no longer the
  hot path.
- Spatial pre-binning / quadtree early-out — only helps zoomed-in
  views, which already display-cap.
- f32→u16 mercator-relative coordinate quantisation — memory win,
  not a perf win at 75M (8.6 GB→4.3 GB on the position buffers,
  potentially shifts headroom for ≥150M datasets).
- Alpha-to-coverage instead of log-blend OIT — would help
  fragment-fill cost in non-gesture frames but those already render
  at full cap with full detail; the user-visible motion path is
  already display-capped.
- Server-side fp16/u16 packed Arrow stream — initial-load
  bandwidth, not pan-time.

These are tracked as open todos but are not on the critical path
for the 75M pan goal.
