<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  interface Props<Selection> {
    data: {
      x: Float32Array<ArrayBuffer>;
      y: Float32Array<ArrayBuffer>;
      /** When set, ``x``/``y`` are ignored and the renderer unpacks
       *  these via a one-shot u32 → f32 GPU compute pass. */
      xPacked?: Uint32Array<ArrayBuffer> | null;
      yPacked?: Uint32Array<ArrayBuffer> | null;
      coordsBoundsX?: [number, number] | null;
      coordsBoundsY?: [number, number] | null;
      category: Uint8Array<ArrayBuffer> | null;
    };
    /** Set when the parent already projected ``data.y`` to Mercator on
     *  the server (GIS Path C optimisation). The internal projection
     *  loop is then a no-op pass-through. */
    yIsAlreadyMercator?: boolean;
    categoryCount: number;
    categoryColors: string[] | null;
    width: number;
    height: number;
    pixelRatio: number;
    theme: ThemeConfig | null;
    config: EmbeddingViewConfig | null;
    totalCount: number | null;
    maxDensity: number | null;
    labels?: Label[] | null;
    queryClusterLabels: ((clusters: Rectangle[][]) => Promise<(LabelContent | null)[]>) | null;
    tooltip: Selection | null;
    selection: Selection[] | null;
    querySelection: ((x: number, y: number, unitDistance: number) => Promise<Selection | null>) | null;
    rangeSelection: Rectangle | Point[] | null;
    defaultViewportState: ViewportState | null;
    viewportState: ViewportState | null;
    customTooltip: CustomComponent<HTMLDivElement, { tooltip: Selection }> | null;
    customOverlay: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;
    onViewportState: ((value: ViewportState) => void) | null;
    onTooltip: ((value: Selection | null) => void) | null;
    onSelection: ((value: Selection[] | null) => void) | null;
    onRangeSelection: ((value: Rectangle | Point[] | null) => void) | null;
    cache: Cache | null;
    cacheIdentifier?: any | null;
  }

  interface Cluster {
    x: number;
    y: number;
    sumDensity: number;
    rects: Rectangle[];
    bandwidth: number;
    content?: LabelContent | null;
  }

  function viewingParameters(
    maxDensity: number,
    minimumDensity: number,
    scale: number,
    pixelWidth: number,
    pixelHeight: number,
    pixelRatio: number,
    userPointSize: number | null,
  ) {
    // Convert max density to per unit point (aka., CSS px unit).
    let viewDimension = Math.max(pixelWidth, pixelHeight) / pixelRatio;
    let maxPointDensity = maxDensity / (scale * scale) / (viewDimension * viewDimension);
    let maxPixelDensity = maxPointDensity / (pixelRatio * pixelRatio);

    let densityScaler = (1 / maxPixelDensity) * 0.2;

    // The scale such that maxPointDensity == minDensity
    let threshold = Math.sqrt(maxDensity / minimumDensity / (viewDimension * viewDimension));
    let thresholdLevel = Math.log(threshold);
    let scaleLevel = Math.log(scale);

    let factor = (Math.min(Math.max((scaleLevel - thresholdLevel) * 2, -1), 1) + 1) / 2;

    let pointSize: number;
    if (userPointSize != null) {
      // Use user-provided point size, scaled by pixel ratio
      pointSize = userPointSize * pixelRatio;
    } else {
      // Use automatic calculation based on density
      let pointSizeAtThreshold = 0.25 / Math.sqrt(maxPointDensity);
      pointSize = Math.max(0.2, Math.min(5, pointSizeAtThreshold)) * pixelRatio;
    }

    let densityAlpha = 1 - factor;
    let pointsAlpha = 0.5 + factor * 0.5;

    return {
      densityScaler,
      densityAlpha,
      contoursAlpha: densityAlpha,
      pointSize,
      pointAlpha: 0.7,
      pointsAlpha: pointsAlpha,
      densityBandwidth: 20,
    };
  }
</script>

<script lang="ts">
  import { deepEquals, interactionHandler, objectHash, type CursorValue } from "@embedding-atlas/utils";
  import { onDestroy, onMount } from "svelte";

  import EditableRectangle from "./EditableRectangle.svelte";
  import Lasso from "./Lasso.svelte";
  import StatusBar from "./StatusBar.svelte";
  import TooltipContainer from "./TooltipContainer.svelte";

  import maplibregl from "maplibre-gl";
  import "maplibre-gl/dist/maplibre-gl.css";
  import { Protocol } from "pmtiles";

  const pmtilesProtocol = new Protocol();
  maplibregl.addProtocol("pmtiles", pmtilesProtocol.tile);

  import { defaultCategoryColors } from "../colors.js";
  import { isPerfEnabled, record as perfRecord, setPointCount as perfSetPointCount } from "./perf_recorder.js";
  import { pointDistance, throttleTooltip, type Point, type Rectangle, type ViewportState } from "../utils.js";
  import { Viewport } from "../viewport_utils.js";
  import { EmbeddingRenderer } from "../webgpu_renderer/renderer.js";
  import { requestWebGPUDevice } from "../webgpu_renderer/utils.js";
  import { customComponentAction, customComponentProps } from "./custom_component_helper.js";
  import type { EmbeddingViewConfig } from "./embedding_view_config.js";
  import { layoutLabels, type LabelWithPlacement } from "./labels.js";
  import { simplifyPolygon } from "./simplify_polygon.js";
  import { resolveTheme, type ThemeConfig } from "./theme.js";
  import type { Cache, CustomComponent, Label, LabelContent, OverlayProxy } from "./types.js";
  import { findClusters } from "./worker/index.js";

  interface SelectionBase {
    x: number;
    y: number;
    category?: number;
    text?: string;
  }

  type Selection = $$Generic<SelectionBase>;

  let {
    data = { x: new Float32Array(), y: new Float32Array(), category: null },
    yIsAlreadyMercator = false,
    categoryCount = 1,
    categoryColors = null,
    width = 800,
    height = 800,
    pixelRatio = 2,
    theme = null,
    config = null,
    totalCount = null,
    maxDensity = null,
    labels = null,
    queryClusterLabels = null,
    tooltip = null,
    selection = null,
    querySelection = null,
    rangeSelection = null,
    defaultViewportState = null,
    viewportState = null,
    customTooltip = null,
    customOverlay = null,
    onViewportState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    cache = null,
    cacheIdentifier = undefined,
  }: Props<Selection> = $props();

  let showClusterLabels = true;

  let colorScheme = $derived(config?.colorScheme ?? "light");
  let resolvedTheme = $derived(resolveTheme(theme, colorScheme));
  let resolvedCategoryColors = $derived(categoryColors ?? defaultCategoryColors(categoryCount));

  let isGis = $derived(config?.isGis ?? false);
  // Mercator-project the y (latitude) array on the JS side so the WGSL
  // shader can stay in projected coordinates. The naive form iterated
  // ``Viewport.projectLat`` 75 M times — ~7 s on Apple-Silicon Chrome and
  // the single biggest synchronous block in the entire cold-start path.
  // The optimised loop inlines the math (no function calls), folds the
  // ``Math.PI`` constants out of the inner loop, and runs ~10× faster.
  // Better still would be doing this in the vertex shader (one Mercator per
  // visible point per frame instead of 75 M up-front); leaving that as a
  // follow-up so we don't reshape the renderer in this change.
  let internalDataY = $derived.by(() => {
    if (isGis && data.y && !yIsAlreadyMercator) {
      const inY = data.y;
      const n = inY.length;
      const y = new Float32Array(n);
      const PI4 = Math.PI / 4;
      const PI180 = Math.PI / 180;
      const RAD_DEG = 180 / Math.PI;
      const log = Math.log;
      const tan = Math.tan;
      const __t0 = typeof performance !== "undefined" ? performance.now() : 0;
      for (let i = 0; i < n; i++) {
        const latRad = inY[i] * PI180;
        y[i] = log(tan(PI4 + latRad * 0.5)) * RAD_DEG;
      }
      if (typeof performance !== "undefined" && n > 1_000_000) {
        const w: any = window as any;
        if (!w.__atlasMercatorLogged) {
          w.__atlasMercatorLogged = true;
          console.log(`[atlas-stage] mercator-loop ${(performance.now() - __t0).toFixed(0)}ms n=${n}`);
        }
      }
      return y;
    }
    return data.y;
  });

  // In GIS mode, the minimum scale corresponds to MapLibre zoom level 1.
  // Below this, MapLibre's jumpTo can't keep up and the map/point layers diverge.
  let gisMinScale = $derived((1024 * 2) / (360 * width));

  let resolvedViewportState = $derived.by(() => {
    let state = viewportState ?? defaultViewportState ?? { x: 0, y: 0, scale: 1 };
    if (isGis) {
      let clamped = false;
      let scale = state.scale;
      let y = state.y;
      if (scale < gisMinScale) {
        scale = gisMinScale;
        clamped = true;
      }
      // Clamp Y so the view stays within Mercator bounds (±85.05° lat → ±180 projected)
      const mercatorMax = 180;
      const sy = width >= height ? (scale * width) / height : scale;
      const visibleHalfY = 1 / sy;
      const yMin = -mercatorMax + visibleHalfY;
      const yMax = mercatorMax - visibleHalfY;
      if (yMin < yMax) {
        if (y < yMin) {
          y = yMin;
          clamped = true;
        }
        if (y > yMax) {
          y = yMax;
          clamped = true;
        }
      }
      if (clamped) {
        return { ...state, scale, y };
      }
    }
    return state;
  });
  let resolvedViewport = $derived(new Viewport(resolvedViewportState, width, height, isGis));
  let pointLocation = $derived(resolvedViewport.pixelLocationFunction());
  let coordinateAtPoint = $derived(resolvedViewport.coordinateAtPixelFunction());

  let preventHover = $state(false);
  // True while the user is actively interacting (drag or recent wheel).
  // Used to drop the effective downsample cap so high-frame-rate pan stays
  // fluent at very large datasets — the configured cap only kicks in once
  // the gesture ends and the renderer has time to draw the full sample.
  let isInteracting = $state(false);
  // Set to true after the first full (non-interactive) render. Gating
  // skipDownsampleCompute on this avoids drawing from an empty
  // compact_indices buffer when a user starts panning before the initial
  // render lands.
  let hasInitialFrame = $state(false);
  // Wall-clock time of the last full downsample compute. Kept for
  // diagnostics / Playwright assertions; the live render path no longer
  // refreshes mid-gesture (see skipDownsampleCompute below) — re-compute
  // happens on gesture release once isInteracting flips false.
  let lastFullComputeAt = $state(0);
  let interactionDecayTimer: ReturnType<typeof setTimeout> | null = null;
  function bumpInteraction() {
    isInteracting = true;
    if (interactionDecayTimer != null) clearTimeout(interactionDecayTimer);
    interactionDecayTimer = setTimeout(() => {
      isInteracting = false;
      interactionDecayTimer = null;
    }, 150);
  }

  // Wheel-zoom suppression. Mid-wheel, every tick changes ``scale`` —
  // and at 322 M points each render is 25-90 s, so a continuous scroll
  // queues N stale renders that the user will never see anyway. Hide
  // the canvas the instant a wheel tick fires, suppress every render
  // until 500 ms after the last tick, then reveal the canvas and let
  // the reactive system fire ONE render at the final zoom level.
  // ``viewportState`` keeps updating throughout the gesture so when
  // we un-suppress, ``$effect.pre`` sees the final scale and renders
  // exactly once.
  let isWheeling = $state(false);
  let wheelEndTimer: ReturnType<typeof setTimeout> | null = null;
  const WHEEL_END_DEBOUNCE_MS = 500;
  function bumpWheel() {
    if (!isWheeling) {
      isWheeling = true;
      // Hide the canvas instantly. CSS opacity is the lightest hammer:
      // it doesn't change layout, it leaves the WebGPU swap-chain
      // texture intact, and the GPU compositor short-circuits the
      // composited frame contribution. visibility:hidden would also
      // disable hit-testing on overlapping SVG handlers, which we
      // still want active for the wheel handler to keep capturing.
      if (canvas) canvas.style.opacity = "0";
    }
    if (wheelEndTimer != null) clearTimeout(wheelEndTimer);
    wheelEndTimer = setTimeout(() => {
      isWheeling = false;
      wheelEndTimer = null;
      if (canvas) canvas.style.opacity = "";
      // Force the reactive render path. ``$effect.pre`` re-runs because
      // ``isWheeling`` is read inside it, but the renderer's own
      // ``setProps`` change-detection may decide nothing changed if
      // we suppressed only one render in the gesture (rare). Calling
      // setNeedsRender explicitly guarantees the gesture's final
      // viewport actually paints.
      setNeedsRender();
    }, WHEEL_END_DEBOUNCE_MS);
  }

  function compareSelection(a: Selection, b: Selection) {
    return a.x == b.x && a.y == b.y && a.category == b.category && a.text == b.text;
  }

  let lockTooltip = $derived(selection?.length == 1 && tooltip != null && compareSelection(selection[0], tooltip));

  function setViewportState(state: ViewportState) {
    if (deepEquals(viewportState, state)) {
      return;
    }
    viewportState = state;
    onViewportState?.(state);
  }

  function setTooltip(newValue: Selection | null) {
    if (deepEquals(tooltip, newValue)) {
      return;
    }
    tooltip = newValue;
    onTooltip?.(newValue);
  }

  function setSelection(newValue: Selection[] | null) {
    if (deepEquals(selection, newValue)) {
      return;
    }
    selection = newValue;
    onSelection?.(newValue);
  }

  function setRangeSelection(newValue: Rectangle | Point[] | null) {
    if (deepEquals(rangeSelection, newValue)) {
      return;
    }
    rangeSelection = newValue;
    onRangeSelection?.(newValue);
  }

  let clusterLabels: LabelWithPlacement[] = $state([]);
  let statusMessage: string | null = $state(null);

  let selectionMode = $state<"marquee" | "lasso" | "none">("none");

  let pixelWidth = $derived(width * pixelRatio);
  let pixelHeight = $derived(height * pixelRatio);

  let canvas: HTMLCanvasElement | null = $state(null);
  let renderer: EmbeddingRenderer | null = $state(null);
  let webGPUError: string | null = $state(null);

  let mapContainer: HTMLElement | undefined = $state();
  let map: maplibregl.Map | undefined = $state();

  let minimumDensity = $derived(config?.minimumDensity ?? 1 / 16);
  // Allow URL params (?downsampleMax=, ?densityWeight=, ?pointSize=) to override
  // the spec for ad-hoc perf experiments without rebuilding.
  let perfOverrides = (typeof window !== "undefined" ? (window as any).__atlasPerfOverrides : null) ?? null;
  let userPointSize = $derived(perfOverrides?.pointSize ?? config?.pointSize ?? null);
  let mode = $derived(perfOverrides?.renderMode ?? config?.mode ?? "points");
  let autoLabelEnabled = $derived(config?.autoLabelEnabled);
  let downsampleMaxPoints = $derived(perfOverrides?.downsampleMaxPoints ?? config?.downsampleMaxPoints ?? 4000000);
  let downsampleDensityWeight = $derived(
    perfOverrides?.downsampleDensityWeight ?? config?.downsampleDensityWeight ?? 5,
  );
  // Cap to use during active drag/wheel. Defaults to a value that keeps
  // 75M-row Overture parquets at >20fps on Apple Silicon WebGPU; opt-out
  // via downsampleMaxPointsInteractive=null in config or ?interactiveCap=0
  // in the URL.
  let downsampleMaxPointsInteractive = $derived(
    // Explicit Infinity / very large value disables the adaptive cap.
    // ?? short-circuits on null and undefined, so we deliberately use a
    // sentinel here: the caller can pass Infinity to opt out without
    // having to know the configured `downsampleMaxPoints`.
    perfOverrides && "downsampleMaxPointsInteractive" in perfOverrides
      ? perfOverrides.downsampleMaxPointsInteractive
      : (config?.downsampleMaxPointsInteractive ?? 200_000),
  );
  let effectiveDownsampleMaxPoints = $derived.by(() => {
    // Once we have an initial frame, ``skipDownsampleCompute`` is true
    // during interaction — the compute pass is skipped and the renderer
    // reuses the previously compacted indices via ``drawIndirect``. In
    // that mode the GPU draws exactly the instance count baked into
    // ``indirectArgsBuffer`` from the LAST full render. Shrinking
    // ``downsampleMaxPoints`` here would shrink ``compactIndicesBuffer``
    // (its size is derived from ``maxPoints``), but ``indirectArgsBuffer``
    // would still carry the old high count — ``points_compacted_vs`` then
    // reads compact_indices[0..oldCount] from the now-smaller buffer,
    // overflows it, and trips the macOS GPU watchdog
    // (``kIOGPUCommandBufferCallbackErrorTimeout``). That cascade was the
    // root cause of the 322 M-row world-view crash on rapid pan/zoom.
    //
    // Skipping the compute pass already makes mid-gesture frames cheap
    // (drawIndirect over the prior compact set is ~2-3 ms), so the
    // interactive cap buys us nothing — keep the buffer at its full
    // allocated size for the entire gesture lifetime.
    if (isInteracting && downsampleMaxPointsInteractive != null && Number.isFinite(downsampleMaxPointsInteractive)) {
      if (hasInitialFrame) {
        return downsampleMaxPoints;
      }
      return Math.min(downsampleMaxPoints, downsampleMaxPointsInteractive as number);
    }
    return downsampleMaxPoints;
  });
  let effectiveDownsampleDensityWeight = $derived.by(() => {
    // Density weighting buys nothing while the user is dragging — the
    // renderer otherwise pays for accumulate + blur over all 75M points
    // every frame just to bias sampling that's about to be redrawn.
    if (isInteracting) return 0;
    // At very large N the accumulate pass (one atomic add per point
    // into a ~480-cell density grid) becomes pathologically slow on
    // wide-angle views. At world-view zoom on the 322 M eubucco file,
    // ~250 M points concentrate over Europe — every grid cell sees
    // ~500 K serialized atomic adds, the single Metal command buffer
    // runs >5 s, and the macOS GPU watchdog kills it (→ device.lost).
    // Skipping density weighting falls back to uniform random
    // sampling: with downsampleMaxPoints=4 M the acceptance rate is
    // ~1 % either way, so the visual difference is invisible at 322 M.
    // ``mode == "density"`` still runs accumulate via wantsDensityOverlay
    // (the user explicitly asked for the density visualisation), so
    // this gate is points-mode only.
    const n = (data.xPacked?.length ?? 0) || (data.x?.length ?? 0);
    if (mode != "density" && n > 50_000_000) return 0;
    return downsampleDensityWeight;
  });
  let mapStyle = $derived(
    isGis ? (config?.mapStyle !== undefined ? config.mapStyle : "https://tiles.openfreemap.org/styles/liberty") : null,
  );
  let basemapAttribution = $derived(
    typeof mapStyle === "string" && mapStyle.toLowerCase().includes("openfreemap")
      ? "OpenFreeMap © OpenMapTiles Data from OpenStreetMap"
      : null,
  );

  let viewingParams = $derived(
    viewingParameters(
      maxDensity ?? (totalCount ?? data.x.length) / 4,
      minimumDensity,
      resolvedViewportState.scale,
      pixelWidth,
      pixelHeight,
      pixelRatio,
      userPointSize,
    ),
  );

  let pointSize = $derived(viewingParams.pointSize);

  let needsUpdateLabels = true;
  let previousLabels: Label[] | null = null;

  // The viewport at time of the last actual WebGPU render. When a gesture
  // is a pure pan (scale unchanged), we skip the compute+draw pipeline
  // entirely and CSS-translate the already-rendered canvas by the pixel
  // delta between renderedViewport and the current viewport. This is
  // analogous to how maplibre's basemap tiles pan: the expensive work
  // (rasterising points at 4M cap) runs once, then pan is pure compositing.
  // On pan end, or if the pan distance exceeds ~half the viewport (at
  // which point the canvas would expose blank edges), we re-render.
  let renderedViewport: { x: number; y: number; scale: number } | null = null;

  function cssPanDelta() {
    if (!renderedViewport) return null;
    const dx = resolvedViewportState.x - renderedViewport.x;
    const dy = resolvedViewportState.y - renderedViewport.y;
    const s = renderedViewport.scale;
    // Same aspect-corrected scaling the Viewport matrix uses — keeps CSS
    // translate pixel-perfect against where a fresh render would have
    // placed the same world coordinates.
    const sx = width < height ? s * (height / width) : s;
    const sy = width < height ? s : s * (width / height);
    const dxCss = -dx * sx * (width / 2);
    const dyCss = dy * sy * (height / 2);
    return { dxCss, dyCss };
  }

  function applyCssPan() {
    if (!canvas) return;
    const d = cssPanDelta();
    if (!d) return;
    canvas.style.transform = `translate(${d.dxCss}px, ${d.dyCss}px)`;
  }

  function clearCssPan() {
    if (canvas) canvas.style.transform = "";
  }

  // Debug counters exposed on window for Playwright — lets a test assert
  // the CSS-pan path actually ran and the clear happened only after GPU
  // present. Cheap, stripped by tree-shaking if __atlasPerfEnabled is
  // never set on window.
  function bumpDbg(key: string) {
    if (typeof window === "undefined") return;
    const dbg = ((window as any).__atlasPanDbg ??= {
      cssPanApplied: 0,
      cssPanSkipped_noInteract: 0,
      cssPanSkipped_noRendered: 0,
      cssPanSkipped_scaleChanged: 0,
      cssPanSkipped_overLimit: 0,
      clearedViaGpuDone: 0,
      renderCalls: 0,
      lastTransform: "",
    });
    dbg[key] = (dbg[key] ?? 0) + 1;
  }

  $effect.pre(() => {
    if (labels !== previousLabels) {
      previousLabels = labels;
      needsUpdateLabels = true;
    }

    // Wheel-zoom suppression: mid-gesture, every tick changes scale
    // and would otherwise queue a 25-90 s render at 322 M points. We
    // hide the canvas in ``bumpWheel`` and short-circuit here. The
    // 500 ms wheel-end timer flips ``isWheeling`` false and triggers
    // setNeedsRender so a SINGLE render fires at the final zoom.
    // Note: read isWheeling INSIDE the effect so Svelte's reactivity
    // re-runs us when the flag flips back to false. ``viewportState``
    // continues to update during the gesture, so the post-flip render
    // sees the final scale + position.
    if (isWheeling) {
      return;
    }
    // CSS-pan fast path: mid-gesture, scale unchanged, and the canvas
    // bitmap still covers the viewport. Zero GPU work — the user's
    // insight that "if I'm only panning, the points are already there".
    if (!isInteracting) {
      bumpDbg("cssPanSkipped_noInteract");
    } else if (renderedViewport == null) {
      bumpDbg("cssPanSkipped_noRendered");
    } else if (Math.abs(resolvedViewportState.scale - renderedViewport.scale) >= 1e-9) {
      bumpDbg("cssPanSkipped_scaleChanged");
    } else if (isInteracting && renderedViewport != null && canvas != null) {
      const d = cssPanDelta();
      if (d != null) {
        // Always apply CSS-pan during interaction — never fall through
        // to a mid-gesture re-render, which causes a visible flash as
        // the canvas briefly shows old content at its un-translated DOM
        // position. Accept blank edges at extreme pan distances; the
        // user either stays within the cached region or tolerates a
        // bit of unrendered margin. The single re-render happens on
        // gesture release.
        const t = `translate(${d.dxCss}px, ${d.dyCss}px)`;
        canvas.style.transform = t;
        bumpDbg("cssPanApplied");
        if (typeof window !== "undefined") {
          ((window as any).__atlasPanDbg as any).lastTransform = t;
        }
        return;
      }
    }
    // Deliberately NOT clearing the CSS transform here. If we cleared it
    // before the new render lands on the canvas, there's a visible gap
    // (CSS style change is synchronous, GPU presentation is not) where
    // the old bitmap sits at its un-translated DOM position while the
    // basemap is already at the new viewport — that's the "points
    // disappear" flash during release. The render() function clears the
    // transform after the GPU has finished presenting the new frame.

    let needsRender = renderer?.setProps({
      mode: mode,
      colorScheme: colorScheme,
      viewportX: resolvedViewportState.x,
      viewportY: resolvedViewportState.y,
      viewportScale: resolvedViewportState.scale,
      width: pixelWidth,
      height: pixelHeight,
      x: data.x,
      y: internalDataY,
      // Packed pass-through. When set, the renderer ignores ``x``/``y``
      // and unpacks via the GPU compute pass — see ``unpack.wgsl``.
      // ``yIsAlreadyMercator`` is enforced upstream in
      // ``EmbeddingViewMosaic`` (always true for the precomputed-u32
      // path), so the renderer never needs to re-project these.
      xPacked: data.xPacked ?? null,
      yPacked: data.yPacked ?? null,
      coordsBoundsX: data.coordsBoundsX ?? null,
      coordsBoundsY: data.coordsBoundsY ?? null,
      category: data.category,
      categoryCount,
      categoryColors: resolvedCategoryColors,
      downsampleMaxPoints: effectiveDownsampleMaxPoints,
      downsampleDensityWeight: effectiveDownsampleDensityWeight,
      // Reuse the last compacted set for the entire duration of an
      // interaction — no downsample/compact passes mid-gesture. At
      // 322 M-row scale a single full compute is multi-hundred-MB of
      // intermediate GPU buffers; running it every 200 ms during a
      // pan was ratcheting the renderer process past macOS jetsam
      // and killing the tab after a few seconds. Trade-off: points
      // entering the viewport during a long zoom-and-drag don't
      // appear until release. Acceptable — the release re-render
      // catches up.
      skipDownsampleCompute:
        isInteracting &&
        hasInitialFrame &&
        // When cap is 0 the stale indirect-args from the last non-zero compute
        // would replay the previous frame's draw count — force compute so the
        // downsample early-return clears the args to zero.
        effectiveDownsampleMaxPoints > 0,
      isGis,
      ...viewingParams,
    });
    if (needsRender) {
      // We can't tell from setProps whether the renderer chose the skip
      // path — the next render() does. Approximate: any frame that wasn't
      // configured to skip is a "full" frame.
      const willCompute = !isInteracting || !hasInitialFrame;
      if (willCompute) {
        lastFullComputeAt = performance.now();
        hasInitialFrame = true;
      }
      // Snapshot the viewport that this full render will reflect — the
      // CSS-pan fast path needs it to compute the translate delta on
      // the next mid-gesture frame.
      renderedViewport = {
        x: resolvedViewportState.x,
        y: resolvedViewportState.y,
        scale: resolvedViewportState.scale,
      };
    }

    if (needsRender) {
      setNeedsRender();
    }

    if (
      (autoLabelEnabled !== false || labels != null) &&
      needsUpdateLabels &&
      renderer != null &&
      data.x != null &&
      data.x.length > 0 &&
      defaultViewportState != null
    ) {
      needsUpdateLabels = false;
      updateLabels(defaultViewportState);
    }
  });

  async function render() {
    _request = null;
    if (!canvas || !renderer) {
      return;
    }
    // Backpressure gate — see ``_renderInFlight`` declaration above for
    // why this exists. Rapid pan-releases at 322 M would otherwise queue
    // multi-second GPU pipelines back-to-back and freeze the tab.
    if (_renderInFlight) {
      _renderPending = true;
      return;
    }
    // Wait for any in-flight u32 → f32 unpack chain to land before
    // submitting the draw. The chain is sequenced (X drain + destroy,
    // then Y) to keep peak GPU residency bounded — without this await
    // the post-data-load frame would race the unpack and read either
    // a half-populated f32 destination (Y still empty zeros → all
    // points collapse to the y-min line) or freshly allocated zero
    // memory (whole scatter at the bbox corner). The promise is a
    // no-op for the WebGL fallback and for steady-state pan/zoom where
    // no new packed buffers have arrived since the last render.
    const unpackPromise = renderer.unpackInFlight;
    if (unpackPromise) {
      const renderToken = _renderToken;
      try {
        await unpackPromise;
      } catch {
        // Chain failures are surfaced inside the renderer; fall through
        // to render whatever the f32 destinations currently hold.
      }
      // Renderer may have been recreated mid-await (device.lost path).
      // The token bumps in the ``$effect(() => { ... })`` above; if it
      // moved, drop this frame — a fresh render() will fire on the new
      // renderer.
      if (renderToken !== _renderToken || !renderer || !canvas) {
        return;
      }
    }
    // Only assign width/height when they actually change. Per HTML spec,
    // even same-value assignment resets the canvas bitmap, which on WebGPU
    // invalidates the swap chain and causes a 1-frame blank at the moment
    // of release.
    if (canvas.width !== renderer.props.width) canvas.width = renderer.props.width;
    if (canvas.height !== renderer.props.height) canvas.height = renderer.props.height;
    const cssW = `${renderer.props.width / pixelRatio}px`;
    const cssH = `${renderer.props.height / pixelRatio}px`;
    if (canvas.style.width !== cssW) canvas.style.width = cssW;
    if (canvas.style.height !== cssH) canvas.style.height = cssH;
    const localCanvas = canvas;
    const dev = (renderer as any).gpuDevice as GPUDevice | undefined;
    const t0 = performance.now();
    // Count must reflect whichever fill path is active. On the packed
    // (u32-direct) path ``props.x`` is the empty sentinel and the real
    // length lives on ``props.xPacked`` — without this, the
    // first-frame flag and the perf log were silently disabled at
    // 322 M scale.
    const count = (renderer.props.xPacked?.length ?? 0) || (renderer.props.x?.length ?? 0);
    const perfOn = isPerfEnabled();
    if (perfOn && count > 1_000_000) {
      const w: any = window as any;
      if (!w.__atlasFirstBigRenderLogged) {
        w.__atlasFirstBigRenderLogged = true;
        console.log(`[atlas-stage] first-big-render-start ${t0.toFixed(0)} count=${count}`);
      }
    }
    renderer.render();
    // Backpressure: this render's submit is now in flight. Block
    // subsequent renders until the GPU drains. WebGL fallback has no
    // ``onSubmittedWorkDone`` — it pipelines implicitly through its own
    // swap chain, so we leave ``_renderInFlight`` false.
    if (dev) {
      _renderInFlight = true;
      const myToken = ++_renderToken;
      let settled = false;
      const finish = (reason: "drain" | "watchdog" | "rejected") => {
        if (settled || myToken !== _renderToken) return;
        settled = true;
        clearTimeout(watchdog);
        _renderInFlight = false;
        if (reason === "watchdog") {
          console.warn(
            "[atlas] render backpressure watchdog fired — onSubmittedWorkDone() did not settle in 300 s. Likely device.lost or GPU process hang; force-resetting so subsequent renders can proceed.",
          );
        }
        if (_renderPending) {
          _renderPending = false;
          setNeedsRender();
        }
      };
      // 300 s (5 min) budget. The no-cap path draws every one of the
      // 322 M eubucco points by chunking the instance draw into 81 ×
      // 4 M-instance cmd buffers. Each cmd buffer is sized to fit
      // Metal's 5 s per-buffer watchdog (~0.6 s wall on Apple GPU at
      // world zoom). Total wall-clock for the whole render is
      // typically 25-90 s, but a contended GPU process or a slow
      // tail-pass can stretch it. The original 30 s budget assumed
      // downsampled drawIndirect (~1 s); 120 s was a halfway bump.
      // 5 min keeps us comfortably above any realistic worst case
      // while still catching genuine device.lost / GPU-process hangs.
      // Premature reset is the trigger for cascading kIOGPU timeouts:
      // it queues a *second* render on top of the in-flight first
      // one, doubling cmd-buffer load past the Metal 5 s ceiling.
      const watchdog = setTimeout(() => finish("watchdog"), 300_000);
      dev.queue.onSubmittedWorkDone().then(
        () => finish("drain"),
        () => finish("rejected"),
      );
    }
    // Always-on "first GPU frame on screen" signal. The viewer's
    // ``EmbeddingAtlas`` column-chart discovery polls
    // ``__atlasFirstBigRenderGpuLogged`` to decide when to mount the side
    // panel — gating it on perf mode meant the panel only appeared with
    // ``?perf=1`` (or after a 60 s safety net). One observer per render
    // until the flag flips, then this branch is dead.
    if (dev && count > 0 && !(window as any).__atlasFirstBigRenderGpuLogged) {
      dev.queue.onSubmittedWorkDone().then(() => {
        const w: any = window as any;
        if (!w.__atlasFirstBigRenderGpuLogged) {
          w.__atlasFirstBigRenderGpuLogged = true;
          if (perfOn && count > 1_000_000) {
            const dur = performance.now() - t0;
            console.log(
              `[atlas-stage] first-big-render-gpu-done ${performance.now().toFixed(0)} took=${dur.toFixed(0)}ms`,
            );
          }
        }
      });
    }
    if (perfOn) {
      const dt = performance.now() - t0;
      const cap = renderer.props.downsampleMaxPoints;
      const downsampled = cap != null && Number.isFinite(cap) && cap > 0 && count > cap;
      perfSetPointCount(count);
      const whenGpuDone = dev ? dev.queue.onSubmittedWorkDone().then(() => performance.now() - t0) : undefined;
      perfRecord({ cpuMs: dt, downsampled, whenGpuDone });
    }
    // Clear the CSS-pan transform after the GPU has finished presenting
    // the fresh frame — only then is it safe to drop the old translated
    // bitmap. Clearing earlier causes a visible misalignment flash.
    // Guarded by !isInteracting so we don't clobber a new in-flight pan
    // that may have started before the prior GPU work finished.
    bumpDbg("renderCalls");
    if (dev) {
      // Defer the CSS-transform clear by one rAF after onSubmittedWorkDone.
      // The rAF fires just BEFORE the next compositor paint, so when the
      // new swap-chain texture and the cleared transform both get committed
      // they land on the same paint cycle — a stale transform on new
      // content would be a visible double-translate flash. Clearing
      // straight from the microtask lets one paint cycle go by with the
      // old texture plus cleared transform, which reads as a "snap back"
      // to pre-drag position.
      dev.queue.onSubmittedWorkDone().then(() => {
        if (localCanvas && !isInteracting) {
          requestAnimationFrame(() => {
            if (localCanvas && !isInteracting) {
              localCanvas.style.transform = "";
              bumpDbg("clearedViaGpuDone");
            }
          });
        }
      });
    } else if (!isInteracting) {
      requestAnimationFrame(() => {
        if (localCanvas && !isInteracting) {
          localCanvas.style.transform = "";
          bumpDbg("clearedViaGpuDone");
        }
      });
    }
  }

  let _request: number | null = null;
  // GPU backpressure. ``renderer.render()`` is fire-and-forget — the
  // submit returns instantly, the GPU drains async. At 322 M points a
  // single full pipeline (accumulate → blur → downsample → draw) takes
  // ~3 s on Apple GPU. Without backpressure, two pan-releases inside
  // 3 s queue ~6 s of GPU work; the macOS Metal command-buffer
  // watchdog can kill long submits and the compositor (same GPU
  // process) starves — Chrome's window goes unresponsive.
  //
  // We coalesce: while ``_renderInFlight`` is true, additional
  // setNeedsRender / requestAnimationFrame fires only flag
  // ``_renderPending``. When the in-flight pipeline drains via
  // ``device.queue.onSubmittedWorkDone()``, we run **one** catch-up
  // render against the latest viewport state. Net: at most one render
  // queued at a time; rapid pan-releases collapse into a single
  // post-drain frame.
  //
  // Resilience: ``onSubmittedWorkDone()`` does NOT resolve when the
  // device is lost (Chromium/Dawn behavior). Without protection,
  // ``_renderInFlight`` stays true forever, every later render goes
  // pending, and the canvas stops updating. Three guards:
  //   1. ``_renderToken`` — late callbacks from a dead device are
  //      ignored (the token doesn't match anymore).
  //   2. 30 s watchdog — force-reset if the Promise never settles.
  //   3. ``$effect`` on ``renderer`` — when the device.lost handler
  //      recreates the renderer, bump the token and clear the flags
  //      so the new device starts with a clean slate.
  let _renderInFlight = false;
  let _renderPending = false;
  let _renderToken = 0;
  function setNeedsRender() {
    if (_request == null) {
      _request = requestAnimationFrame(render);
    }
  }
  $effect(() => {
    // Reset backpressure whenever the renderer reference changes
    // (initial mount + post-device.lost recreation). Bumping the
    // token invalidates any in-flight onSubmittedWorkDone callback
    // bound to the previous device. Flushing ``_renderPending``
    // ensures a queued frame from the old renderer turns into a
    // fresh frame on the new one.
    if (renderer != null) {
      _renderToken++;
      _renderInFlight = false;
      if (_renderPending) {
        _renderPending = false;
        setNeedsRender();
      }
    }
  });

  function setupWebGPURenderer(canvas: HTMLCanvasElement) {
    async function createRenderer() {
      let result = await requestWebGPUDevice();
      if (result == null) {
        console.error("Could not get WebGPU device");
        webGPUError = "WebGPU is unavailable. This view requires a browser with WebGPU support.";
        return;
      }
      const { device, useF16 } = result;

      let context = canvas.getContext("webgpu");
      if (context == null) {
        console.error("Could not get WebGPU canvas context");
        webGPUError = "WebGPU is unavailable. This view requires a browser with WebGPU support.";
        return;
      }

      // Surface any WebGPU validation / shader errors that aren't captured by an
      // explicit error scope, so failures don't manifest as a silently blank canvas.
      device.addEventListener("uncapturederror", (event) => {
        console.error("WebGPU uncaptured error:", (event as GPUUncapturedErrorEvent).error);
      });

      device.lost.then(async (info) => {
        console.info(`WebGPU device was lost: ${info.message}`);
        if (info.reason != "destroyed") {
          renderer?.destroy();
          renderer = null;
          context.unconfigure();
          await createRenderer();
        }
      });

      let format = navigator.gpu.getPreferredCanvasFormat();

      context.configure({
        device: device,
        format: format,
        alphaMode: "premultiplied",
      });

      renderer = new EmbeddingRenderer(context, device, format, pixelWidth, pixelHeight, useF16);
    }

    createRenderer();
  }

  function syncViewportState(defaultViewportState: ViewportState | null) {
    if (defaultViewportState != null && viewportState == null) {
      setViewportState(defaultViewportState);
    }
  }

  $effect.pre(() => syncViewportState(defaultViewportState));

  $effect(() => {
    if (map && resolvedViewportState) {
      const { x, y, scale } = resolvedViewportState;
      const zoom = Math.log2((360 * scale * width) / 1024);

      map.jumpTo({
        center: [x, isGis ? Viewport.unprojectLat(y) : y],
        zoom: zoom,
      });

      // Expose viewport state for E2E drift testing
      if (typeof window !== "undefined") {
        (window as any).__geospatialAtlasViewport = { x, y, scale, width, height, isGis };
      }
    }
  });

  $effect(() => {
    if (map && width && height) {
      map.resize();
    }
  });

  $effect(() => {
    if (mapContainer && mapStyle) {
      if (!map) {
        map = new maplibregl.Map({
          container: mapContainer,
          style: mapStyle,
          center: [
            resolvedViewportState.x,
            isGis ? Viewport.unprojectLat(resolvedViewportState.y) : resolvedViewportState.y,
          ],
          zoom: Math.log2((360 * resolvedViewportState.scale * width) / 1024),
          minZoom: 1,
          interactive: false,
          attributionControl: false,
          // Needed so MCP `get_map_screenshot` (and any other toDataURL-
          // based capture) includes the rendered basemap tiles. Without
          // this, the WebGL drawing buffer is cleared after each frame
          // and the screenshot shows only the overlay.
          canvasContextAttributes: { preserveDrawingBuffer: true },
        });
        // Expose map instance for E2E testing (no-op in production bundles via tree-shaking)
        if (typeof window !== "undefined") {
          (window as any).__geospatialAtlasMap = map;
        }
      } else {
        map.setStyle(mapStyle);
      }
    } else if (map && !mapStyle) {
      map.remove();
      map = undefined;
    }
  });

  onMount(() => {
    if (canvas == null) {
      return;
    }
    // Setup WebGPU renderer
    setupWebGPURenderer(canvas);

    // Override toDataURL. This is because we must submit the render commands before
    // calling toDataURL, to ensure the current image is populated with contents.
    let _toDataURL = canvas.toDataURL;
    canvas.toDataURL = (...args) => {
      render();
      return _toDataURL.apply(canvas, args);
    };
  });

  onDestroy(() => {
    renderer?.destroy();
    renderer = null;
    if (map) {
      map.remove();
      map = undefined;
    }
  });

  function localCoordinates(e: { clientX: number; clientY: number }): Point {
    let rect = canvas?.getBoundingClientRect() ?? { left: 0, top: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    bumpInteraction();
    bumpWheel();
    let { x, y } = localCoordinates(e);
    let scaler = Math.exp(-e.deltaY / 200);
    onZoom(scaler, { x, y });
  }

  function onZoom(scaler: number, position: Point) {
    let { x, y, scale } = resolvedViewportState;
    setTooltip(null);
    let maxScale: number;
    let minScale: number;
    if (isGis) {
      maxScale = Infinity;
      minScale = gisMinScale;
    } else {
      maxScale = (defaultViewportState?.scale ?? 1) * 1e2;
      minScale = (defaultViewportState?.scale ?? 1) * 1e-2;
    }
    let newScale = Math.min(maxScale, Math.max(minScale, scale * scaler));
    let rect = canvas!.getBoundingClientRect();
    let sz = Math.max(rect.width, rect.height);
    let px = ((position.x - rect.width / 2) / sz) * 2;
    let py = ((rect.height / 2 - position.y) / sz) * 2;
    let newX = x + px / scale - px / newScale;
    let newY = y + py / scale - py / newScale;
    setViewportState({
      x: newX,
      y: newY,
      scale: newScale,
    });
  }

  function onDrag(e1: CursorValue) {
    setTooltip(null);

    let mode: "marquee" | "lasso" | "pan" = "pan";
    if (selectionMode != "none") {
      if (!e1.modifiers.shift) {
        mode = selectionMode;
      }
    } else {
      if (e1.modifiers.shift) {
        mode = e1.modifiers.meta ? "lasso" : "marquee";
      }
    }

    let p1 = localCoordinates(e1);

    switch (mode) {
      case "marquee": {
        return {
          move: (e2: CursorValue) => {
            setTooltip(null);
            if (renderer == null) {
              return;
            }
            let p2 = localCoordinates(e2);
            let l1 = coordinateAtPoint(p1.x, p1.y);
            let l2 = coordinateAtPoint(p2.x, p2.y);
            setRangeSelection({
              xMin: Math.min(l1.x, l2.x),
              yMin: Math.min(l1.y, l2.y),
              xMax: Math.max(l1.x, l2.x),
              yMax: Math.max(l1.y, l2.y),
            });
          },
        };
      }
      case "lasso": {
        let points = [coordinateAtPoint(p1.x, p1.y)];
        return {
          move: (e2: CursorValue) => {
            setTooltip(null);
            if (renderer == null) {
              return;
            }
            let p2 = localCoordinates(e2);
            points = [...points, coordinateAtPoint(p2.x, p2.y)];
            if (points.length >= 3) {
              setRangeSelection(simplifyPolygon(points, 24));
            }
          },
        };
      }
      case "pan": {
        let c0 = coordinateAtPoint(0, 0);
        let c1 = coordinateAtPoint(1, 1);
        let sx = c0.x - c1.x;
        // In GIS mode, the y coordinate return by coordinateAtPoint is unprojected (latitude).
        // However, the viewport state uses projected y coordinate (Mercator).
        // Since the projection is isotropic, we can use the x scale (which is linear) to determine the y scale.
        let sy = isGis ? -sx : c0.y - c1.y;
        let x0 = resolvedViewportState.x;
        let y0 = resolvedViewportState.y;
        bumpInteraction();
        return {
          move: (e2: CursorValue) => {
            bumpInteraction();
            setViewportState({
              x: x0 + (e2.clientX - e1.clientX) * sx,
              y: y0 + (e2.clientY - e1.clientY) * sy,
              scale: resolvedViewportState.scale,
            });
          },
          up: () => {
            // With the CSS-pan fast path no GPU work is in flight during
            // the gesture, so we can snap isInteracting false immediately
            // — the next $effect.pre run does the full re-render at the
            // release viewport and clears the canvas CSS transform in one
            // frame. Skipping the decay makes the visual "settle" instant.
            if (interactionDecayTimer != null) {
              clearTimeout(interactionDecayTimer);
              interactionDecayTimer = null;
            }
            isInteracting = false;
          },
        };
      }
    }
  }

  async function onClick(pointer: CursorValue) {
    if (rangeSelection != null) {
      setRangeSelection(null);
    } else {
      const newSelection = await selectionFromPoint(localCoordinates(pointer));
      if (newSelection == null) {
        setSelection([]);
        setTooltip(null);
      } else {
        if (pointer.modifiers.shift || pointer.modifiers.ctrl || pointer.modifiers.meta) {
          // Toggle the point from the selection
          let index = selection?.findIndex((item) => {
            return item.x == newSelection.x && item.y == newSelection.y && item.category == newSelection.category;
          });
          if (selection == null || index == null || index < 0) {
            setSelection([...(selection ?? []), newSelection]);
            setTooltip(newSelection);
          } else {
            setSelection([...selection.slice(0, index), ...selection.slice(index + 1)]);
            setTooltip(null);
          }
        } else {
          setSelection([newSelection]);
          setTooltip(newSelection);
        }
      }
    }
  }

  let onHoverThrottle = throttleTooltip(
    async (pointer: CursorValue | null) => {
      let position = pointer ? localCoordinates(pointer) : null;
      if (selection != null && selection.length == 1) {
        let cSelection = pointLocation(selection[0].x, selection[0].y);
        if (position != null && pointDistance(position, cSelection) < 10) {
          setTooltip(selection[0]);
        }
      } else {
        setTooltip(await selectionFromPoint(position));
      }
    },
    () => tooltip != null,
  );

  function onHover(e: CursorValue | null) {
    if (e != null) {
      if (!preventHover) {
        onHoverThrottle(e);
      }
    } else {
      onHoverThrottle(null);
    }
  }

  $effect.pre(() => {
    if (preventHover) {
      onHoverThrottle(null);
    }
  });

  async function selectionFromPoint(position: Point | null) {
    if (renderer == null || position == null || querySelection == null) {
      return null;
    }
    const { x, y } = coordinateAtPoint(position.x, position.y);
    const dLon = Math.abs(coordinateAtPoint(position.x + 1, position.y).x - x);
    const dLat = Math.abs(coordinateAtPoint(position.x, position.y + 1).y - y);
    const r = Math.max(dLon, dLat);
    return await querySelection(x, y, r);
  }

  async function generateClusters(
    renderer: EmbeddingRenderer,
    bandwidth: number,
    viewport: ViewportState,
    densityThreshold: number = 0.005,
  ): Promise<Cluster[]> {
    let map = await renderer.densityMap(1000, 1000, bandwidth, viewport);
    let cs = await findClusters(map.data, map.width, map.height);
    let collectedClusters: Cluster[] = [];
    for (let idx = 0; idx < cs.length; idx++) {
      let c = cs[idx];
      let coord = map.coordinateAtPixel(c.meanX, c.meanY);
      let rects: Rectangle[] = c.boundaryRectApproximation!.map(([x1, y1, x2, y2]) => {
        let p1 = map.coordinateAtPixel(x1, y1);
        let p2 = map.coordinateAtPixel(x2, y2);
        return {
          xMin: Math.min(p1.x, p2.x),
          xMax: Math.max(p1.x, p2.x),
          yMin: Math.min(p1.y, p2.y),
          yMax: Math.max(p1.y, p2.y),
        };
      });
      collectedClusters.push({
        x: coord.x,
        y: coord.y,
        sumDensity: c.sumDensity,
        rects: rects,
        bandwidth: bandwidth,
      });
    }
    let maxDensity = collectedClusters.reduce((a, b) => Math.max(a, b.sumDensity), 0);
    return collectedClusters.filter((x) => x.sumDensity / maxDensity > densityThreshold);
  }

  async function generateLabels(viewport: ViewportState): Promise<Label[]> {
    if (renderer == null || queryClusterLabels == null) {
      return [];
    }

    let cacheKey = await objectHash({
      autoLabel: {
        version: 3,
        identifier: cacheIdentifier,
        viewport,
        stopWords: config?.autoLabelStopWords,
        densityThreshold: config?.autoLabelDensityThreshold,
      },
    });

    if (cache != null) {
      let cached = await cache.get(cacheKey);
      if (cached != null) {
        return cached;
      }
    }

    let newClusters = await generateClusters(renderer, 10, viewport, config?.autoLabelDensityThreshold ?? 0.005);
    newClusters = newClusters.concat(await generateClusters(renderer, 5, viewport));

    let labels = await queryClusterLabels(newClusters.map((x) => x.rects));
    for (let i = 0; i < newClusters.length; i++) {
      let label = labels[i];
      newClusters[i].content = label;
      if (typeof label == "object" && label != null && "x" in label && "y" in label) {
        if (label.x != null && label.y != null) {
          newClusters[i].x = label.x;
          newClusters[i].y = label.y;
        }
      }
    }

    let result: Label[] = newClusters
      .filter((x) => x.content != null && (typeof x.content !== "string" || x.content.length > 0))
      .map((x) => ({
        x: x.x,
        y: x.y,
        content: x.content!,
        priority: x.sumDensity,
        level: x.bandwidth == 10 ? 0 : 1,
      }));

    if (cache != null) {
      await cache.set(cacheKey, result);
    }

    return result;
  }

  async function updateLabels(viewport: ViewportState) {
    let vp = new Viewport(viewport, 1000, 1000);
    if (renderer == null) {
      return;
    }
    if (labels != null) {
      clusterLabels = await layoutLabels(vp.scale(), labels, resolvedTheme.fontFamily);
    } else {
      statusMessage = "Generating labels...";
      try {
        let result = await generateLabels(viewport);
        clusterLabels = await layoutLabels(vp.scale(), result, resolvedTheme.fontFamily);
      } catch (e) {
        console.error("Error while generating labels", e);
      } finally {
        statusMessage = null;
      }
    }
  }

  class DefaultTooltipRenderer {
    content: HTMLElement;
    constructor(target: HTMLElement, props: { tooltip: Selection; colorScheme: "light" | "dark"; fontFamily: string }) {
      let content = document.createElement("div");
      this.content = content;
      this.update(props);
      target.appendChild(content);
    }

    update(props: { tooltip: Selection; colorScheme: "light" | "dark"; fontFamily: string }) {
      let content = this.content;
      content.style.fontFamily = props.fontFamily;
      if (colorScheme == "light") {
        content.style.color = "#000";
        content.style.background = "#fff";
        content.style.border = "1px solid #000";
      } else {
        content.style.color = "#ccc";
        content.style.background = "#000";
        content.style.border = "1px solid #ccc";
      }
      content.style.borderRadius = "2px";
      content.style.padding = "5px";
      content.style.fontSize = "12px";
      content.style.maxWidth = "300px";
      content.innerText = props.tooltip.text ?? JSON.stringify(props.tooltip);
    }
  }

  /** Apply a workaround to fix a bug where onwheel does not fire on empty SVG areas in Safari */
  function onWheelWorkaround(element: HTMLElement) {
    element.addEventListener("wheel", () => {}, { passive: true });
  }
</script>

<div style:width="{width}px" style:height="{height}px" style:position="relative" use:onWheelWorkaround>
  <div
    bind:this={mapContainer}
    style:width="100%"
    style:height="100%"
    style:position="absolute"
    style:top="0"
    style:left="0"
    style:z-index="0"
    style:background-color={isGis && !mapStyle ? "black" : undefined}
  ></div>
  <canvas
    bind:this={canvas}
    style:position="absolute"
    style:top="0"
    style:left="0"
    style:z-index="1"
    style:pointer-events="none"
  ></canvas>
  <div style:width="{width}px" style:height="{height}px" style:position="absolute" style:top="0" style:left="0">
    {#if customOverlay}
      {@const action = customComponentAction(customOverlay)}
      {@const proxy = { location: pointLocation, width: width, height: height }}
      {#key action}
        <div use:action={customComponentProps(customOverlay, { proxy: proxy })}></div>
      {/key}
    {/if}
  </div>
  <svg
    width={width}
    height={height}
    style:position="absolute"
    style:left="0"
    style:top="0"
    role="none"
    onwheel={onWheel}
    use:interactionHandler={{
      click: onClick,
      drag: onDrag,
      hover: onHover,
    }}
  >
    <!-- Tooltip point -->
    {#if tooltip != null && renderer != null}
      {@const { x, y } = pointLocation(tooltip.x, tooltip.y)}
      {@const r = Math.max(3, pointSize / pixelRatio) + 1}
      {#if isFinite(x) && isFinite(y) && isFinite(r)}
        <circle
          cx={x}
          cy={y}
          r={r}
          style:stroke={colorScheme == "light" ? "#000" : "#fff"}
          style:stroke-width={1}
          style:fill="none"
        />
      {/if}
    {/if}
    <!-- Selection point(s) -->
    {#if selection != null && renderer != null}
      {#each selection as point}
        {@const { x, y } = pointLocation(point.x, point.y)}
        {@const color = point.category != null ? resolvedCategoryColors[point.category] : resolvedCategoryColors[0]}
        {@const r = Math.max(3, pointSize / pixelRatio) + 1}
        {#if isFinite(x) && isFinite(y) && isFinite(r)}
          <circle
            cx={x}
            cy={y}
            r={r}
            style:stroke={colorScheme == "light" ? "#000" : "#fff"}
            style:stroke-width={2}
            style:fill={color}
          />
        {/if}
      {/each}
    {/if}
    <!-- Cluster labels -->
    {#if showClusterLabels}
      <g>
        {#each clusterLabels as label}
          {@const location = pointLocation(label.coordinate.x, label.coordinate.y)}
          {@const scale = resolvedViewport.scale()}
          {@const isVisible =
            label.placement != null && label.placement.minScale <= scale && scale <= label.placement.maxScale}
          <g transform="translate({location.x},{location.y})">
            {#if isVisible}
              {#if typeof label.content !== "string"}
                <image
                  href={label.content.image}
                  x={-label.content.width / 2}
                  y={-label.content.height / 2}
                  width={label.content.width}
                  height={label.content.height}
                  style:user-select="none"
                  style:-webkit-user-select="none"
                  style:opacity={resolvedTheme.clusterLabelOpacity}
                />
              {:else}
                {@const rows = label.content.split("\n")}
                <g>
                  {#each rows as row, index}
                    <text
                      style:paint-order="stroke"
                      style:stroke-width="4"
                      style:stroke-linejoin="round"
                      style:stroke-linecap="round"
                      style:text-anchor="middle"
                      style:fill={resolvedTheme.clusterLabelColor}
                      style:stroke={resolvedTheme.clusterLabelOutlineColor}
                      style:opacity={resolvedTheme.clusterLabelOpacity}
                      style:user-select="none"
                      style:-webkit-user-select="none"
                      style:font-family={resolvedTheme.fontFamily}
                      x={0}
                      y={(index - (rows.length - 1) / 2) * label.fontSize}
                      font-size={label.fontSize}
                      dominant-baseline="middle"
                    >
                      {row}
                    </text>
                  {/each}
                </g>
              {/if}
            {/if}
          </g>
        {/each}
      </g>
    {/if}
    <!-- Range selection interaction and display -->
    {#if rangeSelection != null && renderer != null}
      {#if rangeSelection instanceof Array}
        <Lasso value={rangeSelection} pointLocation={pointLocation} />
      {:else}
        <EditableRectangle
          value={rangeSelection}
          onChange={setRangeSelection}
          pointLocation={pointLocation}
          coordinateAtPoint={coordinateAtPoint}
          preventHover={(value) => {
            preventHover = value;
          }}
        />
      {/if}
    {/if}
  </svg>
  <!-- Tooltip popup -->
  {#if tooltip != null && renderer != null}
    {@const loc = pointLocation(tooltip.x, tooltip.y)}
    <TooltipContainer
      location={loc}
      allowInteraction={lockTooltip}
      targetHeight={Math.max(3, pointSize / pixelRatio)}
      customTooltip={customTooltip ?? {
        class: DefaultTooltipRenderer,
        props: { colorScheme: colorScheme, fontFamily: resolvedTheme.fontFamily },
      }}
      tooltip={tooltip}
    />
  {/if}
  <!-- Status bar -->
  {#if resolvedTheme.statusBar}
    <StatusBar
      resolvedTheme={resolvedTheme}
      statusMessage={statusMessage ?? webGPUError}
      distancePerPoint={1 / (pointLocation(1, 0).x - pointLocation(0, 0).x)}
      pointCount={data.x.length}
      selectionMode={selectionMode}
      onSelectionMode={(v) => (selectionMode = v)}
      basemapAttribution={basemapAttribution}
    />
  {/if}
</div>
