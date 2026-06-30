<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  interface Props<Selection> {
    data: {
      x: Float32Array<ArrayBuffer>;
      y: Float32Array<ArrayBuffer>;
      z?: Float32Array<ArrayBuffer> | null;
      category: Uint8Array<ArrayBuffer> | null;
    };
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
    /** Resolve a point by its instance index (used for 3D pick-based hover/click). */
    queryByIndex?: ((index: number) => Promise<Selection | null>) | null;
    rangeSelection: Rectangle | Point[] | null;
    defaultViewportState: ViewportState | null;
    viewportState: ViewportState | null;
    defaultCamera3DState?: Camera3DState | null;
    camera3DState?: Camera3DState | null;
    customTooltip: CustomComponent<HTMLDivElement, { tooltip: Selection }> | null;
    customOverlay: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;
    onViewportState: ((value: ViewportState) => void) | null;
    onCamera3DState?: ((value: Camera3DState) => void) | null;
    onTooltip: ((value: Selection | null) => void) | null;
    onSelection: ((value: Selection[] | null) => void) | null;
    onRangeSelection: ((value: Rectangle | Point[] | null) => void) | null;
    cache: Cache | null;
  }

  interface Cluster {
    x: number;
    y: number;
    sumDensity: number;
    rects: Rectangle[];
    bandwidth: number;
    content?: LabelContent | null;
  }

  // The coordinate array references that identify a 3D dataset for the sampler.
  interface DataArrays {
    x: Float32Array<ArrayBuffer>;
    y: Float32Array<ArrayBuffer>;
    z: Float32Array<ArrayBuffer> | null;
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
  import { interactionHandler, type CursorValue, type DragHandler } from "@embedding-atlas/utils";
  import { onDestroy, onMount } from "svelte";

  import EditableRectangle from "./EditableRectangle.svelte";
  import Lasso from "./Lasso.svelte";
  import StatusBar from "./StatusBar.svelte";
  import TooltipContainer from "./TooltipContainer.svelte";

  import {
    Camera3D,
    clampPitch,
    dolly,
    fitCamera3D,
    frustumSampleIndicesChunked,
    globalStrideIndices,
    orbit,
    pan,
    type Camera3DState,
  } from "../camera3d.js";
  import { defaultCategoryColors } from "../colors.js";
  import { normalizeDownsampleCap } from "../renderer_interface.js";
  import type { EmbeddingRenderer, RenderMode } from "../renderer_interface.js";
  import {
    cacheKeyForObject,
    deepEquals,
    pointDistance,
    throttleTooltip,
    type Point,
    type Rectangle,
    type ViewportState,
  } from "../utils.js";
  import { Viewport } from "../viewport_utils.js";
  import { EmbeddingRendererWebGL2 } from "../webgl2_renderer/renderer.js";
  import { EmbeddingRendererWebGPU } from "../webgpu_renderer/renderer.js";
  import { requestWebGPUDevice } from "../webgpu_renderer/utils.js";
  import { customComponentAction, customComponentProps } from "./custom_component_helper.js";
  import type { EmbeddingViewConfig } from "./embedding_view_config.js";
  import { layoutLabels, type LabelWithPlacement } from "./labels.js";
  import { simplifyPolygon } from "./simplify_polygon.js";
  import { resolveTheme, type ThemeConfig } from "./theme.js";
  import type { Cache, CustomComponent, DataPointID, Label, LabelContent, OverlayProxy } from "./types.js";
  import { findClusters } from "./worker/index.js";

  interface SelectionBase {
    x: number;
    y: number;
    z?: number;
    category?: number;
    text?: string;
    // Stable row identifier when available. Used to distinguish rows that share
    // coordinates/category (e.g. stacked 3D points) for selection toggles/tooltips.
    identifier?: DataPointID;
    // DuckDB rowid attached to a rowid-resolved 3D pick. An exact per-row identity
    // for toggles/tooltips even when no user identifier column exists.
    rowid?: number;
  }

  type Selection = $$Generic<SelectionBase>;

  let {
    data = { x: new Float32Array(), y: new Float32Array(), category: null },
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
    queryByIndex = null,
    rangeSelection = null,
    defaultViewportState = null,
    viewportState = null,
    defaultCamera3DState = null,
    camera3DState = null,
    customTooltip = null,
    customOverlay = null,
    onViewportState = null,
    onCamera3DState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    cache = null,
  }: Props<Selection> = $props();

  let showClusterLabels = true;

  let colorScheme = $derived(config?.colorScheme ?? "light");
  let resolvedTheme = $derived(resolveTheme(theme, colorScheme));
  let resolvedCategoryColors = $derived(categoryColors ?? defaultCategoryColors(categoryCount));

  let resolvedViewportState = $derived(viewportState ?? defaultViewportState ?? { x: 0, y: 0, scale: 1 });
  let resolvedViewport = $derived(new Viewport(resolvedViewportState, width, height));
  let pointLocation = $derived(resolvedViewport.pixelLocationFunction());
  let coordinateAtPoint = $derived(resolvedViewport.coordinateAtPixelFunction());

  // --- 3D state -------------------------------------------------------------
  // The fit of an empty cloud IS the default three-quarter view, so derive the
  // fallback from fitCamera3D instead of duplicating its constants here.
  const FALLBACK_CAMERA: Camera3DState = fitCamera3D([], [], []);
  let is3D = $derived(data.z != null && (config?.mode ?? "points") === "points-3d");
  // Live, animated camera. null means "follow the resolved external/default state".
  let liveCamera = $state<Camera3DState | null>(null);
  let resolvedCamera3DState = $derived(camera3DState ?? defaultCamera3DState ?? FALLBACK_CAMERA);
  let renderCamera3DState = $derived(liveCamera ?? resolvedCamera3DState);
  let camera3D = $derived(is3D ? new Camera3D(renderCamera3DState, width, height) : null);

  // The (x, y, z?) -> pixel projector handed to custom overlays, so their geometry
  // rotates with the cloud in 3D. In 2D it ignores z and matches pointLocation.
  // Returns NaN for points behind the camera, which the {#if isFinite(...)} guards hide.
  let overlayLocation = $derived.by(() => {
    if (is3D && camera3D != null) {
      let project = camera3D.pixelLocationFunction();
      return (x: number, y: number, z: number = 0) => project(x, y, z);
    }
    return (x: number, y: number, _z?: number) => pointLocation(x, y);
  });

  // Marker-shaped adapter over overlayLocation for the {x, y, z?} selection/tooltip objects.
  let markerLocation = $derived((p: { x: number; y: number; z?: number }) => overlayLocation(p.x, p.y, p.z ?? 0));

  let previousCamera3DStateProp: Camera3DState | null = null;
  let hasSyncedCamera3DStateProp = false;
  $effect.pre(() => {
    // Mirror the controlled camera3DState prop into liveCamera whenever it CHANGES,
    // including a transition back to null (which re-enables auto-fit: liveCamera
    // becomes null, so renderCamera3DState falls back to defaultCamera3DState).
    // When the prop is uncontrolled (stays null), this never fires after the first
    // run, so liveCamera is left null and seeded lazily on first interaction —
    // letting the view keep re-fitting as async data updates defaultCamera3DState.
    if (!hasSyncedCamera3DStateProp || camera3DState !== previousCamera3DStateProp) {
      let firstRun = !hasSyncedCamera3DStateProp;
      hasSyncedCamera3DStateProp = true;
      previousCamera3DStateProp = camera3DState;
      // On the very first run with no controlled value, leave liveCamera null
      // (lazy seed). Otherwise adopt the new value (including null to re-fit).
      if (!(firstRun && camera3DState == null)) {
        liveCamera = camera3DState;
        // A controlled (external) camera update supersedes any in-flight internal
        // animation (reset/double-click tween or orbit inertia); cancel it so the
        // next tick cannot continue the old animation and overwrite the parent's
        // value via onCamera3DState. (A transition to null re-fits and the tick
        // self-cancels on liveCamera == null, so only the non-null case needs this.)
        if (camera3DState != null) {
          cameraTween = null;
          cameraVelYaw = 0;
          cameraVelPitch = 0;
          if (cameraAnimRequest != null) {
            cancelAnimationFrame(cameraAnimRequest);
            cameraAnimRequest = null;
          }
        }
      }
    }
  });

  // Range selection (marquee/lasso) is a 2D-only feature and its overlay is hidden
  // in 3D. We intentionally do NOT clear an active brush when entering 3D — that
  // would irreversibly drop the user's filter/selection state (and any saved 3D
  // chart's brush). To avoid a *silent* hidden filter, the status bar shows an
  // explicit active-filter indicator with a one-click clear whenever a brush is
  // active in 3D (filterActive/onClearFilter below); a click on empty space also
  // clears it (see onClick). So the filter is preserved across mode switches but
  // never invisible.

  let preventHover = $state(false);

  // Identity of two selected points. Prefers a stable identifier when BOTH sides
  // have one, so rows that share coordinates/category (e.g. stacked 3D points the
  // pick-by-index path disambiguates) are treated as distinct. Falls back to
  // coordinate + category + text only when an identifier is unavailable. Shared by
  // tooltip locking and selection toggles so they stay consistent.
  function compareSelection(a: Selection, b: Selection) {
    // A configured identifier is the semantic key, so it wins (mirrors the coordinated
    // predicate). rowid is the exact per-row fallback for 3D picks when no identifier
    // exists; coordinates are the last resort — so co-located rows toggle independently
    // whenever any exact identity is available.
    let aid = (a as SelectionBase).identifier;
    let bid = (b as SelectionBase).identifier;
    if (aid != null && bid != null) {
      return aid === bid;
    }
    let arow = (a as SelectionBase).rowid;
    let brow = (b as SelectionBase).rowid;
    if (arow != null && brow != null) {
      return arow === brow;
    }
    return a.x == b.x && a.y == b.y && (a.z ?? 0) == (b.z ?? 0) && a.category == b.category && a.text == b.text;
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
  let webGPUPrompt: string | null = $state(null);

  let minimumDensity = $derived(config?.minimumDensity ?? 1 / 16);
  let userPointSize = $derived(config?.pointSize ?? null);
  let mode = $derived(config?.mode ?? "points");
  // The renderer only knows "points"/"density"; "points-3d" maps to "points" and
  // is driven through the separate is3D flag.
  let rendererMode: RenderMode = $derived(mode === "density" ? "density" : "points");
  let autoLabelEnabled = $derived(config?.autoLabelEnabled);
  let downsampleMaxPoints = $derived(config?.downsampleMaxPoints ?? 4000000);
  let downsampleDensityWeight = $derived(config?.downsampleDensityWeight ?? 5);

  // --- 3D frustum-aware downsampling ----------------------------------------
  // When the point count exceeds downsampleMaxPoints, a fixed GLOBAL stride would
  // permanently hide the points it skips: zooming into a dense region could never
  // reveal them (they are never drawn, so never pickable either). So we cull to the
  // current camera frustum and stride within THAT set, so a zoomed-in region draws
  // and picks from its own points.
  //
  // The frustum cull scans every point, which would freeze the UI on multi-million
  // point datasets. So: (1) publish a CHEAP global-stride subset immediately for the
  // first frame (O(cap), no scan); (2) run the O(N) frustum cull in a Web Worker,
  // debounced to once per camera/data settle and cancelable (stale responses are
  // dropped); (3) keep the current subset until the worker responds. The per-point
  // arrays are uploaded to the worker once per dataset; each settle only sends a
  // matrix and receives a bounded index set. Bounded by downsampleMaxPoints. null =
  // draw all. The renderers consume this via the `sampleIndices` prop.
  let sampleIndices3D = $state.raw<Uint32Array<ArrayBuffer> | null>(null);
  let sampleIndicesTimer: ReturnType<typeof setTimeout> | null = null;
  // Identify a dataset by its underlying x/y/z ARRAY references, not the `data`
  // wrapper object: parents pass `data` as a fresh object literal every render, so
  // keying on the wrapper would re-upload arrays and drop in-flight worker results on
  // every unrelated update. The arrays only change on a real data swap (and x/y/z can
  // change independently, so all three are tracked).
  let previousSampleData: DataArrays | null = null;
  // Plain (non-reactive) mirror of "do we currently hold a computed set", so the
  // recompute effect never READS sampleIndices3D (which it writes) and self-loops.
  let hasComputedSampleIndices = false;
  // Monotonic request id used to cancel/supersede an in-flight chunked refinement.
  let sampleRequestId = 0;
  // Plain mirror of the current sample length, so the effect can detect when the cap
  // was lowered below the current sample (and re-stride immediately) without reading
  // the $state it writes.
  let currentSampleLength = 0;
  // The frustum-relevant inputs from the previous effect run, so the effect can tell a
  // real change (camera/size/cap) from wrapper-only `data` churn (tooltip/selection
  // re-renders) and avoid needlessly invalidating/rescheduling the refine.
  let previousRefineCamera: Camera3DState | null = null;
  let previousRefineWidth = 0;
  let previousRefineHeight = 0;
  let previousRefineCap: number | null = null;

  // Yields between frustum-cull chunks so the (O(N)) scan never blocks the UI thread.
  // Prefers requestIdleCallback (runs in spare time, with a timeout so it still makes
  // progress under load); falls back to a macrotask where it is unavailable.
  function idleYield(): Promise<void> {
    return new Promise((resolve) => {
      let ric = (globalThis as any).requestIdleCallback;
      if (typeof ric === "function") {
        ric(() => resolve(), { timeout: 100 });
      } else {
        setTimeout(resolve, 0);
      }
    });
  }

  function effectiveDownsampleCap(): number | null {
    return normalizeDownsampleCap(downsampleMaxPoints);
  }

  // Snapshot of the current data's coordinate array references.
  function dataArrays(): DataArrays {
    return { x: data.x, y: data.y, z: data.z ?? null };
  }

  // True when two snapshots reference the same x/y/z buffers (a real data swap
  // changes at least one reference; the `data` wrapper alone is not reliable).
  function sameDataArrays(a: DataArrays | null, b: DataArrays | null): boolean {
    return a != null && b != null && a.x === b.x && a.y === b.y && a.z === b.z;
  }

  function setSampleIndices3D(value: Uint32Array<ArrayBuffer> | null) {
    hasComputedSampleIndices = value != null;
    currentSampleLength = value != null ? value.length : 0;
    sampleIndices3D = value;
  }

  // Kick off (or supersede) a frustum refinement for the current camera. The cull
  // runs as a CHUNKED, idle-scheduled, cancelable pass over the existing coordinate
  // arrays (no copy, no worker, no GPU upload), so it stays complete for arbitrarily
  // large clouds while never blocking the UI and using only the (<= cap) result of
  // extra memory. A newer camera/data change bumps sampleRequestId, which cancels any
  // in-flight pass; until a fresh result installs, the cheap global-stride subset
  // (published synchronously by the effect) remains on screen.
  function requestFrustumRefine() {
    let cam = camera3D;
    let z = data.z;
    let cap = effectiveDownsampleCap();
    if (!is3D || cam == null || z == null || cap == null || data.x.length <= cap) {
      return;
    }
    let reqId = ++sampleRequestId;
    let arrays = dataArrays();
    frustumSampleIndicesChunked(arrays.x, arrays.y, z, cam.viewProjection(), cap, {
      yieldToHost: idleYield,
      // Abandon the pass if a newer refine was requested or the data was replaced.
      shouldCancel: () => reqId !== sampleRequestId || !sameDataArrays(dataArrays(), arrays),
    })
      .then((indices) => {
        // undefined = canceled; null = no longer downsampling. Either way, and on any
        // stale/superseded result, keep the current sample untouched.
        if (reqId !== sampleRequestId || indices == null || !sameDataArrays(dataArrays(), arrays)) {
          return;
        }
        setSampleIndices3D(indices);
      })
      .catch((e) => {
        // Surface the failure and keep the current bounded subset (the global-stride
        // fallback is valid, just not frustum-refined). The next settle retries.
        console.error("3D frustum sample refine failed", e);
      });
  }

  $effect(() => {
    // Re-run whenever something that could change the visible 3D subset changes. We
    // read `data` (a wrapper recreated on every parent render) but decide relevance
    // from its x/y/z arrays, the camera, size, and cap — never the wrapper identity.
    let camera = renderCamera3DState;
    void data;
    let cap = effectiveDownsampleCap();
    let w = width;
    let h = height;
    void is3D;

    let count = data.x.length;
    if (!is3D || data.z == null || cap == null || count <= cap) {
      // 2D, or few enough points to draw all: no downsampling, cancel any refine.
      if (sampleIndicesTimer != null) {
        clearTimeout(sampleIndicesTimer);
        sampleIndicesTimer = null;
      }
      sampleRequestId++; // invalidate any in-flight refine
      if (hasComputedSampleIndices) {
        setSampleIndices3D(null);
      }
      previousSampleData = dataArrays();
      previousRefineCamera = camera;
      previousRefineWidth = w;
      previousRefineHeight = h;
      previousRefineCap = cap;
      return;
    }

    // Determine what ACTUALLY changed (ignoring wrapper-only churn). Size feeds the
    // frustum via the camera's projection aspect, so it counts as a camera change.
    let cur = dataArrays();
    let dataChanged = !sameDataArrays(cur, previousSampleData);
    let cameraChanged = camera !== previousRefineCamera || w !== previousRefineWidth || h !== previousRefineHeight;
    let capChanged = cap !== previousRefineCap;
    previousSampleData = cur;
    previousRefineCamera = camera;
    previousRefineWidth = w;
    previousRefineHeight = h;
    previousRefineCap = cap;

    // Publish a cheap global-stride subset synchronously when it's the first render /
    // a data swap (old indices reference stale rows), OR the cap was just lowered
    // below the current sample (so reducing downsampleMaxPoints to recover from an
    // overloaded view takes effect immediately, not only after the debounced refine).
    // The chunked frustum refine still runs afterward to specialize to the zoomed-in
    // region.
    let resetStride = dataChanged || !hasComputedSampleIndices || currentSampleLength > cap;
    if (resetStride) {
      setSampleIndices3D(globalStrideIndices(count, cap));
    }

    // Only invalidate the in-flight refine and reschedule when a frustum-relevant
    // input actually changed. Wrapper-only updates (tooltip/selection re-renders pass
    // a fresh `data` object with the same x/y/z, camera, size, cap) must NOT drop a
    // pending result or it could never install during active hover/selection.
    if (!(dataChanged || cameraChanged || capChanged || resetStride)) {
      return;
    }

    // Invalidate any in-flight refine: its result is for a now-stale view.
    sampleRequestId++;

    // Refine to the frustum-aware subset once the camera/data settles. Debounced, so
    // continuous interaction reuses the last bounded subset and the chunked scan runs
    // once per settle.
    if (sampleIndicesTimer != null) {
      clearTimeout(sampleIndicesTimer);
    }
    sampleIndicesTimer = setTimeout(() => {
      sampleIndicesTimer = null;
      requestFrustumRefine();
    }, 150);
  });

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

  $effect.pre(() => {
    if (labels !== previousLabels) {
      previousLabels = labels;
      needsUpdateLabels = true;
    }

    let needsRender = renderer?.setProps({
      mode: rendererMode,
      colorScheme: colorScheme,
      viewportX: resolvedViewportState.x,
      viewportY: resolvedViewportState.y,
      viewportScale: resolvedViewportState.scale,
      width: pixelWidth,
      height: pixelHeight,
      x: data.x,
      y: data.y,
      category: data.category,
      categoryCount,
      categoryColors: resolvedCategoryColors,
      downsampleMaxPoints,
      downsampleDensityWeight,
      ...viewingParams,
      // 3D point cloud props. The renderer draws 3D only when z and viewProjection
      // are both non-null, so leaving viewProjection null keeps it on the 2D path.
      z: is3D ? (data.z ?? null) : null,
      viewProjection: is3D && camera3D != null ? camera3D.viewProjection() : null,
      cameraEye: is3D && camera3D != null ? camera3D.eye() : [0, 0, 0],
      cameraDistance: is3D && camera3D != null ? camera3D.cameraDistance() : 1,
      pointSize3D: (userPointSize ?? 6) * pixelRatio,
      fogDensity: config?.fogDensity ?? 0.6,
      sampleIndices: is3D ? sampleIndices3D : null,
    });

    if (needsRender) {
      setNeedsRender();
    }

    if (
      !is3D &&
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

  function render() {
    _request = null;
    if (!canvas || !renderer) {
      return;
    }
    canvas.width = renderer.props.width;
    canvas.height = renderer.props.height;
    canvas.style.width = `${renderer.props.width / pixelRatio}px`;
    canvas.style.height = `${renderer.props.height / pixelRatio}px`;
    renderer.render();
  }

  let _request: number | null = null;
  function setNeedsRender() {
    if (_request == null) {
      _request = requestAnimationFrame(render);
    }
  }

  function setupWebGLRenderer(canvas: HTMLCanvasElement) {
    webGPUPrompt = "WebGPU is unavailable. Falling back to WebGL.";

    let context: WebGL2RenderingContext | null;

    function createRenderer() {
      // `depth: true` provides the depth buffer the 3D point cloud path needs;
      // the 2D path renders to its own framebuffers and ignores it.
      context = canvas.getContext("webgl2", { antialias: false, depth: true });
      if (context == null) {
        console.error("Could not get WebGL 2 context");
        return;
      }
      context.getExtension("EXT_color_buffer_float");
      context.getExtension("EXT_float_blend");
      context.getExtension("OES_texture_float_linear");
      renderer = new EmbeddingRendererWebGL2(context, pixelWidth, pixelHeight);
    }

    createRenderer();

    canvas.addEventListener("webglcontextlost", () => {
      renderer?.destroy();
      renderer = null;
      context = null;
    });

    canvas.addEventListener("webglcontextrestored", () => {
      createRenderer();
    });
  }

  function setupWebGPURenderer(canvas: HTMLCanvasElement) {
    let canFallbackToWebGL = true;

    async function createRenderer() {
      let device = await requestWebGPUDevice();
      if (device == null) {
        console.error("Could not get WebGPU device");
        if (canFallbackToWebGL) {
          setupWebGLRenderer(canvas);
        }
        return;
      }

      let context = canvas.getContext("webgpu");
      if (context == null) {
        console.error("Could not get WebGPU canvas context");
        if (canFallbackToWebGL) {
          setupWebGLRenderer(canvas);
        }
        return;
      }

      // Once we get the context, we can't fallback to setupWebGLRenderer.
      canFallbackToWebGL = false;

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

      renderer = new EmbeddingRendererWebGPU(context, device, format, pixelWidth, pixelHeight);
    }

    createRenderer();
  }

  function syncViewportState(defaultViewportState: ViewportState | null) {
    if (defaultViewportState != null && viewportState == null) {
      setViewportState(defaultViewportState);
    }
  }

  $effect.pre(() => syncViewportState(defaultViewportState));

  onMount(() => {
    if (canvas == null) {
      return;
    }
    // Setup WebGPU renderer (with fallback to WebGL)
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
    if (sampleIndicesTimer != null) {
      clearTimeout(sampleIndicesTimer);
      sampleIndicesTimer = null;
    }
    // Cancel any in-flight camera animation (reset/focus tween or orbit inertia) so
    // its RAF callback cannot mutate liveCamera or fire onCamera3DState after unmount.
    if (cameraAnimRequest != null) {
      cancelAnimationFrame(cameraAnimRequest);
      cameraAnimRequest = null;
    }
    sampleRequestId++; // cancel any in-flight chunked refine
  });

  function localCoordinates(e: { clientX: number; clientY: number }): Point {
    let rect = canvas?.getBoundingClientRect() ?? { left: 0, top: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  // --- 3D camera interaction (orbit / pan / dolly with inertia + tweens) ----
  const ORBIT_SPEED = 0.008; // radians per CSS pixel
  const CAMERA_TWEEN_MS = 450; // reset / focus-on-point animation duration
  // Two-tier orbit-velocity thresholds: a looser one to START inertia on release,
  // a tighter one to STOP the decaying animation.
  const ORBIT_INERTIA_START = 1e-3;
  const ORBIT_INERTIA_STOP = 1e-4;

  let cameraVelYaw = 0;
  let cameraVelPitch = 0;
  let cameraTween: { from: Camera3DState; to: Camera3DState; start: number; duration: number } | null = null;
  let cameraAnimRequest: number | null = null;

  // Seed the live camera lazily, the first time the user interacts. By then async
  // data has loaded and resolvedCamera3DState reflects the correctly fitted default.
  function ensureLiveCamera(): Camera3DState {
    if (liveCamera == null) {
      liveCamera = resolvedCamera3DState;
    }
    return liveCamera;
  }

  function commitCamera() {
    if (liveCamera != null) {
      onCamera3DState?.(liveCamera);
    }
  }

  function lerp(a: number, b: number, t: number) {
    return a + (b - a) * t;
  }

  function lerpCamera(a: Camera3DState, b: Camera3DState, t: number): Camera3DState {
    let dYaw = b.yaw - a.yaw;
    while (dYaw > Math.PI) dYaw -= 2 * Math.PI;
    while (dYaw < -Math.PI) dYaw += 2 * Math.PI;
    return {
      target: [lerp(a.target[0], b.target[0], t), lerp(a.target[1], b.target[1], t), lerp(a.target[2], b.target[2], t)],
      distance: Math.exp(lerp(Math.log(a.distance), Math.log(b.distance), t)),
      yaw: a.yaw + dYaw * t,
      pitch: clampPitch(lerp(a.pitch, b.pitch, t)),
      fov: lerp(a.fov, b.fov, t),
    };
  }

  function startCameraAnim() {
    if (cameraAnimRequest == null) {
      cameraAnimRequest = requestAnimationFrame(cameraAnimTick);
    }
  }

  function cameraAnimTick(now: number) {
    cameraAnimRequest = null;
    if (!is3D || liveCamera == null) {
      cameraVelYaw = 0;
      cameraVelPitch = 0;
      cameraTween = null;
      return;
    }
    let active = false;
    if (cameraTween != null) {
      let t = Math.min(1, (now - cameraTween.start) / cameraTween.duration);
      let e = 1 - Math.pow(1 - t, 3); // ease-out cubic
      liveCamera = lerpCamera(cameraTween.from, cameraTween.to, e);
      if (t < 1) {
        active = true;
      } else {
        cameraTween = null;
        commitCamera();
      }
    } else if (Math.abs(cameraVelYaw) > ORBIT_INERTIA_STOP || Math.abs(cameraVelPitch) > ORBIT_INERTIA_STOP) {
      liveCamera = orbit(liveCamera, cameraVelYaw, cameraVelPitch);
      cameraVelYaw *= 0.9;
      cameraVelPitch *= 0.9;
      active = true;
      if (Math.abs(cameraVelYaw) <= ORBIT_INERTIA_STOP && Math.abs(cameraVelPitch) <= ORBIT_INERTIA_STOP) {
        commitCamera();
      }
    }
    if (active) {
      startCameraAnim();
    }
  }

  function clampDistance(state: Camera3DState): Camera3DState {
    // Anchor zoom bounds to the FITTED default distance, not resolvedCamera3DState:
    // when camera3DState is controlled, the latter tracks the just-zoomed camera, so
    // the bounds would drift with every wheel tick and let zoom escape the range.
    let base = (defaultCamera3DState ?? FALLBACK_CAMERA).distance;
    let distance = Math.min(base * 50, Math.max(base * 0.02, state.distance));
    return { ...state, distance };
  }

  function resetCamera() {
    if (!is3D) {
      return;
    }
    let from = ensureLiveCamera();
    // Always re-fit to the default (fitted) camera. Using resolvedCamera3DState
    // would be a no-op when camera3DState is controlled, since it tracks the live camera.
    let target = defaultCamera3DState ?? FALLBACK_CAMERA;
    cameraVelYaw = 0;
    cameraVelPitch = 0;
    cameraTween = { from: from, to: target, start: performance.now(), duration: CAMERA_TWEEN_MS };
    startCameraAnim();
  }

  function onDrag3D(e1: CursorValue): DragHandler | undefined {
    ensureLiveCamera();
    setTooltip(null);
    cameraTween = null;
    cameraVelYaw = 0;
    cameraVelPitch = 0;
    let panMode = e1.modifiers.shift;
    let last = localCoordinates(e1);
    let vYaw = 0;
    let vPitch = 0;
    return {
      move: (e2: CursorValue) => {
        if (liveCamera == null) {
          return;
        }
        let p2 = localCoordinates(e2);
        let dx = p2.x - last.x;
        let dy = p2.y - last.y;
        last = p2;
        if (panMode) {
          liveCamera = pan(liveCamera, dx, dy, width, height);
        } else {
          let dYaw = -dx * ORBIT_SPEED;
          let dPitch = -dy * ORBIT_SPEED;
          liveCamera = orbit(liveCamera, dYaw, dPitch);
          vYaw = dYaw;
          vPitch = dPitch;
        }
      },
      up: () => {
        if (!panMode && (Math.abs(vYaw) > ORBIT_INERTIA_START || Math.abs(vPitch) > ORBIT_INERTIA_START)) {
          cameraVelYaw = vYaw;
          cameraVelPitch = vPitch;
          startCameraAnim();
        } else {
          commitCamera();
        }
      },
    };
  }

  async function onDoubleClick(e: MouseEvent) {
    if (!is3D) {
      return;
    }
    let sel = await selectionFromPick(localCoordinates(e));
    if (sel == null) {
      return;
    }
    let from = ensureLiveCamera();
    cameraVelYaw = 0;
    cameraVelPitch = 0;
    let to: Camera3DState = { ...from, target: [sel.x, sel.y, sel.z ?? 0] };
    cameraTween = { from: from, to: to, start: performance.now(), duration: CAMERA_TWEEN_MS };
    startCameraAnim();
  }

  // Resolves a 3D pick to a selection. Returns a Selection for a hit, `null` for a
  // confirmed no-hit (empty space), and `undefined` when the result is indeterminate
  // (renderer reported stale/error, or the backing arrays changed mid-pick) — callers
  // must leave selection/tooltip UNCHANGED on `undefined` so an async race during a
  // click does not clear the user's selection.
  async function selectionFromPick(position: Point): Promise<Selection | null | undefined> {
    if (renderer == null || queryByIndex == null) {
      return undefined; // cannot resolve a pick right now → leave state unchanged
    }
    // 3D picking is asynchronous (WebGPU queues a readback), and BOTH the backing
    // arrays and the camera can change during the awaits (the renderer only rejects
    // a stale pick up to when renderer.pick returns — camera inertia/tween/controlled
    // updates or a filter/data refresh can still happen while queryByIndex resolves).
    // Capture the x/y/z buffers AND the camera state up front and treat any mid-resolve
    // change as indeterminate. Key on the x/y/z buffers, NOT the `data` wrapper, which
    // parents recreate on unrelated re-renders.
    let snapshot = dataArrays();
    let cameraSnapshot = renderCamera3DState;
    let stale = () => !sameDataArrays(dataArrays(), snapshot) || renderCamera3DState !== cameraSnapshot;
    let index = await renderer.pick(position.x * pixelRatio, position.y * pixelRatio);
    // Indeterminate: renderer reported a stale/failed pick, or the data/camera changed.
    if (index === undefined || stale()) {
      return undefined;
    }
    // Confirmed no-hit (cursor over empty space).
    if (index === null) {
      return null;
    }
    let result = await queryByIndex(index);
    if (stale()) {
      return undefined; // data or camera changed while the row was being resolved
    }
    // queryByIndex returns null when it cannot resolve the picked row (e.g. failed
    // enrichment); treat that as indeterminate, not an intentional empty click.
    return result ?? undefined;
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    if (is3D) {
      let cam = ensureLiveCamera();
      setTooltip(null);
      cameraTween = null;
      liveCamera = clampDistance(dolly(cam, Math.exp(e.deltaY / 200)));
      commitCamera();
      return;
    }
    let { x, y } = localCoordinates(e);
    let scaler = Math.exp(-e.deltaY / 200);
    onZoom(scaler, { x, y });
  }

  function onZoom(scaler: number, position: Point) {
    let { x, y, scale } = resolvedViewportState;
    setTooltip(null);
    let maxScale = (defaultViewportState?.scale ?? 1) * 1e2;
    let minScale = (defaultViewportState?.scale ?? 1) * 1e-2;
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

  function onDrag(e1: CursorValue): DragHandler | undefined {
    if (is3D) {
      return onDrag3D(e1);
    }
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
        let sy = c0.y - c1.y;
        let x0 = resolvedViewportState.x;
        let y0 = resolvedViewportState.y;
        return {
          move: (e2: CursorValue) => {
            setViewportState({
              x: x0 + (e2.clientX - e1.clientX) * sx,
              y: y0 + (e2.clientY - e1.clientY) * sy,
              scale: resolvedViewportState.scale,
            });
          },
        };
      }
    }
  }

  function applyClickSelection(newSelection: Selection | null, pointer: CursorValue) {
    if (newSelection == null) {
      setSelection([]);
      setTooltip(null);
    } else if (pointer.modifiers.shift || pointer.modifiers.ctrl || pointer.modifiers.meta) {
      // Toggle the point from the selection, using the same identity comparison as
      // tooltip locking (identifier-first) so co-located rows toggle independently.
      let index = selection?.findIndex((item) => compareSelection(item, newSelection));
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

  async function onClick(pointer: CursorValue) {
    // A click first clears any active brush (in both 2D and 3D), giving a
    // click-to-clear path even though the brush overlay is hidden in 3D.
    if (rangeSelection != null) {
      setRangeSelection(null);
      return;
    }
    let newSelection = is3D
      ? await selectionFromPick(localCoordinates(pointer))
      : await selectionFromPoint(localCoordinates(pointer));
    // `undefined` means the 3D pick was indeterminate (the view changed or the pick
    // failed mid-click) — leave the existing selection/tooltip untouched rather than
    // treating it as an intentional empty-space click that would clear them.
    if (newSelection === undefined) {
      return;
    }
    applyClickSelection(newSelection, pointer);
  }

  let onHoverThrottle = throttleTooltip(
    async (pointer: CursorValue | null) => {
      let position = pointer ? localCoordinates(pointer) : null;
      // When a single point is selected, keep its (lockable) tooltip while the
      // cursor is near it, and leave the tooltip untouched otherwise (so moving
      // into an interactive tooltip does not dismiss it). This mirrors 2D; 3D uses
      // the camera projection + pick instead of the viewport.
      if (selection != null && selection.length == 1) {
        let cSelection = is3D ? markerLocation(selection[0]) : pointLocation(selection[0].x, selection[0].y);
        if (
          position != null &&
          isFinite(cSelection.x) &&
          isFinite(cSelection.y) &&
          pointDistance(position, cSelection) < 10
        ) {
          setTooltip(selection[0]);
        }
      } else if (is3D) {
        if (position == null) {
          setTooltip(null);
        } else {
          let picked = await selectionFromPick(position);
          // Leave the tooltip unchanged on an indeterminate pick (view changed / pick
          // failed mid-hover); only update it for a definitive hit or no-hit.
          if (picked !== undefined) {
            setTooltip(picked);
          }
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
    let { x, y } = coordinateAtPoint(position.x, position.y);
    let r = Math.abs(coordinateAtPoint(position.x + 1, position.y).x - x);
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

    let cacheKey = await cacheKeyForObject({
      autoLabel: {
        version: 3,
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
  <canvas bind:this={canvas} style:position="absolute" style:top="0" style:left="0"></canvas>
  <div style:width="{width}px" style:height="{height}px" style:position="absolute" style:top="0" style:left="0">
    {#if customOverlay}
      {@const action = customComponentAction(customOverlay)}
      {@const proxy = { location: overlayLocation, width: width, height: height }}
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
    ondblclick={onDoubleClick}
    use:interactionHandler={{
      click: onClick,
      drag: onDrag,
      hover: onHover,
    }}
  >
    <!-- Tooltip point -->
    {#if tooltip != null && renderer != null}
      {@const { x, y } = markerLocation(tooltip)}
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
        {@const { x, y } = markerLocation(point)}
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
    <!-- Cluster labels (2D only) -->
    {#if showClusterLabels && !is3D}
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
    <!-- Range selection interaction and display (2D only) -->
    {#if rangeSelection != null && renderer != null && !is3D}
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
    {@const loc = markerLocation(tooltip)}
    {#if isFinite(loc.x) && isFinite(loc.y)}
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
  {/if}
  <!-- Status bar -->
  {#if resolvedTheme.statusBar}
    <StatusBar
      resolvedTheme={resolvedTheme}
      statusMessage={statusMessage ?? webGPUPrompt}
      distancePerPoint={1 / (pointLocation(1, 0).x - pointLocation(0, 0).x)}
      pointCount={data.x.length}
      selectionMode={selectionMode}
      onSelectionMode={(v) => (selectionMode = v)}
      is3D={is3D}
      onResetCamera={resetCamera}
      filterActive={rangeSelection != null}
      onClearFilter={() => setRangeSelection(null)}
    />
  {/if}
</div>
