<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  interface Props<Selection> {
    layers: ResolvedLayer[];
    /** Number of points in the main data (for the status bar). */
    pointCount: number;
    /** Category colors of the main data (for the selection markers). */
    categoryCount: number;
    categoryColors: string[] | null;
    width: number;
    height: number;
    pixelRatio: number;
    theme: ThemeConfig | null;
    config: EmbeddingViewConfig | null;
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
    onLayerValues: ((values: LayerValues[]) => void) | null;
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
  import { onDestroy, onMount, untrack } from "svelte";

  import EditableRectangle from "./EditableRectangle.svelte";
  import Lasso from "./Lasso.svelte";
  import StatusBar from "./StatusBar.svelte";
  import TooltipContainer from "./TooltipContainer.svelte";

  import { defaultCategoryColors } from "../colors.js";
  import { pointDistance, throttleTooltip, type Point, type Rectangle, type ViewportState } from "../utils.js";
  import { Viewport } from "../viewport_utils.js";
  import { EmbeddingRenderer, type RendererLayer, type RendererSource } from "../webgpu_renderer/renderer.js";
  import { requestWebGPUDevice } from "../webgpu_renderer/utils.js";
  import { customComponentAction, customComponentProps } from "./custom_component_helper.js";
  import type { EmbeddingViewConfig } from "./embedding_view_config.js";
  import { layoutLabels, type LabelGroup, type LabelWithPlacement } from "./labels.js";
  import { LabelLayers, type LayerLabels } from "./label_layers.js";
  import {
    MAX_DENSITY_CATEGORIES,
    type AutoLabels,
    type LabelSnapshot,
    type LayerValues,
    type ResolvedLabelsLayer,
    type ResolvedLayer,
    type ResolvedSource,
  } from "./layers.js";
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
    layers = [],
    pointCount = 0,
    categoryCount = 1,
    categoryColors = null,
    width = 800,
    height = 800,
    pixelRatio = 2,
    theme = null,
    config = null,
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
    onLayerValues = null,
    cache = null,
  }: Props<Selection> = $props();

  let colorScheme = $derived(config?.colorScheme ?? "light");
  let resolvedTheme = $derived(resolveTheme(theme, colorScheme));
  let resolvedCategoryColors = $derived(categoryColors ?? defaultCategoryColors(categoryCount));

  let resolvedViewportState = $derived(viewportState ?? defaultViewportState ?? { x: 0, y: 0, scale: 1 });
  let resolvedViewport = $derived(new Viewport(resolvedViewportState, width, height));
  let pointLocation = $derived(resolvedViewport.pixelLocationFunction());
  let coordinateAtPoint = $derived(resolvedViewport.coordinateAtPixelFunction());

  let preventHover = $state(false);

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

  let clusterLabels: (LabelWithPlacement & { layer: string })[] = $state([]);
  let labelsGeneratingCount = $state(0);
  let statusMessage: string | null = $derived(labelsGeneratingCount > 0 ? "Generating labels..." : null);

  let selectionMode = $state<"marquee" | "lasso" | "none">("none");

  let pixelWidth = $derived(width * pixelRatio);
  let pixelHeight = $derived(height * pixelRatio);

  let canvas: HTMLCanvasElement | null = $state(null);
  let renderer: EmbeddingRenderer | null = $state(null);
  let webGPUError: string | null = $state(null);

  interface LayerViewingParameters {
    /** Point size for the tooltip / selection markers (before clamping for rendering). */
    pointSize: number;
    renderLayers: RendererLayer[];
    sources: Record<string, RendererSource>;
    /** Effective values of the points and density layers (reported with onLayerValues), by layer key. */
    values: Record<string, LayerValues>;
  }

  function sourceViewingParameters(source: ResolvedSource, minimumDensity: number, userPointSize: number | null) {
    return viewingParameters(
      source.maxDensity ?? (source.totalCount ?? source.x.length) / 4,
      minimumDensity,
      resolvedViewportState.scale,
      pixelWidth,
      pixelHeight,
      pixelRatio,
      userPointSize,
    );
  }

  /** The color of a density layer that has more categories than MAX_DENSITY_CATEGORIES.
   *  A constant array, so the renderer's layer inputs don't change on every frame. */
  const tooManyCategoriesDensityColors = ["#7f7f7f"];

  function computeLayerViewingParameters(layers: ResolvedLayer[]): LayerViewingParameters {
    let sources: Record<string, RendererSource> = {};
    let renderLayers: RendererLayer[] = [];
    let values: LayerViewingParameters["values"] = {};
    let round = (v: number) => Math.round(v * 1000) / 1000;
    // Tooltip and selection markers are for points of the main data: size them like the main data's points
    // layer, or the first points layer if no points layer draws the main data.
    let markerPointSize: number | null = null;
    let mainMarkerPointSize: number | null = null;
    // Points are drawn smaller when a density layer is present (matches the previous density mode).
    let hasDensity = layers.some((l) => l.type == "density");
    for (let layer of layers) {
      if (layer.type == "labels") {
        continue;
      }
      let source = layer.source;
      sources[source.key] = { x: source.x, y: source.y, category: source.category };
      switch (layer.type) {
        case "points": {
          let params = sourceViewingParameters(source, layer.minimumDensity, layer.pointSize);
          markerPointSize = markerPointSize ?? params.pointSize;
          if (source.key == "main") {
            mainMarkerPointSize = mainMarkerPointSize ?? params.pointSize;
          }
          // Points are drawn by area coverage, so any size works (sizes below a pixel fade out smoothly). The
          // automatic size keeps a minimum, so zoomed-out views stay visible; a specified size is used as is.
          let renderedPointSize =
            layer.pointSize != null ? params.pointSize : Math.max(hasDensity ? 1 : 3, params.pointSize);
          let pointsAlpha = layer.opacity == "auto" ? params.pointsAlpha : layer.opacity;
          values[layer.key] = {
            type: "points",
            id: layer.id,
            opacity: round(pointsAlpha),
            pointSize: round(renderedPointSize / pixelRatio),
          };
          renderLayers.push({
            key: layer.key,
            type: "points",
            source: source.key,
            categoryCount: layer.useCategory ? source.categoryCount : 1,
            categoryColors: layer.categoryColors,
            pointSize: renderedPointSize,
            pointAlpha: params.pointAlpha,
            pointsAlpha: pointsAlpha,
            densityScaler: params.densityScaler,
            densityQuantizationStep: 0.1,
            densityAlpha: 0,
            contoursAlpha: 0,
            downsampleMaxPoints: layer.downsampleMaxPoints,
            downsampleDensityWeight: layer.downsampleDensityWeight,
          });
          break;
        }
        case "density": {
          let params = sourceViewingParameters(source, layer.minimumDensity, null);
          let alpha = layer.opacity == "auto" ? params.densityAlpha : layer.opacity;
          let tooManyCategories = layer.useCategory && source.categoryCount > MAX_DENSITY_CATEGORIES;
          let useCategories = layer.useCategory && !tooManyCategories;
          values[layer.key] = { type: "density", id: layer.id, opacity: round(alpha) };
          renderLayers.push({
            key: layer.key,
            type: "density",
            source: source.key,
            categoryCount: useCategories ? source.categoryCount : 1,
            // With too many categories, the density of all categories is drawn together in a neutral color
            // (using the first category's color would suggest it's the density of that category only).
            categoryColors: tooManyCategories ? tooManyCategoriesDensityColors : layer.categoryColors,
            pointSize: params.pointSize,
            pointAlpha: 0,
            pointsAlpha: 0,
            densityScaler: params.densityScaler,
            densityQuantizationStep: 0.1,
            densityAlpha: layer.fill ? alpha : 0,
            contoursAlpha: layer.contours ? alpha : 0,
            downsampleMaxPoints: null,
            downsampleDensityWeight: 0,
          });
          break;
        }
      }
    }
    markerPointSize = mainMarkerPointSize ?? markerPointSize;
    if (markerPointSize == null) {
      let firstSource = layers.find((l) => l.type != "labels")?.source;
      markerPointSize =
        firstSource != null
          ? sourceViewingParameters(firstSource, config?.minimumDensity ?? 1 / 16, config?.pointSize ?? null).pointSize
          : 3 * pixelRatio;
    }
    return { pointSize: markerPointSize, renderLayers, sources, values };
  }

  let layerViewingParams = $derived(computeLayerViewingParameters(layers));

  let pointSize = $derived(layerViewingParams.pointSize);

  $effect.pre(() => {
    let needsRender = renderer?.setProps({
      colorScheme: colorScheme,
      viewportX: resolvedViewportState.x,
      viewportY: resolvedViewportState.y,
      viewportScale: resolvedViewportState.scale,
      width: pixelWidth,
      height: pixelHeight,
      densityBandwidth: 20,
      sources: layerViewingParams.sources,
      layers: layerViewingParams.renderLayers,
    });

    if (needsRender) {
      setNeedsRender();
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

  function setupWebGPURenderer(canvas: HTMLCanvasElement) {
    async function createRenderer() {
      let device = await requestWebGPUDevice();
      if (device == null) {
        console.error("Could not get WebGPU device");
        webGPUError = "WebGPU is unavailable. This view requires a browser with WebGPU support.";
        return;
      }

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

      renderer = new EmbeddingRenderer(context, device, format, pixelWidth, pixelHeight);
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
  });

  function localCoordinates(e: { clientX: number; clientY: number }): Point {
    let rect = canvas?.getBoundingClientRect() ?? { left: 0, top: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
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
    let { x, y } = coordinateAtPoint(position.x, position.y);
    let r = Math.abs(coordinateAtPoint(position.x + 1, position.y).x - x);
    return await querySelection(x, y, r);
  }

  async function generateClusters(
    renderer: EmbeddingRenderer,
    snapshot: LabelSnapshot,
    bandwidth: number,
    densityThreshold: number = 0.005,
  ): Promise<Cluster[]> {
    let map = await renderer.densityMap(snapshot, 1000, 1000, bandwidth, snapshot.viewport);
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

  // Label generations in progress, shared by layers with the same merge key.
  let pendingLabelGenerations = new Map<string, Promise<LayerLabels>>();

  function generateLabels(auto: AutoLabels): Promise<LayerLabels> {
    let pending = pendingLabelGenerations.get(auto.mergeKey);
    if (pending == null) {
      pending = generateLabelsUncached(auto).finally(() => {
        pendingLabelGenerations.delete(auto.mergeKey);
      });
      pendingLabelGenerations.set(auto.mergeKey, pending);
    }
    return pending;
  }

  /** Generate automatic labels: from the persistent cache if they're there (without loading the data), otherwise
   *  from a snapshot of the data (density maps, clusters, and the labels of the clusters). */
  async function generateLabelsUncached(auto: AutoLabels): Promise<LayerLabels> {
    let densityThreshold = auto.densityThreshold ?? config?.autoLabelDensityThreshold ?? null;
    let cacheKey =
      auto.persistKey != null && cache != null
        ? await objectHash({ autoLabels: { version: 5, key: auto.persistKey, densityThreshold } })
        : null;
    if (cacheKey != null) {
      let cached = await cache!.get(cacheKey);
      if (cached != null && Array.isArray(cached.labels) && cached.viewport != null) {
        return cached;
      }
    }

    let snapshot = await auto.loadSnapshot();
    let currentRenderer = renderer;
    if (snapshot.x.length == 0 || currentRenderer == null) {
      return { labels: [], viewport: snapshot.viewport };
    }
    let newClusters = await generateClusters(currentRenderer, snapshot, 10, densityThreshold ?? 0.005);
    newClusters = newClusters.concat(await generateClusters(currentRenderer, snapshot, 5));

    let labels = await auto.queryClusterLabels(
      newClusters.map((x) => x.rects),
      snapshot,
    );
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

    let result: LayerLabels = {
      labels: newClusters
        .filter((x) => x.content != null && (typeof x.content !== "string" || x.content.length > 0))
        .map((x) => ({
          x: x.x,
          y: x.y,
          content: x.content!,
          priority: x.sumDensity,
          level: x.bandwidth == 10 ? 0 : 1,
        })),
      viewport: snapshot.viewport,
    };
    if (cacheKey != null) {
      await cache!.set(cacheKey, result);
    }
    return result;
  }

  /** The labels of each labels layer (reported with onLayerValues), by layer key. */
  let labelResults: Record<string, Label[] | null> = $state.raw({});
  let labelLayoutGeneration = 0;

  const labelLayers = new LabelLayers({
    generate: generateLabels,
    onChange: () => layoutAllLabels(),
    onGenerating: (count) => {
      labelsGeneratingCount = count;
    },
    onError: (error) => console.error("Error while generating labels", error),
  });
  onDestroy(() => labelLayers.destroy());

  $effect.pre(() => {
    let layersWithLabels = layers.filter((l): l is ResolvedLabelsLayer => l.type == "labels");
    // Wait for the renderer (to generate labels) and the default viewport (to lay out static labels).
    if (renderer != null && defaultViewportState != null) {
      untrack(() => labelLayers.sync(layersWithLabels));
    }
  });

  /** Lay out labels from all labels layers together so that labels from different layers don't overlap.
   *  Called whenever the labels of any layer change. */
  async function layoutAllLabels() {
    let results = labelLayers.results();
    labelResults = Object.fromEntries(results.map(({ key, labels }) => [key, labels?.labels ?? null]));
    let generation = ++labelLayoutGeneration;
    let mainViewport = untrack(() => defaultViewportState ?? resolvedViewportState);
    let scaleOf = (viewport: ViewportState) => new Viewport(viewport, 1000, 1000).scale();
    // Each layer's labels are a group: their zoom ranges come from their own levels, relative to the scale they
    // were generated at (static labels: the main viewport), so layers don't change each other's zoom behavior.
    let groups: LabelGroup[] = [];
    let labelLayerKeys: string[] = [];
    for (let { key, labels } of results) {
      if (labels == null || labels.labels.length == 0) {
        continue;
      }
      groups.push({ labels: labels.labels, normalScale: scaleOf(labels.viewport ?? mainViewport) });
      for (let _ of labels.labels) {
        labelLayerKeys.push(key);
      }
    }
    let result = await layoutLabels(groups, resolvedTheme.fontFamily);
    if (generation != labelLayoutGeneration) {
      return;
    }
    clusterLabels = result.map((label, i) => ({ ...label, layer: labelLayerKeys[i] }));
  }

  let labelOpacity = $derived(
    Object.fromEntries(
      layers.filter((l) => l.type == "labels").map((l) => [l.key, l.opacity ?? resolvedTheme.clusterLabelOpacity]),
    ),
  );

  let layerValues: LayerValues[] = $derived(
    layers.flatMap((layer): LayerValues[] => {
      if (layer.type == "labels") {
        let labels = labelResults[layer.key] ?? null;
        return [{ type: "labels", id: layer.id, opacity: labelOpacity[layer.key], labels: labels }];
      }
      let values = layerViewingParams.values[layer.key];
      return values != null ? [values] : [];
    }),
  );

  function shallowEquals(a: Record<string, any>, b: Record<string, any>): boolean {
    let keys = Object.keys(a);
    return keys.length == Object.keys(b).length && keys.every((k) => a[k] === b[k]);
  }

  // Unchanged entries keep their identity, so consumers can compare entries (and labels) by reference:
  // the values change continuously while zooming, but labels change rarely.
  let lastLayerValues: LayerValues[] | null = null;
  $effect(() => {
    let values = layerValues;
    if (onLayerValues == null) {
      return;
    }
    let changed = lastLayerValues == null || values.length != lastLayerValues.length;
    let result = values.map((value, i) => {
      let last = lastLayerValues?.[i];
      if (last != null && shallowEquals(value, last)) {
        return last;
      }
      changed = true;
      return value;
    });
    if (changed) {
      lastLayerValues = result;
      onLayerValues(result);
    }
  });

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
    {#if clusterLabels.length > 0}
      <g>
        {#each clusterLabels as label}
          {@const location = pointLocation(label.coordinate.x, label.coordinate.y)}
          {@const scale = resolvedViewport.scale()}
          {@const isVisible =
            label.placement != null && label.placement.minScale <= scale && scale <= label.placement.maxScale}
          {@const opacity = labelOpacity[label.layer] ?? resolvedTheme.clusterLabelOpacity}
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
                  style:opacity={opacity}
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
                      style:opacity={opacity}
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
      pointCount={pointCount}
      selectionMode={selectionMode}
      onSelectionMode={(v) => (selectionMode = v)}
    />
  {/if}
</div>
