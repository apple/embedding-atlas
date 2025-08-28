<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  interface Props<Selection> {
    data: {
      x: Float32Array;
      y: Float32Array;
      category: Uint8Array | null;
    };
    categoryCount: number;
    categoryColors: string[] | null;
    width: number;
    height: number;
    pixelRatio: number;
    colorScheme: "light" | "dark";
    theme: Theme | null;
    mode: "points" | "density";
    minimumDensity: number;
    totalCount: number | null;
    maxDensity: number | null;
    automaticLabels: AutomaticLabelsConfig | boolean;
    queryClusterLabels: ((rects: Rectangle[]) => Promise<string | null>) | null;
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
  }

  interface Cluster {
    x: number;
    y: number;
    sumDensity: number;
    rects: Rectangle[];
    bandwidth: number;
    label?: string | null;
  }

  interface InitialLabel {
    text: string;
    x: number;
    y: number;
    priority: number;
    level: number;
  }

  interface Label {
    text: string;
    fontSize: number;
    bounds: Rectangle;
    locationAtZero: Point;
    coordinate: Point;
    placement: { minScale: number; maxScale: number } | null;
  }

  function viewingParameters(
    maxDensity: number,
    minimumDensity: number,
    scale: number,
    pixelWidth: number,
    pixelHeight: number,
    pixelRatio: number,
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

    let pointSizeAtThreshold = 0.25 / Math.sqrt(maxPointDensity);
    let pointSize = Math.max(0.2, Math.min(5, pointSizeAtThreshold)) * pixelRatio;
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
  import { onDestroy, onMount } from "svelte";

  import EditableRectangle from "./EditableRectangle.svelte";
  import Lasso from "./Lasso.svelte";
  import StatusBar from "./StatusBar.svelte";
  import TooltipContainer from "./TooltipContainer.svelte";

  import { defaultCategoryColors } from "../colors.js";
  import { measureText } from "../measure_text.js";
  import type { EmbeddingRenderer } from "../renderer_interface.js";
  import {
    cacheKeyForObject,
    deepEquals,
    mouseEventHandlers,
    pointDistance,
    throttleTooltip,
    type MouseModifiers,
    type Point,
    type Rectangle,
    type ViewportState,
  } from "../utils.js";
  import { Viewport } from "../viewport_utils.js";
  import { EmbeddingRendererWebGL2 } from "../webgl2_renderer/renderer.js";
  import { EmbeddingRendererWebGPU } from "../webgpu_renderer/renderer.js";
  import { isWebGPUAvailable } from "../webgpu_renderer/utils.js";
  import { customComponentAction, customComponentProps } from "./custom_component_helper.js";
  import { simplifyPolygon } from "./simplify_polygon.js";
  import { resolveTheme, type Theme } from "./theme.js";
  import type { AutomaticLabelsConfig, CustomComponent, OverlayProxy } from "./types.js";
  import { dynamicLabelPlacement, findClusters } from "./worker/index.js";

  interface SelectionBase {
    x: number;
    y: number;
    category?: number;
    text?: string;
  }

  type Selection = $$Generic<SelectionBase>;

  let {
    data = { x: new Float32Array(), y: new Float32Array(), category: null },
    categoryCount = 1,
    categoryColors = null,
    width = 800,
    height = 800,
    pixelRatio = 2,
    colorScheme = "light",
    theme = null,
    mode = "density",
    minimumDensity = 1 / 16,
    totalCount = null,
    maxDensity = null,
    automaticLabels = false,
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
  }: Props<Selection> = $props();

  let showClusterLabels = true;

  let resolvedTheme = $derived(resolveTheme(theme, colorScheme));
  let resolvedCategoryColors = $derived(categoryColors ?? defaultCategoryColors(categoryCount));

  let resolvedViewportState = $derived(viewportState ?? defaultViewportState ?? { x: 0, y: 0, scale: 1 });
  let resolvedViewport = $derived(new Viewport(resolvedViewportState, width, height));
  let pointLocation = $derived(resolvedViewport.pixelLocationFunction());
  let coordinateAtPoint = $derived(resolvedViewport.coordinateAtPixelFunction());

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

  let clusterLabels: Label[] = $state([]);
  let statusMessage: string | null = $state(null);

  let selectionMode = $state<"marquee" | "lasso" | "none">("none");

  let pixelWidth = $derived(width * pixelRatio);
  let pixelHeight = $derived(height * pixelRatio);

  let canvas: HTMLCanvasElement | null = $state(null);
  let renderer: EmbeddingRenderer | null = $state(null);
  let webGPUPrompt: string | null = $state(null);

  let viewingParams = $derived(
    viewingParameters(
      maxDensity ?? (totalCount ?? data.x.length) / 4,
      minimumDensity,
      resolvedViewportState.scale,
      pixelWidth,
      pixelHeight,
      pixelRatio,
    ),
  );
  let pointSize = $derived(viewingParams.pointSize);

  let needsUpdateLabels = true;

  $effect.pre(() => {
    let needsRender = renderer?.setProps({
      mode: mode,
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
      ...viewingParams,
    });

    if (needsRender) {
      setNeedsRender();
      if (
        automaticLabels !== false &&
        needsUpdateLabels &&
        renderer != null &&
        data.x != null &&
        data.x.length > 0 &&
        defaultViewportState != null
      ) {
        needsUpdateLabels = false;
        updateLabels(defaultViewportState);
      }
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
    let context: WebGL2RenderingContext | null;

    function createRenderer() {
      context = canvas.getContext("webgl2", { antialias: false })!;
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
    async function createRenderer() {
      let context = canvas.getContext("webgpu");
      if (context == null) {
        console.error("Could not get WebGPU canvas context");
        return;
      }

      let adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.error("Could not request WebGPU adapter");
        return;
      }

      let maxBufferSize = 512 * 1048576;
      let maxStorageBufferBindingSize = 512 * 1048576;
      maxBufferSize = Math.min(maxBufferSize, adapter.limits.maxBufferSize);
      maxStorageBufferBindingSize = Math.min(maxStorageBufferBindingSize, adapter.limits.maxStorageBufferBindingSize);
      let descriptor: GPUDeviceDescriptor = {
        requiredLimits: {
          maxBufferSize: maxBufferSize,
          maxStorageBufferBindingSize: maxStorageBufferBindingSize,
        },
        requiredFeatures: ["shader-f16"],
      };
      let device = await adapter.requestDevice(descriptor);

      device.lost.then((info) => {
        console.info(`WebGPU device was lost: ${info.message}`);
        if (info.reason != "destroyed") {
          renderer?.destroy();
          renderer = null;
          createRenderer();
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
    if (isWebGPUAvailable()) {
      setupWebGPURenderer(canvas);
    } else {
      setupWebGLRenderer(canvas);
      webGPUPrompt = "WebGPU is unavailable. If you are using Safari, please enable the WebGPU feature flag.";
    }
  });

  onDestroy(() => {
    renderer?.destroy();
    renderer = null;
  });

  function onZoom(scaler: number, position: Point) {
    let { x, y, scale } = resolvedViewportState;
    setTooltip(null);
    let newScale = Math.min(1e2, Math.max(1e-2, scale * scaler));
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

  function onDrag(p1: Point, modifiers: { shift: boolean; meta: boolean }) {
    setTooltip(null);

    let mode: "marquee" | "lasso" | "pan" = "pan";
    if (selectionMode != "none") {
      if (!modifiers.shift) {
        mode = selectionMode;
      }
    } else {
      if (modifiers.shift) {
        mode = modifiers.meta ? "lasso" : "marquee";
      }
    }

    switch (mode) {
      case "marquee": {
        return {
          move: (p2: Point) => {
            setTooltip(null);
            if (renderer == null) {
              return;
            }
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
          move: (p2: Point) => {
            setTooltip(null);
            if (renderer == null) {
              return;
            }
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
          move: (p2: Point) => {
            setViewportState({
              x: x0 + (p2.x - p1.x) * sx,
              y: y0 + (p2.y - p1.y) * sy,
              scale: resolvedViewportState.scale,
            });
          },
        };
      }
    }
  }

  async function onClick(position: Point | null, modifiers: MouseModifiers) {
    if (rangeSelection != null) {
      setRangeSelection(null);
    } else {
      const newSelection = await selectionFromPoint(position);
      if (newSelection == null) {
        setSelection([]);
        setTooltip(null);
      } else {
        if (modifiers.shift || modifiers.ctrl || modifiers.meta) {
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

  async function onHover(position: Point | null) {
    if (selection != null && selection.length == 1) {
      let cSelection = pointLocation(selection[0].x, selection[0].y);
      if (position != null && pointDistance(position, cSelection) < 10) {
        setTooltip(selection[0]);
      }
    } else {
      setTooltip(await selectionFromPoint(position));
    }
  }

  async function selectionFromPoint(position: Point | null) {
    if (renderer == null || position == null || querySelection == null) {
      return null;
    }
    let { x, y } = coordinateAtPoint(position.x, position.y);
    let r = Math.abs(coordinateAtPoint(position.x + 1, position.y).x - x);
    return await querySelection(x, y, r);
  }

  function hasTooltip() {
    return tooltip != null;
  }

  let mouseHandlers = $derived(
    canvas
      ? mouseEventHandlers(canvas, {
          zoom: onZoom,
          drag: onDrag,
          hover: throttleTooltip(onHover, hasTooltip),
          click: onClick,
        })
      : null,
  );

  async function generateClusters(
    renderer: EmbeddingRenderer,
    bandwidth: number,
    viewport: ViewportState,
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
    let threshold = maxDensity * 0.005;
    return collectedClusters.filter((x) => x.sumDensity > threshold);
  }

  async function generateLabels(viewport: ViewportState): Promise<InitialLabel[]> {
    if (renderer == null) {
      return [];
    }

    let cacheKey = await cacheKeyForObject({ generateLabels: viewport });

    if (typeof automaticLabels == "object" && automaticLabels.cache) {
      let cached = await automaticLabels.cache.get(cacheKey);
      if (cached != null) {
        return cached;
      }
    }

    statusMessage = "Generating clusters...";

    let newClusters = await generateClusters(renderer, 10, viewport);
    newClusters = newClusters.concat(await generateClusters(renderer, 5, viewport));

    statusMessage = "Generating labels (initializing)...";
    if (queryClusterLabels) {
      for (let i = 0; i < newClusters.length; i++) {
        let label = await queryClusterLabels(newClusters[i].rects);
        newClusters[i].label = label;
        statusMessage = `Generating labels (${(((i + 1) / newClusters.length) * 100).toFixed(0)}%)...`;
      }
    }

    let result: InitialLabel[] = newClusters
      .filter((x) => x.label != null)
      .map((x) => ({
        text: x.label!,
        x: x.x,
        y: x.y,
        priority: x.sumDensity,
        level: x.bandwidth == 10 ? 0 : 1,
      }));

    if (typeof automaticLabels == "object" && automaticLabels.cache) {
      await automaticLabels.cache.set(cacheKey, result);
    }

    return result;
  }

  export async function updateLabels(viewport: ViewportState) {
    if (renderer == null) {
      return;
    }
    let vp = new Viewport(viewport, width, height);
    let newClusters = await generateLabels(viewport);

    let currentScale = viewport.scale;
    let maxScale = viewport.scale / 2;
    let scaleThreshold = maxScale * 4;
    let newClusterLabels: Label[] = newClusters.map((cluster) => {
      let p = vp.pixelLocation(cluster.x, cluster.y);
      let fontSize = cluster.level == 0 ? 14 : 12;
      let size = measureText({
        text: cluster.text,
        fontSize: fontSize,
        fontFamily: resolvedTheme.fontFamily,
      });
      size.width += 4;
      size.height += 4;
      let threshold = currentScale / scaleThreshold;
      return {
        text: cluster.text,
        fontSize: fontSize,
        bounds: {
          xMin: p.x - size.width / 2,
          xMax: p.x + size.width / 2,
          yMin: p.y - size.height / 2,
          yMax: p.y + size.height / 2,
        },
        locationAtZero: p,
        priority: cluster.priority,
        minScale: cluster.level == 0 ? threshold / 1.2 : null,
        maxScale: cluster.level == 0 ? null : threshold,
        coordinate: { x: cluster.x, y: cluster.y },
        placement: null,
      };
    });

    let placements = await dynamicLabelPlacement(newClusterLabels, { globalMaxScale: currentScale / maxScale });
    for (let i = 0; i < placements.length; i++) {
      let placement = placements[i];
      if (placement != null) {
        let maxScale = currentScale / placement.minScale;
        let minScale = currentScale / placement.maxScale;
        newClusterLabels[i].placement = { minScale, maxScale };
      }
    }
    clusterLabels = newClusterLabels;
    statusMessage = null;
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
</script>

<div style:width="{width}px" style:height="{height}px" style:position="relative">
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
    onwheel={mouseHandlers?.wheel}
    onmousedown={mouseHandlers?.mousedown}
    onmousemove={mouseHandlers?.mousemove}
    onmouseleave={mouseHandlers?.mouseleave}
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
          {@const rows = label.text.split("\n")}
          {@const location = pointLocation(label.coordinate.x, label.coordinate.y)}
          {@const isVisible =
            label.placement != null &&
            label.placement.minScale <= resolvedViewportState.scale &&
            resolvedViewportState.scale <= label.placement.maxScale}
          <g transform="translate({location.x},{location.y})">
            {#if isVisible}
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
          preventHover={mouseHandlers?.preventHover ?? (() => {})}
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
      statusMessage={statusMessage ?? webGPUPrompt}
      distancePerPoint={1 / (pointLocation(1, 0).x - pointLocation(0, 0).x)}
      pointCount={data.x.length}
      selectionMode={selectionMode}
      onSelectionMode={(v) => (selectionMode = v)}
    />
  {/if}
</div>
