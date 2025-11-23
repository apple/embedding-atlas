<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  interface Props<Selection> {
    data: {
      x: Float32Array<ArrayBuffer>;
      y: Float32Array<ArrayBuffer>;
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
    queryClusterLabels: ((clusters: Rectangle[][]) => Promise<(string | null)[]>) | null;
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
  }

  interface Cluster {
    x: number;
    y: number;
    sumDensity: number;
    rects: Rectangle[];
    bandwidth: number;
    label?: string | null;
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
  import { interactionHandler, type CursorValue } from "@embedding-atlas/utils";
  import { onDestroy, onMount } from "svelte";

  import EditableRectangle from "./EditableRectangle.svelte";
  import Lasso from "./Lasso.svelte";
  import StatusBar from "./StatusBar.svelte";
  import TooltipContainer from "./TooltipContainer.svelte";

  import { defaultCategoryColors } from "../colors.js";
  import type { EmbeddingRenderer } from "../renderer_interface.js";
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
  import { isWebGPUAvailable } from "../webgpu_renderer/utils.js";
  import { customComponentAction, customComponentProps } from "./custom_component_helper.js";
  import type { EmbeddingViewConfig } from "./embedding_view_config.js";
  import { layoutLabels, type LabelWithPlacement } from "./labels.js";
  import { simplifyPolygon } from "./simplify_polygon.js";
  import { resolveTheme, type ThemeConfig } from "./theme.js";
  import type { Cache, CustomComponent, Label, OverlayProxy } from "./types.js";
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
  }: Props<Selection> = $props();

  let showClusterLabels = true;

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
  let autoLabelEnabled = $derived(config?.autoLabelEnabled);

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
      webGPUPrompt = "WebGPU is unavailable. Falling back to WebGL.";
    }
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
        version: 1,
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

    if (queryClusterLabels) {
      let labels = await queryClusterLabels(newClusters.map((x) => x.rects));
      for (let i = 0; i < newClusters.length; i++) {
        newClusters[i].label = labels[i];
      }
    }

    let result: Label[] = newClusters
      .filter((x) => x.label != null && x.label.length > 0)
      .map((x) => ({
        x: x.x,
        y: x.y,
        text: x.label!,
        priority: x.sumDensity,
        level: x.bandwidth == 10 ? 0 : 1,
      }));

    if (cache != null) {
      await cache.set(cacheKey, result);
    }

    return result;
  }

  async function updateLabels(viewport: ViewportState) {
    if (renderer == null) {
      return;
    }
    if (labels != null) {
      clusterLabels = await layoutLabels(width, height, viewport, labels, resolvedTheme.fontFamily);
    } else {
      statusMessage = "Generating labels...";
      let result = await generateLabels(viewport);
      clusterLabels = await layoutLabels(width, height, viewport, result, resolvedTheme.fontFamily);
      statusMessage = null;
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
      statusMessage={statusMessage ?? webGPUPrompt}
      distancePerPoint={1 / (pointLocation(1, 0).x - pointLocation(0, 0).x)}
      pointCount={data.x.length}
      selectionMode={selectionMode}
      onSelectionMode={(v) => (selectionMode = v)}
    />
  {/if}
</div>
