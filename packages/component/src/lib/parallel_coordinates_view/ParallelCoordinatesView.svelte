<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { interactionHandler, type CursorValue } from "@embedding-atlas/utils";
  import { onDestroy, onMount } from "svelte";

  import { defaultCategoryColors, parseColorNormalizedRgb } from "../colors.js";
  import { requestWebGPUDevice } from "../webgpu_renderer/utils.js";
  import type { ParallelCoordinatesViewProps } from "./api.js";
  import { ParallelCoordinatesRendererWebGPU } from "./renderer.js";
  import { resolveTheme } from "./theme.js";

  type AxisLabel = { value: number; label: string; priority?: number };

  let {
    data,
    axisLabels = null,
    axisTitles = null,
    categoryColors = null,
    opacity = null,
    binCount = null,
    width = null,
    height = null,
    pixelRatio = null,
    colorScheme = null,
    theme = null,
    brush = null,
    onBrushChange = null,
    autoOpacity = null,
  }: ParallelCoordinatesViewProps = $props();

  // Horizontal edge padding. The leftmost axis sits `gap` from the left edge (room for its ticks);
  // each axis then owns an equal slot to its right where its tick labels are drawn.
  const gap = 8;
  const labelInset = 4; // gap between an axis/its ticks and the start of its labels
  const gamma = 2.2;

  let resolvedWidth = $derived(width ?? 800);
  let resolvedHeight = $derived(height ?? 500);
  let resolvedPixelRatio = $derived(pixelRatio ?? 2);
  let resolvedOpacity = $derived(opacity ?? 1);
  let resolvedAutoOpacity = $derived(autoOpacity ?? true);
  let resolvedColorScheme = $derived(colorScheme ?? "light");
  let resolvedTheme = $derived(resolveTheme(theme, resolvedColorScheme));

  // Font sizes are themeable; the label/title box heights and the title gap derive from them so the
  // vertical layout scales with the font size. (CSS px throughout.)
  let labelFontSize = $derived(resolvedTheme.labelFontSize); // also the top/bottom margin so end labels aren't clipped
  let titleFontSize = $derived(resolvedTheme.titleFontSize);
  let labelBoxHeight = $derived(Math.ceil(labelFontSize * 1.6)); // height of the centered tick-label box
  let titleBoxHeight = $derived(Math.ceil(titleFontSize * 1.6)); // height of the axis-title box at the very top

  // Reserve a title row at the top only when at least one axis has a title; otherwise fit tight.
  let hasTitles = $derived((axisTitles ?? []).some((t) => t != null && t !== ""));
  // Top margin clears the topmost tick label's upper half, plus the title row + spacing when present.
  let margin = $derived({
    top: hasTitles ? titleBoxHeight + labelBoxHeight / 2 : labelFontSize,
    bottom: labelFontSize,
  });

  // Default bin count: ~1 bin per 2px of axis height, clamped to [32, 256]. Adapts on resize.
  let resolvedBinCount = $derived(
    binCount ?? Math.max(32, Math.min(256, Math.round((resolvedHeight - margin.top - margin.bottom) / 2))),
  );

  let pixelWidth = $derived(Math.max(1, Math.round(resolvedWidth * resolvedPixelRatio)));
  let pixelHeight = $derived(Math.max(1, Math.round(resolvedHeight * resolvedPixelRatio)));

  let numFields = $derived(data.values.length);
  let numRows = $derived(data.values[0]?.length ?? 0);

  // Interleave the column-major values into the renderer's row-major buffer.
  let interleaved = $derived.by(() => {
    let nf = numFields;
    let nr = numRows;
    let out = new Float32Array(nr * nf);
    for (let f = 0; f < nf; f++) {
      let col = data.values[f];
      for (let r = 0; r < nr; r++) {
        out[r * nf + f] = col[r];
      }
    }
    return out;
  });

  // Bake the color LUT + normalized per-row color value.
  let colorData = $derived.by(() => buildColor(data.colorValue ?? null, categoryColors, numRows));

  function setLUT(lut: Uint8Array, i: number, color: { r: number; g: number; b: number; a: number }) {
    lut[i * 4 + 0] = Math.round(color.r * 255);
    lut[i * 4 + 1] = Math.round(color.g * 255);
    lut[i * 4 + 2] = Math.round(color.b * 255);
    lut[i * 4 + 3] = Math.round(color.a * 255);
  }

  function buildColor(
    colorValue: Float32Array | Uint8Array | null,
    categoryColors: string[] | null | undefined,
    numRows: number,
  ): { lut: Uint8Array<ArrayBuffer>; cv: Float32Array<ArrayBuffer>; lutSize: number; interpolate: boolean } {
    let palette = categoryColors ?? defaultCategoryColors(10);
    if (palette.length == 0) {
      palette = ["#888888"];
    }
    // Copy the palette straight into the LUT, one color per texel -- no interpolation here. `lutSize` is
    // the number of real entries; the unused tail repeats the last color so clamped sampling is safe.
    let lutSize = Math.min(256, palette.length);
    let lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      setLUT(lut, i, parseColorNormalizedRgb(palette[Math.min(i, lutSize - 1)]));
    }

    if (colorValue == null) {
      // Single color: every row points at the first texel.
      return { lut, cv: new Float32Array(numRows), lutSize: 1, interpolate: false };
    }

    if (colorValue instanceof Uint8Array) {
      // Categorical: the value is a direct LUT texel index.
      let cv = new Float32Array(numRows);
      for (let r = 0; r < numRows; r++) {
        cv[r] = colorValue[r];
      }
      return { lut, cv, lutSize, interpolate: false };
    }

    // Continuous: the value in [0, 1] interpolates across the LUT's real entries (done shader-side:
    // 0 -> first real texel, 1 -> last real texel).
    let cv = colorValue instanceof Float32Array ? colorValue : new Float32Array(colorValue);
    // Ensure the buffer is plain ArrayBuffer-backed.
    if (!(cv.buffer instanceof ArrayBuffer)) {
      cv = new Float32Array(cv);
    }
    return { lut, cv: cv as Float32Array<ArrayBuffer>, lutSize, interpolate: true };
  }

  /**
   * Auto opacity factor: pick a per-row alpha so a peak-density pixel (~numRows/binCount rows
   * converging at an axis) reaches a readable coverage under the composite's OIT curve
   * 1-(1-alpha)^count. The caller multiplies it by the user opacity.
   */
  function resolveAutoOpacity(numRows: number, binCount: number): number {
    const K_PEAK = 2.5; // peak-pixel coverage (1 - exp(-K_PEAK) ≈ 0.92)
    let peakCount = Math.max(1, numRows / binCount);
    return 1 - Math.exp(-K_PEAK / peakCount);
  }

  // Axis layout (CSS px for the SVG overlay, device px for the renderer).
  let layout = $derived.by(() => {
    let plotTop = margin.top;
    let plotBottom = resolvedHeight - margin.bottom;
    // Each axis owns an equal slot; its line sits at the slot's left edge and its labels fill the
    // slot to the right. The leftmost axis is `gap` from the edge; the rightmost label ends `gap` before it.
    let slotWidth = numFields > 0 ? (resolvedWidth - gap * 2) / numFields : 0;
    let axisXsCss = new Float32Array(numFields);
    for (let i = 0; i < numFields; i++) {
      axisXsCss[i] = gap + i * slotWidth;
    }
    let axisXs = new Float32Array(numFields);
    for (let i = 0; i < numFields; i++) {
      axisXs[i] = axisXsCss[i] * resolvedPixelRatio;
    }
    return {
      axisXsCss,
      axisXs,
      slotWidth,
      plotTop,
      plotBottom,
      plotY1: plotTop * resolvedPixelRatio,
      plotY2: plotBottom * resolvedPixelRatio,
    };
  });

  // Tick labels sit to the right of each axis, clipped to the slot (minus padding) via CSS ellipsis.
  let maxLabelWidth = $derived(Math.max(0, layout.slotWidth - labelInset * 2));

  // ---- Label overlap resolution ----
  // Tick marks are always drawn; only their text labels can crowd together. Each label is centered at
  // its value's Y position in a box `labelBoxHeight` tall, so when boxes overlap we greedily drop the
  // lower-priority labels: process labels in priority order (higher `priority` first; ties broken by
  // their original order within the list) and keep a label only if its box doesn't overlap one that was
  // already kept on that axis. The tick mark itself stays either way. Default priority is 0.
  // `visibleLabels[i]` is the set of tick objects (by reference) whose label should be shown on axis `i`.
  let visibleLabels = $derived.by(() => {
    let plotHeight = layout.plotBottom - layout.plotTop;
    return (axisLabels ?? []).map((ticks) => {
      let kept = new Set<AxisLabel>();
      if (ticks == null) {
        return kept;
      }
      // Priority order: higher priority first, ties broken by original index.
      let order = ticks.map((tick, i) => ({ tick, i }));
      order.sort((a, b) => (b.tick.priority ?? 0) - (a.tick.priority ?? 0) || a.i - b.i);
      let placed: { top: number; bottom: number }[] = [];
      for (let { tick } of order) {
        let y = layout.plotTop + tick.value * plotHeight;
        let top = y - labelBoxHeight / 2;
        let bottom = y + labelBoxHeight / 2;
        if (placed.some((p) => top < p.bottom && bottom > p.top)) {
          continue; // overlaps a higher-priority label that was already placed; discard.
        }
        placed.push({ top, bottom });
        kept.add(tick);
      }
      return kept;
    });
  });

  // ---- Axis brushes (normalized [0,1], 0 = top). Purely an overlay; never affects the data. ----
  const BRUSH_HALF_WIDTH = 8; // half-width of the brush interactive region, CSS px
  const BRUSH_INDICATOR_HALF_WIDTH = 6; // half-width of the visible brush band, CSS px
  const BRUSH_BORDER = 8; // resize handle thickness, CSS px (matches the embedding view)

  let svgElement: SVGSVGElement | null = null;

  function pixelToValue(yPx: number): number {
    let h = layout.plotBottom - layout.plotTop;
    if (h <= 0) {
      return 0;
    }
    return Math.max(0, Math.min(1, (yPx - layout.plotTop) / h));
  }
  function valueToPixel(v: number): number {
    return layout.plotTop + v * (layout.plotBottom - layout.plotTop);
  }
  function sortedBrush(b: [number, number] | null | undefined): [number, number] | null {
    return b == null ? null : [Math.min(b[0], b[1]), Math.max(b[0], b[1])];
  }

  // Emit the full per-axis array with `index` replaced, always normalized so from <= to.
  function emitBrush(index: number, range: [number, number] | null) {
    if (onBrushChange == null) {
      return;
    }
    let next: ([number, number] | null)[] = [];
    for (let k = 0; k < numFields; k++) {
      next[k] = sortedBrush(k == index ? range : (brush?.[k] ?? null));
    }
    onBrushChange(next);
  }

  function localY(e: CursorValue): number {
    let top = svgElement?.getBoundingClientRect().top ?? 0;
    return e.clientY - top;
  }

  function createBrush(index: number) {
    return (e1: CursorValue) => {
      let v0 = pixelToValue(localY(e1));
      return {
        move: (e2: CursorValue) => emitBrush(index, [v0, pixelToValue(localY(e2))]),
        up: (e2: CursorValue) => {
          let v1 = pixelToValue(localY(e2));
          emitBrush(index, Math.abs(v1 - v0) < 1e-4 ? null : [v0, v1]);
        },
      };
    };
  }

  function moveBrush(index: number) {
    return (e1: CursorValue) => {
      let cur = sortedBrush(brush?.[index]);
      if (cur == null) {
        return;
      }
      let h = layout.plotBottom - layout.plotTop;
      let size = cur[1] - cur[0];
      let resolve = (e2: CursorValue): [number, number] => {
        let dv = h > 0 ? (e2.clientY - e1.clientY) / h : 0;
        let lo = cur[0] + dv;
        let hi = cur[1] + dv;
        if (lo < 0) {
          lo = 0;
          hi = size;
        }
        if (hi > 1) {
          hi = 1;
          lo = 1 - size;
        }
        return [lo, hi];
      };
      return {
        move: (e2: CursorValue) => emitBrush(index, resolve(e2)),
        up: (e2: CursorValue) => emitBrush(index, resolve(e2)),
      };
    };
  }

  function resizeBrush(index: number, edge: 0 | 1) {
    return (e1: CursorValue) => {
      let cur = sortedBrush(brush?.[index]);
      if (cur == null) {
        return;
      }
      let resolve = (e2: CursorValue): [number, number] => {
        let ends: [number, number] = [cur[0], cur[1]];
        ends[edge] = pixelToValue(localY(e2));
        return ends;
      };
      return {
        move: (e2: CursorValue) => emitBrush(index, resolve(e2)),
        up: (e2: CursorValue) => emitBrush(index, resolve(e2)),
      };
    };
  }

  let canvas: HTMLCanvasElement | null = $state(null);
  let renderer: ParallelCoordinatesRendererWebGPU | null = $state(null);
  let message: string | null = $state(null);

  $effect.pre(() => {
    let needsRender = renderer?.setProps({
      values: interleaved,
      numFields: numFields,
      colorValue: colorData.cv,
      colorLUT: colorData.lut,
      colorLUTSize: colorData.lutSize,
      colorMode: colorData.interpolate ? 1 : 0,
      opacity: resolvedAutoOpacity ? resolveAutoOpacity(numRows, resolvedBinCount) * resolvedOpacity : resolvedOpacity,
      binCount: resolvedBinCount,
      width: pixelWidth,
      height: pixelHeight,
      axisXs: layout.axisXs,
      plotY1: layout.plotY1,
      plotY2: layout.plotY2,
      backgroundColor: resolvedColorScheme == "light" ? [1, 1, 1] : [0, 0, 0],
      gamma: gamma,
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
    canvas.style.width = `${renderer.props.width / resolvedPixelRatio}px`;
    canvas.style.height = `${renderer.props.height / resolvedPixelRatio}px`;
    renderer.render();
  }

  let _request: number | null = null;
  function setNeedsRender() {
    if (_request == null) {
      _request = requestAnimationFrame(render);
    }
  }

  // The view owns its device; `destroyed` stops an in-flight device request from creating a renderer after unmount.
  let device: GPUDevice | null = null;
  let destroyed = false;

  function setupRenderer(canvas: HTMLCanvasElement) {
    async function createRenderer() {
      // float32-blendable is required: the bin / accumulation targets are blended rgba32float.
      let newDevice = await requestWebGPUDevice(["float32-blendable"]);
      if (destroyed) {
        newDevice?.destroy();
        return;
      }
      device = newDevice;
      if (newDevice == null) {
        message = "WebGPU is not available in this browser.";
        return;
      }
      let context = canvas.getContext("webgpu");
      if (context == null) {
        message = "Could not get a WebGPU canvas context.";
        return;
      }
      message = null;

      newDevice.lost.then(async (info) => {
        if (info.reason != "destroyed") {
          renderer?.destroy();
          renderer = null;
          context.unconfigure();
          await createRenderer();
        }
      });

      let format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device: newDevice, format: format, alphaMode: "premultiplied" });
      renderer = new ParallelCoordinatesRendererWebGPU(context, newDevice, format, pixelWidth, pixelHeight);
      setNeedsRender();
    }
    createRenderer();
  }

  onMount(() => {
    if (canvas == null) {
      return;
    }
    setupRenderer(canvas);

    // Submit the render commands before serializing, so the image is populated.
    let _toDataURL = canvas.toDataURL;
    canvas.toDataURL = (...args) => {
      render();
      return _toDataURL.apply(canvas, args);
    };
  });

  onDestroy(() => {
    destroyed = true;
    if (_request != null) {
      cancelAnimationFrame(_request);
      _request = null;
    }
    renderer?.destroy();
    renderer = null;
    device?.destroy();
    device = null;
  });
</script>

<div style="position:relative;user-select:none" style:width="{resolvedWidth}px" style:height="{resolvedHeight}px">
  <canvas bind:this={canvas} style="position:absolute;top:0;left:0"></canvas>
  <svg
    bind:this={svgElement}
    width={resolvedWidth}
    height={resolvedHeight}
    style="position:absolute;top:0;left:0;pointer-events:none"
  >
    {#each { length: numFields } as _, i (i)}
      {@const x = layout.axisXsCss[i]}
      <line
        x1={x}
        y1={layout.plotTop}
        x2={x}
        y2={layout.plotBottom}
        stroke={resolvedTheme.axisLineColor}
        stroke-width="1"
        stroke-linecap="butt"
      />
      {#if hasTitles && (axisTitles?.[i] ?? null) != null && axisTitles?.[i] !== ""}
        <foreignObject x={x} y={0} width={maxLabelWidth + labelInset} height={titleBoxHeight}>
          <div
            xmlns="http://www.w3.org/1999/xhtml"
            title={axisTitles?.[i] ?? ""}
            style="box-sizing:border-box;padding:0 4px 0 0;height:{titleBoxHeight}px;line-height:{titleBoxHeight}px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:{titleFontSize}px;font-weight:600;color:{resolvedTheme.labelColor};font-family:{resolvedTheme.fontFamily}"
          >
            {axisTitles?.[i]}
          </div>
        </foreignObject>
      {/if}
      {#each axisLabels?.[i] ?? [] as tick (tick.value + ":" + tick.label)}
        {@const y = layout.plotTop + tick.value * (layout.plotBottom - layout.plotTop)}
        <line
          x1={x - 2}
          y1={y}
          x2={x + 2}
          y2={y}
          stroke={resolvedTheme.tickColor}
          stroke-width="1"
          stroke-linecap="butt"
        />
        {#if visibleLabels[i]?.has(tick)}
          <foreignObject x={x + labelInset} y={y - labelBoxHeight / 2} width={maxLabelWidth} height={labelBoxHeight}>
            <div
              xmlns="http://www.w3.org/1999/xhtml"
              style="box-sizing:border-box;padding:0 4px;height:{labelBoxHeight}px;line-height:{labelBoxHeight}px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:{labelFontSize}px;paint-order:stroke;-webkit-text-stroke:2px {resolvedTheme.labelOutlineColor};color:{resolvedTheme.labelColor};font-family:{resolvedTheme.fontFamily}"
            >
              {tick.label}
            </div>
          </foreignObject>
        {/if}
      {/each}

      <!-- Brush: a transparent track to create on, then the brush band + resize handles. -->
      <rect
        x={x - BRUSH_HALF_WIDTH}
        y={layout.plotTop}
        width={BRUSH_HALF_WIDTH * 2}
        height={layout.plotBottom - layout.plotTop}
        fill="none"
        style:pointer-events="fill"
        style:cursor="crosshair"
        role="none"
        use:interactionHandler={{
          drag: createBrush(i),
          click: () => {
            if ((brush?.[i] ?? null) != null) {
              emitBrush(i, null);
            }
          },
        }}
      />
      {@const rb = sortedBrush(brush?.[i])}
      {#if rb}
        {@const y0 = valueToPixel(rb[0])}
        {@const y1 = valueToPixel(rb[1])}
        <!-- Move hit region: full interactive width, transparent. -->
        <rect
          x={x - BRUSH_HALF_WIDTH}
          y={y0}
          width={BRUSH_HALF_WIDTH * 2}
          height={y1 - y0}
          style:stroke="none"
          style:fill="none"
          style:cursor="move"
          style:pointer-events="all"
          role="none"
          use:interactionHandler={{ drag: moveBrush(i) }}
        />
        <!-- Visible band: thin, non-interactive. -->
        <rect
          x={x - BRUSH_INDICATOR_HALF_WIDTH}
          y={y0}
          width={BRUSH_INDICATOR_HALF_WIDTH * 2}
          height={y1 - y0}
          style:stroke="#fff"
          style:fill="rgba(128,128,128,0.25)"
          style:pointer-events="none"
        />
        <rect
          x={x - BRUSH_HALF_WIDTH}
          y={y0 - BRUSH_BORDER / 2}
          width={BRUSH_HALF_WIDTH * 2}
          height={BRUSH_BORDER}
          style:stroke="none"
          style:fill="none"
          style:pointer-events="all"
          style:cursor="ns-resize"
          role="none"
          use:interactionHandler={{ drag: resizeBrush(i, 0) }}
        />
        <rect
          x={x - BRUSH_HALF_WIDTH}
          y={y1 - BRUSH_BORDER / 2}
          width={BRUSH_HALF_WIDTH * 2}
          height={BRUSH_BORDER}
          style:stroke="none"
          style:fill="none"
          style:pointer-events="all"
          style:cursor="ns-resize"
          role="none"
          use:interactionHandler={{ drag: resizeBrush(i, 1) }}
        />
      {/if}
    {/each}
  </svg>
  {#if message}
    <div
      style="position:absolute;top:50%;left:0;width:100%;text-align:center;transform:translateY(-50%);font-size:13px"
      style:color={resolvedTheme.labelColor}
    >
      {message}
    </div>
  {/if}
</div>
