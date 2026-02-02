<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { SVGAttributes } from "svelte/elements";
  import type { ChartTheme } from "../common/theme.js";
  import type { XYFrameProxy } from "../common/types.js";
  import {
    areaPath,
    lineData,
    linePath,
    resolveBand,
    resolvePoint,
    resolveSize,
    resolveStyle,
    type ResolveContext,
  } from "./layer_helper.js";
  import type { LayerOutputs } from "./runtime.js";

  interface Props {
    proxy: XYFrameProxy;
    theme: ChartTheme;
    layer: LayerOutputs;
  }

  let { proxy, theme, layer }: Props = $props();
  let container: SVGGElement;

  function createElement(
    type: string,
    attrs1: Partial<SVGAttributes<SVGElement>>,
    attrs2: Partial<SVGAttributes<SVGElement>>,
  ): SVGElement {
    let el = document.createElementNS("http://www.w3.org/2000/svg", type);
    for (const [k, v] of Object.entries(attrs1)) {
      el.setAttribute(k, v);
    }
    for (const [k, v] of Object.entries(attrs2)) {
      el.setAttribute(k, v);
    }
    return el;
  }

  function* createElements(rctx: ResolveContext, layer: LayerOutputs): Generator<SVGElement> {
    let elements: SVGElement[] = [];
    switch (layer.primitive) {
      case "rect": {
        const x = resolveBand(rctx, "x", layer.xDimension);
        const y = resolveBand(rctx, "y", layer.yDimension);
        const attrs = resolveStyle(rctx, layer.style);
        for (let i = 0; i < layer.data.length; i++) {
          const [x0, x1] = x.at(i);
          const [y0, y1] = y.at(i);
          if (isFinite(x0) && isFinite(x1) && isFinite(y0) && isFinite(y1)) {
            let rx = Math.min(x0, x1);
            let ry = Math.min(y0, y1);
            let rw = Math.abs(x1 - x0);
            let rh = Math.abs(y1 - y0);
            yield createElement("rect", { x: rx, y: ry, width: rw, height: rh }, attrs.at(i));
          }
        }
        break;
      }
      case "point": {
        const x = resolvePoint(rctx, "x");
        const y = resolvePoint(rctx, "y");
        const size = resolveSize(rctx, layer.pointSize);
        const attrs = resolveStyle(rctx, layer.style);
        for (let i = 0; i < layer.data.length; i++) {
          let xi = x.at(i);
          let yi = y.at(i);
          let sizei = size.at(i);
          if (isFinite(xi) && isFinite(yi) && isFinite(sizei)) {
            yield createElement("circle", { cx: xi, cy: yi, r: Math.sqrt(sizei / Math.PI) }, attrs.at(i));
          }
        }
        break;
      }
      case "rule": {
        const x = resolveBand(rctx, "x", layer.xDimension);
        const y = resolveBand(rctx, "y", layer.yDimension);
        const attrs = resolveStyle(rctx, layer.style);
        for (let i = 0; i < layer.data.length; i++) {
          const [x0, x1] = x.at(i);
          const [y0, y1] = y.at(i);
          if (isFinite(x0) && isFinite(x1) && isFinite(y0) && isFinite(y1)) {
            yield createElement("line", { x1: x0, y1: y0, x2: x1, y2: y1 }, attrs.at(i));
          }
        }
        break;
      }
      case "line": {
        const attrs = resolveStyle(rctx, layer.style);
        for (let ids of lineData(layer.data)) {
          let d = linePath(rctx, ids, layer);
          if (d != null) {
            yield createElement("path", { d: d }, attrs.at(ids[0]));
          }
        }
        break;
      }
      case "area": {
        const attrs = resolveStyle(rctx, layer.style);
        for (let ids of lineData(layer.data)) {
          let d = areaPath(rctx, ids, layer);
          if (d != null) {
            yield createElement("path", { d: d }, attrs.at(ids[0]));
          }
        }
        break;
      }
    }
    return elements;
  }

  $effect(() => {
    let elements = createElements({ proxy, data: layer.data, theme }, layer);

    let frag = document.createDocumentFragment();
    for (let el of elements) {
      frag.appendChild(el);
    }
    container.replaceChildren(frag);
  });
</script>

<g bind:this={container}></g>
