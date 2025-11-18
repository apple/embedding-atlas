<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  import * as d3 from "d3";
  import type { SVGAttributes } from "svelte/elements";

  import type { ChartTheme } from "../common/theme.js";
  import type { XYFrameProxy } from "../common/types.js";
  import type { DataTable, LayerOutputs } from "./runtime.js";
  import type { Dimension, Interpolate, MarkStyle } from "./spec.js";

  interface ResolveContext {
    proxy: XYFrameProxy;
    data: DataTable;
    theme: ChartTheme;
  }

  function proxyArray<T>(length: number, value: (index: number) => T): ArrayLike<T> {
    let isValidIndex = (i: string | symbol) => typeof i == "string" && !isNaN(+i);
    return new Proxy(
      { length: length },
      {
        get(_, prop) {
          if (isValidIndex(prop)) {
            return value(+(prop as any));
          }
        },
        has(_, prop) {
          return isValidIndex(prop);
        },
      },
    );
  }

  function mapped<T, V>(column: T[], mapper: (x: T) => V, indices?: number[]): ArrayLike<V> {
    if (indices) {
      return proxyArray(indices.length, (i) => mapper(column[indices[i]]));
    } else {
      return proxyArray(column.length, (i) => mapper(column[i]));
    }
  }

  function resolvePoint({ proxy, data }: ResolveContext, axis: "x" | "y"): ArrayLike<number> {
    let column = data.columns[axis];
    let scale = proxy.scale[axis];
    if (column != undefined && scale != undefined) {
      return mapped(column, scale.apply);
    }
    let constant: number = axis == "x" ? proxy.plotWidth / 2 : proxy.plotHeight / 2;
    return proxyArray(data.length, () => constant);
  }

  function dimensionModifier(dimension?: Dimension): ((v: [number, number]) => [number, number]) | undefined {
    if (typeof dimension == "number") {
      let dv = dimension / 2;
      return ([v1, v2]) => [(v1 + v2) / 2 - dv, (v1 + v2) / 2 + dv];
    } else if (dimension != undefined) {
      if ("gap" in dimension) {
        let { gap, clampToRatio } = dimension;
        return ([v1, v2]) => {
          let dv = Math.min(gap / 2, ((clampToRatio ?? 0) * Math.abs(v1 - v2)) / 2);
          if (v1 < v2) {
            return [v1 + dv, v2 - dv];
          } else {
            return [v1 - dv, v2 + dv];
          }
        };
      } else if ("ratio" in dimension) {
        let s = (1 - dimension.ratio) / 2;
        return ([v1, v2]) => [v1 + (v2 - v1) * s, v2 + (v1 - v2) * s];
      }
    }
  }

  function resolveBand(
    { proxy, data }: ResolveContext,
    axis: "x" | "y",
    dimension?: Dimension,
  ): ArrayLike<[number, number]> {
    let modifier = dimensionModifier(dimension);
    let column = data.columns[axis];
    let scale = proxy.scale[axis];
    if (column != undefined && scale != undefined) {
      if (modifier) {
        return mapped(column, (v) => modifier(scale.applyBand(v)));
      } else {
        return mapped(column, scale.applyBand);
      }
    }
    let c1 = data.columns[axis + "1"];
    let c2 = data.columns[axis + "2"];
    if (c1 != undefined && c2 != undefined && scale != undefined) {
      let zipped = c1.map((v1, i) => [v1, c2[i]]);
      if (modifier) {
        return mapped(zipped, ([v1, v2]) => modifier([scale.apply(v1), scale.apply(v2)]));
      } else {
        return mapped(zipped, ([v1, v2]) => [scale.apply(v1), scale.apply(v2)]);
      }
    }
    let constant: [number, number] = axis == "x" ? [0, proxy.plotWidth] : [0, proxy.plotHeight];
    return proxyArray(data.length, () => constant);
  }

  function resolveColor({ proxy, data, theme }: ResolveContext): ArrayLike<string> {
    if (data.columns.color != undefined && proxy.scale.color != undefined) {
      return mapped(data.columns.color, proxy.scale.color.apply);
    }
    let constant = theme.markColor;
    return proxyArray(data.length, () => constant);
  }

  function resolveSize({ proxy, data }: ResolveContext): ArrayLike<number> {
    if (data.columns.size != undefined && proxy.scale.size != undefined) {
      return mapped(data.columns.size, proxy.scale.size.apply);
    }
    let constant = 100;
    return proxyArray(data.length, () => constant);
  }

  function lineData(data: DataTable): number[][] {
    let groupBy = Object.keys(data.columns).filter((k) => k != "x" && k != "y");
    // Group by everything except x, y.
    let map = new Map<string, number[]>();
    for (let i = 0; i < data.length; i++) {
      let key = JSON.stringify(groupBy.map((x) => data.columns[x][i]));
      let line = map.get(key);
      if (line == undefined) {
        map.set(key, [i]);
      } else {
        line.push(i);
      }
    }
    return Array.from(map.values());
  }

  function resolveInterpolate(value: Interpolate, orientation: "horizontal" | "vertical"): d3.CurveFactory {
    switch (value) {
      case "monotone":
        if (orientation == "horizontal") {
          return d3.curveMonotoneY;
        } else {
          return d3.curveMonotoneX;
        }
      case "basis":
        return d3.curveBasis;
      case "natural":
        return d3.curveNatural;
      case "step":
        return d3.curveStep;
      case "step-before":
        return d3.curveStepBefore;
      case "step-after":
        return d3.curveStepAfter;
      case "catmull-rom":
        return d3.curveCatmullRom;
      case "cardinal":
        return d3.curveCardinal;
      case "linear":
        return d3.curveLinear;
      default:
        console.warn(`Unknown interpolate: ${value}`);
        return d3.curveLinear;
    }
  }

  function linePath(ctx: ResolveContext, indices: number[], layer: LayerOutputs) {
    let orientation = layer.orientation ?? "vertical";
    let x = resolvePoint(ctx, "x");
    let y = resolvePoint(ctx, "y");
    let points = indices.filter((i) => isFinite(x[i]) && isFinite(y[i])).sort((ai, bi) => x[ai] - x[bi]);
    return d3
      .line<number>()
      .x((i) => x[i])
      .y((i) => y[i])
      .curve(resolveInterpolate(layer.interpolate, orientation))(points);
  }

  function areaPath(ctx: ResolveContext, indices: number[], layer: LayerOutputs) {
    let orientation = layer.orientation ?? "vertical";
    if (orientation == "vertical") {
      let x = resolvePoint(ctx, "x");
      let y = resolveBand(ctx, "y", undefined);
      let points = indices
        .filter((i) => isFinite(x[i]) && isFinite(y[i][0]) && isFinite(y[i][1]))
        .sort((ai, bi) => x[ai] - x[bi]);
      let fn = d3
        .area<number>()
        .x((i) => x[i])
        .y0((i) => y[i][0])
        .y1((i) => y[i][1])
        .curve(resolveInterpolate(layer.interpolate, orientation));
      return fn(points);
    } else {
      let x = resolveBand(ctx, "x", undefined);
      let y = resolvePoint(ctx, "y");
      let points = indices
        .filter((i) => isFinite(y[i]) && isFinite(x[i][0]) && isFinite(x[i][1]))
        .sort((ai, bi) => y[ai] - y[bi]);
      let fn = d3
        .area<number>()
        .y((i) => y[i])
        .x0((i) => x[i][0])
        .x1((i) => x[i][1])
        .curve(resolveInterpolate(layer.interpolate, orientation));
      return fn(points);
    }
  }

  function resolveStyle(rctx: ResolveContext, style: MarkStyle): ArrayLike<Partial<SVGAttributes<SVGElement>>> {
    let color = resolveColor(rctx);

    function maybeConstColor(value: string | undefined | null): string | undefined {
      if (value == null) {
        return "none";
      }
      if (value == "$encoding") {
        return undefined;
      }
      if (value.startsWith("$")) {
        return (rctx.theme as any)[value.substring(1)] ?? value;
      }
      return value;
    }

    let consts: Partial<SVGAttributes<SVGElement>> = {
      "stroke-width": style.strokeWidth,
      "stroke-linecap": style.strokeCap,
      "stroke-linejoin": style.strokeJoin,
      "stroke-opacity": style.strokeOpacity,
      "fill-opacity": style.fillOpacity,
      "paint-order": style.paintOrder,
      opacity: style.opacity,
      fill: maybeConstColor(style.fillColor),
      stroke: maybeConstColor(style.strokeColor),
    };

    return proxyArray(rctx.data.length, (index) => {
      let r = {
        ...consts,
      };
      if (style.fillColor == "$encoding") {
        r.fill = color[index];
      }
      if (style.strokeColor == "$encoding") {
        r.stroke = color[index];
      }
      return r;
    });
  }
</script>

<script lang="ts">
  interface Props {
    proxy: XYFrameProxy;
    theme: ChartTheme;
    layer: LayerOutputs;
  }

  let { proxy, theme, layer }: Props = $props();

  let rctx = $derived<ResolveContext>({ proxy, data: layer.data, theme });
</script>

<g>
  {#if layer.primitive == "rect"}
    {@const x = resolveBand(rctx, "x", layer.xDimension)}
    {@const y = resolveBand(rctx, "y", layer.yDimension)}
    {@const attrs = resolveStyle(rctx, layer.style)}
    {#each { length: layer.data.length } as _, i}
      {@const [x0, x1] = x[i]}
      {@const [y0, y1] = y[i]}
      {#if isFinite(x0) && isFinite(x1) && isFinite(y0) && isFinite(y1)}
        <rect
          x={Math.min(x0, x1)}
          y={Math.min(y0, y1)}
          width={Math.abs(x0 - x1)}
          height={Math.abs(y0 - y1)}
          {...attrs[i]}
        />
      {/if}
    {/each}
  {:else if layer.primitive == "point"}
    {@const x = resolvePoint(rctx, "x")}
    {@const y = resolvePoint(rctx, "y")}
    {@const attrs = resolveStyle(rctx, layer.style)}
    {@const size = resolveSize(rctx)}
    {#each { length: layer.data.length } as _, i}
      {#if isFinite(x[i]) && isFinite(y[i])}
        <circle cx={x[i]} cy={y[i]} r={Math.sqrt(size[i] / Math.PI)} {...attrs[i]} />
      {/if}
    {/each}
  {:else if layer.primitive == "rule"}
    {@const x = resolveBand(rctx, "x", layer.xDimension)}
    {@const y = resolveBand(rctx, "y", layer.yDimension)}
    {@const attrs = resolveStyle(rctx, layer.style)}
    {#each { length: layer.data.length } as _, i}
      {@const [x0, x1] = x[i]}
      {@const [y0, y1] = y[i]}
      {#if isFinite(x0) && isFinite(x1) && isFinite(y0) && isFinite(y1)}
        <line x1={x0} y1={y0} x2={x1} y2={y1} {...attrs[i]} />
      {/if}
    {/each}
  {:else if layer.primitive == "line"}
    {@const attrs = resolveStyle(rctx, layer.style)}
    {#each lineData(layer.data) as ids}
      <path d={linePath(rctx, ids, layer)} {...attrs[ids[0]]} />
    {/each}
  {:else if layer.primitive == "area"}
    {@const attrs = resolveStyle(rctx, layer.style)}
    {#each lineData(layer.data) as ids}
      <path d={areaPath(rctx, ids, layer)} {...attrs[ids[0]]} />
    {/each}
  {/if}
</g>
