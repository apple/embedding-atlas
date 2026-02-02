import * as d3 from "d3";
import type { SVGAttributes } from "svelte/elements";

import type { ChartTheme } from "../common/theme.js";
import type { XYFrameProxy } from "../common/types.js";
import type { DataTable, LayerOutputs } from "./runtime.js";
import type { Dimension, Interpolate, MarkStyle } from "./spec.js";

export interface ResolveContext {
  proxy: XYFrameProxy;
  data: DataTable;
  theme: ChartTheme;
}

export interface Mapped<T> {
  length: number;
  at: (index: number) => T;
}

function mapped<T, V>(column: T[], mapper: (x: T) => V, indices?: number[]): Mapped<V> {
  if (indices) {
    return { length: indices.length, at: (i) => mapper(column[indices[i]]) };
  } else {
    return { length: column.length, at: (i) => mapper(column[i]) };
  }
}

function constant<T>(length: number, value: T): Mapped<T> {
  return { length: length, at: () => value };
}

export function resolvePoint({ proxy, data }: ResolveContext, axis: "x" | "y"): Mapped<number> {
  let column = data.columns[axis];
  let scale = proxy.scale[axis];
  if (column != undefined && scale != undefined) {
    return mapped(column, scale.apply);
  }
  return constant(data.length, axis == "x" ? proxy.plotWidth / 2 : proxy.plotHeight / 2);
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

export function resolveBand(
  { proxy, data }: ResolveContext,
  axis: "x" | "y",
  dimension?: Dimension,
): Mapped<[number, number]> {
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
  return constant(data.length, axis == "x" ? [0, proxy.plotWidth] : [0, proxy.plotHeight]);
}

export function resolveColor({ proxy, data, theme }: ResolveContext): Mapped<string> {
  if (data.columns.color != undefined && proxy.scale.color != undefined) {
    return mapped(data.columns.color, proxy.scale.color.apply);
  }
  return constant(data.length, theme.markColor);
}

export function resolveSize({ proxy, data }: ResolveContext, defaultValue?: number): Mapped<number> {
  if (data.columns.size != undefined && proxy.scale.size != undefined) {
    return mapped(data.columns.size, proxy.scale.size.apply);
  }
  return constant(data.length, defaultValue ?? 100);
}

export function lineData(data: DataTable): number[][] {
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

export function linePath(
  ctx: ResolveContext,
  indices: number[],
  layer: LayerOutputs,
  canvasContext?: CanvasRenderingContext2D,
) {
  let orientation = layer.orientation ?? "vertical";
  let x = resolvePoint(ctx, "x");
  let y = resolvePoint(ctx, "y");
  let points = indices.filter((i) => isFinite(x.at(i)) && isFinite(y.at(i))).sort((ai, bi) => x.at(ai) - x.at(bi));
  let fn = d3
    .line<number>()
    .x((i) => x.at(i))
    .y((i) => y.at(i))
    .curve(resolveInterpolate(layer.interpolate, orientation));
  if (canvasContext) {
    fn.context(canvasContext);
  }
  return fn(points);
}

export function areaPath(
  ctx: ResolveContext,
  indices: number[],
  layer: LayerOutputs,
  canvasContext?: CanvasRenderingContext2D,
) {
  let orientation = layer.orientation ?? "vertical";
  if (orientation == "vertical") {
    let x = resolvePoint(ctx, "x");
    let y = resolveBand(ctx, "y", undefined);
    let points = indices
      .filter((i) => isFinite(x.at(i)) && isFinite(y.at(i)[0]) && isFinite(y.at(i)[1]))
      .sort((ai, bi) => x.at(ai) - x.at(bi));
    let fn = d3
      .area<number>()
      .x((i) => x.at(i))
      .y0((i) => y.at(i)[0])
      .y1((i) => y.at(i)[1])
      .curve(resolveInterpolate(layer.interpolate, orientation));
    if (canvasContext) {
      fn.context(canvasContext);
    }
    return fn(points);
  } else {
    let x = resolveBand(ctx, "x", undefined);
    let y = resolvePoint(ctx, "y");
    let points = indices
      .filter((i) => isFinite(y.at(i)) && isFinite(x.at(i)[0]) && isFinite(x.at(i)[1]))
      .sort((ai, bi) => y.at(ai) - y.at(bi));
    let fn = d3
      .area<number>()
      .y((i) => y.at(i))
      .x0((i) => x.at(i)[0])
      .x1((i) => x.at(i)[1])
      .curve(resolveInterpolate(layer.interpolate, orientation));
    if (canvasContext) {
      fn.context(canvasContext);
    }
    return fn(points);
  }
}

export function resolveStyle(rctx: ResolveContext, style: MarkStyle): Mapped<Partial<SVGAttributes<SVGElement>>> {
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

  return {
    length: rctx.data.length,
    at: (index) => {
      let r = {
        ...consts,
      };
      if (style.fillColor == "$encoding") {
        r.fill = color.at(index);
      }
      if (style.strokeColor == "$encoding") {
        r.stroke = color.at(index);
      }
      return r;
    },
  };
}

export function resolveCanvasStyle(
  rctx: ResolveContext,
  style: MarkStyle,
): {
  prepare: (ctx: CanvasRenderingContext2D) => void;
  draw: (ctx: CanvasRenderingContext2D, index: number) => void;
} {
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

  let fillFirst = style.paintOrder != "stroke fill";

  return {
    prepare: (ctx) => {
      ctx.lineWidth = style.strokeWidth ?? 1;
      ctx.lineCap = style.strokeCap ?? "butt";
      ctx.lineJoin = style.strokeJoin ?? "bevel";
      ctx.globalAlpha = style.opacity ?? 1;
      let constFillColor = maybeConstColor(style.fillColor);
      let constStrokeColor = maybeConstColor(style.strokeColor);
      if (constFillColor != undefined) {
        ctx.fillStyle = constFillColor;
      }
      if (constStrokeColor != undefined) {
        ctx.strokeStyle = constStrokeColor;
      }
    },
    draw: (ctx, index) => {
      if (style.fillColor == "$encoding") {
        ctx.fillStyle = color.at(index);
      }
      if (style.strokeColor == "$encoding") {
        ctx.strokeStyle = color.at(index);
      }

      let fillAlpha = (style.opacity ?? 1) * (style.fillOpacity ?? 1);
      let strokeAlpha = (style.opacity ?? 1) * (style.strokeOpacity ?? 1);

      if (fillFirst) {
        if (style.fillColor != undefined && fillAlpha > 0) {
          ctx.globalAlpha = fillAlpha;
          ctx.fill();
        }
        if (style.strokeColor != undefined && strokeAlpha > 0) {
          ctx.globalAlpha = strokeAlpha;
          ctx.stroke();
        }
      } else {
        if (style.strokeColor != undefined && strokeAlpha > 0) {
          ctx.globalAlpha = strokeAlpha;
          ctx.stroke();
        }
        if (style.fillColor != undefined && fillAlpha > 0) {
          ctx.globalAlpha = fillAlpha;
          ctx.fill();
        }
      }
    },
  };
}
