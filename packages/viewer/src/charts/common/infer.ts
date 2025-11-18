// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as d3 from "d3";

import { resolveInterpolate, type ChartTheme } from "./theme.js";
import type {
  AxisConfig,
  ConcretePositionScale,
  ConcreteScale,
  GridLine,
  IntermediatePositionScale,
  Label,
  LinearPosition,
  ScaleConfig,
  ScaleType,
  Tick,
} from "./types.js";

let canvas: HTMLCanvasElement | undefined = undefined;

function sharedGraphicsContext() {
  if (canvas == undefined) {
    canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
  }
  return canvas.getContext("2d")!;
}

function measureText(
  text: string,
  options: { fontFamily: string; fontSize: number; maxWidth: number },
): { size: { width: number; height: number }; fontFamily: string; fontSize: number } {
  let ctx = sharedGraphicsContext();
  ctx.font = `${options.fontSize}px ${options.fontFamily}`;
  let metrics = ctx.measureText(text);
  return {
    size: { width: Math.min(options.maxWidth, metrics.width), height: options.fontSize },
    fontFamily: options.fontFamily,
    fontSize: options.fontSize,
  };
}

export function makeScale(
  spec: ScaleConfig,
  axis: AxisConfig | null | undefined,
  dimension: "x" | "y",
  theme: ChartTheme,
): IntermediatePositionScale {
  switch (spec.type) {
    case "band":
      return makeBandScale(spec, axis, dimension, theme);
    default:
      return makeContinuousScale(spec, axis, dimension, theme);
  }
}

function makeBandScale(
  spec: ScaleConfig,
  axis: AxisConfig | null | undefined,
  dimension: "x" | "y",
  theme: ChartTheme,
): IntermediatePositionScale {
  let domain = [...spec.domain, ...(spec.specialValues ?? [])];
  domain = Array.from(new Set(domain));
  if (dimension == "y") {
    domain = domain.reverse();
  }

  let labels: Label[] = [];
  let ticks: Tick[] = [];

  if (axis) {
    let values: string[] = axis.values ?? domain;
    let labelPadding = axis.labelPadding ?? 6;
    let opts = textOptions(axis, theme);
    labels = values.map((v) => {
      if (dimension == "y") {
        return {
          text: v,
          value: v,
          padding: labelPadding,
          level: 0,
          orientation: "horizontal",
          ...measureText(v, opts),
        };
      } else {
        let tm = measureText(v, opts);
        tm.size = { width: tm.size.height, height: tm.size.width };
        return {
          text: v,
          value: v,
          padding: labelPadding,
          level: 0,
          orientation: "vertical",
          ...tm,
        };
      }
    });
    ticks = values.map((x) => ({ value: x, level: 0 }));
  }

  let scale = d3.scaleBand().domain(domain).padding(0.1);
  if (dimension == "y") {
    scale.range([1, 0]);
  }

  let bandwidth = scale.bandwidth();
  let step = scale.step();

  let base: IntermediatePositionScale["base"] = {
    rangeBands: [
      [
        [0, 0],
        [1, 0],
      ],
    ],
    apply: (x) => {
      return [(scale(x) ?? 0) + bandwidth / 2, 0];
    },
    applyBand: (x) => {
      let p = scale(x) ?? 0;
      return [
        [p, 0],
        [p + bandwidth, 0],
      ];
    },
    invert(position, [r0, r1]) {
      let t = (position - r0) / (r1 - r0);
      if (dimension == "y") {
        t = 1 - t;
      }
      let index = Math.floor(t / step);
      return domain[index];
    },
  };

  return {
    labels: labels,
    gridLines: [],
    ticks: ticks,
    base: base,
    concrete: (range) => makeConcrete(base, "band", spec.domain, spec.specialValues ?? [], range),
  };
}

function lpInterpolator(a: LinearPosition, b: LinearPosition): (t: number) => LinearPosition {
  let [a0, a1] = a;
  let [b0, b1] = b;
  return (t) => [a0 + (b0 - a0) * t, a1 + (b1 - a1) * t];
}

function lpResolver(range: [number, number]): (pos: LinearPosition) => number {
  let [r0, r1] = range;
  return ([a, b]) => a * (r1 - r0) + r0 + b;
}

function makeConcrete(
  base: IntermediatePositionScale["base"],
  type: ScaleType,
  domain: any[],
  specialValues: string[],
  range: [number, number],
): ConcretePositionScale {
  let resolver = lpResolver(range);
  return {
    type: type,
    domain: domain,
    specialValues: specialValues,
    range: range,
    rangeBands: base.rangeBands.map(([a, b]) => [resolver(a), resolver(b)]),
    apply: (x) => {
      return resolver(base.apply(x));
    },
    applyBand: (x) => {
      let [b1, b2] = base.applyBand(x);
      let p1 = resolver(b1);
      let p2 = resolver(b2);
      return p1 < p2 ? [p1, p2] : [p2, p1];
    },
    invert: (x, type) => {
      return base.invert(x, range, type);
    },
  };
}

function textOptions(
  axis: AxisConfig | null | undefined,
  theme: ChartTheme,
): { fontFamily: string; fontSize: number; maxWidth: number } {
  return {
    fontFamily: axis?.labelFontFamily ?? theme.labelFontFamily,
    fontSize: axis?.labelFontSize ?? theme.labelFontSize,
    maxWidth: axis?.labelMaxWidth ?? theme.labelMaxWidth,
  };
}

function makeContinuousScale(
  spec: ScaleConfig,
  axis: AxisConfig | null | undefined,
  dimension: "x" | "y",
  theme: ChartTheme,
): IntermediatePositionScale {
  let scale: d3.ScaleContinuousNumeric<number, number>;

  switch (spec.type) {
    case "linear": {
      scale = d3.scaleLinear().domain(spec.domain);
      break;
    }
    case "log": {
      scale = d3.scaleLog().domain(spec.domain);
      break;
    }
    case "symlog": {
      let constant = spec.constant ?? 1;
      scale = d3.scaleSymlog().constant(constant).domain(spec.domain);
      scale.nice = () => scale;
      scale.ticks = (count) => symlogTicks(scale.domain(), constant, count);
      scale.tickFormat = () => d3.format("~s");
      break;
    }
    default: {
      throw new Error("invalid scale type");
    }
  }

  let labels: Label[] = [];
  let gridLines: GridLine[] = [];

  if (axis) {
    let values: number[] = [];
    if (axis.extendScaleToTicks ?? true) {
      if (axis.values) {
        values = axis.values;
        let all = scale.domain().concat(values);
        scale = scale.domain([
          all.reduce((a, b) => Math.min(a, b), all[0]),
          all.reduce((a, b) => Math.max(a, b), all[0]),
        ]);
      } else {
        let count = axis.desiredTickCount ?? 5;
        if (scale.nice) {
          scale = scale.nice(count);
        }
        values = scale.ticks(count);
      }
    } else {
      if (axis.values) {
        values = axis.values;
      } else {
        let count = axis.desiredTickCount ?? 5;
        values = scale.ticks(count);
      }
      let [dmin, dmax] = scale.domain();
      values = values.filter((x) => x >= dmin && x <= dmax);
    }
    let labelPadding = axis.labelPadding ?? 6;
    let tickFormat = scale.tickFormat(axis.values ? axis.values.length : (axis.desiredTickCount ?? 5));
    let valueLevel = (x: number) => {
      if (spec.type == "log" || spec.type == "symlog") {
        return Math.round(Math.log10(Math.abs(x))) == Math.log10(Math.abs(x)) ? 0 : 1;
      } else {
        return 0;
      }
    };
    let opts = textOptions(axis, theme);
    labels = values.map((v) => {
      let text = tickFormat(v);
      return {
        text: text,
        value: v,
        padding: labelPadding,
        level: valueLevel(v),
        orientation: "horizontal",
        ...measureText(text, opts),
      };
    });
    for (let sp of spec.specialValues ?? []) {
      labels.push({
        text: sp,
        value: sp,
        padding: labelPadding,
        level: 0,
        orientation: "horizontal",
        ...measureText(sp, opts),
      });
    }
    gridLines = values.map((x) => ({ value: x, level: valueLevel(x) }));
  }

  if (dimension == "y") {
    scale.range([1, 0]);
  }

  let { rangeBands, map } = resolveSpecialValues(
    Array.from(new Set(spec.specialValues ?? [])),
    dimension == "x" ? "start" : "end",
  );

  let [r0, r1] = rangeBands[0];
  let pos = lpInterpolator(r0, r1);

  let base: IntermediatePositionScale["base"] = {
    rangeBands: rangeBands,
    apply: (x) => {
      let sp = map.get(x);
      if (sp != undefined) {
        return [(sp[0][0] + sp[1][0]) / 2, (sp[0][1] + sp[1][1]) / 2];
      }
      if (x == undefined) {
        return [NaN, NaN];
      }
      let t = x instanceof Array && x.length >= 2 ? (scale(x[0]) + scale(x[1])) / 2 : scale(x);
      return pos(t);
    },
    applyBand: (x) => {
      let sp = map.get(x);
      if (sp != null) {
        return sp;
      }
      if (x == null) {
        return [
          [NaN, NaN],
          [NaN, NaN],
        ];
      }
      if (x instanceof Array && x.length >= 2) {
        return [pos(scale(x[0])), pos(scale(x[1]))];
      } else {
        let v = pos(scale(x));
        return [v, v];
      }
    },
    invert(position, range, type) {
      let resolver = lpResolver(range);
      if (type != "number") {
        for (let [value, [r0, r1]] of map.entries()) {
          let b0 = resolver(r0);
          let b1 = resolver(r1);
          if (position >= Math.min(b0, b1) && position <= Math.max(b0, b1)) {
            return value;
          }
        }
      }
      if (type != "string") {
        let b0 = resolver(r0);
        let b1 = resolver(r1);
        return scale.invert((position - b0) / (b1 - b0));
      }
    },
  };

  return {
    labels: labels,
    gridLines: gridLines,
    ticks: gridLines,
    base: base,
    concrete: (range) => makeConcrete(base, spec.type, scale.domain(), spec.specialValues ?? [], range),
  };
}

function resolveSpecialValues(
  specialValues: string[],
  placement: "start" | "end",
): {
  rangeBands: [LinearPosition, LinearPosition][];
  map: Map<string, [LinearPosition, LinearPosition]>;
} {
  let baseBand: [LinearPosition, LinearPosition] = [
    [0, 0],
    [1, 0],
  ];
  let specialBand: [LinearPosition, LinearPosition] | undefined;
  let map = new Map<string, [LinearPosition, LinearPosition]>();

  if (specialValues.length > 0) {
    let gap = 8;
    let bandSize = 20;
    let len = specialValues.length * bandSize;

    if (placement == "start") {
      baseBand = [
        [0, len + gap],
        [1, 0],
      ];

      specialBand = [
        [0, 0],
        [0, len],
      ];
    } else {
      baseBand = [
        [0, 0],
        [1, -len - gap],
      ];

      specialBand = [
        [1, -len],
        [1, 0],
      ];
    }

    let interp = lpInterpolator(specialBand[0], specialBand[1]);

    for (let i = 0; i < specialValues.length; i++) {
      let t0 = (i + 0.1) / specialValues.length;
      let t1 = (i + 0.9) / specialValues.length;
      map.set(specialValues[i], [interp(t0), interp(t1)]);
    }
  }

  return {
    rangeBands: specialBand != undefined ? [baseBand, specialBand] : [baseBand],
    map: map,
  };
}

export function resolveLabelOverlap<T>(
  labels: T[],
  span: (label: T) => { center: number; length: number; priority: number },
  options: { gap?: number } = {},
): T[] {
  let gap = options.gap ?? 0;
  let spans = labels.map((d, i) => ({ ...span(d), index: i })).sort((a, b) => a.priority - b.priority);
  let placed = labels.map((_) => false);
  let overlaps = (a: { center: number; length: number }, b: { center: number; length: number }) => {
    return Math.abs(a.center - b.center) < gap + a.length / 2 + b.length / 2;
  };
  for (let i = 0; i < spans.length; i++) {
    let hasOverlap = false;
    for (let j = 0; j < i; j++) {
      if (placed[spans[j].index] && overlaps(spans[i], spans[j])) {
        hasOverlap = true;
        break;
      }
    }
    placed[spans[i].index] = !hasOverlap;
  }
  return labels.filter((_, i) => placed[i]);
}

function symlogTicks(domain: number[], constant: number, count?: number | undefined): number[] {
  count = count ?? 5;

  let min = domain[0];
  let max = domain[1];

  if ((min > 0 && max > 0 && min / max > 0.5) || (min < 0 && max < 0 && max / min > 0.5)) {
    return d3.scaleLinear().domain([min, max]).ticks(count);
  }

  let start = constant * 2;
  let threshold = constant * 5;
  if (min < -threshold && max > threshold) {
    count = Math.ceil(count / 2);
  }
  return [
    ...(min < -threshold
      ? d3
          .scaleLog()
          .domain([start, -min])
          .ticks(count)
          .map((x) => -x)
      : []),
    0,
    ...(max > threshold ? d3.scaleLog().domain([start, max]).ticks(count) : []),
  ].filter((x) => x >= min && x <= max);
}

export function inferColorScale(spec: ScaleConfig, theme: ChartTheme): ConcreteScale<string> {
  switch (spec.type) {
    case "band": {
      let categoryColors = (spec.range as any) ?? theme.categoryColors;

      let map = new Map<string, string>();
      if (typeof categoryColors == "function") {
        let colors = categoryColors(spec.domain.length);
        for (let i = 0; i < spec.domain.length; i++) {
          map.set(spec.domain[i], colors[i]);
        }
      } else {
        for (let i = 0; i < spec.domain.length; i++) {
          map.set(spec.domain[i], categoryColors[i % categoryColors.length]);
        }
      }

      return {
        type: spec.type,
        domain: spec.domain,
        specialValues: spec.specialValues ?? [],
        apply: (value: any) => map.get(value) ?? theme.markColorGray,
      };
    }
    default:
      let intermediate = makeScale({ ...spec, specialValues: [] }, null, "x", theme);
      let concrete = intermediate.concrete([0, 1]);
      let interp = resolveInterpolate((spec.range as any) ?? theme.interpolate);
      let specialValuesSet = new Set(spec.specialValues ?? []);
      return {
        type: spec.type,
        domain: spec.domain,
        specialValues: spec.specialValues ?? [],
        apply: (value: any) => {
          if (specialValuesSet.has(value)) {
            return theme.markColorGray;
          }
          return interp(concrete.apply(value));
        },
      };
  }
}

export function inferSizeScale(spec: ScaleConfig, theme: ChartTheme): ConcreteScale<number> {
  let sizeRange: [number, number] = [0, 1000];

  if (spec.range instanceof Array && spec.range.length == 2) {
    let [r0, r1] = spec.range;
    if (typeof r0 == "number" && typeof r1 == "number") {
      sizeRange = [r0, r1];
    }
  }

  switch (spec.type) {
    case "band": {
      let map = new Map<string, number>();
      for (let i = 0; i < spec.domain.length; i++) {
        map.set(spec.domain[i], ((i + 1) / spec.domain.length) * (sizeRange[1] - sizeRange[0]) + sizeRange[0]);
      }
      return {
        type: spec.type,
        domain: spec.domain,
        specialValues: spec.specialValues ?? [],
        apply: (value: any) => map.get(value) ?? 1,
      };
    }
    default:
      let intermediate = makeScale({ ...spec, specialValues: [] }, null, "x", theme);
      let concrete = intermediate.concrete(sizeRange);
      return {
        type: spec.type,
        domain: spec.domain,
        specialValues: spec.specialValues ?? [],
        apply: (value: any) => {
          return concrete.apply(value);
        },
      };
  }
}
