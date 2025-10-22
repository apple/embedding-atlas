// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { format } from "d3-format";
import { scaleBand, scaleLinear, scaleLog, scaleSymlog, type ScaleBand, type ScaleContinuousNumeric } from "d3-scale";

import { defaultCategoryColors, defaultOrdinalColors } from "./colors.js";
import type {
  AxisSpec,
  ConcreteScale,
  Extents,
  GridLine,
  IntermedateScale,
  Label,
  ScaleSpec,
  ScaleType,
  Tick,
} from "./types.js";

export const FONT_FAMILY = "system-ui";
export const FONT_SIZE = 11;
export const MAX_LABEL_LENGTH = 80;

let canvas: HTMLCanvasElement | undefined = undefined;

function get_context() {
  if (canvas == undefined) {
    canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
  }
  return canvas.getContext("2d")!;
}

function measureText(text: string): { width: number; height: number } {
  let ctx = get_context();
  ctx.font = `${FONT_SIZE}px ${FONT_FAMILY}`;
  let metrics = ctx.measureText(text);
  return {
    width: Math.min(MAX_LABEL_LENGTH, metrics.width),
    height: FONT_SIZE,
  };
}

function axisExtents(labels: Label[], dimension: "x" | "y", padding: number = 0): Extents {
  let maxWidth = 0;
  let maxHeight = 0;
  for (let label of labels) {
    let { width, height } = label.size;
    maxWidth = Math.max(maxWidth, width);
    maxHeight = Math.max(maxHeight, height);
  }
  switch (dimension) {
    case "x": {
      return { left: maxWidth / 2, right: maxWidth / 2, top: 0, bottom: maxHeight + padding };
    }
    case "y": {
      return { left: maxWidth + padding, right: 0, top: maxHeight / 2, bottom: maxHeight / 2 };
    }
  }
}

export function makeScale(spec: ScaleSpec, axis: AxisSpec | null | undefined, dimension: "x" | "y"): IntermedateScale {
  switch (spec.type) {
    case "band":
      return makeBandScale(spec, axis, dimension);
    default:
      return makeContinuousScale(spec, axis, dimension);
  }
}

function makeBandScale(spec: ScaleSpec, axis: AxisSpec | null | undefined, dimension: "x" | "y"): IntermedateScale {
  let domain = [...spec.domain, ...(spec.specialValues ?? [])];
  domain = Array.from(new Set(domain));
  if (dimension == "y") {
    domain = domain.reverse();
  }
  let scale = scaleBand().domain(domain).padding(0.1);

  let labels: Label[] = [];
  let ticks: Tick[] = [];
  let extents: Extents = { left: 0, right: 0, top: 0, bottom: 0 };

  if (axis) {
    let values: string[] = axis.values ?? scale.domain();
    let labelPadding = axis.labelPadding ?? 6;
    labels = values.map((v) => {
      let { width, height } = measureText(v);
      if (dimension == "y") {
        return {
          text: v,
          value: v,
          padding: labelPadding,
          level: 0,
          size: { width, height },
          orientation: "horizontal",
        };
      } else {
        return {
          text: v,
          value: v,
          padding: labelPadding,
          level: 0,
          size: { width: height, height: width },
          orientation: "vertical",
        };
      }
    });
    ticks = values.map((x) => {
      return { value: x, level: 0 };
    });
    extents = unionExtents([extents, axisExtents(labels, dimension, labelPadding)]);
  }

  return {
    extents: extents,
    labels: labels,
    gridLines: [],
    ticks: ticks,
    concrete: (range) => makeConcreteBand(scale, spec.domain, spec.specialValues ?? [], range),
  };
}

function makeContinuousScale(
  spec: ScaleSpec,
  axis: AxisSpec | null | undefined,
  dimension: "x" | "y",
): IntermedateScale {
  let scale: ScaleContinuousNumeric<number, number>;

  switch (spec.type) {
    case "linear": {
      scale = scaleLinear().domain(spec.domain);
      break;
    }
    case "log": {
      scale = scaleLog().domain(spec.domain);
      break;
    }
    case "symlog": {
      let constant = spec.constant ?? 1;
      scale = scaleSymlog().constant(constant).domain(spec.domain);
      scale.nice = () => scale;
      scale.ticks = (count) => symlogTicks(scale.domain(), constant, count);
      scale.tickFormat = () => format("~s");
      break;
    }
    default: {
      throw new Error("invalid scale type");
    }
  }

  let labels: Label[] = [];
  let gridLines: GridLine[] = [];
  let extents: Extents = { left: 0, right: 0, top: 0, bottom: 0 };

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
    labels = values.map((v) => {
      let text = tickFormat(v);
      return {
        text: text,
        value: v,
        padding: labelPadding,
        level: valueLevel(v),
        size: measureText(text),
        orientation: "horizontal",
      };
    });
    for (let sp of spec.specialValues ?? []) {
      labels.push({
        text: sp,
        value: sp,
        padding: labelPadding,
        level: 0,
        size: measureText(sp),
        orientation: "horizontal",
      });
    }
    gridLines = values.map((x) => {
      return { value: x, level: valueLevel(x) };
    });
    extents = unionExtents([extents, axisExtents(labels, dimension, labelPadding)]);
  }
  return {
    extents: extents,
    labels: labels,
    gridLines: gridLines,
    ticks: gridLines,
    concrete: (range) => makeConcreteContinuous(scale, spec.specialValues ?? [], range),
  };
}

export function unionExtents(values: Extents[]): Extents {
  let r: Extents = { left: 0, right: 0, top: 0, bottom: 0 };
  for (let v of values) {
    r.left = Math.max(r.left, v.left);
    r.right = Math.max(r.right, v.right);
    r.top = Math.max(r.top, v.top);
    r.bottom = Math.max(r.bottom, v.bottom);
  }
  return r;
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

function makeConcreteContinuous(
  partial: ScaleContinuousNumeric<number, number>,
  specialValues: string[],
  range: [number, number],
): ConcreteScale {
  specialValues = Array.from(new Set(specialValues));
  let rangeStart = range[0];
  let rangeEnd = range[1];
  let map = new Map<string, [number, number]>();
  let specialValuesRange: [number, number] | undefined = undefined;
  if (specialValues.length > 0) {
    let distance = 22;
    let specialValuesGap = 8;
    let bandSize = 20;
    let specialLength = specialValues.length * bandSize;
    let gap = 2;
    let start = rangeStart;
    if (rangeStart < rangeEnd) {
      rangeStart = rangeStart + specialLength + distance;
      specialValuesRange = [start, start + specialLength + distance - specialValuesGap];
    } else {
      rangeStart = rangeStart - specialLength - distance;
      start = rangeStart + distance;
      specialValuesRange = [start - distance + specialValuesGap, start + specialLength];
    }
    for (let i = 0; i < specialValues.length; i++) {
      map.set(specialValues[i], [start + i * bandSize + gap, start + i * bandSize + bandSize - gap]);
    }
  }
  let r = partial.copy().range([rangeStart, rangeEnd]);
  return {
    domain: partial.domain(),
    specialValues: specialValues ?? [],
    range: [rangeStart, rangeEnd],
    rangeBands: [[rangeStart, rangeEnd], ...(specialValuesRange ? [specialValuesRange] : [])],
    apply: (x) => {
      let sp = map.get(x);
      if (sp != null) {
        return (sp[0] + sp[1]) / 2;
      } else {
        return r(x);
      }
    },
    applyBand: (x) => {
      let sp = map.get(x);
      if (sp != null) {
        return sp;
      } else {
        if (typeof x != "number") {
          return [r(x[0]), r(x[1])];
        } else {
          let v = r(x);
          return [v, v];
        }
      }
    },
    invert: (x, type) => {
      if (type != "number") {
        for (let [value, range] of map.entries()) {
          if (x >= Math.min(...range) && x <= Math.max(...range)) {
            return value;
          }
        }
      }
      return r.invert(x);
    },
  };
}

function makeConcreteBand(
  partial: ScaleBand<string>,
  domain: string[],
  specialValues: string[],
  range: [number, number],
): ConcreteScale {
  let rangeStart = range[0];
  let rangeEnd = range[1];
  let scale = partial.copy().range(range);
  let bandWidth = scale.bandwidth();
  let step = scale.step();
  return {
    domain: domain,
    specialValues: specialValues,
    range: [rangeStart, rangeEnd],
    rangeBands: [[rangeStart, rangeEnd]],
    apply: (x) => {
      return (scale(x) ?? 0) + bandWidth / 2;
    },
    applyBand: (x) => {
      let p = scale(x) ?? 0;
      return [p, p + bandWidth];
    },
    invert: (x) => {
      let t = (x - rangeStart) / (rangeEnd - rangeStart);
      let index = Math.floor((t * Math.abs(rangeEnd - rangeStart)) / step);
      return partial.domain()[index];
    },
  };
}

function symlogTicks(domain: number[], constant: number, count?: number | undefined): number[] {
  count = count ?? 5;

  let min = domain[0];
  let max = domain[1];

  if ((min > 0 && max > 0 && min / max > 0.5) || (min < 0 && max < 0 && max / min > 0.5)) {
    return scaleLinear().domain([min, max]).ticks(count);
  }

  let start = constant * 2;
  let threshold = constant * 5;
  if (min < -threshold && max > threshold) {
    count = Math.ceil(count / 2);
  }
  return [
    ...(min < -threshold
      ? scaleLog()
          .domain([start, -min])
          .ticks(count)
          .map((x) => -x)
      : []),
    0,
    ...(max > threshold ? scaleLog().domain([start, max]).ticks(count) : []),
  ].filter((x) => x >= min && x <= max);
}

export function inferPositionScale(
  stats: {
    min: number;
    minPositive: number;
    max: number;
    median: number;
    count: number;
  },
  type: ScaleType | null | undefined,
): ScaleSpec {
  let { min, max, median, count, minPositive } = stats;
  let scale: ScaleSpec = {
    type: "linear",
    domain: [min, max],
  };
  if (type == null) {
    if (count >= 100 && min >= 0 && median < max * 0.05) {
      scale.type = min > 0 ? "log" : "symlog";
    }
  } else {
    scale.type = type;
  }
  if (scale.type == "log") {
    scale.domain[0] = minPositive;
  }
  return scale;
}

export function inferColorScale(
  domain: (string | [number, number])[],
  options: {
    fade?: string[];
    ordinal?: boolean;
  } = {},
): {
  domain: (string | [number, number])[];
  apply: (value: string | [number, number]) => string;
} {
  let fade = options.fade ?? [];
  let ordinal = options.ordinal ?? false;

  let colorfulKeys: string[] = [];
  let fadeKeys: string[] = [];
  let value2key = (x: any) => JSON.stringify(x);
  for (let item of domain) {
    let key = value2key(item);
    if (typeof item == "string" && fade.indexOf(item) >= 0) {
      fadeKeys.push(key);
    } else {
      colorfulKeys.push(key);
    }
  }
  let colors = ordinal ? defaultOrdinalColors(colorfulKeys.length) : defaultCategoryColors(colorfulKeys.length);
  let map = new Map<string | [number, number], string>(colorfulKeys.map((k, i) => [k, colors[i]]));
  return {
    domain: domain,
    apply: (value) => {
      let key = value2key(value);
      return map.get(key) ?? "#888888";
    },
  };
}
