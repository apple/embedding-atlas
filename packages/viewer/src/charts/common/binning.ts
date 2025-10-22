// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import { scaleLinear, scaleLog } from "d3-scale";
import type { ScaleType } from "./types.js";

export interface Binning {
  scale: ScaleClass & { domain: [number, number]; constant?: number };
  binStart: number;
  binSize: number;
}

export interface ScaleClass {
  type: "linear" | "log" | "symlog";
  expr(x: SQL.ExprNode, constant: number): SQL.ExprNode;
  forward(x: number, constant: number): number;
  reverse(x: number, constant: number): number;
}

const scaleTypes: Record<string, ScaleClass> = {
  linear: { type: "linear", expr: (x) => x, forward: (x) => x, reverse: (x) => x },
  log: {
    type: "log",
    expr: (x) => SQL.cond(SQL.gt(x, 0), SQL.log(x), SQL.literal("nan")),
    forward: (x) => Math.log10(x),
    reverse: (x) => Math.pow(10, x),
  },
  symlog: {
    type: "symlog",
    expr: (x: SQL.ExprNode, constant: number) =>
      SQL.mul(SQL.sign(x), SQL.ln(SQL.add(1, SQL.abs(SQL.div(x, constant))))),
    forward: (x: number, constant: number) => Math.sign(x) * Math.log1p(Math.abs(x) / constant),
    reverse: (x: number, constant: number) => Math.sign(x) * Math.expm1(Math.abs(x)) * constant,
  },
};

function roundToNearest(value: number, array: number[]): number {
  let minV = value;
  let minD = Infinity;
  for (let v of array) {
    let d = Math.abs(value - v);
    if (d < minD) {
      minD = d;
      minV = v;
    }
  }
  return minV;
}

export function inferBinning(
  stats: { min: number; minPositive: number; max: number; median: number; count: number },
  options: {
    scale?: ScaleType | null;
    desiredCount?: number;
  } = {},
): Binning {
  let { min, max, median, count } = stats;

  // Infer scale type
  let scaleType = options.scale;
  if (scaleType == "band") {
    scaleType = null;
  }
  if (scaleType == null) {
    scaleType = "linear";
    if (count >= 100 && min >= 0 && median < max * 0.05) {
      scaleType = min > 0 ? "log" : "symlog";
    }
  }

  if (min <= 0 && scaleType == "log") {
    if (max <= 0) {
      // Log scale with no positive value, we'll just do a default domain of [1, 10]
      min = 1;
      max = 10;
    } else {
      min = Math.min(stats.minPositive, max / 10);
    }
  }

  let desiredCount = options.desiredCount ?? 5;

  switch (scaleType) {
    case "linear": {
      let s = scaleLinear().domain([min, max]).nice(desiredCount);
      let ticks = s.ticks(desiredCount);
      return {
        scale: { ...scaleTypes.linear, domain: s.domain() as any },
        binStart: s.domain()[0],
        binSize: ticks[1] - ticks[0],
      };
    }
    case "log": {
      let s = scaleLog().domain([min, max]).nice();
      let binStart = Math.log10(s.domain()[0]);
      let binSize = (Math.log10(s.domain()[1]) - binStart) / desiredCount;
      binSize = roundToNearest(binSize, [0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]);
      return {
        scale: { ...scaleTypes.log, domain: s.domain() as any },
        binStart: binStart,
        binSize: binSize,
      };
    }
    case "symlog": {
      let absMax = Math.max(Math.abs(min), Math.abs(max));
      let constant = absMax >= 100 ? 1 : absMax > 0 ? absMax / 1e5 : 1;
      let sMin = scaleTypes.symlog.forward(min, constant);
      let sMax = scaleTypes.symlog.forward(max, constant);
      return {
        scale: { ...scaleTypes.symlog, domain: [min, max], constant: constant },
        binStart: sMin,
        binSize: (sMax - sMin) / desiredCount,
      };
    }
    default:
      throw new Error("invalid scale type");
  }
}
