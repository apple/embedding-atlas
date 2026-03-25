// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as d3 from "d3";

import { inferNumberFormatter } from "./formatter.js";

export interface ContinuousTicksOptions {
  /** Scale type */
  type: "linear" | "log" | "symlog" | "time";

  /** Symlog constant */
  constant?: number;

  /** Data domain min */
  dataMin: number;

  /** Data domain max */
  dataMax: number;

  /** Whether we will extend the scale domain to min/max of ticks (if true, may output ticks outside the data domain) */
  extendDomainToTicks?: boolean;

  /** Desired number of ticks */
  desiredCount?: number;

  /** Pre-specified tick values, if specified, use values directly as ticks, but still infer a formatter */
  values?: number[];

  /**
   * For "time" scale type, are the values timezone-aware (default false). If true, treat values as true UTC timestamps (ms since epoch), and use the current timezone to display them.
   * If false, treat values as timestamps with unknown timezone, display them in UTC but do not include any timezone information.
   */
  hasTimezone?: boolean;
}

export interface ContinuousTicksResult {
  /** Extended domain min */
  domainMin: number;

  /** Extended domain max */
  domainMax: number;

  /** Tick values */
  values: number[];

  /** A function to format the ticks */
  format: (value: number) => string;

  /**
   * A function to return the tick level (0 is base level, 1 is lower level, etc.).
   * Currently level 0 has gridlines and labels, level 1 has gridlines only
   */
  level: (value: number) => number;
}

export function continuousTicks(options: ContinuousTicksOptions): ContinuousTicksResult {
  // Special treatment for time
  if (options.type == "time") {
    return timeTicks(options);
  }

  let desiredCount = options.desiredCount ?? 5;

  let scale: d3.ScaleContinuousNumeric<number, number>;

  // Numerical types
  switch (options.type) {
    case "linear": {
      scale = d3.scaleLinear().domain([options.dataMin, options.dataMax]);
      break;
    }
    case "log": {
      scale = d3.scaleLog().domain([options.dataMin, options.dataMax]);
      break;
    }
    case "symlog": {
      let constant = options.constant ?? 1;
      scale = d3.scaleSymlog().constant(constant).domain([options.dataMin, options.dataMax]);
      scale.nice = () => scale;
      scale.ticks = (count) => symlogTicks(scale.domain(), constant, count);
      scale.tickFormat = () => d3.format("~s");
      break;
    }
    default: {
      throw new Error("invalid scale type");
    }
  }

  let values: number[] = [];
  if (options.extendDomainToTicks ?? true) {
    if (options.values) {
      values = options.values;
      let all = scale.domain().concat(values);
      scale = scale.domain([
        all.reduce((a, b) => Math.min(a, b), all[0]),
        all.reduce((a, b) => Math.max(a, b), all[0]),
      ]);
    } else {
      if (scale.nice) {
        scale = scale.nice(desiredCount);
      }
      values = scale.ticks(desiredCount);
    }
  } else {
    values = options.values ?? scale.ticks(desiredCount);
    let [dmin, dmax] = scale.domain();
    values = values.filter((x) => x >= dmin && x <= dmax);
  }

  // let format = scale.tickFormat(options.values ? options.values.length : desiredCount);
  let format = inferNumberFormatter(values);
  let level = (x: number) => {
    if (options.type == "log" || options.type == "symlog") {
      return Math.round(Math.log10(Math.abs(x))) == Math.log10(Math.abs(x)) ? 0 : 1;
    } else {
      return 0;
    }
  };
  return {
    domainMin: scale.domain()[0],
    domainMax: scale.domain()[1],
    values,
    format,
    level,
  };
}

function timeTicks(options: ContinuousTicksOptions): ContinuousTicksResult {
  let desiredCount = options.desiredCount ?? 5;

  let scale = (options.hasTimezone ? d3.scaleTime() : d3.scaleUtc()).domain([options.dataMin, options.dataMax]);

  let values: number[] = [];

  if (options.extendDomainToTicks ?? true) {
    if (options.values) {
      values = options.values;
      let all = scale
        .domain()
        .map((x) => x.getTime())
        .concat(values);
      scale = scale.domain([
        all.reduce((a, b) => Math.min(a, b), all[0]),
        all.reduce((a, b) => Math.max(a, b), all[0]),
      ]);
    } else {
      if (scale.nice) {
        scale = scale.nice(desiredCount);
      }
      values = scale.ticks(desiredCount).map((x) => x.getTime());
    }
  } else {
    values = options.values ?? scale.ticks(desiredCount).map((x) => x.getTime());
    let [dmin, dmax] = scale.domain().map((x) => x.getTime());
    values = values.filter((x) => x >= dmin && x <= dmax);
  }

  let timeFormat = scale.tickFormat(options.values ? options.values.length : desiredCount);
  let format = (v: number) => timeFormat(new Date(v));

  return {
    domainMin: scale.domain()[0].getTime(),
    domainMax: scale.domain()[1].getTime(),
    values,
    format,
    level: () => 0,
  };
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
