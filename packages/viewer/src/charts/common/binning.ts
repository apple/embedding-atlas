// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import * as d3 from "d3";

import type { ScaleType } from "./types.js";

export interface Binning {
  scale: { type: "linear" | "log" | "symlog" | "time"; constant?: number; domain: [number, number] };

  binIndexExpr: (input: SQL.ExprNode) => SQL.ExprNode;
  nullPredicateExpr: (input: SQL.ExprNode) => SQL.ExprNode;

  binIndex: (value: number) => number;
  rangeForIndex: (index: number) => [number, number];
}

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
      let s = d3.scaleLinear().domain([min, max]).nice(desiredCount);
      let ticks = s.ticks(desiredCount);
      let binStart = s.domain()[0];
      let binSize = ticks[1] - ticks[0];
      return {
        scale: { type: "linear", domain: s.domain() as any },
        binIndexExpr: (x) =>
          SQL.cond(
            SQL.isFinite(x), // Ensure finite value
            SQL.floor(SQL.mul(SQL.sub(x, binStart), 1 / binSize)),
            SQL.literal(null),
          ),
        nullPredicateExpr: (x) => SQL.or(SQL.isNull(x), SQL.not(SQL.isFinite(x))),
        binIndex: (x) => Math.floor((x - binStart) / binSize),
        rangeForIndex: (index) => [binStart + binSize * index, binStart + binSize * (index + 1)],
      };
    }
    case "log": {
      let s = d3.scaleLog().domain([min, max]).nice();
      let binStart = Math.log10(s.domain()[0]);
      let binSize = (Math.log10(s.domain()[1]) - binStart) / desiredCount;
      binSize = roundToNearest(binSize, [0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]);
      return {
        scale: { type: "log", domain: s.domain() as any },
        binIndexExpr: (x) =>
          SQL.cond(
            SQL.and(SQL.isFinite(x), SQL.gt(x, 0)), // Ensure positive finite value
            SQL.floor(SQL.mul(SQL.sub(SQL.log(x), binStart), 1 / binSize)),
            SQL.literal(null),
          ),
        nullPredicateExpr: (x) => SQL.or(SQL.isNull(x), SQL.not(SQL.isFinite(x)), SQL.not(SQL.gt(x, 0))),
        binIndex: (x) => Math.floor((Math.log10(x) - binStart) / binSize),
        rangeForIndex: (index) => [
          Math.pow(10, binStart + binSize * index),
          Math.pow(10, binStart + binSize * (index + 1)),
        ],
      };
    }
    case "symlog": {
      let absMax = Math.max(Math.abs(min), Math.abs(max));
      let constant = absMax >= 100 ? 1 : absMax > 0 ? absMax / 1e5 : 1;
      let forward = (x: number) => Math.sign(x) * Math.log1p(Math.abs(x) / constant);
      let reverse = (x: number) => Math.sign(x) * Math.expm1(Math.abs(x)) * constant;
      let sMin = forward(min);
      let sMax = forward(max);
      let binStart = sMin;
      let binSize = (sMax - sMin) / desiredCount;
      return {
        scale: { type: "symlog", constant: constant, domain: [min, max] },
        binIndexExpr: (x) =>
          SQL.cond(
            SQL.isFinite(x), // Ensure finite value
            SQL.floor(
              SQL.mul(
                SQL.sub(SQL.mul(SQL.sign(x), SQL.ln(SQL.add(1, SQL.abs(SQL.div(x, constant))))), binStart),
                1 / binSize,
              ),
            ),
            SQL.literal(null),
          ),
        nullPredicateExpr: (x) => SQL.or(SQL.isNull(x), SQL.not(SQL.isFinite(x))),
        binIndex: (x) => Math.floor((forward(x) - binStart) / binSize),
        rangeForIndex: (index) => [reverse(binStart + binSize * index), reverse(binStart + binSize * (index + 1))],
      };
    }
    default:
      throw new Error("invalid scale type");
  }
}

type TimeStepType = "fixed" | "day" | "month" | "year";

interface TimeStep {
  type: TimeStepType;
  /** For 'fixed': exact step in ms. For 'day': number of days. For 'month': number of months. For 'year': number of years. */
  value: number;
  /** Approximate number of milliseconds for this step */
  approxMs: number;
}

// Time constants in milliseconds
const MS_SECOND = 1000;
const MS_MINUTE = 60 * MS_SECOND;
const MS_HOUR = 60 * MS_MINUTE;
const MS_DAY = 24 * MS_HOUR;
const MS_MONTH_APPROX = 30.44 * MS_DAY;
const MS_YEAR_APPROX = 365.25 * MS_DAY;

/** Candidate time steps: [approxMs, type, value] */
const TIME_STEP_CANDIDATES: TimeStep[] = [
  ...[1, 5, 10, 15, 30].map((k): TimeStep => ({ approxMs: k * MS_SECOND, type: "fixed", value: k * MS_SECOND })),
  ...[1, 5, 10, 15, 30].map((k): TimeStep => ({ approxMs: k * MS_MINUTE, type: "fixed", value: k * MS_MINUTE })),
  ...[1, 3, 6, 12].map((k): TimeStep => ({ approxMs: k * MS_HOUR, type: "fixed", value: k * MS_HOUR })),
  ...[1, 2, 7].map((k): TimeStep => ({ approxMs: k * MS_DAY, type: "day", value: k })),
  ...[1, 2, 3, 6].map((k): TimeStep => ({ approxMs: k * MS_MONTH_APPROX, type: "month", value: k })),
  ...[1, 2, 5, 10, 25, 50, 100].map((k): TimeStep => ({ approxMs: k * MS_YEAR_APPROX, type: "year", value: k })),
];

/** Pick a nice time step that divides the given duration into approximately desiredCount bins. */
function inferTimeStep(durationMs: number, desiredCount: number): TimeStep {
  let targetStepMs = durationMs / Math.max(desiredCount, 1);
  let logTarget = Math.log(Math.max(targetStepMs, 1));

  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < TIME_STEP_CANDIDATES.length; i++) {
    let dist = Math.abs(Math.log(TIME_STEP_CANDIDATES[i].approxMs) - logTarget);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }

  let { type, value, approxMs } = TIME_STEP_CANDIDATES[bestIdx];
  return { type, value, approxMs };
}

function dateHelpers(hasTimezone: boolean) {
  // When hasTimezone is true, shift epoch_ms by the current timezone offset so that
  // UTC date functions (getUTCFullYear, etc.) return local calendar values.
  // This avoids relying on timezone name strings that DuckDB may not support.
  // getTimezoneOffset() returns minutes, positive for west of UTC.
  // For UTC+8 it returns -480, so tzOffsetMs = +28800000.
  let tzOffsetMs = hasTimezone ? -new Date().getTimezoneOffset() * 60_000 : 0;

  // Shift ms to "local-adjusted UTC" so UTC extraction gives local calendar values
  let adjusted = (ms: number) => new Date(ms + tzOffsetMs);

  return {
    tzOffsetMs,
    yearOf: (ms: number) => adjusted(ms).getUTCFullYear(),
    monthOf: (ms: number) => adjusted(ms).getUTCMonth(), // 0-based
    // Construct epoch_ms for midnight of the given calendar date in the target timezone
    makeDate: (y: number, m: number, d: number = 1) => Date.UTC(y, m, d) - tzOffsetMs,
  };
}

export function inferTimeBinning(
  stats: { min: number; max: number; count: number },
  options: {
    desiredCount?: number;
    hasTimezone?: boolean;
  } = {},
): Binning {
  let { min, max } = stats;
  let desiredCount = options.desiredCount ?? 5;
  let hasTimezone = options.hasTimezone ?? false;

  let { type: stepType, value: stepValue } = inferTimeStep(max - min, desiredCount);

  // Date helpers that respect timezone setting
  let { tzOffsetMs, yearOf, monthOf, makeDate } = dateHelpers(hasTimezone);

  // SQL helper: convert epoch_ms to a TIMESTAMP, offset-adjusted so that
  // UTC date extraction (YEAR, MONTH, CAST AS DATE) yields local calendar values.
  let toTimestampSQL = (x: SQL.ExprNode) => {
    if (tzOffsetMs !== 0) {
      return SQL.sql`epoch_ms(CAST(${x} AS BIGINT) + ${SQL.literal(tzOffsetMs)})`;
    }
    return SQL.sql`epoch_ms(CAST(${x} AS BIGINT))`;
  };

  let nullPredicateExpr: Binning["nullPredicateExpr"] = (x) => SQL.or(SQL.isNull(x), SQL.not(SQL.isFinite(x)));

  switch (stepType) {
    case "fixed": {
      let stepMs = stepValue;
      let origin = Math.floor(min / stepMs) * stepMs;
      let domainEnd = Math.ceil(max / stepMs) * stepMs;
      return {
        scale: { type: "time", domain: [origin, domainEnd] },
        binIndexExpr: (x) =>
          SQL.cond(SQL.isFinite(x), SQL.floor(SQL.mul(SQL.sub(x, origin), 1 / stepMs)), SQL.literal(null)),
        nullPredicateExpr,
        binIndex: (x) => Math.floor((x - origin) / stepMs),
        rangeForIndex: (i) => [origin + stepMs * i, origin + stepMs * (i + 1)],
      };
    }
    case "day": {
      let stepDays = stepValue;

      // Day number from epoch_ms: shift by timezone offset, then divide by MS_DAY.
      let msToDayNum = (ms: number) => Math.floor((ms + tzOffsetMs) / MS_DAY);
      // Inverse: day number back to epoch_ms (midnight in target timezone)
      let dayNumToMs = (dayNum: number) => dayNum * MS_DAY - tzOffsetMs;

      let originDayNum = Math.floor(msToDayNum(min) / stepDays) * stepDays;
      let maxDayNum = msToDayNum(max);
      let endDayNum = (Math.floor((maxDayNum - originDayNum) / stepDays) + 1) * stepDays + originDayNum;

      return {
        scale: { type: "time", domain: [dayNumToMs(originDayNum), dayNumToMs(endDayNum)] },
        binIndexExpr: (x) => {
          let dayNum = SQL.sql`FLOOR((CAST(${x} AS BIGINT) + ${SQL.literal(tzOffsetMs)}) / ${SQL.literal(MS_DAY)})`;
          return SQL.cond(
            SQL.isFinite(x),
            SQL.sql`CAST(FLOOR((${dayNum} - ${SQL.literal(originDayNum)}) / ${SQL.literal(stepDays)}) AS INTEGER)`,
            SQL.literal(null),
          );
        },
        nullPredicateExpr,
        binIndex: (x) => Math.floor((msToDayNum(x) - originDayNum) / stepDays),
        rangeForIndex: (i) => [dayNumToMs(originDayNum + stepDays * i), dayNumToMs(originDayNum + stepDays * (i + 1))],
      };
    }
    case "month": {
      let stepMonths = stepValue;
      // yearMonth = year * 12 + month (0-based month)
      let minYM = yearOf(min) * 12 + monthOf(min);
      let originYM = Math.floor(minYM / stepMonths) * stepMonths;
      let maxYM = yearOf(max) * 12 + monthOf(max);
      let endYM = (Math.floor((maxYM - originYM) / stepMonths) + 1) * stepMonths + originYM;

      let ymToMs = (ym: number) => makeDate(Math.floor(ym / 12), ym - Math.floor(ym / 12) * 12);
      let msToYM = (ms: number) => yearOf(ms) * 12 + monthOf(ms);

      return {
        scale: { type: "time", domain: [ymToMs(originYM), ymToMs(endYM)] },
        binIndexExpr: (x) => {
          let ts = toTimestampSQL(x);
          // DuckDB MONTH() returns 1-12, subtract 1 for 0-based
          let ym = SQL.sql`(YEAR(${ts}) * 12 + MONTH(${ts}) - 1)`;
          return SQL.cond(
            SQL.isFinite(x),
            SQL.sql`CAST(FLOOR((${ym} - ${SQL.literal(originYM)}) / ${SQL.literal(stepMonths)}) AS INTEGER)`,
            SQL.literal(null),
          );
        },
        nullPredicateExpr,
        binIndex: (x) => Math.floor((msToYM(x) - originYM) / stepMonths),
        rangeForIndex: (i) => [ymToMs(originYM + stepMonths * i), ymToMs(originYM + stepMonths * (i + 1))],
      };
    }
    case "year": {
      // Year-based binning
      let stepYears = stepValue;
      let originYear = Math.floor(yearOf(min) / stepYears) * stepYears;
      let maxYear = yearOf(max);
      let endYear = (Math.floor((maxYear - originYear) / stepYears) + 1) * stepYears + originYear;

      let yearToMs = (y: number) => makeDate(y, 0);
      let msToYear = (ms: number) => yearOf(ms);

      return {
        scale: { type: "time", domain: [yearToMs(originYear), yearToMs(endYear)] },
        binIndexExpr: (x) => {
          let ts = toTimestampSQL(x);
          return SQL.cond(
            SQL.isFinite(x),
            SQL.sql`CAST(FLOOR((YEAR(${ts}) - ${SQL.literal(originYear)}) / ${SQL.literal(stepYears)}) AS INTEGER)`,
            SQL.literal(null),
          );
        },
        nullPredicateExpr,
        binIndex: (x) => Math.floor((msToYear(x) - originYear) / stepYears),
        rangeForIndex: (i) => [yearToMs(originYear + stepYears * i), yearToMs(originYear + stepYears * (i + 1))],
      };
    }
    default:
      throw new Error("invalid step type");
  }
}
