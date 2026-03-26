// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as d3 from "d3";

/**
 * Infer a number formatter for an array of values.
 * - If all values are integer, display them as integers (with toLocaleString) without decimal points.
 * - For floating point values, find a reasonable precision, and format all values consistently.
 */
export function inferNumberFormatter(values: number[]): (value: number) => string {
  let locale = undefined; // Use default locale.

  let finite = values.filter((v) => Number.isFinite(v));

  if (finite.length === 0) {
    return (value: number) => value.toLocaleString(locale);
  }

  let allInteger = finite.every((v) => Number.isInteger(v));

  let maxAbs = d3.max(finite, (v) => Math.abs(v)) ?? 0;

  if (allInteger && maxAbs < 1e15) {
    // Use locale grouping (e.g. 1,234,567) for reasonable integers.
    return (value: number) => value.toLocaleString(locale, { maximumFractionDigits: 0 });
  }

  // Determine a reasonable number of decimal places.
  let minAbs = d3.min(finite.filter((v) => v !== 0).map((v) => Math.abs(v))) ?? Infinity;

  // For very large or very small numbers, use exponential notation.
  if (maxAbs >= 1e9 || (minAbs > 0 && minAbs < 1e-3)) {
    return d3.format(".3~e");
  }

  // Find a precision that distinguishes the values.
  // Use the range (or max absolute value if range is zero) to determine decimal digits.
  let range = (d3.max(finite) ?? 0) - (d3.min(finite) ?? 0);
  let ref = range > 0 ? range : maxAbs;

  // Number of integer digits in the reference magnitude.
  let intDigits = ref >= 1 ? Math.floor(Math.log10(ref)) + 1 : 0;

  // We want ~4 significant digits total, with at least 1 decimal for floats.
  let sigFigs = Math.max(4, intDigits + 1);
  let decimals = Math.max(1, sigFigs - intDigits);
  // Clamp to a reasonable range.
  decimals = Math.min(decimals, 6);

  // If the values already look "clean" at fewer decimals, reduce.
  // Check if all values round-trip at fewer decimal places.
  for (let d = 1; d < decimals; d++) {
    let factor = 10 ** d;
    if (finite.every((v) => Math.abs(v - Math.round(v * factor) / factor) < 1e-12)) {
      decimals = d;
      break;
    }
  }

  return (value: number) =>
    value.toLocaleString(locale, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/**
 * Infer a time formatter. The inputs are milliseconds since epoch. If `hasTimezone` is true, use local time, otherwise use UTC time.
 * The goal of the inferred format is to have a concise representation of the values.
 * For instance, if all values are rounded year (e.g., 2020-01-01 00:00:00),
 * then output a formatter that displays year only.
 * If all value are rounded to day, month, format up to the unit respectively.
 */
export function inferTimeFormatter(values: number[], hasTimezone: boolean = false): (value: number) => string {
  if (values.length === 0) {
    let fmt = hasTimezone ? d3.timeFormat : d3.utcFormat;
    let formatter = fmt("%Y-%m-%d %H:%M:%S");
    return (value: number) => formatter(new Date(value));
  }

  // When hasTimezone is true, use local time via Date methods; otherwise use UTC methods.
  let tzOffsetMs = hasTimezone ? -new Date().getTimezoneOffset() * 60_000 : 0;
  let adjusted = (ms: number) => new Date(ms + tzOffsetMs);

  let getYear = (ms: number) => adjusted(ms).getUTCFullYear();
  let getMonth = (ms: number) => adjusted(ms).getUTCMonth();
  let getDay = (ms: number) => adjusted(ms).getUTCDate();
  let getHour = (ms: number) => adjusted(ms).getUTCHours();
  let getMinute = (ms: number) => adjusted(ms).getUTCMinutes();
  let getSecond = (ms: number) => adjusted(ms).getUTCSeconds();
  let getMs = (ms: number) => adjusted(ms).getUTCMilliseconds();

  // Determine the finest granularity needed
  let hasSubSecond = values.some((v) => getMs(v) !== 0);
  let hasSeconds = hasSubSecond || values.some((v) => getSecond(v) !== 0);
  let hasMinutes = hasSeconds || values.some((v) => getMinute(v) !== 0);
  let hasHours = hasMinutes || values.some((v) => getHour(v) !== 0);
  let hasDays = hasHours || values.some((v) => getDay(v) !== 1);
  let hasMonths = hasDays || values.some((v) => getMonth(v) !== 0);

  // Check if all values share the same year (for more compact formatting)
  let allSameYear = values.every((v) => getYear(v) === getYear(values[0]));

  // Pick the most concise format specifier
  let specifier: string;
  if (!hasMonths) {
    specifier = "%Y";
  } else if (!hasDays) {
    specifier = "%b %Y";
  } else if (!hasHours) {
    specifier = allSameYear ? "%b %-d" : "%Y-%m-%d";
  } else if (!hasSeconds) {
    specifier = allSameYear ? "%b %-d %H:%M" : "%Y-%m-%d %H:%M";
  } else if (!hasSubSecond) {
    specifier = allSameYear ? "%b %-d %H:%M:%S" : "%Y-%m-%d %H:%M:%S";
  } else {
    specifier = allSameYear ? "%b %-d %H:%M:%S.%L" : "%Y-%m-%d %H:%M:%S.%L";
  }

  let fmt = hasTimezone ? d3.timeFormat : d3.utcFormat;
  let formatter = fmt(specifier);
  return (value: number) => formatter(new Date(value));
}
