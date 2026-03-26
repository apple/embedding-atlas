// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { inferNumberFormatter, inferTimeFormatter } from "../common/formatter.js";

export function inferColumnFormatters(
  data: Record<string, any>[],
  columns: string[],
): Record<string, (v: any) => string> {
  let result: Record<string, (v: any) => string> = {};
  for (let column of columns) {
    let values = data.map((x) => x[column]);
    if (values.every((x) => x == null || typeof x == "number")) {
      result[column] = inferNumberFormatter(values);
    }
    if (values.every((x) => x == null || x instanceof Date)) {
      result[column] = inferTimeFormatter(values);
    }
  }
  return result;
}
