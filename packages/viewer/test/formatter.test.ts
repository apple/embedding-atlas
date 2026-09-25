// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { afterAll, describe, expect, it } from "vitest";

import { inferTimeFormatter } from "../src/charts/common/formatter.js";

// Use a time zone with daylight saving time so winter and summer offsets differ.
const originalTZ = process.env.TZ;
process.env.TZ = "America/New_York";

afterAll(() => {
  if (originalTZ === undefined) {
    delete process.env.TZ;
  } else {
    process.env.TZ = originalTZ;
  }
});

describe("inferTimeFormatter", () => {
  it("uses each value's own offset when hasTimezone is true", () => {
    // Local midnights on the first of the month, one in standard time and one in DST.
    let values = [new Date(2024, 0, 1).getTime(), new Date(2024, 6, 1).getTime()];
    let format = inferTimeFormatter(values, true);
    expect(values.map(format)).toEqual(["Jan 2024", "Jul 2024"]);
  });
});
