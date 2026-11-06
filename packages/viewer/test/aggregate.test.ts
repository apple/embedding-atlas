// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import { describe, expect, test } from "vitest";

import { computeFieldStats } from "../src/charts/common/aggregate.js";

describe("computeFieldStats", () => {
  test("treats DECIMAL columns as quantitative", async () => {
    // Answers the DESCRIBE query with the column type, and every other query with one row of stats.
    let coordinator: any = {
      query: async (query: any) => {
        let row = query.type == "DESCRIBE_QUERY" ? { column_type: "DECIMAL(18,3)" } : { count: 2, min: 1.5, max: 2.5 };
        return { get: () => row };
      },
    };
    let stats = await computeFieldStats(coordinator, "data", SQL.column("price"));
    expect(stats?.kind).toBe("quantitative");
  });
});
