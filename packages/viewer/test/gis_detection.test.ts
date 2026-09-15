// Copyright (c) 2026. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { detectGisColumns } from "../src/utils/gis_detection.js";

describe("detectGisColumns geometry encodings", () => {
  test.each([
    "BLOB",
    "GEOMETRY('OGC:CRS84')",
    "UTINYINT[]",
    "BIGINT[21]",
    "VARCHAR",
    "JSON",
    'STRUCT("type" VARCHAR, coordinates DOUBLE[])',
  ])("recognizes geometry stored as %s", (columnType) => {
    expect(detectGisColumns([{ column_name: "geometry", column_type: columnType }])).toEqual({
      type: "geometry",
      geometryColumn: "geometry",
      xColumn: "lon",
      yColumn: "lat",
    });
  });

  test("keeps numeric lon/lat columns ahead of a geometry column", () => {
    expect(
      detectGisColumns([
        { column_name: "geometry", column_type: "VARCHAR" },
        { column_name: "lon", column_type: "DOUBLE" },
        { column_name: "lat", column_type: "DOUBLE" },
      ]),
    ).toEqual({ type: "columns", xColumn: "lon", yColumn: "lat" });
  });

  test("does not treat unrelated geometry scalars or lists as encoded geometry", () => {
    expect(detectGisColumns([{ column_name: "geometry", column_type: "INTEGER" }])).toBeNull();
    expect(detectGisColumns([{ column_name: "geometry", column_type: "VARCHAR[]" }])).toBeNull();
  });
});
