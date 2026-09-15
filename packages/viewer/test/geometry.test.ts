// Copyright (c) 2026. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { pointCoordinatesFromGeometry } from "../src/utils/geometry.js";

const WKB_POINT_BYTES = [1, 1, 0, 0, 0, 30, 139, 109, 82, 209, 91, 87, 192, 93, 251, 2, 122, 225, 139, 70, 64];

describe("pointCoordinatesFromGeometry", () => {
  test("parses WKB from binary and Parquet integer-list representations", () => {
    const expected = [-93.434651, 45.0928185];
    expect(pointCoordinatesFromGeometry(Uint8Array.from(WKB_POINT_BYTES))).toEqual(expected);
    expect(pointCoordinatesFromGeometry(WKB_POINT_BYTES)).toEqual(expected);
    expect(pointCoordinatesFromGeometry(BigInt64Array.from(WKB_POINT_BYTES.map(BigInt)))).toEqual(expected);
  });

  test("parses GeoJSON objects, strings, and features", () => {
    const point = { type: "Point", coordinates: [-73.99006518, 40.755437] };
    expect(pointCoordinatesFromGeometry(point)).toEqual(point.coordinates);
    expect(pointCoordinatesFromGeometry(JSON.stringify(point))).toEqual(point.coordinates);
    expect(pointCoordinatesFromGeometry({ type: "Feature", geometry: point })).toEqual(point.coordinates);
  });

  test("parses WKT points and rejects unsupported or malformed geometry", () => {
    expect(pointCoordinatesFromGeometry("POINT (-73.99006518 40.755437)")).toEqual([-73.99006518, 40.755437]);
    expect(
      pointCoordinatesFromGeometry({
        type: "LineString",
        coordinates: [
          [0, 0],
          [1, 1],
        ],
      }),
    ).toBeNull();
    expect(pointCoordinatesFromGeometry([1, 2, 3])).toBeNull();
    expect(pointCoordinatesFromGeometry("not geometry")).toBeNull();
  });
});
