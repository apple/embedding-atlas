// Copyright (c) 2026 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { fieldKey, resolveLayerData } from "../src/charts/embedding/layer_data.js";
import type { EmbeddingSpec } from "../src/charts/embedding/types.js";

const spec: EmbeddingSpec = {
  type: "embedding",
  data: { x: "x", y: { sql: "y * 2" }, text: "text", category: "topic" },
  colorScale: { domain: ["a", "b"] },
  color: "#123456",
};

describe("resolveLayerData", () => {
  test("layers on the view's table inherit the fields and the color scale", () => {
    let d = resolveLayerData(spec, { type: "points" }, "t");
    expect(d.x).toBe("x");
    expect(d.y).toEqual({ sql: "y * 2" });
    expect(d.category).toBe("topic");
    expect(d.colorScale).toEqual({ domain: ["a", "b"] });
    expect(d.categoryIsMain).toBe(true);
    expect(d.color).toBe(null);
  });

  test("a layer's own color scale makes its own legend", () => {
    let d = resolveLayerData(spec, { type: "points", colorScale: { domain: ["b"] } }, "t");
    expect(d.category).toBe("topic");
    expect(d.colorScale).toEqual({ domain: ["b"] });
    expect(d.categoryIsMain).toBe(false);
  });

  test("an explicit category equal to the main one shares the main legend", () => {
    let d = resolveLayerData(spec, { type: "density", data: { category: "topic" } }, "t");
    expect(d.categoryIsMain).toBe(true);
    expect(d.colorScale).toEqual({ domain: ["a", "b"] });
    let other = resolveLayerData(spec, { type: "density", data: { category: { sql: "lower(topic)" } } }, "t");
    expect(other.categoryIsMain).toBe(false);
    expect(other.colorScale).toBe(null);
  });

  test("a fixed color opts out of the inherited category; the spec's color is the default", () => {
    let fixed = resolveLayerData(spec, { type: "points", color: "#ff0000" }, "t");
    expect(fixed.category).toBe(null);
    expect(fixed.color).toBe("#ff0000");
    let none = resolveLayerData(spec, { type: "points", data: { category: null } }, "t");
    expect(none.color).toBe("#123456");
  });

  test("layers on other tables don't inherit fields", () => {
    let d = resolveLayerData(spec, { type: "points", data: { table: "u", x: "ux" } }, "t");
    expect(d.isDataTable).toBe(false);
    expect(d.x).toBe("ux");
    expect(d.y).toBe(undefined);
    expect(d.category).toBe(null);
    expect(d.filtered).toBe(false);
    let labels = resolveLayerData(spec, { type: "labels", data: { table: "u" } }, "t");
    expect(labels.text).toBe(null);
  });

  test("labels layers inherit the text field", () => {
    let d = resolveLayerData(spec, { type: "labels" }, "t");
    expect(d.text).toBe("text");
    expect(d.category).toBe(null);
    expect(d.filtered).toBe(false);
  });
});

describe("fieldKey", () => {
  test("columns and expressions are distinct", () => {
    expect(fieldKey("a")).not.toBe(fieldKey({ sql: "a" }));
    expect(fieldKey(null)).toBe(null);
  });
});
