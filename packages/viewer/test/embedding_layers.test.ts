// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { effectiveLayers, updateLayers, type EmbeddingLayers } from "../src/charts/embedding/layers.js";

describe("effectiveLayers", () => {
  it("defaults an empty spec to points and labels without density", () => {
    const expected: EmbeddingLayers = { points: true, density: false, labels: true };
    expect(effectiveLayers({})).toEqual(expected);
  });

  it("treats the legacy points mode the same as an empty spec", () => {
    expect(effectiveLayers({ mode: "points" })).toEqual({ points: true, density: false, labels: true });
  });

  it("expands the legacy density mode to points, density, and labels", () => {
    // Legacy density mode drew points and labels too.
    expect(effectiveLayers({ mode: "density" })).toEqual({ points: true, density: true, labels: true });
  });

  it("respects a full layers object verbatim", () => {
    expect(effectiveLayers({ layers: { points: false, density: true, labels: false } })).toEqual({
      points: false,
      density: true,
      labels: false,
    });
  });

  it("merges partial layers over the legacy mode base", () => {
    expect(effectiveLayers({ mode: "density", layers: { labels: false } })).toEqual({
      points: true,
      density: true,
      labels: false,
    });
    expect(effectiveLayers({ mode: "density", layers: { density: false } })).toEqual({
      points: true,
      density: false,
      labels: true,
    });
  });

  it("merges partial layers over the default base when mode is absent", () => {
    expect(effectiveLayers({ layers: { density: true } })).toEqual({ points: true, density: true, labels: true });
  });

  it("returns a plain JSON-serializable object", () => {
    const result = effectiveLayers({ mode: "density" });
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);
  });

  it("does not alias spec.layers; mutating the result leaves the spec untouched", () => {
    const spec = { layers: { points: true, density: true, labels: true } };
    const result = effectiveLayers(spec);
    result.points = false;
    result.labels = false;
    expect(spec.layers).toEqual({ points: true, density: true, labels: true });
  });
});

describe("updateLayers", () => {
  it("replaces a legacy mode with a complete layers object and deletes mode", () => {
    const spec: { mode?: "points" | "density"; layers?: Partial<EmbeddingLayers> } = { mode: "density" };
    updateLayers(spec, { points: false });
    expect(spec.layers).toEqual({ points: false, density: true, labels: true });
    expect("mode" in spec).toBe(false);
  });

  it("applies the patch over existing layers and preserves untouched keys", () => {
    const spec: { layers?: Partial<EmbeddingLayers> } = { layers: { points: false } };
    updateLayers(spec, { density: true });
    expect(spec.layers).toEqual({ points: false, density: true, labels: true });
  });

  it("leaves unrelated spec fields untouched", () => {
    const data = { table: "items" };
    const spec: Record<string, any> = { mode: "density", pointSize: 3, data };
    updateLayers(spec, { labels: false });
    expect(spec.pointSize).toBe(3);
    expect(spec.data).toBe(data);
    expect(Object.keys(spec).sort()).toEqual(["data", "layers", "pointSize"]);
  });

  it("is idempotent when the same patch is applied twice", () => {
    const spec: { mode?: "points" | "density"; layers?: Partial<EmbeddingLayers> } = { mode: "density" };
    updateLayers(spec, { points: false });
    const first = { ...spec.layers };
    updateLayers(spec, { points: false });
    expect(spec.layers).toEqual(first);
  });
});
