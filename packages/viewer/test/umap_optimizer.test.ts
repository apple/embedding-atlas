// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";
import { withPipeline } from "../src/app/umap_optimizer.js";

describe("umap optimizer", () => {
  it("runs every optimizer on the 0.6 pipeline", () => {
    for (const optimizer of ["adam", "momentum", "sgd"] as const) {
      expect(withPipeline({ optimizer })).toEqual({ optimizer, compatibilityLayout: false });
    }
  });

  it("defaults options without an optimizer to Adam", () => {
    expect(withPipeline(undefined)).toEqual({ optimizer: "adam", compatibilityLayout: false });
    expect(withPipeline({ minDist: 0.2 })).toEqual({ minDist: 0.2, optimizer: "adam", compatibilityLayout: false });
  });

  it("moves saved legacy options to the 0.6 pipeline and keeps their optimizer", () => {
    expect(withPipeline({ compatibilityLayout: true })).toEqual({ optimizer: "adam", compatibilityLayout: false });
    expect(withPipeline({ optimizer: "sgd", compatibilityLayout: true })).toEqual({
      optimizer: "sgd",
      compatibilityLayout: false,
    });
  });
});
