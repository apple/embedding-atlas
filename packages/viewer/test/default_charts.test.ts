// Copyright (c) 2026 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import { describe, expect, test } from "vitest";

import { defaultCharts } from "../src/charts/default_charts.js";

const mockCoordinator = {
  query: async () => [],
} as unknown as Coordinator;

describe("defaultCharts", () => {
  test("preserves projection data fields when config.embedding overrides partial data properties", async () => {
    const charts = await defaultCharts({
      coordinator: mockCoordinator,
      table: "items",
      id: "id",
      projection: {
        x: "proj_x",
        y: "proj_y",
        text: "text_col",
        image: "img_col",
        importance: "imp_col",
        neighbors: "nbr_col",
      },
      config: {
        embedding: {
          title: "Custom Embedding",
          mode: "density",
          data: {
            category: "label",
            text: "override_text",
          },
        },
      },
    });

    const embedding = charts.find((c) => c.type === "embedding");
    expect(embedding).toEqual({
      type: "embedding",
      title: "Custom Embedding",
      mode: "density",
      data: {
        x: "proj_x",
        y: "proj_y",
        text: "override_text",
        image: "img_col",
        importance: "imp_col",
        neighbors: "nbr_col",
        category: "label",
      },
    });
  });

  test("ignores null embedding override at runtime and respects false", async () => {
    const withNull = await defaultCharts({
      coordinator: mockCoordinator,
      table: "items",
      id: "id",
      projection: { x: "x", y: "y" },
      config: {
        embedding: null as unknown as undefined,
      },
    });
    expect(withNull.some((c) => c.type === "embedding")).toBe(true);

    const disabled = await defaultCharts({
      coordinator: mockCoordinator,
      table: "items",
      id: "id",
      projection: { x: "x", y: "y" },
      config: {
        embedding: false,
      },
    });
    expect(disabled.some((c) => c.type === "embedding")).toBe(false);
  });
});
