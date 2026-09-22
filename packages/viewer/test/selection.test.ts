// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { DataPointID } from "@embedding-atlas/component";
import type { Coordinator } from "@uwdata/mosaic-core";
import { get } from "svelte/store";
import { describe, expect, it } from "vitest";

import { EmbeddingAtlasStore } from "../src/stores/embedding_atlas_store.js";

function coordinator(): Coordinator {
  return {
    query: () =>
      Promise.resolve([
        {
          column_name: "id",
          column_type: "INTEGER",
        },
      ]),
  } as unknown as Coordinator;
}

async function createStore(selection: DataPointID[] | null = null): Promise<EmbeddingAtlasStore> {
  const store = new EmbeddingAtlasStore({
    coordinator: coordinator(),
    data: {
      id: "id",
      table: "points",
    },
    initialState: {
      charts: {
        "1": { content: "Selection test", type: "markdown" },
      },
      currentLayout: "1",
      layoutOrder: ["1"],
      layouts: {
        "1": { chartIds: ["1"], name: "Default", type: "list" },
      },
    },
    selection,
  });
  await store.ready;
  return store;
}

describe("EmbeddingAtlas controlled selection", () => {
  it("initializes the shared highlight from selection with a defensive copy", async () => {
    const selection: DataPointID[] = [7, "point-8", 9n];
    const store = await createStore(selection);

    expect(get(store.chartContext.highlight)).toEqual(selection);
    expect(get(store.chartContext.highlight)).not.toBe(selection);
  });

  it("updates on ordered changes and deduplicates ordered-equal arrays", async () => {
    const store = await createStore([7, "point-8", 9n]);
    const calls: (DataPointID[] | null)[] = [];
    const unsubscribe = store.chartContext.highlight.subscribe((value) => calls.push(value));

    store.setHighlight([7, "point-8", 9n]);
    expect(calls).toEqual([[7, "point-8", 9n]]);

    const reordered: DataPointID[] = ["point-8", 7, 9n];
    store.setHighlight(reordered);
    reordered[0] = "mutated-outside";

    expect(calls).toEqual([
      [7, "point-8", 9n],
      ["point-8", 7, 9n],
    ]);
    unsubscribe();
  });

  it("clears on null or undefined without duplicate notifications", async () => {
    const store = await createStore([7]);
    const calls: (DataPointID[] | null)[] = [];
    const unsubscribe = store.chartContext.highlight.subscribe((value) => calls.push(value));

    store.setHighlight(null);
    store.setHighlight(undefined);

    expect(calls).toEqual([[7], null]);
    expect(get(store.chartContext.highlight)).toBeNull();
    unsubscribe();
  });
});
