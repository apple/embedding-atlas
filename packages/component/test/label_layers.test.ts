import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { LabelLayers, type LayerLabels } from "../src/lib/embedding_view/label_layers.js";
import type { AutoLabels, ResolvedLabelsLayer } from "../src/lib/embedding_view/layers.js";

const viewport = { x: 0, y: 0, scale: 1 };

function auto(requestKey: string, queryKey: string, ready = true): AutoLabels {
  return {
    requestKey,
    queryKey,
    mergeKey: requestKey,
    persistKey: null,
    densityThreshold: null,
    ready,
    loadSnapshot: async () => ({ x: new Float32Array(), y: new Float32Array(), viewport }),
    queryClusterLabels: async () => [],
  };
}

function layer(key: string, a: AutoLabels | null, labels: any[] | null = null): ResolvedLabelsLayer {
  return { type: "labels", key, id: key, opacity: null, labels, auto: a };
}

/** Generations that resolve when the test says so. */
function setup() {
  let calls: { requestKey: string; resolve: (labels: LayerLabels) => void }[] = [];
  let changes = 0;
  let generating: number[] = [];
  let layers = new LabelLayers({
    generate: (a) => new Promise((resolve) => calls.push({ requestKey: a.requestKey, resolve })),
    onChange: () => (changes += 1),
    onGenerating: (n) => generating.push(n),
    debounce: 300,
  });
  let finish = async (i: number, content: string) => {
    calls[i].resolve({ labels: [{ x: 0, y: 0, content }], viewport });
    await Promise.resolve();
    await Promise.resolve();
  };
  let shown = (key = "a") =>
    layers
      .results()
      .find((r) => r.key == key)
      ?.labels?.labels.map((l) => l.content) ?? null;
  return { layers, calls, finish, shown, changes: () => changes, generating };
}

describe("LabelLayers", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  test("static labels are shown as they are", () => {
    let t = setup();
    let labels = [{ x: 0, y: 0, content: "static" }];
    t.layers.sync([layer("a", null, labels)]);
    expect(t.shown()).toEqual(["static"]);
    expect(t.calls.length).toBe(0);
  });

  test("automatic labels are generated right away the first time, and once per request", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    expect(t.calls.length).toBe(1);
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    expect(t.calls.length).toBe(1);
    await t.finish(0, "one");
    expect(t.shown()).toEqual(["one"]);
    expect(t.generating).toEqual([1, 0]);
  });

  test("a new data revision keeps the labels and is debounced", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    await t.finish(0, "one");
    t.layers.sync([layer("a", auto("r2", "q1"))]);
    t.layers.sync([layer("a", auto("r3", "q1"))]);
    expect(t.calls.length).toBe(1);
    expect(t.shown()).toEqual(["one"]);
    vi.advanceTimersByTime(300);
    expect(t.calls.map((c) => c.requestKey)).toEqual(["r1", "r3"]);
    await t.finish(1, "three");
    expect(t.shown()).toEqual(["three"]);
  });

  test("a new query clears the labels and is generated right away", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    await t.finish(0, "one");
    t.layers.sync([layer("a", auto("r2", "q2"))]);
    expect(t.shown()).toBe(null);
    expect(t.calls.length).toBe(2);
  });

  test("results of superseded requests are dropped", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    t.layers.sync([layer("a", auto("r2", "q2"))]);
    await t.finish(0, "stale");
    expect(t.shown()).toBe(null);
    await t.finish(1, "current");
    expect(t.shown()).toEqual(["current"]);
  });

  test("a layer that isn't ready waits, then generates right away", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1", false))]);
    expect(t.calls.length).toBe(0);
    t.layers.sync([layer("a", auto("r1", "q1", true))]);
    expect(t.calls.length).toBe(1);
  });

  test("streaming data: only the latest revision is generated, after the updates pause", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r0", "q"))]);
    for (let i = 1; i <= 10; i++) {
      t.layers.sync([layer("a", auto(`r${i}`, "q"))]);
      vi.advanceTimersByTime(100);
    }
    expect(t.calls.map((c) => c.requestKey)).toEqual(["r0"]);
    vi.advanceTimersByTime(300);
    expect(t.calls.map((c) => c.requestKey)).toEqual(["r0", "r10"]);
  });

  test("removing a layer drops its labels and pending generation", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1")), layer("b", auto("s1", "p1"))]);
    t.layers.sync([layer("b", auto("s1", "p1"))]);
    await t.finish(0, "a");
    expect(t.layers.results().map((r) => r.key)).toEqual(["b"]);
    expect(t.shown("a")).toBe(null);
  });

  test("switching from static to automatic labels clears the static labels", () => {
    let t = setup();
    t.layers.sync([layer("a", null, [{ x: 0, y: 0, content: "static" }])]);
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    expect(t.shown()).toBe(null);
    expect(t.calls.length).toBe(1);
  });

  test("a layer without automatic labels shows nothing", () => {
    let t = setup();
    t.layers.sync([layer("a", null)]);
    expect(t.shown()).toBe(null);
    expect(t.calls.length).toBe(0);
  });
});

describe("LabelLayers errors", () => {
  test("a failed generation is retried once after a delay, not on every sync", async () => {
    vi.useFakeTimers();
    try {
      let attempts = 0;
      let errors: unknown[] = [];
      let layers = new LabelLayers({
        generate: async () => {
          attempts += 1;
          throw new Error("failing");
        },
        onChange: () => {},
        onError: (e) => errors.push(e),
        retryDelay: 1000,
      });
      layers.sync([layer("a", auto("r1", "q1"))]);
      await vi.advanceTimersByTimeAsync(0);
      // Syncs with the same request don't retry.
      layers.sync([layer("a", auto("r1", "q1"))]);
      layers.sync([layer("a", auto("r1", "q1"))]);
      await vi.advanceTimersByTimeAsync(0);
      expect(attempts).toBe(1);
      // One retry after the delay, then no more.
      await vi.advanceTimersByTimeAsync(1000);
      expect(attempts).toBe(2);
      await vi.advanceTimersByTimeAsync(5000);
      layers.sync([layer("a", auto("r1", "q1"))]);
      await vi.advanceTimersByTimeAsync(0);
      expect(attempts).toBe(2);
      expect(errors.length).toBe(2);
      // A new request is generated again.
      layers.sync([layer("a", auto("r2", "q2"))]);
      await vi.advanceTimersByTimeAsync(0);
      expect(attempts).toBe(3);
    } finally {
      vi.useRealTimers();
    }
  });

  test("a transient failure succeeds on the retry", async () => {
    vi.useFakeTimers();
    try {
      let attempts = 0;
      let layers = new LabelLayers({
        generate: async () => {
          attempts += 1;
          if (attempts == 1) {
            throw new Error("transient");
          }
          return { labels: [{ x: 0, y: 0, content: "ok" }], viewport };
        },
        onChange: () => {},
        onError: () => {},
        retryDelay: 1000,
      });
      layers.sync([layer("a", auto("r1", "q1"))]);
      await vi.advanceTimersByTimeAsync(1000);
      expect(attempts).toBe(2);
      expect(layers.results()[0].labels?.labels.map((l) => l.content)).toEqual(["ok"]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("errors of cancelled generations are not reported", async () => {
    let errors: unknown[] = [];
    let reject: ((e: unknown) => void) | null = null;
    let layers = new LabelLayers({
      generate: (a) =>
        a.requestKey == "r1" ? new Promise((_, r) => (reject = r)) : Promise.resolve({ labels: [], viewport }),
      onChange: () => {},
      onError: (e) => errors.push(e),
    });
    layers.sync([layer("a", auto("r1", "q1"))]);
    layers.sync([layer("a", auto("r2", "q2"))]);
    reject!(new Error("cancelled"));
    await new Promise((r) => setTimeout(r, 0));
    expect(errors.length).toBe(0);
  });
});

describe("LabelLayers waiting", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  test("a pending regeneration is dropped when the data becomes not ready", async () => {
    let t = setup();
    t.layers.sync([layer("a", auto("r1", "q1"))]);
    await t.finish(0, "one");
    t.layers.sync([layer("a", auto("r2", "q1"))]); // debounced
    t.layers.sync([layer("a", auto("r3", "q1", false))]); // the data is gone before the timer fires
    vi.advanceTimersByTime(1000);
    expect(t.calls.map((c) => c.requestKey)).toEqual(["r1"]);
    expect(t.shown()).toEqual(["one"]);
    t.layers.sync([layer("a", auto("r4", "q1", true))]);
    expect(t.calls.map((c) => c.requestKey)).toEqual(["r1", "r4"]);
  });
});
