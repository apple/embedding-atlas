// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { loadEmbeddingModel } from "../src/inference/embedding.js";

interface FakeResponseInit {
  status?: number;
  headers?: Record<string, string>;
  body?: any;
}

function fakeResponse({ status = 200, headers = {}, body = {} }: FakeResponseInit): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k: string) => headers[k.toLowerCase()] ?? headers[k] ?? null },
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    json: async () => body,
  } as unknown as Response;
}

function embedBody(texts: string[], dimensions = 4): { data: { embedding: number[]; index: number }[] } {
  return {
    data: texts.map((_, i) => ({
      index: i,
      embedding: Array.from({ length: dimensions }, (_, d) => i * 100 + d),
    })),
  };
}

describe("loadEmbeddingModel — openai", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    vi.useFakeTimers();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.useRealTimers();
  });

  it("merges synchronously fanned-out calls into one fetch", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: string[] };
      return fakeResponse({ body: embedBody(body.input) });
    });
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "k" }, "text");

    const a = model.embeddings(["alpha"]);
    const b = model.embeddings(["beta", "gamma"]);
    const c = model.embeddings(["delta"]);

    const [ra, rb, rc] = await Promise.all([a, b, c]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const sentBody = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(sentBody.input).toEqual(["alpha", "beta", "gamma", "delta"]);

    // Each caller gets vectors only for their own inputs.
    expect(ra.dimensions).toBe(4);
    expect(ra.vectors.length).toBe(1 * 4);
    expect(rb.vectors.length).toBe(2 * 4);
    expect(rc.vectors.length).toBe(1 * 4);
    // The coalesce slicer maps offsets correctly: caller `c` gets the 4th input's vector.
    expect(Array.from(rc.vectors)).toEqual([300, 301, 302, 303]);
  });

  it("sequential awaited calls land in separate fetches", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: string[] };
      return fakeResponse({ body: embedBody(body.input) });
    });
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "k" }, "text");

    await model.embeddings(["a"]);
    await model.embeddings(["b"]);

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("merges calls that arrive while a flush is in flight", async () => {
    let releaseFirstFetch: () => void = () => {};
    const firstHeld = new Promise<void>((r) => (releaseFirstFetch = r));
    let calls = 0;
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const n = ++calls;
      const body = JSON.parse(init.body as string) as { input: string[] };
      if (n === 1) {
        await firstHeld;
      }
      return fakeResponse({ body: embedBody(body.input) });
    });
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "k" }, "text");

    const a = model.embeddings(["a"]);
    // Let the first flush kick off (microtask + the inner awaits in the loader).
    await Promise.resolve();
    await Promise.resolve();
    const b = model.embeddings(["b"]);
    const c = model.embeddings(["c"]);

    releaseFirstFetch();
    const [, rb, rc] = await Promise.all([a, b, c]);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const secondCall = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string);
    expect(secondCall.input).toEqual(["b", "c"]);
    // Sliced correctly: caller b is index 0, caller c is index 1 in the merged batch.
    expect(rb.vectors.length).toBe(4);
    expect(rc.vectors.length).toBe(4);
    expect(Array.from(rc.vectors)).toEqual([100, 101, 102, 103]);
  });

  it("retries on 429 with Retry-After and eventually resolves", async () => {
    let calls = 0;
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      calls++;
      if (calls === 1) {
        return fakeResponse({ status: 429, headers: { "Retry-After": "0" }, body: { error: "rate_limit" } });
      }
      const body = JSON.parse(init.body as string) as { input: string[] };
      return fakeResponse({ body: embedBody(body.input) });
    });
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "k" }, "text");
    const promise = model.embeddings(["hello"]);

    // advance past the coalesce window + the retry backoff
    await vi.advanceTimersByTimeAsync(2000);
    const result = await promise;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.dimensions).toBe(4);
    expect(result.vectors.length).toBe(4);
  });

  it("surfaces non-retryable errors immediately", async () => {
    const fetchMock = vi.fn(async () => fakeResponse({ status: 401, body: { error: { message: "invalid api key" } } }));
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "bad" }, "text");
    await expect(model.embeddings(["x"])).rejects.toThrow(/401/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("dispose rejects pending coalesced callers", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: string[] };
      return fakeResponse({ body: embedBody(body.input) });
    });
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "k" }, "text");
    const pending = model.embeddings(["x"]);
    await model.dispose();

    await expect(pending).rejects.toThrow(/disposed/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("dispose waits for an in-flight flush to settle before resolving", async () => {
    let releaseFetch: () => void = () => {};
    const held = new Promise<void>((r) => (releaseFetch = r));
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string) as { input: string[] };
      await held;
      return fakeResponse({ body: embedBody(body.input) });
    });
    globalThis.fetch = fetchMock as any;

    const model = await loadEmbeddingModel("text-embedding-3-small", { apiKey: "k" }, "text");

    const pending = model.embeddings(["x"]);
    // Let the flush kick off (microtask + the loader's inner awaits) so the
    // fetch is actually in flight when we dispose.
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    let disposeSettled = false;
    const disposeDone = model.dispose().then(() => {
      disposeSettled = true;
    });

    // While the fetch is held, dispose must not resolve — otherwise the caller
    // would tear down the underlying extractor mid-inference.
    await Promise.resolve();
    await Promise.resolve();
    expect(disposeSettled).toBe(false);

    releaseFetch();
    await disposeDone;
    expect(disposeSettled).toBe(true);

    // The in-flight caller still resolves normally.
    const result = await pending;
    expect(result.vectors.length).toBe(4);
  });
});
