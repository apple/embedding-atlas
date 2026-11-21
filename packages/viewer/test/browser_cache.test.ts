// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { createBrowserCache, createMemoryCache, datasetCacheNamespace } from "../src/utils/browser_cache";

/** Minimal in-memory IDBFactory test double supporting store/index/cursor operations. */
function createFakeIndexedDB() {
  const databases = new Map<string, Map<string, Map<string, any>>>();

  function makeRequest<T>(executor: (req: any) => T): IDBRequest<T> {
    const req: any = {
      result: undefined,
      error: null,
      onsuccess: null,
      onerror: null,
    };
    queueMicrotask(() => {
      try {
        req.result = executor(req);
        req.onsuccess?.({ target: req });
      } catch (err) {
        req.error = err;
        req.onerror?.({ target: req });
      }
    });
    return req;
  }

  const factory: any = {
    open(name: string) {
      const req: any = {
        result: undefined,
        error: null,
        onsuccess: null,
        onerror: null,
        onblocked: null,
        onupgradeneeded: null,
        transaction: null,
      };
      queueMicrotask(() => {
        const isNew = !databases.has(name);
        let stores = databases.get(name);
        if (!stores) {
          stores = new Map();
          databases.set(name, stores);
        }

        const makeStore = (storeName: string) => {
          const getMap = () => stores!.get(storeName)!;
          const storeApi: any = {
            indexNames: { contains: () => true },
            createIndex: () => ({}),
            get(key: string) {
              return makeRequest(() => getMap().get(key));
            },
            put(record: any) {
              return makeRequest(() => {
                getMap().set(record.key, structuredClone(record));
                return record.key;
              });
            },
            count() {
              return makeRequest(() => getMap().size);
            },
            index() {
              return {
                openCursor() {
                  const entries = Array.from(getMap().values()).sort((a, b) => a.updatedAt - b.updatedAt);
                  let i = 0;
                  const cursorReq: any = { result: null, error: null, onsuccess: null, onerror: null };
                  const step = () => {
                    queueMicrotask(() => {
                      if (i >= entries.length) {
                        cursorReq.result = null;
                        cursorReq.onsuccess?.({ target: cursorReq });
                        return;
                      }
                      const entry = entries[i++];
                      cursorReq.result = {
                        value: entry,
                        delete() {
                          getMap().delete(entry.key);
                        },
                        continue: step,
                      };
                      cursorReq.onsuccess?.({ target: cursorReq });
                    });
                  };
                  step();
                  return cursorReq;
                },
              };
            },
          };
          return storeApi;
        };

        const db: any = {
          objectStoreNames: { contains: (storeName: string) => stores!.has(storeName) },
          createObjectStore(storeName: string) {
            stores!.set(storeName, new Map());
            return makeStore(storeName);
          },
          transaction(storeName: string) {
            const tx: any = {
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore: () => makeStore(storeName),
            };
            setTimeout(() => tx.oncomplete?.(), 0);
            return tx;
          },
        };

        req.result = db;
        if (isNew) {
          req.transaction = db.transaction("entries");
          req.onupgradeneeded?.({ target: req });
        }
        req.onsuccess?.({ target: req });
      });
      return req;
    },
  };

  return { factory: factory as IDBFactory, databases };
}

describe("datasetCacheNamespace", () => {
  it("distinguishes URLs and local file metadata", () => {
    const a = datasetCacheNamespace([{ url: "https://example.com/a.parquet" }]);
    const b = datasetCacheNamespace([{ url: "https://example.com/b.parquet" }]);
    expect(a).toBe(datasetCacheNamespace([{ url: "https://example.com/a.parquet" }]));
    expect(a).not.toBe(b);

    const f1 = { name: "data.parquet", size: 100, lastModified: 1000 } as File;
    const f2 = { name: "data.parquet", size: 100, lastModified: 2000 } as File;
    const f3 = { name: "other.parquet", size: 100, lastModified: 1000 } as File;
    expect(datasetCacheNamespace([f1])).not.toBe(datasetCacheNamespace([f2]));
    expect(datasetCacheNamespace([f1])).not.toBe(datasetCacheNamespace([f3]));
  });
});

describe("createMemoryCache", () => {
  it("stores, refreshes recency, and evicts oldest entries", async () => {
    const cache = createMemoryCache(2);
    expect(await cache.get("missing")).toBeNull();

    await cache.set("a", [1]);
    await cache.set("b", [2]);
    expect(await cache.get("a")).toEqual([1]); // "a" becomes most recently used
    await cache.set("c", [3]); // "b" should be evicted

    expect(await cache.get("b")).toBeNull();
    expect(await cache.get("a")).toEqual([1]);
    expect(await cache.get("c")).toEqual([3]);
  });
});

describe("createBrowserCache", () => {
  it("persists entries across cache instances and isolates namespaces", async () => {
    const { factory } = createFakeIndexedDB();
    const labels = [{ x: 1, y: 2, content: "Cluster A", priority: 10, level: 0 }];

    const writer = createBrowserCache('["url","https://example.com/one.parquet"]', { indexedDB: factory });
    await writer.set("hash-1", labels);

    const sameDatasetReader = createBrowserCache('["url","https://example.com/one.parquet"]', {
      indexedDB: factory,
    });
    expect(await sameDatasetReader.get("hash-1")).toEqual(labels);

    const otherDatasetReader = createBrowserCache('["url","https://example.com/two.parquet"]', {
      indexedDB: factory,
    });
    expect(await otherDatasetReader.get("hash-1")).toBeNull();
  });

  it("evicts oldest entries in IndexedDB when maxEntries is exceeded", async () => {
    const { factory, databases } = createFakeIndexedDB();
    const cache = createBrowserCache("ns", { indexedDB: factory, maxEntries: 2 });

    await cache.set("k1", [1]);
    await cache.set("k2", [2]);
    await cache.set("k3", [3]);

    const stored = databases.get("embedding-atlas-cache")!.get("entries")!;
    expect(stored.size).toBe(2);
    expect(stored.has(JSON.stringify(["ns", "k1"]))).toBe(false);
    expect(stored.has(JSON.stringify(["ns", "k2"]))).toBe(true);
    expect(stored.has(JSON.stringify(["ns", "k3"]))).toBe(true);
  });

  it("falls back cleanly to memory when IndexedDB is unavailable or fails", async () => {
    const cacheNoIdb = createBrowserCache("ns", { indexedDB: null });
    await cacheNoIdb.set("k1", { ok: true });
    expect(await cacheNoIdb.get("k1")).toEqual({ ok: true });

    const brokenIdb = {
      open() {
        throw new Error("SecurityError: storage disabled");
      },
    } as unknown as IDBFactory;
    const cacheBrokenIdb = createBrowserCache("ns", { indexedDB: brokenIdb });
    await cacheBrokenIdb.set("k2", { ok: 2 });
    expect(await cacheBrokenIdb.get("k2")).toEqual({ ok: 2 });
    expect(await cacheBrokenIdb.get("missing")).toBeNull();
  });
});
