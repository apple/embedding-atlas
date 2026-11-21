// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Cache } from "../api.js";

const DEFAULT_DB_NAME = "embedding-atlas-cache";
const STORE_NAME = "entries";
const UPDATED_AT_INDEX = "updatedAt";
const DEFAULT_MAX_ENTRIES = 128;

export interface BrowserCacheOptions {
  /** IndexedDB database name. Defaults to `"embedding-atlas-cache"`. */
  dbName?: string;
  /** Maximum number of entries retained in memory and IndexedDB. Defaults to 128. */
  maxEntries?: number;
  /** Optional IndexedDB factory override (primarily for testing). */
  indexedDB?: IDBFactory | null;
}

/**
 * Derive a deterministic cache namespace from the loaded dataset inputs so
 * distinct files or URLs using the same `"dataset"` table do not collide.
 */
export function datasetCacheNamespace(inputs: (File | { url: string })[]): string {
  const parts = inputs.map((input) => {
    if ("url" in input) {
      return ["url", input.url];
    }
    return ["file", input.name, input.size, input.lastModified];
  });
  return JSON.stringify(parts);
}

/**
 * Bounded in-memory LRU cache implementing {@link Cache}.
 */
export function createMemoryCache(maxEntries: number = DEFAULT_MAX_ENTRIES): Cache {
  const limit = Math.max(1, maxEntries);
  const entries = new Map<string, any>();

  return {
    async get(key: string): Promise<any | null> {
      if (!entries.has(key)) {
        return null;
      }
      const value = entries.get(key);
      // Refresh LRU order.
      entries.delete(key);
      entries.set(key, value);
      return value ?? null;
    },
    async set(key: string, value: any): Promise<void> {
      if (entries.has(key)) {
        entries.delete(key);
      }
      entries.set(key, value);
      while (entries.size > limit) {
        const oldestKey = entries.keys().next().value;
        if (oldestKey === undefined) {
          break;
        }
        entries.delete(oldestKey);
      }
    },
  };
}

function openDatabase(factory: IDBFactory, dbName: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(dbName, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      const store = db.objectStoreNames.contains(STORE_NAME)
        ? request.transaction!.objectStore(STORE_NAME)
        : db.createObjectStore(STORE_NAME, { keyPath: "key" });
      if (!store.indexNames.contains(UPDATED_AT_INDEX)) {
        store.createIndex(UPDATED_AT_INDEX, UPDATED_AT_INDEX, { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Failed to open IndexedDB cache"));
    request.onblocked = () => reject(new Error("IndexedDB cache open blocked"));
  });
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed"));
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  });
}

async function evictOldestEntries(store: IDBObjectStore, excess: number): Promise<void> {
  if (excess <= 0) {
    return;
  }
  const index = store.index(UPDATED_AT_INDEX);
  let remaining = excess;
  await new Promise<void>((resolve, reject) => {
    const cursorReq = index.openCursor();
    cursorReq.onsuccess = () => {
      const cursor = cursorReq.result;
      if (!cursor || remaining <= 0) {
        resolve();
        return;
      }
      cursor.delete();
      remaining--;
      cursor.continue();
    };
    cursorReq.onerror = () => reject(cursorReq.error ?? new Error("IndexedDB cursor failed"));
  });
}

/**
 * Persistent browser {@link Cache} backed by IndexedDB with an in-memory LRU
 * mirror and graceful fallback when IndexedDB is unavailable or blocked.
 */
export function createBrowserCache(namespace: string, options: BrowserCacheOptions = {}): Cache {
  const maxEntries = Math.max(1, options.maxEntries ?? DEFAULT_MAX_ENTRIES);
  const dbName = options.dbName ?? DEFAULT_DB_NAME;
  const memory = createMemoryCache(maxEntries);
  const idbFactory =
    options.indexedDB !== undefined
      ? options.indexedDB
      : typeof globalThis !== "undefined" && "indexedDB" in globalThis
        ? globalThis.indexedDB
        : null;

  let dbPromise: Promise<IDBDatabase> | null = null;
  let monotonicStamp = 0;

  function scopedKey(key: string): string {
    return JSON.stringify([namespace, key]);
  }

  function getDatabase(): Promise<IDBDatabase> | null {
    if (!idbFactory) {
      return null;
    }
    if (!dbPromise) {
      dbPromise = openDatabase(idbFactory, dbName).catch((err) => {
        dbPromise = null;
        throw err;
      });
    }
    return dbPromise;
  }

  return {
    async get(key: string): Promise<any | null> {
      const cacheKey = scopedKey(key);
      const cached = await memory.get(cacheKey);
      if (cached !== null) {
        return cached;
      }

      try {
        const db = await getDatabase();
        if (!db) {
          return null;
        }
        const tx = db.transaction(STORE_NAME, "readonly");
        const done = transactionDone(tx);
        const store = tx.objectStore(STORE_NAME);
        const record = await requestToPromise(store.get(cacheKey));
        await done;
        if (record != null && record.value !== undefined) {
          await memory.set(cacheKey, record.value);
          return record.value;
        }
      } catch {
        // Ignore storage errors and behave as a cache miss.
      }
      return null;
    },

    async set(key: string, value: any): Promise<void> {
      const cacheKey = scopedKey(key);
      await memory.set(cacheKey, value);

      try {
        const db = await getDatabase();
        if (!db) {
          return;
        }
        const tx = db.transaction(STORE_NAME, "readwrite");
        const done = transactionDone(tx);
        const store = tx.objectStore(STORE_NAME);
        const updatedAt = Date.now() * 1000 + (monotonicStamp++ % 1000);
        await requestToPromise(store.put({ key: cacheKey, value, updatedAt }));
        const total = await requestToPromise(store.count());
        if (total > maxEntries) {
          await evictOldestEntries(store, total - maxEntries);
        }
        await done;
      } catch {
        // Ignore quota / clone / storage-blocked errors; in-memory cache still holds the value.
      }
    },
  };
}
