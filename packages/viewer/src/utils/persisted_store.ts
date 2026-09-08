// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { writable, type Writable } from "svelte/store";

/**
 * A `writable` store hydrated from `localStorage` on creation and persisted
 * back on every change. Use for global user preferences (model config, UI
 * prefs) that should survive a page reload.
 *
 * On corrupt JSON, write failures, or environments without `window` (SSR),
 * the store falls back gracefully — never throws from this module.
 */
export function persistedWritable<T>(key: string, defaultValue: T): Writable<T> {
  let initial = defaultValue;
  if (typeof window !== "undefined") {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw != null) {
        initial = JSON.parse(raw) as T;
      }
    } catch {
      // corrupt JSON or storage disabled — keep default
    }
  }
  const store = writable<T>(initial);
  if (typeof window !== "undefined") {
    store.subscribe((value) => {
      try {
        window.localStorage.setItem(key, JSON.stringify(value));
      } catch {
        // quota exceeded or storage disabled — silently ignore
      }
    });
  }
  return store;
}
