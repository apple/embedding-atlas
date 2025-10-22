// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type Writable } from "svelte/store";

/**
 * Wraps an existing Svelte writable store and returns a new store
 * that ignores its own updates when notifying subscribers.
 *
 * Subscribers of the returned store will **not be called** when
 * the wrapped store is updated through this wrapper's `set` or `update` methods.
 * Updates made directly to the original store will still notify subscribers.
 *
 * This is useful when a component both writes to and subscribes from the same store,
 * and you want to prevent its own writes from re-triggering its callbacks.
 */
export function isolatedWritable<T>(wrapped: Writable<T>): Writable<T> {
  let counter = 0;
  function withGate(perform: () => void) {
    counter += 1;
    try {
      perform();
    } finally {
      counter -= 1;
    }
  }
  return {
    set(value: T) {
      withGate(() => {
        wrapped.set(value);
      });
    },
    update(updater: (value: T) => T) {
      withGate(() => {
        wrapped.update(updater);
      });
    },
    subscribe(run: (value: T) => void) {
      return wrapped.subscribe((value) => {
        if (counter == 0) {
          run(value);
        }
      });
    },
  };
}
