// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { derived, writable, type Readable, type Writable } from "svelte/store";

type Stores = Readable<any> | [Readable<any>, ...Readable<any>[]];
type StoresValues<T> = T extends Readable<infer U> ? U : { [K in keyof T]: T[K] extends Readable<infer U> ? U : never };

/**
 * Wraps a Readable store so that subscribers are only notified when the
 * value actually changes by strict reference equality (`!==`).
 *
 * Svelte's built-in stores use `safe_not_equal`, which always considers
 * objects and functions as "changed" even when the reference is the same.
 * This wrapper suppresses those redundant notifications.
 */
function dedup<T>(store: Readable<T>): Readable<T> {
  return {
    subscribe(run) {
      let first = true;
      let current: T;
      return store.subscribe((value) => {
        if (first || value !== current) {
          first = false;
          current = value;
          run(value);
        }
      });
    },
  };
}

/**
 * A writable store that only notifies subscribers when the value changes
 * by strict reference equality (`!==`).
 *
 * Unlike Svelte's built-in `writable`, calling `set` or `update` with the
 * same object reference will **not** trigger subscriber callbacks. This is
 * useful for stores holding objects or arrays where identity-based change
 * detection is sufficient.
 *
 * For primitive values (numbers, strings, booleans), this behaves
 * identically to the built-in `writable` since `safe_not_equal` already
 * skips equal primitives.
 */
export function stableWritable<T>(value: T): Writable<T> {
  const inner = writable(value);
  const readable = dedup(inner);
  return {
    set: inner.set,
    update: inner.update,
    subscribe: readable.subscribe,
  };
}

/**
 * A derived store that only notifies subscribers when the derived value
 * changes by strict reference equality (`!==`).
 *
 * Unlike Svelte's built-in `derived`, if the derivation function returns
 * the same object reference after an upstream change, subscribers will
 * **not** be re-notified. This prevents unnecessary cascading updates
 * when a derivation produces the same result.
 *
 * @example
 * ```ts
 * const items = stableWritable([1, 2, 3]);
 * const count = stableDerived(items, ($items) => $items.length);
 * // Replacing items with a new array of the same length
 * // will not notify count's subscribers.
 * ```
 */
export function stableDerived<S extends Stores, T>(
  stores: S,
  fn: (values: StoresValues<S>) => T,
  initialValue?: T,
): Readable<T> {
  return dedup(derived(stores, fn, initialValue));
}

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
