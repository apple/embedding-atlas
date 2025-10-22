// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { deepEquals } from "./equals.js";

/**
 * Recursively merges updates into a value immutably.
 *
 * For each key in `updates`:
 * - If `updates[key] === undefined`, the key is removed from the result.
 * - If `updates[key] !== value[key]`, the key is updated.
 * - If both `value[key]` and `updates[key]` are plain objects, the merge happens recursively.
 *
 * Arrays and other non-object values are treated atomically and replaced if they differ.
 * The function never mutates `value` or `updates`. Only new objects are created if a change is needed.
 * If no changes are required, `undefined` is returned.
 *
 * @template T - The type of the original value.
 * @param {T} value - The original value to merge updates into.
 * @param {Partial<T>} updates - The updates to apply.
 * @returns {T | undefined} A new value with updates applied, or `undefined` if no updates were necessary.
 *
 * @example
 * const value = { a: 1, b: { c: 2, d: 3 } };
 * const updates = { b: { c: 4 }, a: undefined };
 * const result = mergeUpdates(value, updates);
 * // result: { b: { c: 4, d: 3 } }
 */
export function mergeUpdates<T>(value: T, updates: Partial<T>): T | undefined {
  if (isObject(value) && isObject(updates)) {
    let result: T | undefined = undefined;
    for (let key in updates) {
      if (updates[key] !== value[key]) {
        if (updates[key] === undefined) {
          if (result === undefined) {
            result = { ...value };
          }
          delete result[key];
        } else {
          let m = mergeUpdates(value[key], updates[key]);
          if (m !== undefined) {
            if (result === undefined) {
              result = { ...value };
            }
            result[key] = m;
          }
        }
      }
    }
    return result;
  } else if (!deepEquals(value, updates)) {
    return updates as any;
  } else {
    return undefined;
  }
}

export function applyUpdatesIfNeeded<T, R>(
  value: T,
  updates: Partial<T>,
  mode: "merge" | "replace",
  setter: (value: T) => R,
): R | undefined {
  switch (mode) {
    case "merge": {
      let r = mergeUpdates(value, updates);
      if (r !== undefined) {
        return setter(r);
      } else {
        return undefined;
      }
    }
    case "replace": {
      if (value !== updates) {
        return setter(updates as any);
      } else {
        return undefined;
      }
    }
  }
}

export function applyUpdatesForKeyIfNeeded<T, R>(
  records: Record<string, T>,
  key: string,
  update: Partial<T>,
  mode: "merge" | "replace",
  setter: (value: Record<string, T>) => R,
): R | undefined {
  switch (mode) {
    case "merge": {
      let r = mergeUpdates(records, { [key]: update as any });
      if (r !== undefined) {
        return setter(r);
      } else {
        return undefined;
      }
    }
    case "replace": {
      if (records[key] !== update) {
        let newValue = { ...records };
        newValue[key] = update as any;
        return setter(newValue);
      } else {
        return undefined;
      }
    }
  }
}

function isObject(value: any): boolean {
  return value != null && typeof value == "object" && !Array.isArray(value);
}
