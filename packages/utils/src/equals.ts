// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Performs a deep equality comparison between two values.
 *
 * Recursively compares objects and arrays by their contents rather than reference.
 * Handles null values and primitive types appropriately.
 *
 * @param a - First value to compare
 * @param b - Second value to compare
 * @returns `true` if the values are deeply equal, `false` otherwise
 */
export function deepEquals(a: any, b: any): boolean {
  if (a === b) {
    return true;
  }
  // If either of them is null or not an object, they are not equal
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
    return false;
  }
  // If the objects/arrays have a different number of keys, they are not equal
  if (Object.keys(a).length !== Object.keys(b).length) {
    return false;
  }
  for (let key in a) {
    if (b.hasOwnProperty(key)) {
      if (!deepEquals(a[key], b[key])) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

/**
 * Creates a memoized version of a function that uses deep equality comparison.
 *
 * The memoized function caches its most recent result and returns the cached value
 * if the new result is deeply equal to the previous one. This is useful for
 * maintaining referential stability when the computed value hasn't meaningfully changed.
 *
 * @example
 * ```ts
 * // Use with Svelte's $derived.by() to maintain referential stability
 * let items = $derived.by(deepMemo(() => computeItems(source)));
 * ```
 *
 * @template Args - The argument types of the function
 * @template T - The return type of the function
 * @param fn - The function to memoize
 * @returns A memoized version of the function that returns the cached result if deeply equal to the previous result
 */
export function deepMemo<Args extends any[], T>(fn: (...args: Args) => T): (...args: Args) => T {
  let memo: T | undefined = undefined;
  return (...args) => {
    let current = fn(...args);
    if (memo !== undefined && deepEquals(current, memo)) {
      return memo;
    }
    memo = current;
    return current;
  };
}
