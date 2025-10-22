// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

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
