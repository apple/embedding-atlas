// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/** Debounce the given function. */
export function debounce<T extends any[]>(func: (...args: T) => void, time: number = 1000): (...args: T) => void {
  let timeout: any | undefined = undefined;
  let perform = (...args: T) => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => {
      func(...args);
    }, time);
  };
  return perform;
}
