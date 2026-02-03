// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Wraps an async function so that only the latest invocation's result
 * will be passed to onResult. Earlier pending results are ignored. Errors are also ignored.
 */
export function latestAsync<Args extends any[], R>(
  asyncFn: (...args: Args) => Promise<R>,
  onResult: (result: R) => void,
): (...args: Args) => void {
  let latestId = 0;
  return (...args: Args) => {
    const id = ++latestId;
    asyncFn(...args)
      .then((result) => {
        if (id === latestId) {
          onResult(result);
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };
}
