// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export function findUnusedId(object: Record<string, unknown>, prefix: string = ""): string {
  let counter = 1;
  let id: string;
  do {
    id = prefix + counter;
    counter++;
  } while (id in object);
  return id;
}
