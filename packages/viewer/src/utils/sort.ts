// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Sort items by the given order. If item not in the given order, sort by appearance.
 * Items with a given order should appear before items without.
 */
export function reorder(items: string[], order?: string[] | undefined): string[] {
  if (!order || order.length === 0) {
    return items;
  }
  let itemsSet = new Set(items);
  let orderSet = new Set(order);
  return [...order.filter((x) => itemsSet.has(x)), ...items.filter((x) => !orderSet.has(x))];
}
