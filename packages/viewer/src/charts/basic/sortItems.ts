// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface SortableItem {
  x: string;
  total: number;
  selected: number;
  isSpecial: boolean;
}

/**
 * Sorts items for display in count plots.
 * Regular items are sorted by the specified field and order.
 * Special items (null, other) are kept at the end, unsorted.
 *
 * @param items - Array of items to sort
 * @param firstSpecialIndex - Index where special items start (-1 if no special items)
 * @param sortBy - Field to sort by ("total" or "selected"), defaults to "total"
 * @param sortOrder - Sort order ("asc" or "desc"), defaults to "desc"
 * @returns Sorted array with regular items sorted and special items at the end
 */
export function sortItems(
  items: SortableItem[],
  firstSpecialIndex: number,
  sortBy: "total" | "selected" | undefined,
  sortOrder: "asc" | "desc" | undefined,
): SortableItem[] {
  // Split into regular items and special items using slice (more efficient than filter)
  let splitIndex = firstSpecialIndex === -1 ? items.length : firstSpecialIndex;
  let regularItems = items.slice(0, splitIndex);
  let specialItems = items.slice(splitIndex);

  // Sort regular items
  let sortField = sortBy ?? "total";
  let order = sortOrder ?? "desc";
  regularItems = regularItems.slice().sort((a, b) => {
    let diff = a[sortField] - b[sortField];
    return order === "asc" ? diff : -diff;
  });

  // Special items stay at the end, unsorted
  return [...regularItems, ...specialItems];
}
