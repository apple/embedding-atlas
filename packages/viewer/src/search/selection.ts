// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

/** Add search result IDs as a cross-filter clause without replacing other filters. */
export function applySearchResultsSelection(
  filter: Selection,
  source: object,
  table: string,
  idColumn: string,
  ids: any[],
): void {
  const uniqueIds = Array.from(new Set(ids));
  if (uniqueIds.length === 0) return;

  filter.update({
    source,
    clients: new Set(),
    value: uniqueIds,
    predicate: SQL.isIn(
      SQL.column(idColumn, table),
      uniqueIds.map((id) => SQL.literal(id)),
    ),
    meta: { type: "point" },
  });
}
