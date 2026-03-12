// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ColumnStyle } from "../../renderers/types.js";

export type SortOrder = { column: string; direction: "ascending" | "descending" }[];

export interface InstancesSpec {
  type: "instances";
  title?: string;

  /**
   * Columns to show in the instance view.
   * If specified, the table and card views will be limited to the given columns, and custom card template will only receive the given columns as data.
   * If not specified, include all columns from the dataset (or query result is `query` is specified).
   */
  columns?: string[];

  /** Sort order. If not specified, use original data order. */
  sort?: SortOrder;

  /** View mode, defaults to "table" */
  viewMode?: "table" | "cards";

  /** Optional custom SQL query to filter or transform the data */
  query?: string;

  /** Number of items per page, defaults to 100 */
  pageSize?: number;

  /** Default height in pixels, defaults to 500. This value is used when the view's height is flexible. */
  defaultHeight?: number;

  /** Column styles specific to this instance view. These will override global column styles. */
  columnStyles?: Record<string, ColumnStyle>;

  /**
   * Liquid template for the cards (rendered with liquidjs).
   * Use a Liquid template instead of column styles for custom cards.
   * If not specified, use the tooltip view as card.
   */
  cardTemplate?: string;
}

export interface InstancesState {
  offset?: number;
}
