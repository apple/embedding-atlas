// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { SQLField } from "../spec/spec.js";

export interface CountPlotSpec {
  type: "count-plot";
  title?: string;

  data: {
    /** The data field */
    field: SQLField;

    /** Indicate if the field contains list[str] data, default false */
    isList?: boolean;
  };

  /** The max number of categories to show, default 10 */
  limit?: number;

  /** Labeling method, '%': percentage, '#': count, '#/#': selected count over total count */
  labels?: "%" | "#" | "#/#";

  /** Order the categories by total count, selected count, alphabetical, or custom order, default 'total-descending' */
  order?:
    | "total-descending"
    | "total-ascending"
    | "selected-descending"
    | "selected-ascending"
    | "alphabetical"
    | string[];
}

export interface PredicatesSpec {
  type: "predicates";
  title?: string;
  items?: { name: string; predicate: string }[];
}

export interface MarkdownSpec {
  type: "markdown";
  title?: string;
  content: string;
}

export interface ContentViewerSpec {
  type: "content-viewer";
  title?: string;
  field: string;
}
