// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface CountPlotSpec {
  type: "count-plot" | "count-plot-list";
  title?: string;
  data: {
    field: string;
  };
  expanded?: boolean;
  percentage?: boolean;
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
