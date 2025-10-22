// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ScaleType } from "../common/types.js";

export interface BoxPlotSpec {
  type: "box-plot";
  title?: string;
  data: {
    x: string;
    y: string;
  };
  xBinCount?: number;
  xScaleType?: ScaleType | null;
  yScaleType?: ScaleType | null;
}

export interface CountPlotSpec {
  type: "count-plot" | "count-plot-list";
  title?: string;
  data: {
    field: string;
  };
  expanded?: boolean;
  percentage?: boolean;
}

export interface HistogramSpec {
  type: "histogram";
  title?: string;
  data: {
    field: string;
  };
  binCount?: number;
  xScaleType?: ScaleType | null;
}

export interface Histogram2DSpec {
  type: "histogram-2d";
  title?: string;
  data: {
    x: string;
    y: string;
  };
  xBinCount?: number;
  yBinCount?: number;
  xScaleType?: ScaleType | null;
  yScaleType?: ScaleType | null;
  normalization?: "x" | "y" | null;
}

export interface HistogramStackSpec {
  type: "histogram-stack";
  title?: string;
  data: {
    x: string;
    group: string;
  };
  xBinCount?: number;
  groupBinCount?: number;
  xScaleType?: ScaleType | null;
  normalization?: "x" | null;
}

export interface PredicatesSpec {
  type: "predicates";
  title?: string;
  items?: { name: string; predicate: string }[];
}
