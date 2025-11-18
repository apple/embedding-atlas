// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/** Mark type */
export type MarkType = "bar" | "rect" | "line" | "area" | "point" | "rule";

/** Field from the data table, can be a column name or a SQL expression */
export type SQLField = string | { sql: string };

/** Table name or SQL expression that produces a table */
export type SQLTable = string | { sql: string };

/** Data value */
export type DataValue = string | number | [number, number];

/** Encoding channel */
export type Channel = "x" | "y" | "color" | "size";

/** Mark attribute */
export type Attribute = "x" | "y" | "x1" | "x2" | "y1" | "y2" | "color" | "size" | "group";

export type AggregateFn =
  | "count"
  | "distinct"
  | "min"
  | "max"
  | "mean"
  | "average"
  | "median"
  | "stdev"
  | "stdevp"
  | "variance"
  | "variancep"
  | "sum"
  | "product"
  | "quantile"
  | "ecdf-value"
  | "ecdf-rank";

/** Interpolate method for line or area */
export type Interpolate =
  | "linear"
  | "cardinal"
  | "catmull-rom"
  | "natural"
  | "monotone"
  | "basis"
  | "step"
  | "step-before"
  | "step-after";

/** Encoding */
export type Encoding =
  | {
      /** The data field */
      field: SQLField;

      bin?: {
        /** Desired bin count */
        desiredCount?: number;
      };
    }
  | {
      /** Aggregate type */
      aggregate: AggregateFn | { sql: string };

      /** The data field for the aggregate */
      field?: SQLField;

      /** For "quantile" aggregate, the quantile value (0-1) */
      quantile?: number;

      /** Normalize the value by x or y */
      normalize?: "x" | "y";
    }
  | {
      value: DataValue;
    };

/** Mark dimension for width and height */
export type Dimension = { gap: number; clampToRatio?: number } | { ratio: number } | number;

/** Mark style */
export interface MarkStyle {
  /** Fill color. If `null`, disable fill. Default is based on mark type. */
  fillColor?: string | null;
  /** Fill opacity */
  fillOpacity?: number;

  /** Stroke color. If `null`, disable stroke. Default is based on mark type. */
  strokeColor?: string | null;
  /** Stroke width */
  strokeWidth?: number;
  /** Stroke opacity */
  strokeOpacity?: number;
  /** Stroke cap */
  strokeCap?: "butt" | "round" | "square";
  /** Stroke join */
  strokeJoin?: "round" | "miter-clip" | "miter" | "bevel";

  /** Paint order */
  paintOrder?: string;

  /** Opacity */
  opacity?: number;
}

export interface Layer {
  /** Data source, default to the main data table */
  from?: SQLTable;

  /** Filter the data. Use $filter to refer to the shared filter */
  filter?: "$filter";

  /** Mark type */
  mark: MarkType;

  /** Mark style */
  style?: MarkStyle;

  /**
   * z-index indicating the layer order. Default value is 0.
   * If the value is negative, the mark will be drawn below grid lines.
   */
  zIndex?: number;

  /** Orientation of bar marks */
  orientation?: "vertical" | "horizontal";

  /** Interpolate method for line and area marks */
  interpolate?: Interpolate;

  /** Width of bar / rect marks */
  width?: Dimension;

  /** Height of bar / rect marks */
  height?: Dimension;

  /** Encoding */
  encoding?: Partial<Record<Attribute, Encoding>>;
}

/** Scale type */
export type ScaleType = "linear" | "log" | "symlog" | "band";

/** Scale */
export interface Scale {
  /** Scale type for quantitative scales */
  type?: ScaleType;

  /** Scale domain */
  domain?: DataValue[];

  /** Special values. All represented as strings. */
  specialValues?: string[];

  /** symlog constant. */
  constant?: number;

  /**
   * Scale range. Currently do not apply for x and y scales.
   * For size scales, this should be [min, max] size.
   * For nominal color scales, this should be a list of colors.
   * For quantitative color scales, this should be a predefined interpolate scheme, or a list of colors to interpolate.
   */
  range?: (string | number)[] | string;
}

export interface Axis {
  /** Axis title */
  title?: string;

  /** Values for ticks, grid lines, and labels */
  values?: any[];

  /** Desired number of ticks. Default 5. */
  desiredTickCount?: number;

  /** Extend scale to ticks. Default true. */
  extendScaleToTicks?: boolean;

  /** Padding to label */
  labelPadding?: number;

  /** Label font family */
  labelFontFamily?: string;

  /** Label font size */
  labelFontSize?: number;

  /** Label max width */
  labelMaxWidth?: number;
}

/** Chart selection */
export interface Selection {
  encoding: "x" | "y" | "xy";
}

/** Widget for editing a chart */
export type Widget =
  | {
      type: "scale.type";
      channel: Channel;
    }
  | {
      type: "encoding.normalize";
      layer: number | number[];
      attribute: Attribute;
      options: ("x" | "y")[];
    };

/** Chart specification */
export interface ChartSpec {
  /** The title of the chart */
  title?: string;

  /** Size configuration */
  plotSize?: {
    /** Width of the plot area */
    width?: number;

    /** Height of the plot area */
    height?: number;

    /** Aspect ratio of the plot area */
    aspectRatio?: number;
  };

  /** Layers */
  layers?: Layer[];

  /** Scale configurations */
  scale?: Partial<Record<Channel, Scale>>;

  /** Axis configurations */
  axis?: Partial<Record<"x" | "y", Axis>>;

  /** Selections */
  selection?: Record<string, Selection>;

  /** Widgets */
  widgets?: Widget[];
}
