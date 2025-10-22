// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type ScaleType = "linear" | "log" | "symlog" | "band";

export interface ScaleSpec {
  /** Scale type. */
  type: ScaleType;

  /** The data domain. */
  domain: any[];

  /** Special values. All represented as strings. */
  specialValues?: string[];

  /** symlog constant. */
  constant?: number;
}

export interface AxisSpec {
  values?: any[];
  /** Desired number of ticks. Default 5. */
  desiredTickCount?: number;
  /** Extend scale to ticks. Default true. */
  extendScaleToTicks?: boolean;
  /** Padding to label. */
  labelPadding?: number;
}

export interface Extents {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface Label {
  text: string;
  padding: number;
  value: any;
  level: number;
  orientation: "horizontal" | "vertical";
  size: { width: number; height: number };
}

export interface GridLine {
  value: any;
  level: number;
}

export interface Tick {
  value: any;
  level: number;
}

export interface IntermedateScale {
  extents: Extents;
  labels: Label[];
  gridLines: GridLine[];
  ticks: Tick[];
  concrete(range: [number, number]): ConcreteScale;
}

export interface ConcreteScale {
  domain: any[];
  specialValues: string[];

  range: [number, number];
  rangeBands: [number, number][];

  apply(value: any): number;
  applyBand(value: any): [number, number];
  invert(position: number, type?: "string" | "number"): any;
}

export interface XYFrameProxy {
  plotWidth: number;
  plotHeight: number;
  xScale?: ConcreteScale;
  yScale?: ConcreteScale;
}
