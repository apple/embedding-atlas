// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ScaleType } from "../spec/spec.js";

export { type ScaleType };

export interface ScaleConfig {
  /** Scale type. */
  type: ScaleType;

  /** The data domain. */
  domain: any[];

  /** Special values. All represented as strings. */
  specialValues?: string[];

  /** symlog function constant. */
  constant?: number;

  /**
   * Scale range. Currently do not apply for x and y scales.
   * For size scales, this should be [min, max] size.
   * For nominal color scales, this should be a list of colors.
   * For quantitative color scales, this should be a predefined interpolate scheme, or a list of colors to interpolate.
   */
  range?: (string | number)[] | string;
}

export interface AxisConfig {
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

export interface Extents {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface Label {
  text: string;
  fontFamily: string;
  fontSize: number;
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

export interface ConcreteScale<Output> {
  type: ScaleType;

  domain: any[];
  specialValues: string[];

  apply(value: any): Output;
}

export type LinearPosition = [number, number];

export interface IntermediatePositionScale {
  labels: Label[];
  gridLines: GridLine[];
  ticks: Tick[];

  base: {
    rangeBands: [LinearPosition, LinearPosition][];
    apply(value: any): LinearPosition;
    applyBand(value: any): [LinearPosition, LinearPosition];
    invert(position: number, range: [number, number], type?: "string" | "number"): any;
  };

  concrete(range: [number, number]): ConcretePositionScale;
}

export interface ConcretePositionScale extends ConcreteScale<number> {
  domain: any[];
  specialValues: string[];

  range: [number, number];
  rangeBands: [number, number][];

  apply(value: any): number;
  applyBand(value: any): [number, number];
  invert(position: number, type?: "string" | "number"): any;
}

export interface PlotLayout {
  width?: number;
  height?: number;
  plotWidth?: number;
  plotHeight?: number;
  plotAspectRatio?: number;
}

export interface XYFrameProxy {
  plotWidth: number;
  plotHeight: number;
  scale: {
    x?: ConcretePositionScale;
    y?: ConcretePositionScale;
    color?: ConcreteScale<string>;
    size?: ConcreteScale<number>;
  };
}
