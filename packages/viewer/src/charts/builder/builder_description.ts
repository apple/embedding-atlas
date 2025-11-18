// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { JSType } from "../../utils/database.js";
import type { ChartSpec } from "../spec/spec.js";

export interface Field {
  name: string;
  type: "continuous" | "discrete" | "discrete[]";
}

export type UIElement =
  | {
      field: {
        key: string;
        label: string;
        required?: boolean;
        types?: JSType[] | null;
      };
    }
  | {
      code: {
        key: string;
        language?: string;
        jsonSchema?: any;
      };
    }
  | {
      spec: { key: string };
    };

// Helpers to infer the type of values from the UI description.
type UIValue<E> = E extends { field: { key: infer K extends string; required: true } }
  ? { [P in K]: Field }
  : E extends { field: { key: infer K extends string } }
    ? { [P in K]: Field | undefined }
    : E extends { code: { key: infer K extends string } }
      ? { [P in K]: string }
      : E extends { spec: { key: infer K extends string } }
        ? { [P in K]: ChartSpec }
        : never;

type UnionToIntersection<U> = (U extends any ? (x: U) => void : never) extends (x: infer I) => void ? I : never;
type UIValues<A extends readonly UIElement[]> = UnionToIntersection<UIValue<A[number]>>;

export interface ChartBuilderDescription<Spec, UI extends readonly UIElement[]> {
  /** An icon for the chart type */
  icon: string;

  /** Description of the chart type  */
  description: string;

  /** UI elements */
  ui: UI;

  /** Whether to show a preview chart (default true) */
  preview?: boolean;

  /** A function to create a chart spec from the given values */
  create: (values: UIValues<UI>, context: { table: string; id: string }) => Spec | undefined;
}
