// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type DataPointID = string | number | bigint;

export interface DataPoint {
  x: number;
  y: number;
  category?: number;
  text?: string;
  identifier?: DataPointID;
  fields?: Record<string, any>;
}

export type DataField = string | { sql: string };

export interface Cache {
  get: (key: string) => Promise<any | null>;
  set: (key: string, value: any) => Promise<void>;
}

/** The content of a label: either a text string or an image with display dimensions (and optionally x, y coordinates). */
export type LabelContent = string | { x?: number; y?: number; image: string; width: number; height: number };

export interface Label {
  /** X coordinate. */
  x: number;
  /** Y coordinate. */
  y: number;
  /** Label content: a text string or an image reference. */
  content: LabelContent;
  /** Label level. The label will be shown around 2^level zoom factor. */
  level?: number | null;
  /** Placement priority. */
  priority?: number | null;
}

export interface OverlayProxy {
  location: (x: number, y: number) => { x: number; y: number };
  width: number;
  height: number;
}

type CustomComponentClass<N, P> = new (node: N, props: P) => { update?: (props: P) => void; destroy?: () => void };

export type CustomComponent<N, P> =
  | {
      class: CustomComponentClass<N, P & any>;
      props?: Record<string, any>;
    }
  | CustomComponentClass<N, P>;
