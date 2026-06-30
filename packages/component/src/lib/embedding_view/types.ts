// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type DataPointID = string | number | bigint;

export interface DataPoint {
  x: number;
  y: number;
  /** Z coordinate, present for points in a 3D embedding view. */
  z?: number;
  category?: number;
  text?: string;
  identifier?: DataPointID;
  fields?: Record<string, any>;
  /**
   * Stable DuckDB rowid of the source row, attached when a 3D pick was resolved by
   * rowid. Lets coordinated-selection predicates target the exact row even without a
   * user identifier column, so duplicate/stacked points are not over-selected. A
   * number (rowids fit well within 2^53) so the selection stays JSON-serializable.
   */
  rowid?: number;
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
  /** Projects a data-space point to view (CSS) pixels. In 3D, pass `z` so overlay
   *  geometry rotates with the point cloud; in 2D `z` is ignored. */
  location: (x: number, y: number, z?: number) => { x: number; y: number };
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
