// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { LayoutSpec } from "../layout.js";

export interface Placement {
  /** Start x location (in grid index) of the chart */
  x: number;
  /** Start y location (in grid index) of the chart */
  y: number;
  /** Width (in grid cell count) of the chart */
  width: number;
  /** Height (in grid cell count) of the chart */
  height: number;
}

export interface DashboardLayoutSpec extends LayoutSpec {
  /** Layout type */
  type: "dashboard";

  /** Layout name */
  name: string;

  /** IDs of charts in this layout */
  chartIds: string[];

  /** The number of columns in the dashboard grid, default is 24 */
  numColumns?: number;

  /** The number of rows in the dashboard grid, default is 18 */
  numRows?: number;

  /**
   * The grid, keyed by `${numColumns}x${numRows}`, e.g., "24x18".
   * Charts can be placed in additional rows, but the user would need to scroll down to view them.
   */
  grids?: Record<
    string,
    {
      /** Placements of charts, keyed by chart ids, value is a Placement struct. */
      placements?: Record<string, Placement>;
      /** Order of the charts, affects overlapping charts. */
      order?: string[];
    }
  >;
}
