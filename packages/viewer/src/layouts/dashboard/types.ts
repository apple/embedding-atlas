// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type Placement } from "./placement.js";

export interface DashboardLayoutState {
  numRows?: number;
  numColumns?: number;
  grids?: Record<string, { placements?: Record<string, Placement>; order?: string[] }>;
}
