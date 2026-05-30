// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { LayoutSpec } from "../layout.js";

export type Section = "embedding" | "table" | "chart";

export interface ListLayoutSpec extends LayoutSpec {
  /** Layout type */
  type: "list";

  /** Layout name */
  name: string;

  /** IDs of charts in this layout */
  chartIds: string[];

  showTable?: boolean;
  showEmbedding?: boolean;
  showCharts?: boolean;

  chartsOrder?: string[];
  chartVisibility?: Record<string, boolean>;
}
