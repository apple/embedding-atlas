// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Snippet } from "svelte";

export interface LayoutSpec {
  /** Layout type */
  type: string;
  /** Layout name */
  name: string;
  /** IDs of charts in this layout */
  chartIds: string[];
}

export interface LayoutProps {
  /** The layout id */
  layout: string;

  /** A snippet that renders a given chart. */
  chartView: Snippet<
    [{ id: string; width?: number | "container"; height?: number | "container"; mode?: "view" | "edit" }]
  >;
}

export type LayoutOptionsProps = Omit<LayoutProps, "chartView">;
