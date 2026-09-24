// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createClassComponent } from "svelte/legacy";

import Component from "./ParallelCoordinatesView.svelte";

import type { ParallelCoordinatesViewThemeConfig } from "./theme.js";

export interface ParallelCoordinatesViewProps {
  /** The data. */
  data: {
    /**
     * Column-major values: one `Float32Array` per axis, each of length `numRows`.
     * Values must be in `[0, 1]` where `0` maps to the top of the plot and `1` to the bottom.
     * The number of axes is `values.length`.
     */
    values: Float32Array[];

    /**
     * Per-row color value.
     *  - `Uint8Array`: a `0–255` index directly into `categoryColors` (categorical).
     *  - `Float32Array`: a `0–1` position interpolated across `categoryColors`
     *    (`0` → first color, `1` → last color).
     *  - `null`/absent: a single color is used.
     */
    colorValue?: Float32Array | Uint8Array | null;
  };

  /**
   * Tick labels per axis. `axisLabels[i]` is the list of ticks for axis `i`.
   * Each tick has a `value` in `[0, 1]` (`0` = top) and a display `label`. An optional `priority`
   * (default `0`) controls overlap resolution: when labels would overlap on the Y axis, higher-priority
   * labels are kept and lower-priority ones are dropped (ties broken by the order within the list).
   */
  axisLabels?: ({ value: number; label: string; priority?: number }[] | null | undefined)[] | null;

  /** Axis titles. */
  axisTitles?: (string | null | undefined)[] | null;

  /** The colors for the categories / the palette to interpolate. If not specified, default colors are used. */
  categoryColors?: string[] | null;

  /**
   * Opacity control. When `autoOpacity` is enabled this multiplies the automatic opacity; otherwise it is
   * the exact opacity passed to the renderer. Default `1`.
   */
  opacity?: number | null;

  /**
   * Automatically choose the opacity from the number of points (so the density stays readable regardless
   * of dataset size); `opacity` then acts as a multiplier. When disabled, `opacity` is used exactly.
   * Default `true`.
   */
  autoOpacity?: boolean | null;

  /** Number of density bins per axis. Defaults to ~1 bin per 2px of axis height (clamped 32–256). */
  binCount?: number | null;

  /** The width of the view. */
  width?: number | null;

  /** The height of the view. */
  height?: number | null;

  /** The pixel ratio of the view. */
  pixelRatio?: number | null;

  /** The color scheme. Default `light`. */
  colorScheme?: "light" | "dark" | null;

  /** Theme overrides for axis line, tick, and label colors (with optional light/dark variants). */
  theme?: ParallelCoordinatesViewThemeConfig | null;

  /**
   * Per-axis brush selections, indexed like `data.values` (one entry per axis). Each entry is a
   * `[from, to]` range in normalized [0, 1] coordinates (0 = top); `null` (or a missing entry)
   * means no brush on that axis. A range may be given as `from > to`. Controlled — the view
   * renders what is passed here and does not filter or change the rendered data.
   */
  brush?: ([number, number] | null)[] | null;

  /**
   * Called when the user creates, moves, resizes, or clears a brush. Receives the full updated
   * per-axis brush array, always normalized so `from <= to`. Purely a notification — the view
   * does not filter or otherwise change the rendered ribbons.
   */
  onBrushChange?: ((brush: ([number, number] | null)[]) => void) | null;
}

export class ParallelCoordinatesView {
  private component: any;
  private currentProps: ParallelCoordinatesViewProps;

  constructor(target: HTMLElement, props: ParallelCoordinatesViewProps) {
    this.currentProps = { ...props };
    this.component = createClassComponent({ component: Component, target: target, props: props });
  }

  update(props: Partial<ParallelCoordinatesViewProps>) {
    let updates: Partial<ParallelCoordinatesViewProps> = {};
    for (let key in props) {
      if ((props as any)[key] !== (this.currentProps as any)[key]) {
        (updates as any)[key] = (props as any)[key];
        (this.currentProps as any)[key] = (props as any)[key];
      }
    }
    this.component.$set(updates);
  }

  destroy() {
    this.component.$destroy();
  }
}
