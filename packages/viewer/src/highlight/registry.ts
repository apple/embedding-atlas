// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { interpolateRgb, rgb } from "d3";

/**
 * Global registry of CSS Custom Highlight buckets and the stylesheets that
 * paint them. Highlights are quantized into `levels` buckets; each bucket is a
 * single document-global `Highlight` object (a maplike of `Range`s) styled via
 * `::highlight(ea-hl-<k>)`.
 *
 * Styling is split into two parts:
 *  - The `::highlight()` *rules* obey normal style scoping and must be injected
 *    into every tree root that contains highlighted text (document or shadow
 *    root). See {@link ensureStylesFor}.
 *  - The color *values* are CSS custom properties defined once at the document
 *    level (`:root` for light, `.dark` for dark). Custom properties inherit —
 *    including across shadow boundaries — so they reach text inside shadow
 *    roots, and `::highlight` inherits them from the originating element. Dark
 *    mode is therefore automatic via a `.dark` ancestor class; no `@media`, no
 *    re-registering highlights on theme change.
 */

const PREFIX = "ea-hl";
const RULES_STYLE_ID = "ea-highlight-styles";
const VARS_STYLE_ID = "ea-highlight-vars";

export type HighlightInterpolator = (t: number) => string;

export interface HighlightScheme {
  /** Interpolator sampled for the light palette. */
  light: HighlightInterpolator;
  /** Interpolator sampled for the dark palette. */
  dark: HighlightInterpolator;
}

/** Medium orange the highlight background fades up to at full intensity. */
const HIGHLIGHT_COLOR = "#f59e0b";
/** Opacity at full intensity. Kept well below 1 so highlights stay subtle and
 * the black/light or white/dark text on top remains readable without us ever
 * touching the text color. */
const MAX_ALPHA = 0.5;

/**
 * A d3 interpolator from fully transparent to a medium orange, ramping only the
 * alpha channel (the rgb stays constant, so the hue never shifts). Used for both
 * light and dark palettes: a translucent orange reads correctly over a light or
 * a dark background alike.
 */
function alphaRamp(color: string, maxAlpha: number): HighlightInterpolator {
  const from = rgb(color);
  from.opacity = 0;
  const to = rgb(color);
  to.opacity = maxAlpha;
  return interpolateRgb(from, to);
}

export const defaultScheme: HighlightScheme = {
  light: alphaRamp(HIGHLIGHT_COLOR, MAX_ALPHA),
  dark: alphaRamp(HIGHLIGHT_COLOR, MAX_ALPHA),
};

/** Whether the CSS Custom Highlight API is available in this environment. */
export const highlightApiSupported: boolean =
  typeof Highlight !== "undefined" && typeof CSS !== "undefined" && "highlights" in CSS;

let warned = false;
export function warnUnsupportedOnce(): void {
  if (!warned) {
    warned = true;
    console.warn("[highlight] CSS Custom Highlight API is not supported; highlighting is disabled.");
  }
}

let currentLevels = 0;
let buckets: Highlight[] = [];
/** Bumped whenever the bucket count changes, to force per-root rule refresh. */
let rulesGeneration = 0;

let varsScheme: HighlightScheme | null = null;
let varsLevels = -1;
let varsStyleEl: HTMLStyleElement | null = null;

/** Background color for bucket `k` of `levels`. The interpolator ramps alpha,
 * so the text underneath keeps its own color — we never set a foreground. */
function bgFor(interp: HighlightInterpolator, k: number, levels: number): string {
  // Spread buckets across the whole ramp: bucket 0 maps to t = 0 (fully
  // transparent, so a zero score paints nothing) and the top bucket to t = 1
  // (full intensity).
  const t = levels <= 1 ? 1 : k / (levels - 1);
  return rgb(interp(t)).formatRgb();
}

/**
 * Ensure `levels` highlight buckets exist and the color variables match
 * `scheme`. Rebuilds buckets if the level count changed.
 */
export function ensureBuckets(levels: number, scheme: HighlightScheme = defaultScheme): void {
  if (!highlightApiSupported) {
    return;
  }
  if (levels !== currentLevels) {
    for (let k = 0; k < buckets.length; k++) {
      CSS.highlights.delete(`${PREFIX}-${k}`);
    }
    buckets = [];
    for (let k = 0; k < levels; k++) {
      const highlight = new Highlight();
      // Higher score paints on top where ranges overlap.
      highlight.priority = k;
      CSS.highlights.set(`${PREFIX}-${k}`, highlight);
      buckets.push(highlight);
    }
    currentLevels = levels;
    rulesGeneration++;
  }
  ensureVarStyle(levels, scheme);
}

/** Define the document-level color variables for light and dark modes. */
function ensureVarStyle(levels: number, scheme: HighlightScheme): void {
  if (varsStyleEl != null && varsScheme === scheme && varsLevels === levels) {
    return;
  }
  const light: string[] = [];
  const dark: string[] = [];
  for (let k = 0; k < levels; k++) {
    light.push(`--${PREFIX}-bg-${k}:${bgFor(scheme.light, k, levels)};`);
    dark.push(`--${PREFIX}-bg-${k}:${bgFor(scheme.dark, k, levels)};`);
  }
  const css = `:root{${light.join("")}}\n.dark{${dark.join("")}}`;
  if (varsStyleEl == null) {
    varsStyleEl = document.createElement("style");
    varsStyleEl.id = VARS_STYLE_ID;
    document.head.appendChild(varsStyleEl);
  }
  varsStyleEl.textContent = css;
  varsScheme = scheme;
  varsLevels = levels;
}

/**
 * Ensure the `::highlight()` rules are present in `root`'s tree scope. Works for
 * both the document (rules go to `document.head`) and shadow roots (rules go to
 * the root itself), including nested shadow roots.
 */
export function ensureStylesFor(root: Document | ShadowRoot): void {
  if (!highlightApiSupported || currentLevels === 0) {
    return;
  }
  const host: ParentNode = root instanceof Document ? root.head : root;
  let styleEl = host.querySelector(`#${RULES_STYLE_ID}`) as HTMLStyleElement | null;
  if (styleEl != null && styleEl.dataset.generation === String(rulesGeneration)) {
    return;
  }
  if (styleEl == null) {
    styleEl = document.createElement("style");
    styleEl.id = RULES_STYLE_ID;
    host.appendChild(styleEl);
  }
  const rules: string[] = [];
  for (let k = 0; k < currentLevels; k++) {
    // Background only — text keeps its own color (black in light, white in dark).
    rules.push(`::highlight(${PREFIX}-${k}){background-color:var(--${PREFIX}-bg-${k});}`);
  }
  styleEl.textContent = rules.join("\n");
  styleEl.dataset.generation = String(rulesGeneration);
}

/** The `Highlight` object for bucket `k`, or `null` if out of range. */
export function bucketHighlight(k: number): Highlight | null {
  return buckets[k] ?? null;
}
