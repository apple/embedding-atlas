// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { mergeBucketedSpans, type BucketedSpan } from "./merge.js";
import {
  bucketHighlight,
  ensureBuckets,
  ensureStylesFor,
  highlightApiSupported,
  warnUnsupportedOnce,
  type HighlightScheme,
} from "./registry.js";
import { queryKeyOf, schedule, type HighlightScorer, type HighlightSpan } from "./scheduler.js";
import { DEFAULT_EXCLUDE, extractSegments, rangeForSegment } from "./text-map.js";

export type { HighlightScheme } from "./registry.js";
export type { HighlightScorer, HighlightSpan } from "./scheduler.js";

export interface HighlightOptions<Q = unknown> {
  /**
   * Batched scoring function. Receives the unique texts collected across all
   * `use:highlight` instances sharing this scorer + query in one debounce
   * window, plus the current `query`, and returns one {@link HighlightSpan}
   * array per input text.
   */
  scorer: HighlightScorer<Q>;
  /** Recompute trigger, scorer argument, and batch discriminator (e.g. the search query). */
  query?: Q;
  /** Disable highlighting (clears existing highlights). Default `true`. */
  enabled?: boolean;
  /** Number of heatmap buckets. Default `7`. */
  levels?: number;
  /**
   * Domain used to normalize `value` into `[0, 1]`. `"auto"` normalizes per
   * result using its min/max. Default `[0, 1]`.
   */
  domain?: [number, number] | "auto";
  /** Skip spans whose normalized score is below this. Default `0`. */
  minScore?: number;
  /** d3 interpolators for the light and dark palettes. */
  scheme?: HighlightScheme;
  /** Batch flush / recompute debounce in milliseconds. Default `150`. */
  debounce?: number;
  /** Re-extract and re-highlight on DOM mutations. Default `true`. */
  observe?: boolean;
  /** Selector for subtrees whose text is ignored. Default `"script,style,.no-highlight"`. */
  exclude?: string;
  /**
   * Selector limiting which subtrees are highlighted: when set, only text inside
   * a matching element is scored (`exclude` still wins). Default `undefined`
   * highlights everything.
   */
  include?: string;
}

const DEFAULT_LEVELS = 7;
const DEFAULT_DEBOUNCE = 150;

/**
 * Svelte action that paints per-token model scores as a background heatmap over
 * the element's text using the CSS Custom Highlight API. Never mutates the DOM.
 *
 * Scoring runs only while the root element is on screen: an off-screen element's
 * work is deferred and replayed when it next scrolls into view, so highlighting
 * many virtualized/scrolled rows doesn't score elements the user can't see.
 *
 * @example
 * <p use:highlight={{ scorer, query }}>…</p>
 */
export function highlight<Q = unknown>(node: HTMLElement, options: HighlightOptions<Q>) {
  let opts = options;
  let generation = 0;
  let controller: AbortController | null = null;
  let myRanges: { bucket: number; range: Range }[] = [];
  let observer: MutationObserver | null = null;
  let observerTimer: ReturnType<typeof setTimeout> | null = null;
  let visibilityObserver: IntersectionObserver | null = null;
  // Whether the root element is currently on screen. Starts pessimistic: the
  // IntersectionObserver reports the real state on its first (async) callback.
  let visible = false;
  // A run was requested while off screen; replay it once the element is visible.
  let pending = false;

  function clearMine(): void {
    for (const { bucket, range } of myRanges) {
      bucketHighlight(bucket)?.delete(range);
    }
    myRanges = [];
  }

  async function run(): Promise<void> {
    if (!(opts.enabled ?? true)) {
      clearMine();
      return;
    }
    if (!highlightApiSupported) {
      warnUnsupportedOnce();
      return;
    }

    const levels = opts.levels ?? DEFAULT_LEVELS;
    ensureBuckets(levels, opts.scheme);
    ensureStylesFor(node.getRootNode() as Document | ShadowRoot);

    const segments = extractSegments(node, opts.exclude ?? DEFAULT_EXCLUDE, opts.include).filter(
      (segment) => segment.text.length > 0,
    );
    if (segments.length === 0) {
      clearMine();
      return;
    }

    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    const myGeneration = ++generation;

    // Each text node is scored independently so unrelated runs of text (e.g.
    // separate table cells or tooltip fields) never share a scoring context.
    let perSegment: HighlightSpan[][];
    try {
      perSegment = await Promise.all(
        segments.map((segment) =>
          schedule(
            opts.scorer,
            queryKeyOf(opts.query),
            opts.query,
            segment.text,
            opts.debounce ?? DEFAULT_DEBOUNCE,
            signal,
          ),
        ),
      );
    } catch {
      // Aborted (superseded) or scorer rejected — leave current highlights as-is.
      return;
    }
    if (myGeneration !== generation) {
      return; // A newer run started while awaiting.
    }

    const [min, max] = resolveDomain(opts.domain, perSegment.flat());
    const minScore = opts.minScore ?? 0;

    clearMine();
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      // Quantize each span to a bucket, then merge consecutive same-bucket spans
      // so the browser tracks far fewer highlight ranges.
      const bucketed: BucketedSpan[] = [];
      for (const span of perSegment[i]) {
        const norm = max > min ? (span.value - min) / (max - min) : 0;
        if (norm < minScore) {
          continue;
        }
        const clamped = norm < 0 ? 0 : norm > 1 ? 1 : norm;
        const bucket = Math.min(levels - 1, Math.floor(clamped * levels));
        // Bucket 0 is the transparent floor (a zero score paints nothing), so
        // skip it — no point adding invisible ranges to the highlight registry.
        if (bucket <= 0) {
          continue;
        }
        bucketed.push({ start: span.start, end: span.end, bucket });
      }
      for (const span of mergeBucketedSpans(bucketed, segment.text)) {
        const range = rangeForSegment(segment, span.start, span.end);
        if (range == null) {
          continue;
        }
        const highlightObject = bucketHighlight(span.bucket);
        if (highlightObject == null) {
          continue;
        }
        highlightObject.add(range);
        myRanges.push({ bucket: span.bucket, range });
      }
    }
  }

  /**
   * Run only when the element is on screen; otherwise remember that a run is due
   * and replay it when the element next becomes visible. Re-extraction reads the
   * current DOM, so a single deferred flag captures any number of intervening
   * param/DOM changes.
   */
  function requestRun(): void {
    if (!visible) {
      // Params/DOM may have changed since the in-flight run captured them; abort it
      // and bump the generation so its scorer can't resolve and paint stale ranges
      // (mirrors the MutationObserver path). The deferred run re-reads everything.
      invalidateInFlight();
      pending = true;
      return;
    }
    pending = false;
    void run();
  }

  function scheduleObserverRun(): void {
    if (observerTimer != null) {
      clearTimeout(observerTimer);
    }
    observerTimer = setTimeout(() => {
      observerTimer = null;
      requestRun();
    }, opts.debounce ?? DEFAULT_DEBOUNCE);
  }

  /**
   * Discard any in-flight run immediately. {@link extractSegments} captured live
   * `Text` nodes plus offsets computed against their content at that instant; a
   * DOM mutation can shrink or replace those nodes while the (debounced, possibly
   * slow) scoring is still awaiting, leaving the captured offsets stale. Aborting
   * the scorer signal makes `run`'s await reject into its catch, and bumping the
   * generation makes the post-await guard drop the result if it had already
   * resolved — either way no stale ranges get painted against mutated nodes. The
   * fresh run that re-reads the DOM is scheduled separately (debounced).
   */
  function invalidateInFlight(): void {
    controller?.abort();
    controller = null;
    generation++;
  }

  function startObserver(): void {
    if (observer != null) {
      return;
    }
    // Safe from feedback loops: we never mutate the observed DOM.
    observer = new MutationObserver(() => {
      invalidateInFlight();
      scheduleObserverRun();
    });
    observer.observe(node, { characterData: true, childList: true, subtree: true });
  }

  function stopObserver(): void {
    observer?.disconnect();
    observer = null;
    if (observerTimer != null) {
      clearTimeout(observerTimer);
      observerTimer = null;
    }
  }

  /**
   * Track the root element's visibility (the element only, not its descendants)
   * so scoring runs are deferred for off-screen elements until they scroll in.
   */
  function startVisibilityTracking(): void {
    if (visibilityObserver != null) {
      return;
    }
    if (typeof IntersectionObserver === "undefined") {
      // No IntersectionObserver (e.g. non-DOM env): assume always visible.
      visible = true;
      return;
    }
    visibilityObserver = new IntersectionObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (entry.isIntersecting === visible) {
        return;
      }
      visible = entry.isIntersecting;
      if (visible && pending) {
        pending = false;
        void run();
      }
    });
    visibilityObserver.observe(node);
  }

  function stopVisibilityTracking(): void {
    visibilityObserver?.disconnect();
    visibilityObserver = null;
  }

  startVisibilityTracking();
  if (opts.observe ?? true) {
    startObserver();
  }
  requestRun();

  return {
    update(next: HighlightOptions<Q>) {
      opts = next;
      if (next.observe ?? true) {
        startObserver();
      } else {
        stopObserver();
      }
      requestRun();
    },
    destroy() {
      controller?.abort();
      stopObserver();
      stopVisibilityTracking();
      clearMine();
    },
  };
}

function resolveDomain(domain: [number, number] | "auto" | undefined, spans: HighlightSpan[]): [number, number] {
  if (domain == null) {
    return [0, 1];
  }
  if (domain !== "auto") {
    return domain;
  }
  let min = Infinity;
  let max = -Infinity;
  for (const span of spans) {
    if (span.value < min) {
      min = span.value;
    }
    if (span.value > max) {
      max = span.value;
    }
  }
  return Number.isFinite(min) && max > min ? [min, max] : [0, 1];
}
