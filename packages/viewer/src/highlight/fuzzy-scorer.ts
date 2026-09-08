// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import uFuzzy from "@leeoniya/ufuzzy";

import type { HighlightScorer, HighlightSpan } from "./scheduler.js";

/**
 * Default highlight scorer backed by uFuzzy. Unlike a bitap matcher (e.g.
 * fuse.js), uFuzzy matches the needle's terms as bounded contiguous runs rather
 * than scattering across any shared characters, so a search for "cherry" no
 * longer lights up "ec" in "objective" or "rr" in "narrative".
 *
 * uFuzzy hands back match offsets directly: `info.ranges[k]` is a flat
 * `[start0, end0, start1, end1, …]` array (exclusive ends) for the haystack
 * entry `info.idx[k]`, which maps straight onto {@link HighlightSpan}. Matching
 * is binary, so every span gets `value: 1`.
 *
 * The batch scorer runs one `filter`/`info` pass over all texts and maps the
 * ranges back per text by `info.idx`. `options` pass straight through to the
 * uFuzzy constructor (`intraMode`, `intraIns`, `interLft`/`interRgt` bounds,
 * `unicode`, …).
 */

export type FuzzyMatchOptions = NonNullable<ConstructorParameters<typeof uFuzzy>[0]>;

const DEFAULT_OPTIONS: FuzzyMatchOptions = {
  intraMode: 1, // SingleError: tolerate at most one typo per term...
  intraIns: 1, // ...an extra inserted character,
  intraSub: 1, // ...a substituted character,
  intraTrn: 1, // ...a transposition,
  intraDel: 1, // ...or an omitted character.
};

function createMatcher(options?: FuzzyMatchOptions): uFuzzy {
  return new uFuzzy({ ...DEFAULT_OPTIONS, ...options });
}

function spansForRanges(ranges: readonly number[] | undefined): HighlightSpan[] {
  if (ranges == null) {
    return [];
  }
  const spans: HighlightSpan[] = [];
  // uFuzzy ranges are flat [start0, end0, start1, end1, …] with exclusive ends,
  // matching HighlightSpan's exclusive `end`.
  for (let i = 0; i + 1 < ranges.length; i += 2) {
    spans.push({ start: ranges[i], end: ranges[i + 1], value: 1 });
  }
  return spans;
}

/** Run a single filter/info pass and return spans per text, aligned by index. */
function scoreBatch(matcher: uFuzzy, texts: string[], query: string): HighlightSpan[][] {
  const result: HighlightSpan[][] = texts.map(() => []);
  const idxs = matcher.filter(texts, query);
  if (idxs == null || idxs.length === 0) {
    return result;
  }
  const info = matcher.info(idxs, texts, query);
  // `info` may drop entries that fail bound checks, so iterate its own idx list.
  for (let k = 0; k < info.idx.length; k++) {
    result[info.idx[k]] = spansForRanges(info.ranges[k]);
  }
  return result;
}

/**
 * Find fuzzy matches of `query` in `text` as highlight spans. Each matched run
 * becomes one span with `value: 1`. Returns an empty array for an empty query or
 * no match.
 */
export function fuzzyMatch(
  text: string,
  query: string | null | undefined,
  options?: FuzzyMatchOptions,
): HighlightSpan[] {
  if (query == null || query.trim() === "") {
    return [];
  }
  return scoreBatch(createMatcher(options), [text], query)[0];
}

/**
 * Build a {@link HighlightScorer} backed by uFuzzy. The action passes the
 * current `query` to the returned scorer, so a single instance handles changing
 * queries.
 *
 * @example
 * <p use:highlight={{ scorer: fuzzyHighlightScorer(), query }}>…</p>
 */
export function fuzzyHighlightScorer(options?: FuzzyMatchOptions): HighlightScorer<string> {
  return async (texts, query) => {
    if (query == null || query.trim() === "") {
      return texts.map(() => []);
    }
    return scoreBatch(createMatcher(options), texts, query);
  };
}
