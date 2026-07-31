// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Simplification pass over bucketed highlight spans. Scores are quantized into a
 * handful of discrete buckets, so neighboring spans (e.g. the per-token spans of
 * an embedding scorer) frequently land in the same bucket. Collapsing each run
 * of same-bucket spans into a single span sharply cuts the number of CSS Custom
 * Highlight ranges the browser has to track.
 */

export interface BucketedSpan {
  /** Inclusive start character offset within the text. */
  start: number;
  /** Exclusive end character offset within the text. */
  end: number;
  /** Heatmap bucket index. */
  bucket: number;
}

/**
 * Merge consecutive same-bucket spans into single spans, bridging gaps that
 * contain only whitespace. Spans need not be sorted. Two spans are merged only
 * when they share a bucket and the text between them is empty or all whitespace,
 * so distinct highlighted regions separated by real text never fuse. Painting a
 * whitespace gap the same color as the words around it is visually identical to
 * leaving it blank for a background heatmap, so this is effectively lossless.
 */
export function mergeBucketedSpans(spans: BucketedSpan[], text: string): BucketedSpan[] {
  if (spans.length <= 1) {
    return spans;
  }
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: BucketedSpan[] = [];
  let run: BucketedSpan = { ...sorted[0] };
  for (let i = 1; i < sorted.length; i++) {
    const span = sorted[i];
    // `slice` yields "" when the next span overlaps or abuts the run, which the
    // whitespace test accepts — so overlapping/adjacent same-bucket spans merge
    // too.
    const gapIsWhitespace = /^\s*$/.test(text.slice(run.end, span.start));
    if (span.bucket === run.bucket && gapIsWhitespace) {
      if (span.end > run.end) {
        run.end = span.end;
      }
    } else {
      merged.push(run);
      run = { ...span };
    }
  }
  merged.push(run);
  return merged;
}
