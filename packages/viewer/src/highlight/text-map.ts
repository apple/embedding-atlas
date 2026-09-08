// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Collects an element's text content as a list of block-grouped segments,
 * without mutating the DOM. We walk the descendant text nodes (the reusable
 * idea borrowed from mark.js) but, unlike mark.js, we never split nodes.
 *
 * Text nodes are grouped by their nearest block-level container: a run of
 * inline content (e.g. `The <b>quick</b> fox`) becomes one segment so a scorer
 * sees the whole phrase as context, while text in separate blocks (table cells,
 * tooltip fields, paragraphs) stays in separate segments and is never blended.
 *
 * Each segment carries the composite text of its nodes plus per-node offsets, so
 * a character span within a segment maps back to a DOM `Range` that may cross
 * several nodes of that block via {@link rangeForSegment}.
 */

export interface SegmentNode {
  node: Text;
  /** Inclusive start offset of this node within the segment's composite text. */
  start: number;
  /** Exclusive end offset of this node within the segment's composite text. */
  end: number;
}

export interface TextSegment {
  /** The nearest block-level container shared by this segment's nodes. */
  container: Element;
  /** Composite text of the segment's nodes, in document order. */
  text: string;
  /** Per-node offset entries within `text`, in document order. */
  nodes: SegmentNode[];
}

export const DEFAULT_EXCLUDE = "script,style,.no-highlight";

/** Tag names treated as block-level boundaries when grouping text nodes. */
// prettier-ignore
const BLOCK_TAGS = new Set([
  "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "DD", "DETAILS", "DIALOG", "DIV", "DL", "DT",
  "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM", "H1", "H2", "H3", "H4", "H5", "H6",
  "HEADER", "HGROUP", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE", "SECTION",
  "TABLE", "TBODY", "TD", "TFOOT", "TH", "THEAD", "TR", "UL",
]);

/**
 * Nearest block-level ancestor of `node`, searching up to (but not past) `root`.
 * Returns `root` when no block element sits between the node and the root, so
 * inline content directly under the root groups together.
 */
function nearestBlock(node: Text, root: Element): Element {
  let element = node.parentElement;
  while (element != null && element !== root) {
    if (BLOCK_TAGS.has(element.tagName)) {
      return element;
    }
    element = element.parentElement;
  }
  return root;
}

/**
 * Walk `root`'s descendant text nodes and group them into {@link TextSegment}s
 * by their nearest block-level container. Text nodes whose parent matches (or is
 * inside something matching) the `exclude` selector are skipped. When `include`
 * is given, only text nodes inside an element matching it are kept (`exclude`
 * still wins on conflicts).
 */
export function extractSegments(root: Element, exclude: string = DEFAULT_EXCLUDE, include?: string): TextSegment[] {
  const excludeSelector = exclude.trim();
  const includeSelector = include?.trim() ?? "";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = (node as Text).parentElement;
      if (parent == null) {
        return NodeFilter.FILTER_REJECT;
      }
      if (excludeSelector !== "" && parent.closest(excludeSelector) != null) {
        return NodeFilter.FILTER_REJECT;
      }
      if (includeSelector !== "" && parent.closest(includeSelector) == null) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const segments: TextSegment[] = [];
  let current: TextSegment | null = null;
  for (let node = walker.nextNode(); node != null; node = walker.nextNode()) {
    const textNode = node as Text;
    const container = nearestBlock(textNode, root);
    // Consecutive nodes sharing a block container join one segment; a change of
    // container (crossing a block boundary) starts a new one.
    if (current == null || current.container !== container) {
      current = { container, text: "", nodes: [] };
      segments.push(current);
    }
    const start = current.text.length;
    current.text += textNode.data;
    current.nodes.push({ node: textNode, start, end: current.text.length });
  }
  return segments;
}

/** Binary-search for the node whose [start, end) range contains `offset`. */
function findNode(nodes: SegmentNode[], offset: number): SegmentNode | null {
  let lo = 0;
  let hi = nodes.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const entry = nodes[mid];
    if (offset < entry.start) {
      hi = mid - 1;
    } else if (offset >= entry.end) {
      lo = mid + 1;
    } else {
      return entry;
    }
  }
  return null;
}

/**
 * Build a DOM `Range` covering the character span `[start, end)` within
 * `segment`'s composite text. The range may cross several of the segment's text
 * nodes. Returns `null` for empty or out-of-bounds spans.
 */
export function rangeForSegment(segment: TextSegment, start: number, end: number): Range | null {
  start = Math.max(0, Math.floor(start));
  end = Math.min(segment.text.length, Math.floor(end));
  if (end <= start || segment.nodes.length === 0) {
    return null;
  }

  const startEntry = findNode(segment.nodes, start);
  // Use end - 1 so we land on the node containing the last included character.
  const endEntry = findNode(segment.nodes, end - 1);
  if (startEntry == null || endEntry == null) {
    return null;
  }

  const startOffset = start - startEntry.start;
  const endOffset = end - endEntry.start;

  // Offsets were computed against each node's text at extraction time. If a node
  // was mutated (e.g. shrank) while async scoring ran, an offset can now overrun
  // the live node and make setStart/setEnd throw IndexSizeError. Bail instead.
  if (startOffset > startEntry.node.length || endOffset > endEntry.node.length) {
    return null;
  }

  const range = document.createRange();
  range.setStart(startEntry.node, startOffset);
  range.setEnd(endEntry.node, endOffset);
  return range;
}
