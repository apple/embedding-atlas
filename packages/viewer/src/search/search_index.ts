// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Charset, Index, type IndexOptions } from "flexsearch";

const options: IndexOptions = {
  tokenize: "forward",
  encoder: Charset.LatinBalance,
};

/**
 * A query parsed into exact phrases and the remaining free text.
 *
 * Double-quoted runs become exact phrases, everything outside the quotes is
 * collected as free text. For example `"aldi" store` parses to one phrase
 * (`aldi`) plus the free text `store`.
 */
export interface ParsedQuery {
  phrases: string[];
  freeText: string;
}

/**
 * Parse a query into exact phrases and free text.
 *
 * A double-quoted run (e.g. `"aldi"`) means the user wants an exact,
 * case-insensitive substring match instead of the default fuzzy token search.
 * The default encoder maps similar-looking words to the same token (for example
 * "aldi" and "aldea"), which is great for fuzzy recall but surfaces unwanted
 * matches when the user knows exactly what they are looking for. Quoting opts
 * out of that behavior for the quoted run while leaving any unquoted words on
 * the fuzzy path, so `"aldi" store` requires the exact substring "aldi" and
 * fuzzy-matches "store".
 *
 * Empty quotes (`""`) contribute no phrase. An unterminated trailing quote is
 * treated as a literal character of the free text so a half-typed query still
 * searches.
 */
export function parseQuery(query: string): ParsedQuery {
  let phrases: string[] = [];
  let freeText: string[] = [];
  let rest = query;

  while (true) {
    let open = rest.indexOf('"');
    if (open < 0) {
      freeText.push(rest);
      break;
    }
    let close = rest.indexOf('"', open + 1);
    if (close < 0) {
      // No closing quote, keep the remainder as free text verbatim.
      freeText.push(rest);
      break;
    }
    freeText.push(rest.slice(0, open));
    let inner = rest.slice(open + 1, close);
    if (inner.length > 0) {
      phrases.push(inner);
    }
    rest = rest.slice(close + 1);
  }

  return { phrases, freeText: freeText.join(" ").trim() };
}

/**
 * Full text search index backed by flexsearch.
 *
 * In addition to the default fuzzy token search, the index keeps the original
 * text for each point so a quoted query can perform an exact, case-insensitive
 * substring match.
 */
export class SearchIndex {
  private index: Index;
  private texts: Map<string | number, string>;

  constructor() {
    this.index = new Index(options);
    this.texts = new Map();
  }

  clear() {
    this.index.clear();
    this.index.cleanup();
    this.index = new Index(options);
    this.texts = new Map();
  }

  addPoints(points: { id: string | number; text: string }[]) {
    for (let p of points) {
      this.index.add(p.id, p.text);
      this.texts.set(p.id, p.text);
    }
  }

  query(query: string, limit: number): (string | number)[] {
    let { phrases, freeText } = parseQuery(query);

    // No exact phrases: keep the original fuzzy path untouched.
    if (phrases.length === 0) {
      return this.index.search(freeText, { limit });
    }

    // Candidate ids that contain every required phrase as a substring. When
    // there is also free text, narrow to the fuzzy hits for that text so the
    // phrases act as a filter on top of the normal ranking.
    let candidates: Iterable<string | number>;
    if (freeText.length > 0) {
      candidates = this.index.search(freeText) as (string | number)[];
    } else {
      candidates = this.texts.keys();
    }

    let needles = phrases.map((p) => p.toLowerCase());
    let result: (string | number)[] = [];
    for (let id of candidates) {
      let text = this.texts.get(id);
      if (text == null) {
        continue;
      }
      let haystack = text.toLowerCase();
      if (needles.every((needle) => haystack.includes(needle))) {
        result.push(id);
        if (result.length >= limit) {
          break;
        }
      }
    }
    return result;
  }
}
