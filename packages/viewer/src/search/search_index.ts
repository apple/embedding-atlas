// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Charset, Index, type IndexOptions } from "flexsearch";

const options: IndexOptions = {
  tokenize: "forward",
  encoder: Charset.LatinBalance,
};

/**
 * Parse a query string for an exact-phrase request.
 *
 * A query wrapped in double quotes (e.g. `"aldi"`) means the user wants an
 * exact, case-insensitive substring match instead of the default fuzzy token
 * search. The default encoder maps similar-looking words to the same token
 * (for example "aldi" and "aldea"), which is great for fuzzy recall but
 * surfaces unwanted matches when the user knows exactly what they are looking
 * for. Quoting opts out of that behavior.
 *
 * Returns the inner phrase when the query is a quoted exact phrase, otherwise
 * null. An empty quoted string (`""`) is treated as not a phrase so the caller
 * falls back to the regular path.
 */
export function parseExactPhrase(query: string): string | null {
  if (query.length >= 2 && query.startsWith('"') && query.endsWith('"')) {
    let inner = query.slice(1, -1);
    return inner.length > 0 ? inner : null;
  }
  return null;
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
    let phrase = parseExactPhrase(query);
    if (phrase != null) {
      return this.exactSearch(phrase, limit);
    }
    return this.index.search(query, { limit });
  }

  private exactSearch(phrase: string, limit: number): (string | number)[] {
    let needle = phrase.toLowerCase();
    let result: (string | number)[] = [];
    for (let [id, text] of this.texts) {
      if (text.toLowerCase().includes(needle)) {
        result.push(id);
        if (result.length >= limit) {
          break;
        }
      }
    }
    return result;
  }
}
