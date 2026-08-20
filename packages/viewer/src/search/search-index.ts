// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Charset, Index, type IndexOptions } from "flexsearch";

const options: IndexOptions = {
  tokenize: "forward",
  // LatinDefault performs case-folding and diacritic normalization without
  // phonetic merging. The fuzzier presets (LatinBalance/Advanced/Extra/Soundex)
  // collapse unrelated words that share a prefix and similar-sounding letters
  // (e.g. "aldi" and "aldea") into false-positive matches.
  encoder: Charset.LatinDefault,
};

export class SearchIndex {
  private index: Index;

  constructor() {
    this.index = new Index(options);
  }

  clear() {
    this.index.clear();
    this.index.cleanup();
    this.index = new Index(options);
  }

  addPoints(points: { id: string | number; text: string }[]) {
    for (let p of points) {
      this.index.add(p.id, p.text);
    }
  }

  query(query: string, limit: number): (string | number)[] {
    return this.index.search(query, { limit });
  }
}
