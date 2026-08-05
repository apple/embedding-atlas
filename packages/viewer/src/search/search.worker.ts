// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createWorkerRuntime } from "@embedding-atlas/utils";
import { Charset, Index, type IndexOptions } from "flexsearch";

let { handler, registerClass } = createWorkerRuntime();

onmessage = handler;

const options: IndexOptions = {
  tokenize: "forward",
  encoder: Charset.LatinBalance,
};

class SearchIndex {
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

export type { SearchIndex };

registerClass("SearchIndex", () => new SearchIndex());
