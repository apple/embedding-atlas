// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Index } from "flexsearch";

let index = new Index();

export interface ClearRequest {
  type: "clear";
  identifier: string;
}

export interface PointsRequest {
  type: "points";
  identifier: string;
  points: { id: string | number; text: string }[];
}

export interface QueryRequest {
  type: "query";
  identifier: string;
  query: string;
  limit: number;
}

self.onmessage = (e: MessageEvent<ClearRequest | PointsRequest | QueryRequest>) => {
  switch (e.data.type) {
    case "clear":
      index = new Index();
      postMessage({ identifier: e.data.identifier });
      break;
    case "points":
      for (let p of e.data.points) {
        index.add(p.id, p.text);
      }
      postMessage({ identifier: e.data.identifier });
      break;
    case "query":
      let result = index.search(e.data.query, { limit: e.data.limit });
      postMessage({ identifier: e.data.identifier, result: result });
      break;
  }
};
