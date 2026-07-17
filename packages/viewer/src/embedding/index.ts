// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { connectWorker, type WorkerConnection, type WorkerProxy } from "@embedding-atlas/utils";

import type { EmbeddingProjector, EmbeddingProjectorArgs } from "./projector.js";
import type { EmbeddingScorer, EmbeddingScorerArgs } from "./scorer.js";

export type EmbeddingProjectorHandle = WorkerProxy<EmbeddingProjector>;
export type EmbeddingScorerHandle = WorkerProxy<EmbeddingScorer>;

/** Global worker instance, created as needed. */
let _connection: Promise<WorkerConnection> | null = null;

function connect(): Promise<WorkerConnection> {
  if (_connection == null) {
    let worker = new Worker(new URL("./embedding.worker.js", import.meta.url), { type: "module" });
    let p = connectWorker(worker);
    _connection = p;
    // Evict a failed connection so the next caller can retry instead of
    // reusing a permanently-rejected promise.
    p.catch(() => {
      if (_connection === p) {
        _connection = null;
      }
    });
  }
  return _connection;
}

/** Create an embedding projector that computes embedding and projects the result with UMAP. */
export async function createEmbeddingProjector(args: EmbeddingProjectorArgs): Promise<EmbeddingProjectorHandle> {
  let conn = await connect();
  return conn.create<EmbeddingProjector>("EmbeddingProjector", args);
}

/** Create an embedding scorer that computes character-level similarity scores. */
export async function createEmbeddingScorer(args: EmbeddingScorerArgs): Promise<EmbeddingScorerHandle> {
  let conn = await connect();
  return conn.create<EmbeddingScorer>("EmbeddingScorer", args);
}
