// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createWorkerRuntime } from "@embedding-atlas/utils";

import { EmbeddingProjector } from "./projector.js";
import { EmbeddingScorer } from "./scorer.js";
import { FeatureSimilarity } from "./similarity.js";

let { handler, registerClass } = createWorkerRuntime();

onmessage = handler;

registerClass("EmbeddingProjector", (options) => EmbeddingProjector.create(options));
registerClass("EmbeddingScorer", (options) => EmbeddingScorer.create(options));
registerClass("FeatureSimilarity", (options) => FeatureSimilarity.create(options));
