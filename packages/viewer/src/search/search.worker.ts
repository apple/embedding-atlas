// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createWorkerRuntime } from "@embedding-atlas/utils";

import { SearchIndex } from "./search-index.js";

let { handler, registerClass } = createWorkerRuntime();

onmessage = handler;

export type { SearchIndex };

registerClass("SearchIndex", () => new SearchIndex());
