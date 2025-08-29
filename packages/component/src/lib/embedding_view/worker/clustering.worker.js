// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import {
  dynamicLabelPlacement,
  findClusters,
  textSummarizerAdd,
  textSummarizerCreate,
  textSummarizerDestroy,
  textSummarizerSummarize,
} from "./worker_functions.js";

/** @type Record<string, (...args: any[]) => any> */
let functions = {
  dynamicLabelPlacement,
  findClusters,
  textSummarizerCreate,
  textSummarizerAdd,
  textSummarizerDestroy,
  textSummarizerSummarize,
};

onmessage = async (msg) => {
  if (functions[msg.data.name]) {
    let func = functions[msg.data.name];
    let args = msg.data.payload;
    let result = func(...args);
    if (result instanceof Promise) {
      result = await result;
    }
    postMessage({ id: msg.data.id, payload: result });
  }
};

postMessage({
  ready: true,
});

export {};
