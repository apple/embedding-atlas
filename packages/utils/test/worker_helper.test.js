// Copyright (c) 2026 Apple Inc. Licensed under MIT License.

import { Worker } from "node:worker_threads";

import { connectWorker } from "@embedding-atlas/utils";
import { afterEach, describe, expect, it } from "vitest";

const runtimeUrl = new URL("../dist/worker_helper.js", import.meta.url).href;
const threads = [];

function startWorker(body) {
  const thread = new Worker(
    `
      const { parentPort } = require("node:worker_threads");
      globalThis.postMessage = (message, options) => parentPort.postMessage(message, options?.transfer);
      parentPort.on("message", (data) => globalThis.onmessage?.({ data }));
      (async () => {
        const { createWorkerRuntime } = await import(${JSON.stringify(runtimeUrl)});
        ${body}
      })().catch((error) => { setImmediate(() => { throw error; }); });
    `,
    { eval: true },
  );
  threads.push(thread);

  // Adapt Node's real worker events to the browser Worker interface.
  const worker = {
    onmessage: null,
    onerror: null,
    postMessage(message, options) {
      thread.postMessage(message, options?.transfer);
    },
  };
  thread.on("message", (data) => worker.onmessage?.({ data }));
  thread.on("error", (error) => worker.onerror?.({ error, message: error.message }));
  return worker;
}

afterEach(async () => {
  await Promise.all(threads.splice(0).map((thread) => thread.terminate()));
});

describe("connectWorker", () => {
  it("completes the handshake and forwards RPC results", async () => {
    const worker = startWorker(`
      const runtime = createWorkerRuntime();
      globalThis.onmessage = runtime.handler;
      runtime.registerFunction("sum", (a, b) => a + b);
    `);
    const connection = await connectWorker(worker);
    await expect(connection.call("sum", 2, 3)).resolves.toBe(5);
  });

  it("rejects the handshake when the worker fails to start", async () => {
    const worker = startWorker(`throw new Error("startup failed");`);
    await expect(connectWorker(worker)).rejects.toThrow("startup failed");
  });

  it("rejects pending and subsequent calls when the worker crashes", async () => {
    const worker = startWorker(`
      const runtime = createWorkerRuntime();
      globalThis.onmessage = runtime.handler;
      runtime.registerFunction("wait", () => new Promise(() => {}));
      runtime.registerFunction("crash", () => new Promise(() => {
        setImmediate(() => { throw new Error("runtime failed"); });
      }));
    `);
    const connection = await connectWorker(worker);
    await Promise.all([
      expect(connection.call("wait")).rejects.toThrow("runtime failed"),
      expect(connection.call("crash")).rejects.toThrow("runtime failed"),
    ]);
    await expect(connection.call("wait")).rejects.toThrow("runtime failed");
  });
});
