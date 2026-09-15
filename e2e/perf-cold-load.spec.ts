/**
 * Granular cold-load benchmark — captures every server-side and browser-side
 * timing log to pinpoint the bottleneck on the path from URL load → first
 * rendered point.
 *
 * Skipped unless PERF_PARQUET_FILE is set.
 *
 * Usage:
 *   PERF_PARQUET_FILE=/abs/path/to/big.parquet \
 *     npx playwright test e2e/perf-cold-load.spec.ts \
 *     --headed --workers=1 --project=perf-chrome
 */

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcess } from "child_process";
import path from "path";
import { writeFileSync, mkdirSync } from "fs";
import { waitForServer, teardown, waitForCanvas } from "./helpers.js";
import { E2E_CONSTANTS } from "../playwright.config.js";

const PARQUET = process.env.PERF_PARQUET_FILE;
const BASE_URL = `http://localhost:${E2E_CONSTANTS.SERVER_PORT}`;
const TAG = process.env.PERF_TAG ?? `cold-${Date.now()}`;
const ITERATIONS = Number(process.env.PERF_ITERATIONS ?? 1);

let server: ChildProcess | undefined;
const serverLines: string[] = [];

test.beforeAll(async () => {
  if (!PARQUET) {
    test.skip();
    return;
  }
  const { BACKEND_DIR, STATIC_DIR, SERVER_PORT } = E2E_CONSTANTS;
  server = spawn(
    "uv",
    [
      "run",
      "--directory",
      BACKEND_DIR,
      "geospatial-atlas",
      PARQUET,
      "--x",
      "lon",
      "--y",
      "lat",
      "--port",
      String(SERVER_PORT),
      "--no-auto-port",
      "--static",
      STATIC_DIR,
      "--disable-projection",
      "--no-mcp",
    ],
    {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, GSA_DEBUG_SQL: "1", PYTHONUNBUFFERED: "1" },
    },
  );
  server.stdout?.on("data", (b) => {
    const s = b.toString();
    serverLines.push(s);
    process.stdout.write(`[server] ${s}`);
  });
  server.stderr?.on("data", (b) => {
    const s = b.toString();
    serverLines.push(s);
    process.stderr.write(`[server-err] ${s}`);
  });
  await waitForServer(`${BASE_URL}/data/metadata.json`, 5 * 60 * 1000);
});

test.afterAll(async () => {
  await teardown(server);
});

test("cold-load granular", async ({ page }) => {
  test.setTimeout(20 * 60 * 1000);

  const browserLines: string[] = [];
  page.on("console", (msg) => {
    const t = msg.type();
    const text = msg.text();
    browserLines.push(`[${t}] ${text}`);
    // Always echo perf-relevant signals so we see them live.
    if (
      text.includes("[scatter]") ||
      text.includes("[mosaic]") ||
      text.includes("[atlas") ||
      text.includes("[bg-mat]") ||
      text.includes("WebGPU") ||
      text.includes("buffer") ||
      t === "error" ||
      t === "warning"
    ) {
      console.log(`[browser:${t}] ${text}`);
    }
  });

  const summaries: any[] = [];

  for (let i = 0; i < ITERATIONS; i++) {
    console.log(`\n=== Iteration ${i + 1}/${ITERATIONS} ===`);
    // Reset server-side state by re-loading. We can't truly cold-restart the
    // python server between iterations from inside one test, but cache
    // invalidation via URL hash query forces re-fetch on the JS side.
    if (process.env.PERF_DISABLE_COLUMN_CHARTS) {
      await page.addInitScript(() => {
        (window as any).__atlasDisableColumnDiscovery = true;
      });
    }
    const t0 = Date.now();
    await page.goto(`${BASE_URL}?perf=1&iter=${i}_${Date.now()}`);
    const tCanvas = await waitForCanvas(page, 5 * 60 * 1000).then(() => Date.now());
    await page.waitForFunction(
      () => {
        const s = (window as any).__atlasPerf?.summary?.();
        return s != null && s.count > 0 && s.pointCount > 0;
      },
      null,
      { timeout: 5 * 60 * 1000, polling: 250 },
    );
    const tFirstFrame = Date.now();
    const summary = await page.evaluate(() => (window as any).__atlasPerf?.summary?.() ?? null);
    summaries.push({
      iter: i,
      goto_to_canvas_ms: tCanvas - t0,
      goto_to_first_frame_ms: tFirstFrame - t0,
      summary,
    });
    console.log(`Iter ${i}: goto→canvas=${tCanvas - t0}ms, goto→first-frame=${tFirstFrame - t0}ms`);
    if (summary) {
      console.log(`  pointCount=${summary.pointCount} count=${summary.count} fps=${summary.fps?.toFixed?.(2)}`);
    }
  }

  // Persist to file.
  const outDir = path.resolve(__dirname, "perf-results");
  mkdirSync(outDir, { recursive: true });
  const report = {
    tag: TAG,
    parquet: PARQUET,
    iterations: summaries,
    serverLines: serverLines
      .join("")
      .split("\n")
      .filter((s) => s.length),
    browserLines,
    timestamp: new Date().toISOString(),
  };
  writeFileSync(path.join(outDir, `${TAG}.json`), JSON.stringify(report, null, 2));

  console.log("\n========== COLD-LOAD SUMMARY ==========");
  for (const s of summaries) {
    console.log(`iter ${s.iter}: canvas=${s.goto_to_canvas_ms}ms first-frame=${s.goto_to_first_frame_ms}ms`);
  }
  console.log("=======================================\n");

  expect(summaries[0].summary).not.toBeNull();
});
