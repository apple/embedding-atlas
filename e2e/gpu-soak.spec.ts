/**
 * GPU stability soak — reproduces the Metal "ignored submissions" cascade
 * the user hit in the Electron desktop app after loading the 75M places
 * file. The bug surfaces only after first render, when:
 *
 *   1. The 2 s deferred density refinement fires and reactively updates
 *      ``maxDensity`` / ``categoryCount`` (causing buffer reallocs).
 *   2. The user pans/zooms, forcing fresh scatter queries with new
 *      arrow buffers and viewport-derived dataflow updates.
 *
 * The shared WebGPU code path lives in ``packages/component`` and runs
 * identically inside the standalone backend, frontend-only distro, and
 * desktop sidecar. Reproducing here against the standalone backend is
 * the cheapest, most deterministic harness.
 *
 * The new ``window.__atlasGpuErrors`` array (installed by
 * ``requestWebGPUDevice``) records every uncapturederror + device.lost
 * event with timestamp. A passing run leaves it empty.
 *
 * Skipped unless GPU_SOAK_PARQUET is set.
 *
 * Usage::
 *
 *   GPU_SOAK_PARQUET=/abs/path/to/big.parquet \
 *     npx playwright test e2e/gpu-soak.spec.ts \
 *     --headed --workers=1 --project=perf-chrome
 *
 * Tunables::
 *   GPU_SOAK_DURATION_MS  default 60000 — total interaction window
 *   GPU_SOAK_LON  default lon
 *   GPU_SOAK_LAT  default lat
 */

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcess } from "child_process";
import { writeFileSync, mkdirSync } from "fs";
import path from "path";
import { waitForServer, teardown } from "./helpers.js";
import { E2E_CONSTANTS } from "../playwright.config.js";

const PARQUET = process.env.GPU_SOAK_PARQUET;
const X_COL = process.env.GPU_SOAK_LON ?? "lon";
const Y_COL = process.env.GPU_SOAK_LAT ?? "lat";
const DURATION_MS = Number(process.env.GPU_SOAK_DURATION_MS ?? 60_000);
const TAG = process.env.GPU_SOAK_TAG ?? `soak-${Date.now()}`;
const BASE_URL = `http://localhost:${E2E_CONSTANTS.SERVER_PORT}`;

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
      X_COL,
      "--y",
      Y_COL,
      "--port",
      String(SERVER_PORT),
      "--no-auto-port",
      "--static",
      STATIC_DIR,
      "--disable-projection",
      "--no-mcp",
    ],
    { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, PYTHONUNBUFFERED: "1", GSA_DEBUG_SQL: "1" } },
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
  await waitForServer(`${BASE_URL}/data/metadata.json`, 10 * 60 * 1000);
});

test.afterAll(async () => {
  await teardown(server);
});

test("gpu soak — no metal cascade after deferred refinement + pan/zoom", async ({ page }) => {
  test.setTimeout(20 * 60 * 1000);

  const browserLines: string[] = [];
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    const t = msg.type();
    const text = msg.text();
    browserLines.push(`[${t}] ${text}`);
    if (t === "error") consoleErrors.push(text);
    if (
      /atlas-stage|atlas-gpu|first-big-render|deferred-density|scatter|RangeError|out of memory|ArrayBuffer/i.test(text)
    ) {
      process.stdout.write(`[browser] ${text}\n`);
    }
  });
  page.on("pageerror", (err) => {
    consoleErrors.push(`pageerror: ${err.message}`);
    process.stderr.write(`[browser-err] ${err.message}\n`);
  });

  // 1. Load the viewer and wait for first big render. ``?perf=1``
  // toggles the perf-recorder gating so first-big-render-* logs fire.
  const t0 = Date.now();
  await page.goto(`${BASE_URL}/?perf=1`, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: 6 * 60 * 1000,
    polling: 250,
  });
  const tFirstFrame = Date.now() - t0;
  console.log(`[soak] first-big-render-gpu-done at +${tFirstFrame}ms`);

  // Sanity: GPU device info exposed and at least one canvas non-zero.
  const gpuInfo = await page.evaluate(() => (window as any).__atlasGpuDeviceInfo ?? null);
  console.log(`[soak] gpu device info: ${JSON.stringify(gpuInfo)}`);
  const canvasInfo = await page.evaluate(() => {
    const cs = Array.from(document.querySelectorAll("canvas"));
    return cs.map((c) => ({ w: (c as HTMLCanvasElement).width, h: (c as HTMLCanvasElement).height }));
  });
  console.log(`[soak] canvases at first frame: ${JSON.stringify(canvasInfo)}`);
  expect(canvasInfo.some((c) => c.w > 0 && c.h > 0)).toBe(true);

  // 2. Locate the largest canvas — that's the WebGPU scatter surface.
  const targetBox = await page.evaluate(() => {
    const cs = Array.from(document.querySelectorAll("canvas"));
    let best: { x: number; y: number; w: number; h: number } | null = null;
    let bestArea = 0;
    for (const c of cs) {
      const r = (c as HTMLCanvasElement).getBoundingClientRect();
      const a = r.width * r.height;
      if (a > bestArea) {
        bestArea = a;
        best = { x: r.x, y: r.y, w: r.width, h: r.height };
      }
    }
    return best;
  });
  expect(targetBox, "no canvas found").not.toBeNull();
  const cx = targetBox!.x + targetBox!.w / 2;
  const cy = targetBox!.y + targetBox!.h / 2;
  console.log(`[soak] driving interactions on canvas centered at (${cx.toFixed(0)},${cy.toFixed(0)})`);

  // Capture the canvas IMMEDIATELY at first frame, before any pan/zoom — this
  // is what the user sees on cold-load and is the right thing to snapshot
  // for visual regression. The post-soak screenshot is dominated by where
  // the soak's wheel-zoom happened to land.
  {
    const outDir = path.resolve(__dirname, "perf-results");
    mkdirSync(outDir, { recursive: true });
    await page.screenshot({ path: path.join(outDir, `gpu-soak-${TAG}-firstframe.png`), fullPage: false });
  }

  // 3. Soak loop: drag-pan + wheel-zoom for the full DURATION_MS. We
  // intentionally cross the t+2s mark where the deferred density
  // refinement reassigns categoryCount/maxDensity — the suspected
  // trigger of the destroy-while-in-flight cascade.
  await page.mouse.move(cx, cy);
  const tStart = Date.now();
  let dragging = false;
  let frame = 0;
  while (Date.now() - tStart < DURATION_MS) {
    frame++;
    const phase = ((Date.now() - tStart) / DURATION_MS) * 2 * Math.PI;
    const dx = Math.cos(phase * 6) * 200;
    const dy = Math.sin(phase * 4) * 150;
    if (frame % 60 === 0) {
      // Periodic mid-soak status
      const errs = await page.evaluate(() => (window as any).__atlasGpuErrors?.length ?? 0);
      console.log(`[soak] +${Date.now() - tStart}ms frame=${frame} gpuErrors=${errs}`);
    }
    if (frame % 25 === 0) {
      // Wheel-zoom occasionally — exercises mosaic re-query at a new
      // viewport scale, which yields a fresh scatter SQL not in the
      // prewarm cache.
      await page.mouse.wheel(0, frame % 50 === 0 ? -120 : 120);
    } else {
      // Drag-pan. Toggle drag state every ~30 frames so we mix CSS-pan
      // (held) and committed pan (released) code paths.
      if (!dragging && frame % 30 === 0) {
        await page.mouse.down();
        dragging = true;
      } else if (dragging && frame % 30 === 15) {
        await page.mouse.up();
        dragging = false;
      }
      await page.mouse.move(cx + dx, cy + dy);
    }
    await page.waitForTimeout(16);
  }
  if (dragging) await page.mouse.up();

  // 4. Drain — give pending command buffers and any error events a final
  // chance to flush before we sample.
  await page.waitForTimeout(2000);

  // 5. Collect verdict.
  const final = await page.evaluate(() => ({
    gpuErrors: (window as any).__atlasGpuErrors ?? [],
    deviceInfo: (window as any).__atlasGpuDeviceInfo ?? null,
    canvases: Array.from(document.querySelectorAll("canvas")).map((c) => ({
      w: (c as HTMLCanvasElement).width,
      h: (c as HTMLCanvasElement).height,
    })),
  }));

  // Persist artifacts so I can inspect across iterations.
  const outDir = path.resolve(__dirname, "perf-results");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    path.join(outDir, `gpu-soak-${TAG}.json`),
    JSON.stringify(
      {
        tag: TAG,
        parquet: PARQUET,
        durationMs: DURATION_MS,
        tFirstFrameMs: tFirstFrame,
        gpuErrors: final.gpuErrors,
        deviceInfo: final.deviceInfo,
        canvases: final.canvases,
        consoleErrorCount: consoleErrors.length,
        consoleErrorsSample: consoleErrors.slice(0, 25),
        browserLinesTail: browserLines.slice(-100),
        serverLinesTail: serverLines.slice(-50),
      },
      null,
      2,
    ),
  );
  await page.screenshot({ path: path.join(outDir, `gpu-soak-${TAG}.png`), fullPage: false });

  console.log(`[soak] FINAL — gpuErrors=${final.gpuErrors.length} consoleErrors=${consoleErrors.length}`);
  if (final.gpuErrors.length) {
    for (const e of final.gpuErrors) {
      console.log(`  - ${JSON.stringify(e)}`);
    }
  }

  // 6. Assertions: zero GPU errors and the canvas is still alive.
  expect(final.gpuErrors.length, "WebGPU uncapturederror / device-lost during soak").toBe(0);
  expect(
    final.canvases.some((c) => c.w > 0 && c.h > 0),
    "no live canvas after soak (likely Metal cascade nuked it)",
  ).toBe(true);

  // Console-error gate: tolerate a handful of mosaic warnings, fail on
  // Metal/WebGPU specific signatures that indicate a real fault.
  const fatalPatterns = /ignored submissions|kIOGPUCommand|device.*lost|WebGPU.*error|Failed to.*GPU/i;
  const fatal = consoleErrors.filter((e) => fatalPatterns.test(e));
  expect(fatal, `fatal console errors: ${JSON.stringify(fatal)}`).toEqual([]);
});
