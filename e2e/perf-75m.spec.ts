/**
 * Performance baseline / regression test for the Geospatial Atlas viewer
 * against very large datasets (75M+ points).
 *
 * Skipped by default — requires:
 *   - PERF_PARQUET_FILE=/abs/path/to/big.parquet env var
 *   - WebGPU-capable browser (Chrome >= 113 with --enable-unsafe-webgpu)
 *
 * Usage:
 *   PERF_PARQUET_FILE=/Users/dome/work/overture/places_data/places_simplified.parquet \
 *     npx playwright test e2e/perf-75m.spec.ts --headed --workers=1 \
 *     --project=perf-chrome
 *
 * Output: writes per-frame summary to stdout AND e2e/perf-results.json so
 * subsequent runs can diff against a baseline.
 */

import { test, expect, type Page } from "@playwright/test";
import { type ChildProcess } from "child_process";
import { writeFileSync, mkdirSync } from "fs";
import path from "path";
import { startBackendServer, waitForServer, teardown, waitForCanvas } from "./helpers.js";
import { E2E_CONSTANTS } from "../playwright.config.js";

const PARQUET = process.env.PERF_PARQUET_FILE;
const BASE_URL = `http://localhost:${E2E_CONSTANTS.SERVER_PORT}`;
const PAN_SECONDS = Number(process.env.PERF_PAN_SECONDS ?? 5);
const PAN_STEPS_PER_SECOND = 60;

let server: ChildProcess | undefined;

test.describe.configure({ mode: "serial" });

test.beforeAll(async () => {
  if (!PARQUET) {
    test.skip();
    return;
  }
  server = startBackendServer(PARQUET);
  // Stream stdout so we can see DuckDB load progress
  server.stdout?.on("data", (b) => process.stdout.write(`[server] ${b}`));
  server.stderr?.on("data", (b) => process.stderr.write(`[server] ${b}`));
  // 75M dataset cold-load can take a while; bump the timeout.
  await waitForServer(`${BASE_URL}/data/metadata.json`, 5 * 60 * 1000);
});

test.afterAll(async () => {
  await teardown(server);
});

interface ChannelSummary {
  count: number;
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
}
interface PerfSummary {
  count: number;
  windowSeconds: number;
  fps: number;
  pointCount: number;
  downsampleRatio: number;
  cpu: ChannelSummary;
  interval: ChannelSummary;
  gpu: ChannelSummary;
}

async function readPerfSummary(page: Page): Promise<PerfSummary | null> {
  return page.evaluate(() => {
    return (window as any).__atlasPerf?.summary?.() ?? null;
  });
}

async function resetPerf(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__atlasPerf?.reset?.();
    (window as any).__atlasPanDbg = undefined;
  });
}

/**
 * Wait for every submitted instrumented WebGPU frame to reach the screen.
 * A render is recorded when it is submitted, while the matching GPU timing
 * is appended only after queue.onSubmittedWorkDone() resolves. Waiting only
 * for `summary().count > 0` therefore starts the gesture in the middle of the
 * initial full-fidelity draw when Max Points is set to All.
 */
async function waitForGpuIdle(page: Page): Promise<void> {
  const deadline = Date.now() + 5 * 60 * 1000;
  let previousCount = -1;
  while (Date.now() < deadline) {
    const state = await page.evaluate(() => {
      const summary = (window as any).__atlasPerf?.summary?.();
      return {
        firstFramePresented: (window as any).__atlasFirstBigRenderGpuLogged === true,
        submitted: summary?.count ?? 0,
        completed: summary?.gpu?.count ?? 0,
      };
    });
    if (
      state.firstFramePresented &&
      state.submitted > 0 &&
      state.completed >= state.submitted &&
      state.submitted === previousCount
    ) {
      return;
    }
    previousCount = state.completed >= state.submitted ? state.submitted : -1;
    await page.waitForTimeout(500);
  }
  throw new Error("WebGPU queue did not become idle before the performance gesture");
}

async function gpuInfo(page: Page): Promise<unknown> {
  return page.evaluate(async () => {
    const adapter = await (navigator as any).gpu?.requestAdapter?.();
    if (!adapter) return { ok: false, reason: "no adapter" };
    return {
      ok: true,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      adapterInfo: adapter.info
        ? {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            device: adapter.info.device,
            description: adapter.info.description,
          }
        : null,
      features: [...adapter.features],
    };
  });
}

async function dragPan(page: Page, seconds: number): Promise<unknown> {
  // Drive the viewport via DOM mouse events from inside the page. The pan
  // handler in interaction_handler.ts attaches mousemove/mouseup to `window`
  // after a mousedown on the SVG, so the events MUST be dispatched on the
  // right targets (down → SVG, move/up → window) and the first move has to
  // exceed the 2px DRAG_THRESHOLD before the drag handler kicks in.
  //
  // Doing this inside page.evaluate avoids the ~50ms-per-event Playwright
  // IPC round-trip that would cap mouse-driven panning at ~15fps and
  // dominate measurements.
  return await page.evaluate(async (seconds) => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) throw new Error("no canvas");
    // Find the interactive SVG by hunting for one whose parent contains the canvas.
    let svg: SVGElement | null = null;
    const allSvgs = Array.from(document.querySelectorAll("svg"));
    for (const s of allSvgs) {
      if (s.parentElement?.contains(canvas)) {
        svg = s as SVGElement;
        break;
      }
    }
    if (!svg) {
      // Fallback: walk up from the canvas
      svg = canvas.parentElement?.querySelector("svg") as SVGElement | null;
    }
    const downTarget: HTMLElement | SVGElement = svg ?? canvas;
    const rect = downTarget.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const radius = Math.min(rect.width, rect.height) * 0.25;

    let movesFired = 0;
    let viewportChanges = 0;
    const startState = JSON.stringify((window as any).__geospatialAtlasViewport ?? null);
    let actualDurationMs = 0;
    let panDbgBeforeRelease: Record<string, unknown> | null = null;

    function fire(target: EventTarget, type: string, x: number, y: number, buttons: number) {
      const ev = new MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: x,
        clientY: y,
        button: 0,
        buttons,
      });
      target.dispatchEvent(ev);
    }

    fire(downTarget, "mousedown", cx, cy, 1);
    // Threshold-busting first move so the drag handler arms.
    fire(window, "mousemove", cx + 6, cy + 6, 1);

    const start = performance.now();
    let lastViewport = startState;
    await new Promise<void>((resolve) => {
      const tick = () => {
        const elapsed = (performance.now() - start) / 1000;
        if (elapsed >= seconds) {
          actualDurationMs = performance.now() - start;
          panDbgBeforeRelease = structuredClone((window as any).__atlasPanDbg ?? null);
          fire(window, "mouseup", cx, cy, 0);
          resolve();
          return;
        }
        const t = elapsed * Math.PI * 4;
        const x = cx + Math.cos(t) * radius;
        const y = cy + Math.sin(t) * radius;
        fire(window, "mousemove", x, y, 1);
        movesFired++;
        const cur = JSON.stringify((window as any).__geospatialAtlasViewport ?? null);
        if (cur !== lastViewport) {
          viewportChanges++;
          lastViewport = cur;
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });

    return {
      svgFound: svg != null,
      svgTagName: svg?.tagName ?? null,
      downTargetRect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
      movesFired,
      viewportChanges,
      actualDurationMs,
      gestureFps: actualDurationMs > 0 ? (movesFired * 1000) / actualDurationMs : 0,
      panDbgBeforeRelease,
      startState,
      endState: (window as any).__geospatialAtlasViewport ?? null,
    };
  }, seconds);
}

/**
 * Runs the pan benchmark under several URL-driven configurations and
 * collects results into a single JSON report. The configurations are
 * intentionally independent so we can compare raw effect sizes.
 */
interface SweepConfig {
  tag: string;
  query: string;
  description: string;
  targetScale?: number;
}
const CONFIGS: SweepConfig[] = [
  { tag: "world", query: "perf=1", description: "world view, full fidelity" },
  { tag: "region", query: "perf=1", description: "region view, full fidelity", targetScale: 0.5 },
  { tag: "city", query: "perf=1", description: "city view, full fidelity", targetScale: 5.0 },
  { tag: "neighborhood", query: "perf=1", description: "neighborhood, full fidelity", targetScale: 50.0 },
  {
    tag: "world-capped",
    query: "perf=1&downsampleMax=4000000&densityWeight=0",
    description: "world view, 4M representative sample",
  },
];

async function setViewport(page: Page, scale: number) {
  // Use real Playwright wheel events — synthetic WheelEvent dispatch through
  // page.evaluate doesn't reach Svelte's onwheel handler reliably (the
  // synthetic event's `deltaY` ends up zero when it crosses the bridge).
  const canvas = page.locator("canvas").first();
  const box = await canvas.boundingBox();
  if (!box) return;
  const startScale = await page.evaluate(() => (window as any).__geospatialAtlasViewport?.scale);
  if (!startScale) return;
  const ratio = scale / startScale;
  // Each wheel event of |deltaY|=100 multiplies scale by exp(0.5) ≈ 1.65
  // (zoom-in: deltaY negative). Total deltaY needed ≈ -200 * ln(ratio).
  const totalDeltaY = -Math.log(ratio) * 200;
  const stepCount = Math.max(1, Math.ceil(Math.abs(totalDeltaY) / 50));
  const stepDeltaY = totalDeltaY / stepCount;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  for (let i = 0; i < stepCount; i++) {
    await page.mouse.wheel(0, stepDeltaY);
    await page.waitForTimeout(20);
  }
  await page.waitForTimeout(800);
}

async function runConfig(
  page: Page,
  config: { tag: string; query: string; description: string; targetScale?: number },
) {
  console.log(`\n=== Running [${config.tag}]: ${config.description} ===`);
  await page.goto(`${BASE_URL}?${config.query}`);
  await waitForCanvas(page, 5 * 60 * 1000);
  await page.waitForFunction(
    () => {
      const s = (window as any).__atlasPerf?.summary?.();
      return s != null && s.count > 0 && s.pointCount > 0;
    },
    null,
    { timeout: 5 * 60 * 1000, polling: 500 },
  );
  // Do not begin a gesture while the initial full-fidelity frame (or a
  // layout-triggered follow-up frame) is still monopolising the GPU.
  await waitForGpuIdle(page);
  if (config.targetScale != null) {
    await setViewport(page, config.targetScale);
    await waitForGpuIdle(page);
  }
  await resetPerf(page);
  const dragInfo = await dragPan(page, PAN_SECONDS);
  // Mouse-up submits one catch-up render and clears the CSS transform only
  // after that frame is presented. This is both a correctness assertion and
  // a clean boundary before the next navigation.
  await page.waitForFunction(() => ((window as any).__atlasPanDbg?.clearedViaGpuDone ?? 0) > 0, null, {
    timeout: 5 * 60 * 1000,
    polling: 250,
  });
  const panSummary = await readPerfSummary(page);
  console.log(`[${config.tag}] drag: ${JSON.stringify(dragInfo)}`);
  console.log(`[${config.tag}] pan: ${JSON.stringify(panSummary)}`);
  return { config, dragInfo, panSummary };
}

test("75M perf sweep", async ({ page }) => {
  test.setTimeout(30 * 60 * 1000);

  const fatalBrowserErrors: string[] = [];
  page.on("console", (msg) => {
    const t = msg.type();
    const message = msg.text();
    if (/device.*lost|kIOGPU|ignored submissions|out of memory|RangeError|WebGPU.*error/i.test(message)) {
      fatalBrowserErrors.push(message);
    }
    if (t === "error" || t === "warning" || message.includes("WebGPU") || message.includes("buffer")) {
      console.log(`[browser:${t}] ${msg.text()}`);
    }
  });
  page.on("pageerror", (error) => fatalBrowserErrors.push(`pageerror: ${error.message}`));
  page.on("response", (response) => {
    if (response.status() >= 400) {
      console.log(`[browser:http-${response.status()}] ${response.url()}`);
    }
  });

  // Initial load — warm DuckDB, capture GPU info
  const tLoadStart = Date.now();
  await page.goto(`${BASE_URL}?perf=1`);
  console.log("---");
  console.log("GPU info:", JSON.stringify(await gpuInfo(page), null, 2));
  console.log("---");
  await waitForCanvas(page, 5 * 60 * 1000);
  await page.waitForFunction(
    () => {
      const s = (window as any).__atlasPerf?.summary?.();
      return s != null && s.count > 0 && s.pointCount > 0;
    },
    null,
    { timeout: 5 * 60 * 1000, polling: 500 },
  );
  await waitForGpuIdle(page);
  const loadDuration = Date.now() - tLoadStart;
  const coldSummary = await readPerfSummary(page);
  console.log(`Load to first-frame: ${loadDuration} ms`);
  console.log("Cold summary:", JSON.stringify(coldSummary, null, 2));

  const results: any[] = [];
  for (const config of CONFIGS) {
    results.push(await runConfig(page, config));
  }

  // Persist combined report
  const outDir = path.resolve(__dirname, "perf-results");
  mkdirSync(outDir, { recursive: true });
  const tag = process.env.PERF_TAG ?? `sweep-${Date.now()}`;
  const report = {
    parquet: PARQUET,
    loadDurationMs: loadDuration,
    gpu: await gpuInfo(page),
    coldSummary,
    panSeconds: PAN_SECONDS,
    timestamp: new Date().toISOString(),
    gitHead: process.env.GIT_HEAD ?? null,
    runs: results,
  };
  writeFileSync(path.join(outDir, `${tag}.json`), JSON.stringify(report, null, 2));

  // Print a table-style summary at the end so we can eyeball the deltas.
  console.log("\n========== SWEEP SUMMARY ==========");
  console.log(`load: ${loadDuration}ms  parquet: ${PARQUET}`);
  console.log("config          gesture_fps moves   gpu_ms  ds_ratio");
  for (const r of results) {
    const ps = r.panSummary;
    if (!ps) {
      console.log(`${r.config.tag.padEnd(18)} (no data)`);
      continue;
    }
    const drag = r.dragInfo;
    console.log(
      `${r.config.tag.padEnd(18)}${drag.gestureFps.toFixed(1).padStart(8)}` +
        `${String(drag.movesFired).padStart(7)}` +
        `${ps.gpu.meanMs.toFixed(1).padStart(9)}` +
        `${(ps.downsampleRatio * 100).toFixed(0).padStart(8)}%`,
    );
  }
  console.log("===================================\n");

  // Sanity
  for (const r of results) {
    expect(r.panSummary).not.toBeNull();
    expect(r.dragInfo.viewportChanges, `${r.config.tag}: pan did not update the viewport`).toBeGreaterThan(
      PAN_SECONDS * 20,
    );
    expect(r.dragInfo.gestureFps, `${r.config.tag}: gesture RAF stalled`).toBeGreaterThan(20);
    const dbg = r.dragInfo.panDbgBeforeRelease;
    expect(dbg, `${r.config.tag}: CSS-pan instrumentation missing`).not.toBeNull();
    expect(dbg.cssPanApplied, `${r.config.tag}: CSS-pan fast path did not run`).toBeGreaterThan(PAN_SECONDS * 20);
    expect(dbg.renderCalls, `${r.config.tag}: too many WebGPU renders during CSS-pan`).toBeLessThan(
      Math.max(5, dbg.cssPanApplied / 5),
    );
  }
  expect(
    results.find((r) => r.config.tag === "world-capped")?.panSummary?.downsampleRatio,
    "explicit 4M cap did not exercise the downsampled renderer path",
  ).toBe(1);
  expect(fatalBrowserErrors, `fatal browser errors: ${JSON.stringify(fatalBrowserErrors)}`).toEqual([]);
});
