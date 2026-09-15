/**
 * Workgroup-size sweep on a 75 M scatter. Iterates through SIMD-aligned
 * sizes for (a) the downsample family — viewport-cull, density-sample,
 * compact_accepted — and (b) the accumulate / Gaussian-blur kernels.
 *
 * Each configuration runs the same 5 s in-page drag used by
 * `perf-75m.spec.ts` and reports median, p95 and p99 frame intervals
 * + the cold GPU frame time.
 *
 * Usage:
 *   PERF_PARQUET_FILE=/abs/path/to/big.parquet \
 *     npx playwright test e2e/wg-sweep.spec.ts --project=perf-chrome
 *
 * Output: JSON at e2e/perf-results/wg-sweep-<timestamp>.json.
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

let server: ChildProcess | undefined;
test.describe.configure({ mode: "serial" });

test.beforeAll(async () => {
  if (!PARQUET) {
    test.skip();
    return;
  }
  server = startBackendServer(PARQUET);
  server.stdout?.on("data", (b) => process.stdout.write(`[server] ${b}`));
  server.stderr?.on("data", (b) => process.stderr.write(`[server] ${b}`));
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
  return page.evaluate(() => (window as any).__atlasPerf?.summary?.() ?? null);
}
async function resetPerf(page: Page): Promise<void> {
  await page.evaluate(() => (window as any).__atlasPerf?.reset?.());
}

async function dragPan(page: Page, seconds: number) {
  return page.evaluate(async (seconds) => {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) throw new Error("no canvas");
    let svg: SVGElement | null = null;
    for (const s of Array.from(document.querySelectorAll("svg"))) {
      if (s.parentElement?.contains(canvas)) {
        svg = s as SVGElement;
        break;
      }
    }
    if (!svg) svg = canvas.parentElement?.querySelector("svg") as SVGElement | null;
    const target: HTMLElement | SVGElement = svg ?? canvas;
    const rect = target.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const radius = Math.min(rect.width, rect.height) * 0.25;
    function fire(t: EventTarget, type: string, x: number, y: number, buttons: number) {
      t.dispatchEvent(
        new MouseEvent(type, {
          bubbles: true,
          cancelable: true,
          view: window,
          clientX: x,
          clientY: y,
          button: 0,
          buttons,
        }),
      );
    }
    fire(target, "mousedown", cx, cy, 1);
    fire(window, "mousemove", cx + 6, cy + 6, 1);
    const start = performance.now();
    await new Promise<void>((resolve) => {
      const tick = () => {
        const e = (performance.now() - start) / 1000;
        if (e >= seconds) {
          fire(window, "mouseup", cx, cy, 0);
          resolve();
          return;
        }
        const t = e * Math.PI * 4;
        fire(window, "mousemove", cx + Math.cos(t) * radius, cy + Math.sin(t) * radius, 1);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }, seconds);
}

interface WgCell {
  wgDs: number;
  wgAcc: number;
  wgBlur: number;
}

// Sweep matrix: keep it small but representative. Stress the three
// knobs independently. All values are 32-aligned (Apple SIMD width).
const CELLS: WgCell[] = [
  { wgDs: 64, wgAcc: 64, wgBlur: 64 },
  { wgDs: 128, wgAcc: 64, wgBlur: 64 },
  { wgDs: 256, wgAcc: 64, wgBlur: 64 }, // baseline
  { wgDs: 512, wgAcc: 64, wgBlur: 64 },
  { wgDs: 256, wgAcc: 128, wgBlur: 64 },
  { wgDs: 256, wgAcc: 256, wgBlur: 64 },
  { wgDs: 256, wgAcc: 64, wgBlur: 128 },
  { wgDs: 256, wgAcc: 64, wgBlur: 256 },
  { wgDs: 128, wgAcc: 128, wgBlur: 128 },
];

async function runCell(page: Page, cell: WgCell) {
  const url = `${BASE_URL}?perf=1&wgDs=${cell.wgDs}&wgAcc=${cell.wgAcc}&wgBlur=${cell.wgBlur}`;
  console.log(`\n-- ${JSON.stringify(cell)}`);
  await page.goto(url);
  await waitForCanvas(page, 5 * 60 * 1000);
  // Wait for the first recorded frame so we know the compute pipeline built.
  await page.waitForFunction(
    () => {
      const s = (window as any).__atlasPerf?.summary?.();
      return s != null && s.count > 0 && s.pointCount > 0;
    },
    null,
    { timeout: 5 * 60 * 1000, polling: 500 },
  );
  await page.waitForTimeout(2500); // settle
  const cold = await readPerfSummary(page);
  await resetPerf(page);
  await dragPan(page, PAN_SECONDS);
  await page.waitForTimeout(500);
  const pan = await readPerfSummary(page);
  console.log(`cold: fps=${cold?.fps.toFixed(1)} gpu_p50=${cold?.gpu.p50Ms.toFixed(1)}ms`);
  console.log(
    `pan:  fps=${pan?.fps.toFixed(1)} interval_p50=${pan?.interval.p50Ms.toFixed(1)}ms p95=${pan?.interval.p95Ms.toFixed(1)}ms`,
  );
  return { cell, cold, pan };
}

test("workgroup-size sweep", async ({ page }) => {
  test.setTimeout(60 * 60 * 1000);
  page.on("console", (msg) => {
    if (msg.type() === "error") console.log(`[browser] ${msg.text()}`);
  });

  const results: any[] = [];
  for (const cell of CELLS) {
    results.push(await runCell(page, cell));
  }

  const outDir = path.resolve(__dirname, "perf-results");
  mkdirSync(outDir, { recursive: true });
  const tag = process.env.PERF_TAG ?? `wg-sweep-${Date.now()}`;
  const report = {
    parquet: PARQUET,
    timestamp: new Date().toISOString(),
    panSeconds: PAN_SECONDS,
    runs: results,
  };
  writeFileSync(path.join(outDir, `${tag}.json`), JSON.stringify(report, null, 2));

  console.log("\n========== WG SWEEP ==========");
  console.log("wgDs  wgAcc wgBlur  fps   int_p50 int_p95 int_p99 gpu_p50  cold_gpu_p50");
  for (const r of results) {
    if (!r.pan) {
      console.log(`${r.cell.wgDs} ${r.cell.wgAcc} ${r.cell.wgBlur} (no data)`);
      continue;
    }
    console.log(
      `${String(r.cell.wgDs).padStart(4)}  ${String(r.cell.wgAcc).padStart(5)}  ${String(r.cell.wgBlur).padStart(6)}` +
        `${r.pan.fps.toFixed(1).padStart(7)}` +
        `${r.pan.interval.p50Ms.toFixed(1).padStart(9)}` +
        `${r.pan.interval.p95Ms.toFixed(1).padStart(8)}` +
        `${r.pan.interval.p99Ms.toFixed(1).padStart(8)}` +
        `${r.pan.gpu.p50Ms.toFixed(1).padStart(9)}` +
        `${(r.cold?.gpu.p50Ms ?? NaN).toFixed(1).padStart(13)}`,
    );
  }
  console.log("==============================\n");

  for (const r of results) expect(r.pan).not.toBeNull();
});
