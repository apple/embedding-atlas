/**
 * Pan-release crash repro on the 322M eubucco file.
 *
 * User reports: with the 322M Downloads file loaded, the app renders
 * fluently — until they pan the canvas with the mouse and release. The
 * post-release re-render flickers and crashes the browser.
 *
 * This spec hits a *running* sidecar (default http://127.0.0.1:5088)
 * so it inherits the exact env the user is testing against, no fresh
 * spawn. Skipped unless the server is reachable.
 *
 * Captures, on each pan iteration:
 *   - JS heap usage (used / total / limit MB)
 *   - __atlasGpuErrors (uncapturederror + device.lost)
 *   - canvas alive
 *   - any new console errors
 *   - network bytes since last sample (if available)
 *
 * Pan strategy: each iteration is a discrete press → drag → release,
 * then a 1.5 s settle so the post-release re-render and any maplibre
 * tile fetches finish before we sample. We deliberately avoid the
 * continuous-drag pattern of gpu-soak — the user-reported failure is
 * specifically *on release*, so we want each release to be its own
 * observable event.
 */

import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";
import { writeFileSync, mkdirSync } from "fs";
import path from "path";

const BASE_URL = process.env.PAN_RELEASE_URL ?? "http://127.0.0.1:5088";
const TAG = process.env.PAN_RELEASE_TAG ?? `pan-release-${Date.now()}`;
const ITERS = Number(process.env.PAN_RELEASE_ITERS ?? 8);
const SETTLE_MS = Number(process.env.PAN_RELEASE_SETTLE_MS ?? 1500);
const DRAG_DURATION_MS = Number(process.env.PAN_RELEASE_DRAG_MS ?? 600);
const PAN_STEP_PX = Number(process.env.PAN_RELEASE_STEP_PX ?? 250);

interface Snapshot {
  iter: number;
  phase: "init" | "post-pan";
  tElapsedMs: number;
  gpuErrors: any[];
  consoleErrors: number;
  canvasAlive: boolean;
  heap: { usedMb: number; totalMb: number; limitMb: number } | null;
  panOffset?: { dx: number; dy: number };
}

async function snapshot(
  page: Page,
  iter: number,
  phase: "init" | "post-pan",
  t0: number,
  consoleErrors: string[],
  panOffset?: { dx: number; dy: number },
): Promise<Snapshot> {
  const data = await page.evaluate(() => {
    const cs = Array.from(document.querySelectorAll("canvas")) as HTMLCanvasElement[];
    let alive = false;
    let largest = 0;
    for (const c of cs) {
      const a = c.width * c.height;
      if (a > largest) {
        largest = a;
        alive = c.width > 0 && c.height > 0;
      }
    }
    const heap = (performance as any).memory
      ? {
          usedMb: ((performance as any).memory.usedJSHeapSize / 1024 / 1024) | 0,
          totalMb: ((performance as any).memory.totalJSHeapSize / 1024 / 1024) | 0,
          limitMb: ((performance as any).memory.jsHeapSizeLimit / 1024 / 1024) | 0,
        }
      : null;
    return {
      gpuErrors: (window as any).__atlasGpuErrors ?? [],
      canvasAlive: alive,
      heap,
    };
  });
  return {
    iter,
    phase,
    tElapsedMs: Date.now() - t0,
    gpuErrors: data.gpuErrors,
    consoleErrors: consoleErrors.length,
    canvasAlive: data.canvasAlive,
    heap: data.heap,
    panOffset,
  };
}

test("pan-release soak — 322M scatter survives discrete pan-release cycles", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000);

  const consoleErrors: string[] = [];
  const browserLines: string[] = [];
  const onConsole = (msg: ConsoleMessage) => {
    const t = msg.type();
    const text = msg.text();
    browserLines.push(`[${t}] ${text}`);
    if (t === "error") consoleErrors.push(text);
    if (
      /atlas-stage|atlas-gpu|first-big-render|deferred-density|scatter|RangeError|out of memory|ArrayBuffer|kIOGPU|ignored submissions|device.*lost|destroyed|MTLDevice|WebGPU/i.test(
        text,
      )
    ) {
      process.stdout.write(`[browser] ${text}\n`);
    }
  };
  page.on("console", onConsole);
  page.on("pageerror", (err) => {
    consoleErrors.push(`pageerror: ${err.message}`);
    process.stderr.write(`[browser-err] ${err.message}\n`);
  });
  page.on("crash", () => {
    process.stderr.write(`[browser-crash] page crashed\n`);
  });

  // Sanity: server reachable.
  const probe = await page.request.get(`${BASE_URL}/data/metadata.json`);
  expect(probe.ok(), `server unreachable at ${BASE_URL}`).toBeTruthy();
  const meta = await probe.json();
  console.log(`[pan-soak] server metadata projection:`, JSON.stringify(meta.props.data.projection.viewportHint));

  // Load with ?perf=1 to also unlock the [scatter] / [atlas-stage] logs
  // that surface arrow-buffer pressure and re-query firing.
  const t0 = Date.now();
  console.log(`[pan-soak] loading ${BASE_URL}/?perf=1`);
  await page.goto(`${BASE_URL}/?perf=1`, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: 6 * 60 * 1000,
    polling: 250,
  });
  console.log(`[pan-soak] first big render at +${Date.now() - t0}ms`);

  // Locate primary canvas (largest = scatter surface).
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
  console.log(`[pan-soak] canvas center (${cx.toFixed(0)},${cy.toFixed(0)})`);

  const snapshots: Snapshot[] = [];

  // Initial settle so the deferred density refine (if any) is done.
  await page.waitForTimeout(SETTLE_MS);
  snapshots.push(await snapshot(page, 0, "init", t0, consoleErrors));

  // First-frame screenshot (clean, before any pan).
  const outDir = path.resolve(__dirname, "perf-results");
  mkdirSync(outDir, { recursive: true });
  await page.screenshot({ path: path.join(outDir, `pan-release-${TAG}-00-init.png`), fullPage: false });

  // Discrete pan-release cycles. Each one: cursor at canvas center →
  // mouse down → drag in a varied direction → mouse up → settle.
  await page.mouse.move(cx, cy);
  for (let i = 1; i <= ITERS; i++) {
    const angle = (i * 47 * Math.PI) / 180;
    const dx = Math.cos(angle) * PAN_STEP_PX;
    const dy = Math.sin(angle) * PAN_STEP_PX;
    console.log(`[pan-soak] iter ${i} — pan dx=${dx.toFixed(0)} dy=${dy.toFixed(0)}`);

    const stepStart = Date.now();
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    // Drag in 6 micro-steps so maplibre + the canvas CSS-transform see
    // a believable mousemove sequence.
    const STEPS = 6;
    for (let s = 1; s <= STEPS; s++) {
      const t = s / STEPS;
      await page.mouse.move(cx + dx * t, cy + dy * t);
      await page.waitForTimeout(DRAG_DURATION_MS / STEPS);
    }
    await page.mouse.up();
    console.log(`[pan-soak] iter ${i} — released at +${Date.now() - stepStart}ms`);

    // Settle window: the post-release re-render fires here. Watch GPU
    // errors mid-settle to catch a crash before sampling the heap.
    const settleStart = Date.now();
    let cascadeAt: number | null = null;
    while (Date.now() - settleStart < SETTLE_MS) {
      await page.waitForTimeout(100);
      const errCount = await page.evaluate(() => (window as any).__atlasGpuErrors?.length ?? 0).catch(() => -1);
      if (errCount > 0) {
        cascadeAt = Date.now() - settleStart;
        console.error(`[pan-soak] iter ${i} — GPU errors detected at +${cascadeAt}ms (count=${errCount})`);
        break;
      }
      if (errCount === -1) {
        console.error(`[pan-soak] iter ${i} — page.evaluate failed (page crashed?)`);
        break;
      }
    }

    const snap = await snapshot(page, i, "post-pan", t0, consoleErrors, { dx, dy });
    snapshots.push(snap);
    console.log(
      `  iter=${i} canvas=${snap.canvasAlive} gpuErrors=${snap.gpuErrors.length} consoleErrors=${snap.consoleErrors} heap=${snap.heap?.usedMb}/${snap.heap?.totalMb}/${snap.heap?.limitMb}MB`,
    );

    // Snapshot the screen after the pan so I can inspect what flicker looks like.
    await page
      .screenshot({ path: path.join(outDir, `pan-release-${TAG}-${String(i).padStart(2, "0")}.png`), fullPage: false })
      .catch(() => {});

    if (snap.gpuErrors.length > 0 || !snap.canvasAlive) {
      console.error(`[pan-soak] iter ${i} — CASCADE OR DEAD CANVAS, halting soak`);
      // Dump the GPU error records.
      for (const e of snap.gpuErrors) {
        console.error(`  - ${JSON.stringify(e)}`);
      }
      break;
    }
  }

  // Persist artifacts.
  writeFileSync(
    path.join(outDir, `pan-release-${TAG}.json`),
    JSON.stringify(
      {
        tag: TAG,
        baseUrl: BASE_URL,
        iters: ITERS,
        settleMs: SETTLE_MS,
        dragMs: DRAG_DURATION_MS,
        panStepPx: PAN_STEP_PX,
        snapshots,
        consoleErrorCount: consoleErrors.length,
        consoleErrorsSample: consoleErrors.slice(0, 50),
        browserLinesTail: browserLines.slice(-300),
      },
      null,
      2,
    ),
  );

  console.log(`[pan-soak] FINAL — ${snapshots.length} snapshots, consoleErrors=${consoleErrors.length}`);
  for (const s of snapshots) {
    console.log(
      `  iter=${s.iter}/${s.phase} canvas=${s.canvasAlive} gpuErrors=${s.gpuErrors.length} heap=${s.heap?.usedMb}/${s.heap?.totalMb}/${s.heap?.limitMb}MB`,
    );
  }

  // Hard assertions.
  for (const s of snapshots) {
    expect(s.gpuErrors.length, `iter ${s.iter}/${s.phase}: GPU error`).toBe(0);
    expect(s.canvasAlive, `iter ${s.iter}/${s.phase}: canvas dead`).toBe(true);
  }
  const fatalPatterns = /ignored submissions|kIOGPUCommand|device.*lost|WebGPU.*error|Failed to.*GPU/i;
  const fatal = consoleErrors.filter((e) => fatalPatterns.test(e));
  expect(fatal, `fatal console errors: ${JSON.stringify(fatal)}`).toEqual([]);
});
