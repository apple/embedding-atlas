/**
 * Reload-induced flicker / crash repro.
 *
 * The user reported: on the plain http://127.0.0.1:5088/ URL (no ?perf=1),
 * a manual browser reload re-triggers the Metal "ignored submissions"
 * cascade and crashes the tab. The defer-destroy fix in
 * ``packages/component/src/lib/webgpu_renderer/utils.ts`` covers reactive
 * realloc; this test surfaces any teardown-related path that still
 * destroys GPU resources synchronously while submissions are in flight.
 *
 * Talks to a sidecar that is *already running* on RELOAD_SOAK_URL
 * (default http://127.0.0.1:5088). Skips otherwise.
 *
 * Usage::
 *
 *   RELOAD_SOAK_URL=http://127.0.0.1:5088 \
 *     npx playwright test e2e/reload-soak.spec.ts \
 *     --project=perf-chrome --workers=1 --headed
 */

import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";
import { writeFileSync, mkdirSync } from "fs";
import path from "path";

const BASE_URL = process.env.RELOAD_SOAK_URL ?? "http://127.0.0.1:5088";
const TAG = process.env.RELOAD_SOAK_TAG ?? `reload-${Date.now()}`;
const RELOADS = Number(process.env.RELOAD_SOAK_RELOADS ?? 3);
const SETTLE_MS = Number(process.env.RELOAD_SOAK_SETTLE_MS ?? 4000);

interface Snapshot {
  iter: number;
  url: string;
  tFirstFrameMs: number | null;
  sidePanelMounted: boolean;
  sidePanelTitles: string[];
  gpuErrors: any[];
  consoleErrorsSample: string[];
  canvasCount: number;
  primaryCanvasNonZero: boolean;
  heap: { usedMb: number; totalMb: number; limitMb: number } | null;
}

async function waitForFirstFrame(page: Page, timeoutMs: number): Promise<number> {
  const t0 = Date.now();
  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: timeoutMs,
    polling: 200,
  });
  return Date.now() - t0;
}

async function snapshot(
  page: Page,
  iter: number,
  url: string,
  tFirstFrameMs: number | null,
  consoleErrors: string[],
): Promise<Snapshot> {
  const data = await page.evaluate(() => {
    const cs = Array.from(document.querySelectorAll("canvas")) as HTMLCanvasElement[];
    let primaryCanvasNonZero = false;
    let largest = 0;
    for (const c of cs) {
      const a = c.width * c.height;
      if (a > largest) {
        largest = a;
        primaryCanvasNonZero = c.width > 0 && c.height > 0;
      }
    }
    // Side panel detection: ListChartPanel renders titles inside .font-mono.font-medium
    // wrappers. SQL Predicates is one of the primary specs, so we expect AT
    // LEAST 1 (predicates) + 1 column chart once discovery fires.
    const titleEls = Array.from(document.querySelectorAll(".font-mono.font-medium"));
    const titles = titleEls.map((e) => e.textContent?.trim() || "");
    const heap = (performance as any).memory
      ? {
          usedMb: ((performance as any).memory.usedJSHeapSize / 1024 / 1024) | 0,
          totalMb: ((performance as any).memory.totalJSHeapSize / 1024 / 1024) | 0,
          limitMb: ((performance as any).memory.jsHeapSizeLimit / 1024 / 1024) | 0,
        }
      : null;
    return {
      gpuErrors: (window as any).__atlasGpuErrors ?? [],
      canvasCount: cs.length,
      primaryCanvasNonZero,
      titles,
      heap,
    };
  });
  const sidePanelMounted = data.titles.length >= 2;
  return {
    iter,
    url,
    tFirstFrameMs,
    sidePanelMounted,
    sidePanelTitles: data.titles,
    gpuErrors: data.gpuErrors,
    consoleErrorsSample: consoleErrors.slice(-25),
    canvasCount: data.canvasCount,
    primaryCanvasNonZero: data.primaryCanvasNonZero,
    heap: data.heap,
  };
}

test("reload soak — no metal cascade, side panel mounts on plain URL", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000);

  const consoleErrors: string[] = [];
  const browserLines: string[] = [];
  const onConsole = (msg: ConsoleMessage) => {
    const t = msg.type();
    const text = msg.text();
    browserLines.push(`[${t}] ${text}`);
    if (t === "error") consoleErrors.push(text);
    if (
      /atlas-stage|atlas-gpu|first-big-render|deferred-density|scatter|RangeError|out of memory|ArrayBuffer/i.test(text)
    ) {
      process.stdout.write(`[browser] ${text}\n`);
    }
  };
  page.on("console", onConsole);
  page.on("pageerror", (err) => {
    consoleErrors.push(`pageerror: ${err.message}`);
    process.stderr.write(`[browser-err] ${err.message}\n`);
  });

  // Sanity: server reachable.
  const probe = await page.request.get(`${BASE_URL}/data/metadata.json`);
  expect(probe.ok(), `server unreachable at ${BASE_URL}/data/metadata.json`).toBeTruthy();

  const snapshots: Snapshot[] = [];

  // Iter 0 — initial load, plain URL (no ?perf=1).
  console.log(`[reload-soak] iter 0 — initial load ${BASE_URL}/`);
  await page.goto(`${BASE_URL}/`, { waitUntil: "domcontentloaded" });
  let tFF = await waitForFirstFrame(page, 6 * 60 * 1000);
  console.log(`[reload-soak] iter 0 — first frame at +${tFF}ms`);
  // Side panel should fill within a few seconds of first frame (no longer
  // gated on perf flag, no longer waiting for the 60 s safety net).
  await page.waitForTimeout(SETTLE_MS);
  snapshots.push(await snapshot(page, 0, `${BASE_URL}/`, tFF, consoleErrors));

  // Iters 1..N — repeated reloads. Each must come back cleanly with no
  // new GPU errors and a non-zero canvas. The soak deliberately spans
  // the deferred-density-refine 2 s mark before reloading to maximise
  // the chance of catching a destroy-in-flight on a still-running query.
  for (let i = 1; i <= RELOADS; i++) {
    console.log(`[reload-soak] iter ${i} — reload`);
    const reloadStart = Date.now();
    try {
      await page.reload({ waitUntil: "domcontentloaded" });
    } catch (e) {
      console.error(`[reload-soak] iter ${i} — reload threw: ${(e as Error).message}`);
      throw e;
    }
    let tFFi: number | null = null;
    try {
      tFFi = await waitForFirstFrame(page, 90_000);
      console.log(`[reload-soak] iter ${i} — first frame at +${tFFi}ms (reload took ${Date.now() - reloadStart}ms)`);
    } catch (e) {
      console.error(`[reload-soak] iter ${i} — first frame TIMEOUT after reload (${(e as Error).message})`);
    }
    await page.waitForTimeout(SETTLE_MS);
    snapshots.push(await snapshot(page, i, `${BASE_URL}/ (reload)`, tFFi, consoleErrors));
  }

  // Persist artifacts.
  const outDir = path.resolve(__dirname, "perf-results");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    path.join(outDir, `reload-soak-${TAG}.json`),
    JSON.stringify(
      {
        tag: TAG,
        baseUrl: BASE_URL,
        reloads: RELOADS,
        settleMs: SETTLE_MS,
        snapshots,
        consoleErrorCount: consoleErrors.length,
        consoleErrorsSample: consoleErrors.slice(0, 50),
        browserLinesTail: browserLines.slice(-200),
      },
      null,
      2,
    ),
  );
  await page.screenshot({ path: path.join(outDir, `reload-soak-${TAG}.png`), fullPage: false });

  console.log(`[reload-soak] DONE — ${snapshots.length} snapshots, consoleErrors=${consoleErrors.length}`);
  for (const s of snapshots) {
    const heap = s.heap ? `heap=${s.heap.usedMb}/${s.heap.totalMb}/${s.heap.limitMb}MB` : "heap=?";
    console.log(
      `  iter=${s.iter} firstFrame=${s.tFirstFrameMs}ms sidePanel=${s.sidePanelMounted}(${s.sidePanelTitles.length}) canvasOK=${s.primaryCanvasNonZero} gpuErrors=${s.gpuErrors.length} ${heap}`,
    );
  }

  // Hard assertions: no GPU errors at any point, every iteration produced
  // a first frame, every iteration left a live canvas, and the side panel
  // mounted on the initial plain-URL load (the bug #2 user-visible symptom).
  for (const s of snapshots) {
    expect(s.gpuErrors.length, `iter ${s.iter}: WebGPU uncapturederror/device-lost`).toBe(0);
    expect(s.tFirstFrameMs, `iter ${s.iter}: never reached first big render`).not.toBeNull();
    expect(s.primaryCanvasNonZero, `iter ${s.iter}: canvas is zero-sized`).toBe(true);
  }
  expect(snapshots[0].sidePanelMounted, "side panel did not mount on plain URL within settle window").toBe(true);

  const fatalPatterns = /ignored submissions|kIOGPUCommand|device.*lost|WebGPU.*error|Failed to.*GPU/i;
  const fatal = consoleErrors.filter((e) => fatalPatterns.test(e));
  expect(fatal, `fatal console errors: ${JSON.stringify(fatal)}`).toEqual([]);
});
