/**
 * Rapid pan-release stress against the 322M eubucco file.
 *
 * User report: with the 322M file loaded, the FIRST pan-release rendered
 * after ~3 s and worked. The SECOND pan-release froze the Chrome window
 * (unresponsive, can't even open DevTools). Root cause: each release
 * triggers a multi-second compute pipeline (accumulate + blur + downsample
 * + draw) and ``device.queue.submit()`` is fire-and-forget — two releases
 * inside the first one's GPU drain stack the queue, the macOS Metal
 * watchdog kills long submits, the compositor (same GPU process) starves.
 *
 * The fix in EmbeddingViewImpl.svelte adds backpressure: while a render's
 * submit is in flight, additional renders are coalesced into a single
 * ``_renderPending`` flag and fired once the GPU drains via
 * ``device.queue.onSubmittedWorkDone()``.
 *
 * This spec performs N rapid press → drag → release cycles **without**
 * waiting for the post-release GPU work to finish between iterations.
 * Then asserts:
 *   - the page never becomes unresponsive (every page.evaluate succeeds)
 *   - the canvas stays alive across all iterations
 *   - no GPU error / device.lost cascade
 *
 * Hits a *running* sidecar (default http://127.0.0.1:5088) so it inherits
 * the exact env the user is testing against.
 */

import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";

const BASE_URL = process.env.PAN_RAPID_URL ?? "http://127.0.0.1:5088";
const ITERS = Number(process.env.PAN_RAPID_ITERS ?? 5);
const DRAG_DURATION_MS = Number(process.env.PAN_RAPID_DRAG_MS ?? 200);
const PAN_STEP_PX = Number(process.env.PAN_RAPID_STEP_PX ?? 200);
// Time between releases. The bug was: ~150 ms between releases (faster
// than one render completes on 322M) → freeze. We deliberately use 200 ms
// so that the second release lands well before the first's GPU work
// drains.
const INTER_RELEASE_MS = Number(process.env.PAN_RAPID_INTER_MS ?? 200);
// How long we give the page to recover after the storm. With backpressure
// in place this should be small; without it the page is dead.
const RESPONSIVENESS_TIMEOUT_MS = Number(process.env.PAN_RAPID_RESP_MS ?? 5000);

test("rapid pan-release storm — backpressure keeps Chrome responsive on 322M", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000);

  const consoleErrors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
    const text = msg.text();
    if (/atlas-stage|atlas-gpu|device.*lost|kIOGPU|ignored submissions/i.test(text)) {
      process.stdout.write(`[browser] ${text}\n`);
    }
  });
  page.on("pageerror", (err) => {
    consoleErrors.push(`pageerror: ${err.message}`);
    process.stderr.write(`[browser-err] ${err.message}\n`);
  });
  page.on("crash", () => {
    process.stderr.write(`[browser-crash] page crashed\n`);
  });

  const probe = await page.request.get(`${BASE_URL}/data/metadata.json`);
  expect(probe.ok(), `server unreachable at ${BASE_URL}`).toBeTruthy();

  console.log(`[pan-rapid] loading ${BASE_URL}/?perf=1`);
  await page.goto(`${BASE_URL}/?perf=1`, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: 6 * 60 * 1000,
    polling: 250,
  });
  console.log(`[pan-rapid] first big render landed`);

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

  await page.mouse.move(cx, cy);

  // Storm loop. Each iteration is press → drag → release, then wait
  // INTER_RELEASE_MS (deliberately less than one full GPU pipeline at
  // 322M) before starting the next. We do NOT wait for the post-release
  // re-render to finish — that's the whole point of the test.
  for (let i = 1; i <= ITERS; i++) {
    const angle = (i * 47 * Math.PI) / 180;
    const dx = Math.cos(angle) * PAN_STEP_PX;
    const dy = Math.sin(angle) * PAN_STEP_PX;
    console.log(`[pan-rapid] iter ${i} — press → drag dx=${dx.toFixed(0)} dy=${dy.toFixed(0)} → release`);

    await page.mouse.move(cx, cy);
    await page.mouse.down();
    const STEPS = 4;
    for (let s = 1; s <= STEPS; s++) {
      const t = s / STEPS;
      await page.mouse.move(cx + dx * t, cy + dy * t);
      await page.waitForTimeout(DRAG_DURATION_MS / STEPS);
    }
    await page.mouse.up();

    // Tight inter-release pause — does NOT wait for GPU drain.
    await page.waitForTimeout(INTER_RELEASE_MS);

    // After each release, verify the page is still responsive. Without
    // backpressure, this page.evaluate would hang/timeout once the GPU
    // process is saturated and IPC stalls.
    const respCheck = await Promise.race([
      page.evaluate(() => ({
        gpuErrors: ((window as any).__atlasGpuErrors ?? []).length,
        canvasW: (document.querySelector("canvas") as HTMLCanvasElement | null)?.width ?? 0,
      })),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), RESPONSIVENESS_TIMEOUT_MS)),
    ]);
    expect(
      respCheck,
      `iter ${i}: page unresponsive (page.evaluate did not return inside ${RESPONSIVENESS_TIMEOUT_MS}ms)`,
    ).not.toBeNull();
    expect(respCheck!.canvasW, `iter ${i}: canvas dead`).toBeGreaterThan(0);
    expect(respCheck!.gpuErrors, `iter ${i}: GPU error cascade`).toBe(0);
    console.log(
      `[pan-rapid] iter ${i} — page responsive, canvas alive (${respCheck!.canvasW}px), gpuErrors=${respCheck!.gpuErrors}`,
    );
  }

  // Final settle: wait for backpressure-coalesced renders to drain, then
  // verify nothing exploded after the fact.
  console.log(`[pan-rapid] settling for 5 s after storm`);
  await page.waitForTimeout(5000);
  const finalCheck = await page.evaluate(() => ({
    gpuErrors: (window as any).__atlasGpuErrors ?? [],
    canvasAlive: ((document.querySelector("canvas") as HTMLCanvasElement | null)?.width ?? 0) > 0,
  }));
  expect(finalCheck.canvasAlive, "final: canvas dead").toBe(true);
  expect(finalCheck.gpuErrors.length, `final: GPU errors ${JSON.stringify(finalCheck.gpuErrors)}`).toBe(0);

  const fatalConsole = consoleErrors.filter((e) =>
    /device.*lost|kIOGPU|ignored submissions|external Instance/i.test(e),
  );
  expect(fatalConsole, `fatal console errors: ${JSON.stringify(fatalConsole)}`).toEqual([]);
  console.log(`[pan-rapid] DONE — survived ${ITERS} rapid releases, ${consoleErrors.length} non-fatal console errors`);
});
