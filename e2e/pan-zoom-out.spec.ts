/**
 * Pan-twice → zoom-out-to-world freeze repro on the 322M eubucco file.
 *
 * User report: after the render-backpressure fix, panning works smoothly,
 * BUT — pan twice (release each), then wheel-zoom out so the entire
 * dataset comes back into view, and the canvas freezes (basemap still
 * paints, but the scatter overlay never updates).
 *
 * Two compounded bugs in the ship-it backpressure:
 *
 *   1. **Metal command-buffer watchdog at world view.** At zoom-out the
 *      ``accumulate`` pass does 322 M atomic adds into a ~480-cell density
 *      grid; the dense Europe region hits ~500 K serialized atomics per
 *      cell, the single command buffer crosses macOS's GPU watchdog (~3-5
 *      s), the OS kills it → ``device.lost``.
 *
 *   2. **Backpressure deadlocks on device.lost.** Chromium/Dawn does NOT
 *      resolve ``device.queue.onSubmittedWorkDone()`` when the device is
 *      lost; ``_renderInFlight`` stays true forever; subsequent renders
 *      go pending and the screen freezes on the stale frame.
 *
 * Fix:
 *   - ``effectiveDownsampleDensityWeight`` forces 0 when N > 50 M (points
 *     mode), skipping the doomed accumulate pass.
 *   - ``_renderToken`` + 30 s watchdog + ``$effect`` on ``renderer`` so a
 *     dropped Promise never deadlocks the loop.
 *
 * The test: pan twice, wheel-zoom-out to world view, confirm the canvas
 * is alive and ``_renderInFlight`` does not deadlock (the page must keep
 * processing renders — verified by checking that a follow-up viewport
 * change (e.g. small wheel-in) triggers a fresh render).
 */

import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";

const BASE_URL = process.env.PAN_ZOOM_URL ?? "http://127.0.0.1:5088";
const SETTLE_MS = 1500;

async function getRenderCalls(page: Page): Promise<number> {
  return await page.evaluate(() => (window as any).__atlasPanDbg?.renderCalls ?? 0);
}

test("pan-twice + zoom-out — backpressure recovers from device.lost / Metal watchdog", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000);

  const consoleErrors: string[] = [];
  const consoleWarns: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    const text = msg.text();
    if (msg.type() === "error") consoleErrors.push(text);
    if (msg.type() === "warning") consoleWarns.push(text);
    if (/atlas-stage|atlas-gpu|atlas\] render|device.*lost|watchdog/i.test(text)) {
      process.stdout.write(`[browser] ${text}\n`);
    }
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));
  page.on("crash", () => process.stderr.write(`[browser-crash] page crashed\n`));

  const probe = await page.request.get(`${BASE_URL}/data/metadata.json`);
  expect(probe.ok(), `server unreachable at ${BASE_URL}`).toBeTruthy();

  console.log(`[pan-zoom] loading ${BASE_URL}/?perf=1`);
  await page.goto(`${BASE_URL}/?perf=1`, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: 6 * 60 * 1000,
    polling: 250,
  });
  console.log(`[pan-zoom] first big render landed`);

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

  // Pan #1
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 200, cy + 100, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(SETTLE_MS);
  console.log(`[pan-zoom] pan #1 settled`);

  // Pan #2
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx - 250, cy + 80, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(SETTLE_MS);
  console.log(`[pan-zoom] pan #2 settled`);

  const renderCallsBeforeZoom = await getRenderCalls(page);
  console.log(`[pan-zoom] renderCalls before zoom-out: ${renderCallsBeforeZoom}`);

  // Zoom out aggressively to world view. Each wheel event with deltaY
  // > 0 zooms out (per onWheel handler: scaler = exp(-deltaY/200)). 30
  // events × deltaY=300 takes us through ~e^45 = 3e19 zoom-out factor,
  // way beyond gisMinScale clamp — exactly the user scenario.
  await page.mouse.move(cx, cy);
  for (let i = 0; i < 30; i++) {
    await page.mouse.wheel(0, 300);
    await page.waitForTimeout(20);
  }
  console.log(`[pan-zoom] zoomed out — waiting up to 45 s for canvas to recover`);

  // Wait for the post-interaction render to land. Budget covers worst
  // case: pipeline runs ~5 s, hits Metal watchdog, device.lost, recover,
  // re-render, all gated by 30 s watchdog. With both fixes total should
  // be < 10 s; we give 45 s headroom for shared-host CI noise.
  const recoveryDeadline = Date.now() + 45_000;
  let recovered = false;
  let lastRenderCalls = renderCallsBeforeZoom;
  while (Date.now() < recoveryDeadline) {
    await page.waitForTimeout(500);
    const calls = await getRenderCalls(page);
    if (calls > renderCallsBeforeZoom) {
      lastRenderCalls = calls;
      // Need TWO calls past pre-zoom — one for the initial wheel event,
      // one to confirm post-interaction settled render landed without
      // deadlocking.
      if (calls > renderCallsBeforeZoom + 1) {
        recovered = true;
        break;
      }
    }
  }
  expect(
    recovered,
    `canvas froze after zoom-out — renderCalls stuck at ${lastRenderCalls} (was ${renderCallsBeforeZoom} pre-zoom). This is the backpressure-deadlock-on-device.lost bug.`,
  ).toBe(true);
  console.log(`[pan-zoom] canvas updating again — renderCalls=${lastRenderCalls}`);

  // Liveness probe: another small interaction should still produce a
  // render. If the deadlock has merely been masked by recovery (and we'd
  // see a re-freeze on the next event), this catches it.
  const callsBeforeProbe = await getRenderCalls(page);
  await page.mouse.wheel(0, -100);
  await page.waitForTimeout(2000);
  const callsAfterProbe = await getRenderCalls(page);
  expect(
    callsAfterProbe,
    `liveness probe: subsequent wheel-in did not trigger a render (calls ${callsBeforeProbe} → ${callsAfterProbe}). Backpressure may be re-deadlocked.`,
  ).toBeGreaterThan(callsBeforeProbe);

  // Final state: canvas alive, no fatal errors. Watchdog warnings ARE
  // tolerated — they're a fallback mechanism, but if they fire we want
  // to know.
  const finalCheck = await page.evaluate(() => ({
    gpuErrors: (window as any).__atlasGpuErrors ?? [],
    canvasAlive: ((document.querySelector("canvas") as HTMLCanvasElement | null)?.width ?? 0) > 0,
    renderCalls: (window as any).__atlasPanDbg?.renderCalls ?? 0,
  }));
  expect(finalCheck.canvasAlive, "canvas dead at end").toBe(true);
  const fatalConsole = consoleErrors.filter((e) => /external Instance|kIOGPU|ignored submissions/i.test(e));
  expect(fatalConsole, `fatal errors: ${JSON.stringify(fatalConsole)}`).toEqual([]);
  const watchdogFired = consoleWarns.filter((w) => /watchdog fired/i.test(w));
  console.log(
    `[pan-zoom] DONE — renderCalls=${finalCheck.renderCalls}, gpuErrors=${finalCheck.gpuErrors.length}, watchdogFired=${watchdogFired.length}`,
  );
});
