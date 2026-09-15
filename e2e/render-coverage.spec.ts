/**
 * Visual-regression gate: actual scatter pixels reach the screen.
 *
 * The other 322M e2e tests (u32-precision, tooltip-latency, pan-*)
 * passed even when the renderer produced an empty canvas because they
 * only check the ``__atlasFirstBigRenderGpuLogged`` flag — which fires
 * the moment the first GPU command buffer drains, regardless of whether
 * its output was empty. The "0 points displayed, side panel empty"
 * regression I shipped on Apr 26 2026 slipped through the existing
 * gate this way.
 *
 * This test is the actual gate. It samples the largest canvas's pixel
 * data after first big render + a generous quiesce, and asserts that a
 * meaningful fraction of pixels are non-background — i.e. there are
 * actually points on the screen.
 *
 * Backed by an HTTP sidecar at ``http://127.0.0.1:5088`` (override via
 * ``RENDER_COVERAGE_URL``). Skips with a clear message if the sidecar
 * is not reachable, so the test is safe to leave in the default suite.
 */

import { test, expect, type ConsoleMessage } from "@playwright/test";

const BASE_URL = process.env.RENDER_COVERAGE_URL ?? "http://127.0.0.1:5088";
const FIRST_RENDER_TIMEOUT_MS = Number(process.env.RENDER_COVERAGE_FIRST_RENDER_MS ?? 6 * 60 * 1000);
// Post-first-render wait. Enough to let any deferred refinement, side-
// panel discovery, or rebind chain finish writing pixels.
const QUIESCE_MS = Number(process.env.RENDER_COVERAGE_QUIESCE_MS ?? 5_000);
// Minimum fraction of canvas pixels that must be non-background. The
// eubucco scatter at default zoom paints a Europe-shaped land mass —
// well over 5% non-empty, but on a sparse dataset this could drop. 1%
// is a conservative floor that still trips the "scatter is empty" bug
// (which produces ~0% coverage — only the basemap, no points).
const MIN_COVERAGE_FRACTION = Number(process.env.RENDER_COVERAGE_MIN ?? 0.01);

test("render coverage — scatter actually paints points on the canvas", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000);

  const consoleErrors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));

  // Sidecar reachability probe — early-skip with a useful message if
  // the dev forgot to start the server.
  let metaOk = false;
  try {
    const probe = await page.request.get(`${BASE_URL}/data/metadata.json`, { timeout: 3000 });
    metaOk = probe.ok();
  } catch {
    metaOk = false;
  }
  test.skip(!metaOk, `sidecar unreachable at ${BASE_URL}`);

  console.log(`[render] loading ${BASE_URL}/?perf=1`);
  await page.goto(`${BASE_URL}/?perf=1`, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: FIRST_RENDER_TIMEOUT_MS,
    polling: 250,
  });
  console.log(`[render] first big render landed; quiescing ${QUIESCE_MS}ms`);
  await page.waitForTimeout(QUIESCE_MS);

  // We can't read pixels from the WebGPU canvas via ``drawImage``
  // (its swap-chain texture isn't accessible from a 2D context — it
  // composites the same content but the JS-readable backing buffer
  // is empty). Instead we take a Playwright OS-level screenshot of
  // the visible page (which the compositor renders normally,
  // including WebGPU output) and analyse THAT. The screenshot is
  // PNG bytes; we feed them back into the page as an ``Image`` so
  // Canvas2D's ``getImageData`` can read them.
  const pngBytes = await page.screenshot({ fullPage: false });
  const dataUrl = `data:image/png;base64,${pngBytes.toString("base64")}`;
  const result = await page.evaluate(async (url) => {
    const img = new Image();
    await new Promise<void>((res, rej) => {
      img.onload = () => res();
      img.onerror = (e) => rej(new Error(`image load failed: ${String(e)}`));
      img.src = url;
    });
    // Resample to a fixed size so the heuristic is independent of
    // viewport scaling and the byte cost is bounded.
    const SAMPLE_W = 800;
    const SAMPLE_H = 600;
    const scratch = document.createElement("canvas");
    scratch.width = SAMPLE_W;
    scratch.height = SAMPLE_H;
    const ctx = scratch.getContext("2d", { willReadFrequently: true });
    if (!ctx) return { ok: false, reason: "no 2d context" };
    ctx.drawImage(img, 0, 0, SAMPLE_W, SAMPLE_H);
    const data = ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H).data;

    const total = SAMPLE_W * SAMPLE_H;
    // Build a 4-bit-per-channel RGB histogram (4096 buckets) and pick
    // the dominant bucket as "background". Pixels whose RGB delta
    // from the background exceeds 75 (roughly an octant in 8-bit
    // space) count as "scatter content". The empty-canvas regression
    // (0 points displayed) collapses to ~0 % coverage; the healthy
    // 322 M Europe scatter at default zoom paints ~10-30 % non-bg
    // because the continent fills a large area of the page.
    const buckets = new Uint32Array(4096);
    for (let i = 0; i < data.length; i += 4) {
      const r4 = data[i] >> 4;
      const g4 = data[i + 1] >> 4;
      const b4 = data[i + 2] >> 4;
      buckets[(r4 << 8) | (g4 << 4) | b4]++;
    }
    let domIdx = 0;
    let domCount = 0;
    for (let k = 0; k < buckets.length; k++) {
      if (buckets[k] > domCount) {
        domCount = buckets[k];
        domIdx = k;
      }
    }
    const domR = ((domIdx >> 8) & 0xf) << 4;
    const domG = ((domIdx >> 4) & 0xf) << 4;
    const domB = (domIdx & 0xf) << 4;
    let nonBg = 0;
    for (let i = 0; i < data.length; i += 4) {
      const dr = Math.abs(data[i] - domR);
      const dg = Math.abs(data[i + 1] - domG);
      const db = Math.abs(data[i + 2] - domB);
      if (dr + dg + db > 75) nonBg++;
    }
    return {
      ok: true,
      sampleW: SAMPLE_W,
      sampleH: SAMPLE_H,
      total,
      domR,
      domG,
      domB,
      nonBgPixels: nonBg,
      coverage: nonBg / total,
      domShare: domCount / total,
    };
  }, dataUrl);

  console.log(`[render] coverage stats: ${JSON.stringify(result)}`);

  // Save a screenshot regardless so failures have a visual artefact to
  // inspect alongside the numerical output.
  await page.screenshot({ path: "e2e/test-results/render-coverage.png", fullPage: false });

  expect(result.ok, `pixel sample failed: ${(result as any).reason ?? "unknown"}`).toBe(true);
  if (result.ok) {
    expect(
      result.coverage,
      `scatter coverage ${(result.coverage * 100).toFixed(2)}% < threshold ${(MIN_COVERAGE_FRACTION * 100).toFixed(2)}% — points are not painting`,
    ).toBeGreaterThan(MIN_COVERAGE_FRACTION);
  }

  // Don't fail on console errors here — that's the u32-wire-verify
  // job. We only care about pixel coverage.
  if (consoleErrors.length) {
    console.log(`[render] (informational) console errors: ${consoleErrors.length}`);
  }
});
