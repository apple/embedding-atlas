/**
 * Frontend fluency probe for the synthetic 300M-row Europe dataset.
 *
 * Pre-req: server is up on PORT (default 5055) serving
 * /tmp/gsa_bench/europe_300m.parquet via the fast loader.
 *
 * Validates the perf contract the user asked for:
 *   1. Tab survives the initial load + scatter pull.
 *   2. The atlas WebGPU canvas exists at non-zero size and the dataset is
 *      announced as 300M points (Mosaic finished its initial query).
 *   3. A 10-second simulated pan triggers many CSS-pan applies and at most
 *      a handful of GPU re-renders (verifies the recompute-during-pan fix).
 *   4. Color-by completes without crashing the tab.
 *
 * (We deliberately do NOT pixel-sample the WebGPU canvas — drawImage on a
 * non-preserveDrawingBuffer WebGPU context returns transparent in headless
 * Chrome, so the existing canvasNonBlankPixels probe is unreliable here.)
 *
 * Run::
 *
 *   PORT=5055 npx playwright test e2e/europe-300m.spec.ts \
 *     --project=perf-chrome --workers=1
 */

import { test, expect, type Page } from "@playwright/test";

const PORT = Number(process.env.PORT ?? 5055);
const URL = `http://127.0.0.1:${PORT}`;

async function readPanDbg(page: Page) {
  return page.evaluate(() => (window as any).__atlasPanDbg ?? null);
}

interface CanvasInfo {
  count: number;
  sizes: { w: number; h: number; cls: string }[];
  hasNonZero: boolean;
}

async function inspectCanvases(page: Page): Promise<CanvasInfo> {
  return page.evaluate(() => {
    const canvases = Array.from(document.querySelectorAll("canvas"));
    const sizes = canvases.map((c) => ({
      w: (c as HTMLCanvasElement).width,
      h: (c as HTMLCanvasElement).height,
      cls: c.className || c.getAttribute("data-id") || "(no class)",
    }));
    return {
      count: canvases.length,
      sizes,
      hasNonZero: sizes.some((s) => s.w > 0 && s.h > 0),
    };
  });
}

test("europe 300m frontend fluency", async ({ page }) => {
  test.setTimeout(420_000);
  const errors: { type: string; text: string }[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") {
      errors.push({ type: m.type(), text: m.text() });
    }
  });
  page.on("pageerror", (err) => errors.push({ type: "pageerror", text: err.message }));

  await page.goto(URL, { waitUntil: "domcontentloaded" });

  // Phase A: wait until the row-counter chip reads "300,000,000 points"
  // (Mosaic has finished its initial COUNT(*) round-trip) and the atlas
  // canvas has resolved to a non-zero size.
  const startedAt = Date.now();
  const POLL_DEADLINE = startedAt + 240_000;
  let lastInfo: CanvasInfo | null = null;
  let mosaicReady = false;
  while (Date.now() < POLL_DEADLINE) {
    await page.waitForTimeout(3_000);
    lastInfo = await inspectCanvases(page);
    const headerText = await page
      .locator("body")
      .innerText()
      .catch(() => "");
    mosaicReady = /300,000,000\s*points/.test(headerText);
    if (mosaicReady && lastInfo.hasNonZero) break;
  }
  console.log(`canvases at t+${((Date.now() - startedAt) / 1000) | 0}s: ${JSON.stringify(lastInfo)}`);
  expect(mosaicReady, "Mosaic never announced 300M point count").toBe(true);
  expect(lastInfo!.hasNonZero, "no canvas reached non-zero dimensions").toBe(true);

  await page.screenshot({ path: "e2e/test-results/europe-300m-initial.png", fullPage: true });

  // Phase B: simulate a 10-second pan; assert CSS-pan dominates over GPU
  // renders (this is the recompute-during-pan fix).
  // Pan path is at line "region 'Map'" — find it and pan inside it.
  const mapRegion = page.locator('[role="region"][aria-label="Map"], canvas').first();
  const box = await mapRegion.boundingBox();
  if (!box) throw new Error("no map region bbox");
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  await page.evaluate(() => {
    (window as any).__atlasPanDbg = undefined;
  });
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  const PAN_MS = 10_000;
  const t0 = Date.now();
  let dx = 0;
  while (Date.now() - t0 < PAN_MS) {
    dx = ((dx + 17) % 200) - 100;
    const dy = (dx * 0.3) | 0;
    await page.mouse.move(cx + dx, cy + dy);
    await page.waitForTimeout(50);
  }
  await page.mouse.up();
  await page.waitForTimeout(2_000);

  const dbg = await readPanDbg(page);
  console.log(`pan dbg after 10s drag: ${JSON.stringify(dbg)}`);
  expect(dbg, "pan instrumentation never registered").not.toBeNull();
  expect(dbg.cssPanApplied, "CSS-pan never triggered").toBeGreaterThan(20);
  // After the recompute-during-pan fix, mid-gesture renderCalls should be
  // tiny (just the release frame). Keep the bound generous to avoid
  // flakiness on slower CI: at least a 5× ratio of CSS-pans to renders.
  expect(dbg.renderCalls, `too many real renders mid-pan (${dbg.renderCalls})`).toBeLessThan(
    Math.max(5, dbg.cssPanApplied / 5),
  );

  // Tab survived the pan.
  const info2 = await inspectCanvases(page);
  expect(info2.hasNonZero, "canvases gone after pan (tab crashed?)").toBe(true);
  await page.screenshot({ path: "e2e/test-results/europe-300m-after-pan.png", fullPage: true });

  // Phase C: trigger color-by on `category`.
  const colorSelect = page
    .locator("select")
    .filter({ hasText: /category|^--$/ })
    .first();
  await colorSelect.selectOption({ label: /category/ }).catch(async () => {
    // fallback: enumerate
    const selects = await page.locator("select").all();
    for (const sel of selects) {
      const opts = await sel.locator("option").allInnerTexts();
      if (opts.some((o) => o.includes("category"))) {
        await sel.selectOption({ label: opts.find((o) => o.includes("category"))! });
        return;
      }
    }
    throw new Error("category select not found");
  });
  await page.waitForTimeout(40_000); // ALTER+UPDATE on 300M ~ 10-15s + render
  const info3 = await inspectCanvases(page);
  expect(info3.hasNonZero, "canvases gone after color-by").toBe(true);
  await page.screenshot({ path: "e2e/test-results/europe-300m-color.png", fullPage: true });

  console.log(`=== console errors (last 10) ===`);
  for (const e of errors.slice(-10)) {
    console.log(`[${e.type}] ${e.text.slice(0, 240)}`);
  }
});
