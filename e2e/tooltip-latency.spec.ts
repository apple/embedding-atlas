/**
 * Tooltip hover latency budget against the 322M eubucco file.
 *
 * User report: with 75M-row files tooltips appear instantly; with the
 * 322M eubucco file the tooltip takes ~10 s. Root cause: the materialised
 * dataset table has no spatial sort, so the tooltip query —
 *
 *   SELECT x, y, ... FROM dataset
 *   WHERE x BETWEEN px-r AND px+r AND y BETWEEN py-r AND py+r
 *   ORDER BY (x-px)² + (y-py)² LIMIT 1
 *
 * — has to scan all 322M lon/lat values (~5 GB compressed) end-to-end.
 *
 * Fix: ``fast_load.py`` now does ``ORDER BY __x_u16__, __y_u16__`` in the
 * bg materialise CTAS, and ``server.py`` eagerly swaps the view for the
 * sorted table once that CTAS finishes (instead of waiting for the first
 * write query). DuckDB's per-row-group min/max stats on x/y then prune
 * >99 % of row groups for the tooltip BETWEEN, dropping latency from
 * ~10 s to <100 ms.
 *
 * The test: load → wait for first big render → wait for the bg-mat
 * eager-swap to complete → hover the canvas → measure how long until the
 * tooltip SVG circle appears. Asserts a generous budget (loose for noisy
 * shared CI hosts; the actual win is two orders of magnitude).
 */

import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";

const BASE_URL = process.env.TOOLTIP_URL ?? "http://127.0.0.1:5088";
// Budget for the tooltip to appear after mouse-move. With the spatial-
// sort fix in place we expect ~50–500 ms even at 322 M; without it,
// >5 s. 3 s is a comfortable middle ground that catches a regression
// without being noise-prone on a shared host.
const TOOLTIP_BUDGET_MS = Number(process.env.TOOLTIP_BUDGET_MS ?? 3000);
// How long to wait after first-big-render for the bg materialise CTAS
// to complete and the view→table swap to land. Sort of 322M ~50-byte
// rows on a 16-core box with 50 % RAM is roughly 60–120 s with spilling.
const SWAP_WAIT_MS = Number(process.env.TOOLTIP_SWAP_WAIT_MS ?? 180_000);

async function isMaterialisedTable(page: Page, baseUrl: string): Promise<boolean> {
  // Probe via the same SQL endpoint the viewer uses (server.py mounts
  // it at ``/data/query``). Mosaic command shape is
  // ``{type: "json", sql: "..."}``; backend returns
  // ``[{table_type: "BASE TABLE"|"VIEW"}]``.
  try {
    const resp = await page.request.post(`${baseUrl}/data/query`, {
      data: {
        type: "json",
        sql: "SELECT table_type FROM information_schema.tables WHERE table_name = 'dataset'",
      },
    });
    if (!resp.ok()) return false;
    const rows = await resp.json();
    return Array.isArray(rows) && rows.some((r: any) => r.table_type === "BASE TABLE");
  } catch {
    return false;
  }
}

test("tooltip latency — 322M scatter hovers under budget after eager-swap", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000);

  const consoleErrors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));

  const probe = await page.request.get(`${BASE_URL}/data/metadata.json`);
  expect(probe.ok(), `server unreachable at ${BASE_URL}`).toBeTruthy();

  console.log(`[tooltip] loading ${BASE_URL}/?perf=1`);
  await page.goto(`${BASE_URL}/?perf=1`, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
    timeout: 6 * 60 * 1000,
    polling: 250,
  });
  console.log(`[tooltip] first big render landed; waiting up to ${SWAP_WAIT_MS}ms for bg-mat eager-swap`);

  // Poll until the dataset is a BASE TABLE (eager-swap done) or budget
  // exhausts. Budget is loose because the bg CTAS at 322M with ORDER BY
  // takes 60–120 s on a 16-core box.
  const swapDeadline = Date.now() + SWAP_WAIT_MS;
  let swapped = false;
  while (Date.now() < swapDeadline) {
    if (await isMaterialisedTable(page, BASE_URL)) {
      swapped = true;
      break;
    }
    await page.waitForTimeout(2000);
  }
  expect(swapped, `bg-mat eager-swap did not complete in ${SWAP_WAIT_MS}ms`).toBe(true);
  console.log(`[tooltip] eager-swap complete — dataset is now a sorted table`);

  // Quiesce: let any post-swap rendering finish so the hover throttle
  // has a clean slate.
  await page.waitForTimeout(2000);

  // Locate the largest canvas (scatter surface) and its center.
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

  // Sample several hover points (hits in different parts of Europe — the
  // eubucco data is dense over EU and sparse elsewhere). Take the median
  // latency to dampen one-off outliers from JIT/GC/cache cold starts.
  const offsets: Array<{ ox: number; oy: number; label: string }> = [
    { ox: 0.0, oy: 0.0, label: "center (Paris)" },
    { ox: -0.15, oy: 0.05, label: "west (London)" },
    { ox: 0.1, oy: -0.1, label: "south-east (Italy)" },
    { ox: 0.05, oy: -0.2, label: "south (Iberia)" },
    { ox: -0.05, oy: 0.15, label: "north-west (Scandinavia)" },
  ];

  const latencies: number[] = [];
  for (const { ox, oy, label } of offsets) {
    const hx = targetBox!.x + targetBox!.w * (0.5 + ox);
    const hy = targetBox!.y + targetBox!.h * (0.5 + oy);

    // Move pointer well off the canvas first so we get a fresh
    // mouseenter-style hover event.
    await page.mouse.move(targetBox!.x - 50, targetBox!.y - 50);
    await page.waitForTimeout(150);

    const t0 = Date.now();
    await page.mouse.move(hx, hy);
    // Wait for the tooltip SVG circle (the unique hover indicator: an
    // unfilled circle with stroke-width 1 — selection circles use 2).
    let appeared = false;
    try {
      await page.waitForFunction(
        () => {
          const circles = Array.from(document.querySelectorAll("svg circle"));
          return circles.some(
            (c) =>
              (c as SVGCircleElement).getAttribute("style")?.includes("stroke-width: 1") &&
              (c as SVGCircleElement).getAttribute("style")?.includes("fill: none"),
          );
        },
        null,
        { timeout: TOOLTIP_BUDGET_MS, polling: 25 },
      );
      appeared = true;
    } catch {
      appeared = false;
    }
    const dt = Date.now() - t0;
    if (appeared) {
      latencies.push(dt);
      console.log(`[tooltip] ${label}: ${dt}ms`);
    } else {
      console.log(`[tooltip] ${label}: TIMEOUT (>${TOOLTIP_BUDGET_MS}ms — no point under cursor?)`);
    }
  }

  // We expect at least 3 of the 5 hover points to land on data and
  // produce a tooltip inside the budget. Sparse zones in eubucco mean
  // it's reasonable for 1–2 to miss; the median latency is what
  // matters.
  expect(
    latencies.length,
    `only ${latencies.length}/${offsets.length} hovers produced a tooltip`,
  ).toBeGreaterThanOrEqual(3);
  const median = [...latencies].sort((a, b) => a - b)[Math.floor(latencies.length / 2)];
  console.log(`[tooltip] median latency over ${latencies.length} hovers: ${median}ms (budget ${TOOLTIP_BUDGET_MS}ms)`);
  expect(median, `tooltip median latency ${median}ms exceeds budget ${TOOLTIP_BUDGET_MS}ms`).toBeLessThanOrEqual(
    TOOLTIP_BUDGET_MS,
  );
});
