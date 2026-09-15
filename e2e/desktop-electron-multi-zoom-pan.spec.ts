/**
 * Multi-zoom 10-second-interval pan harness — the user's contract:
 *   "panning should work on every zoom level and rendering (after
 *    initial render) should work in 10s intervals!"
 *
 * Pattern:
 *   1. Cold load 322 M, wait for first big render.
 *   2. For each zoom level (out / mid / in):
 *        - Pan in 4 directions, 10 s settle between each pan.
 *        - After EACH 10 s settle: screenshot + assert
 *            (a) zero crash signals from main process,
 *            (b) non-white coverage > MIN_COVERAGE — proves points
 *                actually rendered (not just empty basemap).
 *   3. Final summary screenshot.
 *
 * Run::
 *
 *   DATASET=/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet \
 *     npx playwright test e2e/desktop-electron-multi-zoom-pan.spec.ts \
 *     --project=desktop-electron --workers=1
 */

import { test, expect, _electron as electron, type ConsoleMessage } from "@playwright/test";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPO = join(__dirname, "..");
const DESKTOP = join(REPO, "apps/desktop");
const DATASET = process.env.DATASET ?? "/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet";

const PACKAGED_BIN = join(DESKTOP, "release/mac-arm64/Geospatial Atlas.app/Contents/MacOS/Geospatial Atlas");

const OUT_DIR = join(REPO, "e2e/test-results/multi-zoom-pan");
mkdirSync(OUT_DIR, { recursive: true });

// Coverage threshold: count "scatter ink" pixels — either truly dark
// (R+G+B < 120, e.g. dense additive cores, labels, borders) OR
// saturated blue (B > R+60 && R+G+B < 500, e.g. additive blue blends
// from the scatter overlay). The basemap ocean is *light* blue (B-R
// ≈ 45, R+G+B ≈ 610) so it stays below the threshold; only the
// scatter overlay produces saturated blue at world-out zoom. With
// real scatter visible (millions of additive dots over the basemap)
// this jumps from a basemap-only floor of ~0.1% to several %.
//
// World-out only captures Europe (a small slice of the canvas), so
// 0.5% is the practical floor. At country/city zooms the scatter
// dominates and coverage easily exceeds 5%.
const MIN_COVERAGE = 0.005;

interface CrashSignal {
  t: number;
  source: "stdout" | "stderr";
  line: string;
  kind: string;
}

const CRASH_PATTERNS: { re: RegExp; kind: string }[] = [
  { re: /\[renderer-gone\]/, kind: "renderer-gone" },
  { re: /\[child-gone\] type=GPU/, kind: "gpu-process-gone" },
  { re: /\[renderer-unresponsive\]/, kind: "renderer-unresponsive" },
  { re: /atlas-gpu\] device\.lost/, kind: "device-lost" },
  { re: /atlas-gpu\] uncapturederror/, kind: "uncaptured-gpu-error" },
  { re: /kIOGPUCommandBufferCallbackErrorTimeout/, kind: "metal-watchdog-timeout" },
  { re: /kIOGPUCommandBufferCallbackErrorSubmissionsIgnored/, kind: "metal-poison-cascade" },
  { re: /Out of memory/i, kind: "oom" },
  { re: /WebGPU.*lost/i, kind: "webgpu-lost" },
];

function detectCrash(line: string, source: "stdout" | "stderr"): CrashSignal | null {
  for (const { re, kind } of CRASH_PATTERNS) {
    if (re.test(line)) return { t: Date.now(), source, line: line.trim(), kind };
  }
  return null;
}

interface PanResult {
  zoomLevel: string;
  panIdx: number;
  dirX: number;
  dirY: number;
  coverage: number;
  ok: boolean;
  reason?: string;
}

test("multi-zoom 10s-interval pan", async () => {
  // 3 zoom levels × 4 pans × 10s = 120s + bootstrap (~120s) + slack
  test.setTimeout(25 * 60 * 1000);

  expect(existsSync(DATASET)).toBe(true);
  expect(existsSync(PACKAGED_BIN)).toBe(true);

  const t0 = Date.now();
  const stdoutLog: string[] = [];
  const stderrLog: string[] = [];
  const crashes: CrashSignal[] = [];
  const results: PanResult[] = [];

  const app = await electron.launch({
    cwd: DESKTOP,
    args: [DATASET],
    executablePath: PACKAGED_BIN,
    env: {
      ...process.env,
      GEOSPATIAL_ATLAS_INITIAL_DATASET: DATASET,
      GEOSPATIAL_ATLAS_DEBUG_PORT: "9223",
      GEOSPATIAL_ATLAS_METRICS_INTERVAL: "5000",
      ELECTRON_ENABLE_LOGGING: "1",
    },
  });

  const proc = app.process();
  const ingestStdout = (chunk: Buffer | string, source: "stdout" | "stderr") => {
    const text = typeof chunk === "string" ? chunk : chunk.toString("utf8");
    for (const line of text.split(/\r?\n/)) {
      if (!line) continue;
      (source === "stdout" ? stdoutLog : stderrLog).push(`+${Date.now() - t0}ms ${line}`);
      const c = detectCrash(line, source);
      if (c) crashes.push(c);
    }
  };
  proc.stdout?.on("data", (b) => ingestStdout(b, "stdout"));
  proc.stderr?.on("data", (b) => ingestStdout(b, "stderr"));

  let renderLanded = false;

  try {
    const win = await app.firstWindow();
    win.on("console", (msg: ConsoleMessage) => {
      const line = `[console-${msg.type()}] ${msg.text().slice(0, 600)}`;
      stdoutLog.push(`+${Date.now() - t0}ms ${line}`);
      const c = detectCrash(line, "stdout");
      if (c) crashes.push(c);
    });
    win.on("pageerror", (e) => {
      const line = `[pageerror] ${e.name}: ${e.message}`;
      stderrLog.push(`+${Date.now() - t0}ms ${line}`);
      crashes.push({ t: Date.now(), source: "stderr", line, kind: "pageerror" });
    });

    // Wait for sidecar.
    const sidecarDeadline = Date.now() + 240_000;
    while (Date.now() < sidecarDeadline) {
      await win.waitForTimeout(800);
      const url = win.url();
      if (/127\.0\.0\.1:\d+/.test(url) && !url.includes(":1420")) {
        console.log(`[harness] sidecar URL: ${url}`);
        break;
      }
    }

    // First big render.
    try {
      await win.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
        timeout: 5 * 60 * 1000,
        polling: 250,
      });
      renderLanded = true;
      console.log(`[harness] first big render landed at +${Date.now() - t0}ms`);
    } catch (e) {
      console.log(`[harness] first big render NEVER landed: ${(e as Error).message}`);
    }

    if (crashes.length > 0) {
      throw new Error(`crash during cold load: ${JSON.stringify(crashes[0])}`);
    }
    if (!renderLanded) {
      throw new Error("first big render never landed");
    }

    await win.waitForTimeout(2_000);
    await win.screenshot({ path: join(OUT_DIR, "00-baseline.png") });

    const canvases = await win.evaluate(() =>
      Array.from(document.querySelectorAll("canvas")).map((c) => {
        const r = (c as HTMLCanvasElement).getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height };
      }),
    );
    const target = canvases.length > 0 ? canvases.reduce((a, b) => (a.w * a.h > b.w * b.h ? a : b)) : null;
    expect(target).not.toBeNull();
    const cx = target!.x + target!.w / 2;
    const cy = target!.y + target!.h / 2;
    console.log(`[harness] target canvas: ${JSON.stringify(target)}`);

    // Crop the screenshot to ONLY the scatter canvas region — exclude
    // the sidebar (which has chart bars / text / borders that would
    // pollute the dark-pixel count) and the top toolbar (search bar).
    // Then count near-black pixels: scatter renders as dark dots,
    // basemap is mid-tone (tans/blues), so dark pixels ~= scatter ink.
    const sampleCoverage = async (canvasRect: { x: number; y: number; w: number; h: number }): Promise<number> => {
      try {
        const png = await win.screenshot({
          fullPage: false,
          clip: { x: canvasRect.x, y: canvasRect.y, width: canvasRect.w, height: canvasRect.h },
        });
        return await win.evaluate(
          async (b64) => {
            const img = new Image();
            await new Promise<void>((res, rej) => {
              img.onload = () => res();
              img.onerror = () => rej(new Error("img"));
              img.src = b64;
            });
            const W = img.width;
            const H = img.height;
            const cv = document.createElement("canvas");
            cv.width = W;
            cv.height = H;
            const ctx = cv.getContext("2d", { willReadFrequently: true });
            if (!ctx) return -1;
            // Pre-fill white so any transparent pixels in the screenshot
            // (Playwright sometimes emits alpha=0 from a hung renderer) are
            // forced opaque-white, not opaque-zero. Without this, an all-
            // transparent canvas would read as RGBA(0,0,0,0) → counted as
            // pure black and report 100 % "dark coverage".
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, W, H);
            ctx.drawImage(img, 0, 0, W, H);
            const d = ctx.getImageData(0, 0, W, H).data;
            let dark = 0;
            for (let p = 0; p < d.length; p += 4) {
              const r = d[p],
                g = d[p + 1],
                b = d[p + 2];
              // Truly dark (cores, borders, labels) OR saturated blue
              // (additive scatter blend over basemap). Basemap ocean is
              // light blue (B-R ≈ 45) so it fails the B > R+60 test.
              if (r + g + b < 120 || (b > r + 60 && r + g + b < 500)) dark++;
            }
            return dark / (W * H);
          },
          `data:image/png;base64,${png.toString("base64")}`,
        );
      } catch {
        return -1;
      }
    };

    // Helper: deliberate single-pan gesture (~400ms drag).
    const singlePan = async (dx: number, dy: number) => {
      await win.mouse.move(cx, cy);
      await win.mouse.down();
      const STEPS = 40;
      for (let s = 1; s <= STEPS; s++) {
        await win.mouse.move(cx + (dx * s) / STEPS, cy + (dy * s) / STEPS);
        await win.waitForTimeout(10);
      }
      await win.mouse.up();
    };

    const setZoom = async (deltaY: number, ticks: number) => {
      await win.mouse.move(cx, cy);
      for (let i = 0; i < ticks; i++) {
        await win.mouse.wheel(0, deltaY);
        await win.waitForTimeout(20);
      }
    };

    const ZOOM_LEVELS = [
      { label: "world-out", setupZoom: 30, deltaY: 80 }, // zoom OUT (deltaY > 0)
      { label: "country-mid", setupZoom: 15, deltaY: -40 }, // zoom IN partially
      // city-in: 10 ticks → "town/district" scale (~10 km viewport).
      // Earlier 20 ticks landed us at <1 km, where rural pans dropped
      // outside any building cluster and the renderer correctly drew
      // 0 points — a vacuous-but-failed coverage assert.
      { label: "city-in", setupZoom: 10, deltaY: -50 },
    ];

    const PANS = [
      { dx: -200, dy: 0 },
      { dx: 200, dy: 0 },
      { dx: 0, dy: -150 },
      { dx: 0, dy: 150 },
    ];

    let screenshotIdx = 1;
    for (const zoom of ZOOM_LEVELS) {
      console.log(`\n[harness] === zoom level: ${zoom.label} (ticks=${zoom.setupZoom} delta=${zoom.deltaY}) ===`);

      // Apply zoom (relative to current state).
      await setZoom(zoom.deltaY, zoom.setupZoom);
      await win.waitForTimeout(10_000);
      const cov = await sampleCoverage(target!);
      const setupOk = crashes.length === 0 && cov >= MIN_COVERAGE;
      console.log(
        `[harness] zoom ${zoom.label} settle: cov=${(cov * 100).toFixed(2)}% crashes=${crashes.length} ok=${setupOk}`,
      );
      await win.screenshot({
        path: join(OUT_DIR, `${String(screenshotIdx++).padStart(2, "0")}-zoom-${zoom.label}.png`),
      });
      results.push({
        zoomLevel: zoom.label,
        panIdx: -1,
        dirX: 0,
        dirY: 0,
        coverage: cov,
        ok: setupOk,
        reason: setupOk ? undefined : crashes.length > 0 ? "crash" : "low-coverage",
      });
      if (!setupOk) {
        if (crashes.length > 0) throw new Error(`CRASH after ${zoom.label} setup: ${JSON.stringify(crashes[0])}`);
        throw new Error(`LOW COVERAGE (${(cov * 100).toFixed(2)}%) after ${zoom.label} setup`);
      }

      for (let p = 0; p < PANS.length; p++) {
        const { dx, dy } = PANS[p];
        console.log(`[harness] ${zoom.label} pan ${p + 1}/${PANS.length}: (${dx}, ${dy})`);
        await singlePan(dx, dy);
        // Exact 10s settle for each pan as per user contract.
        await win.waitForTimeout(10_000);
        const cov = await sampleCoverage(target!);
        const ok = crashes.length === 0 && cov >= MIN_COVERAGE;
        console.log(
          `[harness]   ${zoom.label} pan ${p + 1} done: cov=${(cov * 100).toFixed(2)}% crashes=${crashes.length} ok=${ok}`,
        );
        await win.screenshot({
          path: join(OUT_DIR, `${String(screenshotIdx++).padStart(2, "0")}-${zoom.label}-pan-${p + 1}.png`),
        });
        results.push({
          zoomLevel: zoom.label,
          panIdx: p,
          dirX: dx,
          dirY: dy,
          coverage: cov,
          ok,
          reason: ok ? undefined : crashes.length > 0 ? "crash" : "low-coverage",
        });
        if (!ok) {
          if (crashes.length > 0) {
            throw new Error(`CRASH after ${zoom.label} pan ${p + 1}: ${JSON.stringify(crashes[0])}`);
          }
          throw new Error(`LOW COVERAGE (${(cov * 100).toFixed(2)}%) after ${zoom.label} pan ${p + 1}`);
        }
      }
    }

    await win.screenshot({ path: join(OUT_DIR, `${String(screenshotIdx++).padStart(2, "0")}-final.png`) });
    console.log(`\n[harness] all zoom levels + pans complete`);
  } finally {
    try {
      writeFileSync(join(OUT_DIR, "stdout.log"), stdoutLog.join("\n"));
      writeFileSync(join(OUT_DIR, "stderr.log"), stderrLog.join("\n"));
      writeFileSync(
        join(OUT_DIR, "summary.json"),
        JSON.stringify(
          {
            durationMs: Date.now() - t0,
            renderLanded,
            results,
            crashes,
            stdoutLines: stdoutLog.length,
            stderrLines: stderrLog.length,
          },
          null,
          2,
        ),
      );
    } catch {}
    try {
      await app.close();
    } catch {}
  }

  console.log(`\n[harness] FINAL renderLanded=${renderLanded}`);
  console.log(`[harness] FINAL results:`);
  for (const r of results) {
    const tag = r.panIdx < 0 ? `${r.zoomLevel} setup` : `${r.zoomLevel} pan${r.panIdx + 1}`;
    console.log(
      `  ${tag.padEnd(24)} cov=${(r.coverage * 100).toFixed(1).padStart(5)}% ok=${r.ok}${r.reason ? ` reason=${r.reason}` : ""}`,
    );
  }
  console.log(`[harness] FINAL crashes: ${crashes.length}`);
  for (const c of crashes.slice(0, 20)) {
    console.log(`  CRASH(${c.kind}) +${c.t - t0}ms ${c.source}: ${c.line}`);
  }

  expect(renderLanded).toBe(true);
  expect(crashes.length).toBe(0);
  for (const r of results) {
    expect(r.ok, `${r.zoomLevel} ${r.panIdx < 0 ? "setup" : `pan${r.panIdx + 1}`}: ${r.reason ?? "?"}`).toBe(true);
  }
});
