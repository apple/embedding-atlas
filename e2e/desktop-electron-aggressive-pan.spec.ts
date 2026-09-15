/**
 * Aggressive pan harness — designed to reliably reproduce the
 * `kIOGPUCommandBufferCallbackErrorTimeout` crash that the gentler
 * `desktop-electron-real-pan.spec.ts` misses. The user observed that 3+
 * pans in dense areas (~200M points visible) are needed to trigger.
 *
 * Differences vs the gentler harness:
 *   - Resets viewport to a deliberately dense area (Berlin metro)
 *     before panning so 100M+ points are guaranteed in-viewport.
 *   - Pans 12 times rapidly with NO settle window between most pans —
 *     only a 200ms breath every 4 pans.
 *   - Interleaves pan + zoom cycles to thrash the renderer's downsample
 *     compute path (each viewport change forces a full re-compute on
 *     pan-release, since `skipDownsampleCompute` only skips DURING the
 *     gesture).
 *   - Drives high-frequency wheel events (trackpad emulation) over the
 *     densest visible region.
 *   - Captures `kIOGPUCommandBufferCallback*` errors and exits IMMEDIATELY
 *     on the first hit — no point continuing after the device is poisoned.
 *
 * Run::
 *
 *   DATASET=/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet \
 *     npx playwright test e2e/desktop-electron-aggressive-pan.spec.ts \
 *     --project=desktop-electron --workers=1
 */

import { test, expect, _electron as electron, type ConsoleMessage } from "@playwright/test";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPO = join(__dirname, "..");
const DESKTOP = join(REPO, "apps/desktop");
const DATASET = process.env.DATASET ?? "/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet";

const PACKAGED_BIN = join(DESKTOP, "release/mac-arm64/Geospatial Atlas.app/Contents/MacOS/Geospatial Atlas");

const OUT_DIR = join(REPO, "e2e/test-results/aggressive-pan");
mkdirSync(OUT_DIR, { recursive: true });

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

test("aggressive pan crash repro", async () => {
  test.setTimeout(15 * 60 * 1000);

  expect(existsSync(DATASET), `dataset missing at ${DATASET}`).toBe(true);
  expect(existsSync(PACKAGED_BIN), `packaged .app missing at ${PACKAGED_BIN}`).toBe(true);

  const t0 = Date.now();
  const stdoutLog: string[] = [];
  const stderrLog: string[] = [];
  const crashes: CrashSignal[] = [];

  const app = await electron.launch({
    cwd: DESKTOP,
    args: [DATASET],
    executablePath: PACKAGED_BIN,
    env: {
      ...process.env,
      GEOSPATIAL_ATLAS_INITIAL_DATASET: DATASET,
      GEOSPATIAL_ATLAS_DEBUG_PORT: "9223",
      GEOSPATIAL_ATLAS_METRICS_INTERVAL: "2000",
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
  let panSummary: string[] = [];

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

    // Wait for sidecar URL.
    const sidecarDeadline = Date.now() + 240_000;
    while (Date.now() < sidecarDeadline) {
      await win.waitForTimeout(800);
      const url = win.url();
      if (/127\.0\.0\.1:\d+/.test(url) && !url.includes(":1420")) {
        console.log(`[harness] sidecar URL: ${url}`);
        break;
      }
    }

    // Wait for first big render (5 min generous on the 322M bootstrap).
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

    if (renderLanded) {
      await win.waitForTimeout(2_000);
      await win.screenshot({ path: join(OUT_DIR, "01-baseline.png") });

      // Find the largest canvas (the WebGPU scatter overlay).
      const canvases = await win.evaluate(() => {
        return Array.from(document.querySelectorAll("canvas")).map((c) => {
          const r = (c as HTMLCanvasElement).getBoundingClientRect();
          return { x: r.x, y: r.y, w: r.width, h: r.height };
        });
      });
      const target = canvases.length > 0 ? canvases.reduce((a, b) => (a.w * a.h > b.w * b.h ? a : b)) : null;
      console.log(`[harness] target canvas: ${JSON.stringify(target)}`);
      expect(target, "no canvas found").not.toBeNull();
      const cx = target!.x + target!.w / 2;
      const cy = target!.y + target!.h / 2;

      // Helper: dump app metrics as a side-channel diagnostic.
      const sample = async (label: string) => {
        const cov = await win.screenshot({ fullPage: false }).then((png) =>
          win.evaluate(
            async (b64) => {
              const img = new Image();
              await new Promise<void>((res, rej) => {
                img.onload = () => res();
                img.onerror = () => rej(new Error("img"));
                img.src = b64;
              });
              const cv = document.createElement("canvas");
              cv.width = 800;
              cv.height = 600;
              const ctx = cv.getContext("2d", { willReadFrequently: true });
              if (!ctx) return -1;
              ctx.drawImage(img, 0, 0, 800, 600);
              const d = ctx.getImageData(0, 0, 800, 600).data;
              let nonWhite = 0;
              for (let p = 0; p < d.length; p += 4) {
                if (d[p] < 240 || d[p + 1] < 240 || d[p + 2] < 240) nonWhite++;
              }
              return nonWhite / (800 * 600);
            },
            `data:image/png;base64,${png.toString("base64")}`,
          ),
        );
        console.log(`[harness] ${label} coverage=${(cov * 100).toFixed(1)}%`);
        return cov;
      };

      const checkCrash = (label: string) => {
        if (crashes.length > 0) {
          throw new Error(`CRASH after ${label}: ${JSON.stringify(crashes[0])}`);
        }
      };

      // ---- Phase 1: Zoom OUT to maximize visible point count ----
      // Map starts at user's last viewport. We zoom out 6x to ensure
      // we're at world/continent scale where 100M+ are visible.
      console.log(`[harness] phase 1: zoom out to world scale`);
      {
        const start = Date.now();
        await win.mouse.move(cx, cy);
        for (let i = 0; i < 30; i++) {
          // Negative deltaY = zoom out in maplibre wheel handler
          await win.mouse.wheel(0, 80);
        }
        await win.waitForTimeout(2_500);
        await win.screenshot({ path: join(OUT_DIR, "02-zoomed-out.png") });
        const cov = await sample("zoom-out");
        panSummary.push(`zoom-out=${Date.now() - start}ms cov=${(cov * 100).toFixed(1)}%`);
        checkCrash("zoom-out");
      }

      // ---- Phase 2: 12 RAPID consecutive flick pans ----
      // No settle between pans except a 200ms breath every 4. The point
      // is to keep the renderer constantly re-running its compute on
      // pan-release while the GPU process is already under load.
      console.log(`[harness] phase 2: 12 rapid flick pans`);
      for (let i = 0; i < 12; i++) {
        const start = Date.now();
        const dirX = (i % 4) - 2; // -2, -1, 0, 1
        const dirY = ((i + 1) % 3) - 1; // -1, 0, 1
        await win.mouse.move(cx, cy);
        await win.mouse.down();
        for (let s = 1; s <= 24; s++) {
          await win.mouse.move(cx + s * 12 * dirX, cy + s * 8 * dirY);
        }
        await win.mouse.up();
        if (i % 4 === 3) await win.waitForTimeout(200);
        const elapsed = Date.now() - start;
        if (i === 5 || i === 11) {
          await win.waitForTimeout(800);
          await win.screenshot({ path: join(OUT_DIR, `03-pan-${i.toString().padStart(2, "0")}.png`) });
        }
        panSummary.push(`pan-${i}=${elapsed}ms dir=(${dirX},${dirY})`);
        checkCrash(`pan-${i}`);
      }

      // ---- Phase 3: Trackpad-style high-frequency wheel pan ----
      // 120 wheel events with no breaks. This is the gesture style
      // that mimics a real two-finger trackpad swipe.
      console.log(`[harness] phase 3: trackpad-style 120 wheel events`);
      {
        const start = Date.now();
        await win.mouse.move(cx, cy);
        for (let burst = 0; burst < 8; burst++) {
          for (let s = 0; s < 15; s++) {
            await win.mouse.wheel(burst % 2 === 0 ? 30 : -30, burst % 3 === 0 ? 25 : -25);
          }
          // Tiny breath between bursts — real trackpad has 16ms inter-event
          // gaps. Without this maplibre coalesces and drops events.
          await win.waitForTimeout(16);
        }
        await win.waitForTimeout(2_000);
        await win.screenshot({ path: join(OUT_DIR, "04-after-trackpad.png") });
        const cov = await sample("trackpad");
        panSummary.push(`trackpad=${Date.now() - start}ms cov=${(cov * 100).toFixed(1)}%`);
        checkCrash("trackpad");
      }

      // ---- Phase 4: Zoom-pan-zoom cycles ----
      // Forces fresh viewport_cull on every transition.
      console.log(`[harness] phase 4: 5 zoom-pan-zoom cycles`);
      for (let cyc = 0; cyc < 5; cyc++) {
        const start = Date.now();
        // Zoom in
        await win.mouse.move(cx, cy);
        for (let s = 0; s < 8; s++) await win.mouse.wheel(0, -40);
        await win.waitForTimeout(300);
        // Pan
        await win.mouse.down();
        for (let s = 1; s <= 20; s++) {
          await win.mouse.move(cx + s * 10 * (cyc % 2 ? 1 : -1), cy);
        }
        await win.mouse.up();
        await win.waitForTimeout(300);
        // Zoom back out
        for (let s = 0; s < 8; s++) await win.mouse.wheel(0, 40);
        await win.waitForTimeout(500);
        panSummary.push(`zoom-pan-zoom-${cyc}=${Date.now() - start}ms`);
        checkCrash(`zoom-pan-zoom-${cyc}`);
      }

      // ---- Phase 5: Final settle + screenshot ----
      await win.waitForTimeout(3_000);
      await win.screenshot({ path: join(OUT_DIR, "05-final.png") });
      console.log(`[harness] all phases complete; crashes=${crashes.length}`);
    }
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
            panSummary,
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

  console.log(`[harness] FINAL renderLanded=${renderLanded}`);
  console.log(`[harness] FINAL pans:\n  ${panSummary.join("\n  ")}`);
  console.log(`[harness] FINAL crashes: ${crashes.length}`);
  for (const c of crashes.slice(0, 20)) {
    console.log(`  CRASH(${c.kind}) +${c.t - t0}ms ${c.source}: ${c.line}`);
  }

  expect(renderLanded, "first big render never landed").toBe(true);
  expect(crashes.length, `${crashes.length} crash signals; first: ${JSON.stringify(crashes[0] ?? null)}`).toBe(0);
});
