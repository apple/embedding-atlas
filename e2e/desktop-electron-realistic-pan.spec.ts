/**
 * Realistic-cadence pan harness — what a real human does.
 *
 * Pattern: ONE pan → wait 30 s for the full re-compute to settle → next
 * pan. This exercises the pan-release path (where ``skipDownsampleCompute``
 * flips back to false and the FULL 3-pass downsample compute re-runs over
 * all 322 M points at the new viewport) which is exactly what the user
 * hit in interactive use.
 *
 * Differences vs the gentler real-pan and the aggressive harnesses:
 *   - Each pan is a single deliberate gesture, not a flick storm.
 *   - 30 s settle between pans — long enough for the renderer's full
 *     compute chain to actually finish on a 322M dataset (Metal cmd
 *     buffer queue can take ~5–15 s to drain, plus async rerender).
 *   - 8 distinct pan targets across dense European regions so each
 *     re-compute lands a fresh viewport.
 *   - Captures crash signals after every pan + screenshot for diff.
 *
 * Run::
 *
 *   DATASET=/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet \
 *     npx playwright test e2e/desktop-electron-realistic-pan.spec.ts \
 *     --project=desktop-electron --workers=1
 */

import { test, expect, _electron as electron, type ConsoleMessage } from "@playwright/test";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPO = join(__dirname, "..");
const DESKTOP = join(REPO, "apps/desktop");
const DATASET = process.env.DATASET ?? "/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet";

const PACKAGED_BIN = join(DESKTOP, "release/mac-arm64/Geospatial Atlas.app/Contents/MacOS/Geospatial Atlas");

const OUT_DIR = join(REPO, "e2e/test-results/realistic-pan");
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

test("realistic single-pan + 30s settle", async () => {
  // Conservative timeout: 8 pans × 30 s + bootstrap + slack.
  test.setTimeout(20 * 60 * 1000);

  expect(existsSync(DATASET)).toBe(true);
  expect(existsSync(PACKAGED_BIN)).toBe(true);

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
  const panLog: { i: number; ms: number; ok: boolean }[] = [];

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

    if (renderLanded) {
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

      // First, zoom out to world scale so all 322M are in viewport —
      // matches the user's repro condition.
      console.log(`[harness] zoom out to world`);
      await win.mouse.move(cx, cy);
      for (let i = 0; i < 30; i++) await win.mouse.wheel(0, 80);
      await win.waitForTimeout(30_000); // long settle for the world-scale full compute
      await win.screenshot({ path: join(OUT_DIR, "01-world-zoomed.png") });
      if (crashes.length > 0) {
        throw new Error(`CRASH after zoom-out: ${JSON.stringify(crashes[0])}`);
      }

      // 8 deliberate single pans, each followed by 30s settle.
      // Each pan moves a different distance/direction so the new viewport
      // is genuinely different — forces a fresh full re-compute.
      const pans = [
        { dx: -200, dy: 100 },
        { dx: 250, dy: 50 },
        { dx: 0, dy: -300 },
        { dx: 350, dy: 200 },
        { dx: -400, dy: -150 },
        { dx: 100, dy: 300 },
        { dx: -150, dy: -250 },
        { dx: 300, dy: -100 },
      ];

      for (let i = 0; i < pans.length; i++) {
        const { dx, dy } = pans[i];
        const start = Date.now();
        console.log(`[harness] pan ${i + 1}/${pans.length}: (${dx}, ${dy})`);

        // Single deliberate drag — 40 steps over ~400ms, like a real
        // unhurried trackpad gesture.
        await win.mouse.move(cx, cy);
        await win.mouse.down();
        for (let s = 1; s <= 40; s++) {
          await win.mouse.move(cx + (dx * s) / 40, cy + (dy * s) / 40);
          await win.waitForTimeout(10);
        }
        await win.mouse.up();

        const dragMs = Date.now() - start;
        console.log(`[harness]   drag complete in ${dragMs}ms; settling 30s...`);

        // Wait 30s for the full pan-release re-compute to finish.
        // The 3 chunked downsample passes + draw + gamma should land
        // well within this window even at 322M.
        await win.waitForTimeout(30_000);

        const settleMs = Date.now() - start;
        const ok = crashes.length === 0;
        panLog.push({ i, ms: settleMs, ok });
        await win.screenshot({
          path: join(OUT_DIR, `${(i + 2).toString().padStart(2, "0")}-after-pan-${i + 1}.png`),
        });
        console.log(`[harness]   pan ${i + 1} done; total=${settleMs}ms ok=${ok} crashes=${crashes.length}`);

        if (!ok) {
          throw new Error(`CRASH after pan ${i + 1}: ${JSON.stringify(crashes[0])}`);
        }
      }
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
            panLog,
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
  console.log(`[harness] FINAL pans:\n  ${panLog.map((p) => `pan ${p.i + 1}: ${p.ms}ms ok=${p.ok}`).join("\n  ")}`);
  console.log(`[harness] FINAL crashes: ${crashes.length}`);
  for (const c of crashes.slice(0, 20)) {
    console.log(`  CRASH(${c.kind}) +${c.t - t0}ms ${c.source}: ${c.line}`);
  }

  expect(renderLanded).toBe(true);
  expect(crashes.length).toBe(0);
});
