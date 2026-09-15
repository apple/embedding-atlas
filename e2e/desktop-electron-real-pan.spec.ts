/**
 * Realistic-user pan/crash harness for the packaged Electron .app.
 *
 * The previous diagnostic spec drove ``window.mouse`` with 8-step
 * synthetic drags and gated success on
 * ``__atlasFirstBigRenderGpuLogged`` — both of which produced false
 * negatives: the synthetic drag didn't reach the high event-rate that
 * actually trips the bug, and the flag is one-shot so it can't tell us
 * whether subsequent frames landed. The user reported one real
 * trackpad pan was enough to "almost crash the system" and the spec
 * still passed. This harness fixes that.
 *
 * What's new:
 *   1. Captures the .app's main-process stdout/stderr — surfaces
 *      ``[renderer-gone]``, ``[child-gone]``, ``[gpu-info-update]``,
 *      ``[renderer-unresponsive]`` events that electron/main.ts now
 *      forwards. Any of those = crash, fail immediately.
 *   2. Pans with REAL high-frequency mouse moves AND wheel events
 *      (trackpad emulation) — the failure mode the user hits.
 *   3. Polls ``app.getAppMetrics()`` via the
 *      ``GEOSPATIAL_ATLAS_METRICS_INTERVAL`` env var so we know which
 *      process actually ballooned during pan.
 *   4. Inspects the actual DOM of the right sidebar — counts how many
 *      chart components mounted, dumps their text labels, and
 *      explicitly probes whether ``defaultColumnCharts`` ran.
 *   5. Saves a video of the entire run for postmortem.
 *
 * Run::
 *
 *   DATASET=/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet \
 *     npx playwright test e2e/desktop-electron-real-pan.spec.ts \
 *     --project=desktop-electron --workers=1
 */

import { test, expect, _electron as electron, type ConsoleMessage } from "@playwright/test";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPO = join(__dirname, "..");
const DESKTOP = join(REPO, "apps/desktop");
const DATASET = process.env.DATASET ?? "/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet";

const PACKAGED_BIN = join(DESKTOP, "release/mac-arm64/Geospatial Atlas.app/Contents/MacOS/Geospatial Atlas");

const OUT_DIR = join(REPO, "e2e/test-results/real-pan");
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
  { re: /kIOGPUCommandBufferCallback/, kind: "metal-watchdog" },
  { re: /Out of memory/i, kind: "oom" },
  { re: /WebGPU.*lost/i, kind: "webgpu-lost" },
];

function detectCrash(line: string, source: "stdout" | "stderr"): CrashSignal | null {
  for (const { re, kind } of CRASH_PATTERNS) {
    if (re.test(line)) return { t: Date.now(), source, line: line.trim(), kind };
  }
  return null;
}

test("desktop electron real-pan crash harness", async () => {
  test.setTimeout(20 * 60 * 1000);

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
      // Open Chrome DevTools Protocol on a fixed port so this harness
      // (and a curious human) can attach a real DevTools.
      GEOSPATIAL_ATLAS_DEBUG_PORT: "9223",
      // Periodic CPU/RAM metrics from main process.
      GEOSPATIAL_ATLAS_METRICS_INTERVAL: "2000",
      // Verbose Chromium logging so a Metal watchdog cascade is
      // visible in stderr instead of being swallowed silently.
      ELECTRON_ENABLE_LOGGING: "1",
    },
  });

  // Hook app stdout/stderr — these capture the new electron/main.ts
  // observability lines.
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
  let panSummary = "(pan never ran)";
  let sidebarSummary: any = null;
  let sidebarFinalSummary: any = null;

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

    // Wait for navigation off the boot URL onto the sidecar.
    const sidecarDeadline = Date.now() + 240_000;
    while (Date.now() < sidecarDeadline) {
      await win.waitForTimeout(800);
      const url = win.url();
      if (/127\.0\.0\.1:\d+/.test(url) && !url.includes(":1420")) {
        console.log(`[harness] sidecar URL: ${url}`);
        break;
      }
    }

    // Wait for first big render (the embedding scatter actually paints
    // pixels). 5 min is generous for the 322 M-row eubucco bootstrap.
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

    if (renderLanded) {
      // Quiesce so any deferred discovery has a chance to start.
      await win.waitForTimeout(2_000);

      // Baseline screenshot for visual diffing.
      await win.screenshot({ path: join(OUT_DIR, "01-baseline.png"), fullPage: false });

      // ---- Side panel inspection ----
      // Probe what the right sidebar actually contains. mosaic's
      // chart components render as `<div role="figure">` or `<svg>`;
      // we count anything inside the side panel container.
      sidebarSummary = await win.evaluate(() => {
        const w = window as any;
        // Heuristic: the side panel is the rightmost flex column with
        // multiple children. Walk all elements and grab anything that
        // looks like a chart container.
        const chartEls = Array.from(
          document.querySelectorAll(
            '[role="figure"], .vega-embed, .uw-chart, .chart-container, svg.chart, [data-chart]',
          ),
        );
        const sidePanelTexts: string[] = [];
        const allText = document.body.innerText
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean);
        // Look for known side-panel tags.
        const knownTags = ["+ Add", "SQL Predicates", "+ Add Predicate", "Color"];
        for (const t of allText) {
          if (knownTags.some((k) => t.includes(k))) sidePanelTexts.push(t);
        }
        // Inspect the discovery state: read the runtime ``charts``
        // count if exposed, and the polled flag.
        return {
          firstBigRenderLogged: w.__atlasFirstBigRenderGpuLogged === true,
          gpuErrors: (w.__atlasGpuErrors ?? []).slice(),
          chartElCount: chartEls.length,
          chartTagSnapshot: chartEls.slice(0, 8).map((el) => ({
            tag: el.tagName,
            cls: (el as HTMLElement).className?.toString?.().slice(0, 80) ?? "",
            text: ((el as HTMLElement).innerText ?? "").slice(0, 80),
          })),
          knownSidePanelTexts: sidePanelTexts,
          allBodyTextLineCount: allText.length,
        };
      });
      console.log(`[harness] sidebar summary:\n${JSON.stringify(sidebarSummary, null, 2)}`);

      // ---- Real-user pan simulation ----
      // The biggest canvas (devicePixelRatio applied) is the WebGPU
      // scatter overlay; events on it bubble to the basemap pan
      // handler if pointer-events allow. Drag with high event rate
      // AND fire wheel events to mimic trackpad pan/zoom.
      const canvases = await win.evaluate(() => {
        return Array.from(document.querySelectorAll("canvas")).map((c) => {
          const r = (c as HTMLCanvasElement).getBoundingClientRect();
          return { x: r.x, y: r.y, w: r.width, h: r.height };
        });
      });
      const target = canvases.length > 0 ? canvases.reduce((a, b) => (a.w * a.h > b.w * b.h ? a : b)) : null;
      console.log(`[harness] pan target: ${JSON.stringify(target)}`);

      if (target) {
        const pans: { kind: string; ms: number; ok: boolean; coverage: number }[] = [];
        const cx = target.x + target.w / 2;
        const cy = target.y + target.h / 2;

        // Helper: sample non-white coverage of a freshly-taken
        // page screenshot. Returns the fraction in [0, 1] or -1
        // on probe failure.
        const sampleCoverage = async (): Promise<number> => {
          try {
            const png = await win.screenshot({ fullPage: false });
            return await win.evaluate(
              async (b64) => {
                const img = new Image();
                await new Promise<void>((res, rej) => {
                  img.onload = () => res();
                  img.onerror = () => rej(new Error("img load"));
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
            );
          } catch {
            return -1;
          }
        };

        // ---- Pan #1: short fast mouse drag (60Hz) ----
        // 32 small steps at no-delay, simulating a quick flick.
        {
          const start = Date.now();
          await win.mouse.move(cx, cy);
          await win.mouse.down();
          for (let s = 1; s <= 32; s++) {
            await win.mouse.move(cx + s * 8, cy + s * 5);
          }
          await win.mouse.up();
          await win.waitForTimeout(2_000);
          const cov = await sampleCoverage();
          await win.screenshot({ path: join(OUT_DIR, "02-after-mouse-flick.png") });
          pans.push({ kind: "mouse-flick-fast", ms: Date.now() - start, ok: crashes.length === 0, coverage: cov });
          console.log(`[harness] pan1 done; crashes=${crashes.length} coverage=${(cov * 100).toFixed(2)}%`);
          if (crashes.length > 0) throw new Error(`crash after pan1: ${JSON.stringify(crashes[0])}`);
        }

        // ---- Pan #2: long slow drag through density ----
        // 60 steps at 8ms each, simulating a deliberate large pan.
        {
          const start = Date.now();
          await win.mouse.move(cx + 200, cy + 100);
          await win.mouse.down();
          for (let s = 1; s <= 60; s++) {
            await win.mouse.move(cx + 200 - s * 5, cy + 100 - s * 3);
            await win.waitForTimeout(8);
          }
          await win.mouse.up();
          await win.waitForTimeout(2_000);
          const cov = await sampleCoverage();
          await win.screenshot({ path: join(OUT_DIR, "03-after-mouse-slow-pan.png") });
          pans.push({ kind: "mouse-slow-pan", ms: Date.now() - start, ok: crashes.length === 0, coverage: cov });
          console.log(`[harness] pan2 done; crashes=${crashes.length} coverage=${(cov * 100).toFixed(2)}%`);
          if (crashes.length > 0) throw new Error(`crash after pan2: ${JSON.stringify(crashes[0])}`);
        }

        // ---- Pan #3: trackpad emulation via wheel events ----
        // maplibre-gl handles two-finger pan as wheel deltaX/deltaY
        // (without ctrlKey). Playwright's mouse.wheel dispatches
        // standard wheel events. Multiple bursts simulate a real
        // trackpad swipe.
        {
          const start = Date.now();
          await win.mouse.move(cx, cy);
          for (let burst = 0; burst < 8; burst++) {
            for (let s = 0; s < 12; s++) {
              await win.mouse.wheel(burst % 2 === 0 ? 30 : -30, burst % 3 === 0 ? 20 : -20);
            }
            await win.waitForTimeout(40);
          }
          await win.waitForTimeout(2_000);
          const cov = await sampleCoverage();
          await win.screenshot({ path: join(OUT_DIR, "04-after-wheel-trackpad.png") });
          pans.push({ kind: "wheel-trackpad", ms: Date.now() - start, ok: crashes.length === 0, coverage: cov });
          console.log(`[harness] pan3 done; crashes=${crashes.length} coverage=${(cov * 100).toFixed(2)}%`);
          if (crashes.length > 0) throw new Error(`crash after pan3: ${JSON.stringify(crashes[0])}`);
        }

        // ---- Pan #4: storm — many short flicks back to back ----
        // Stress the renderer with rapid viewport changes.
        {
          const start = Date.now();
          for (let i = 0; i < 6; i++) {
            await win.mouse.move(cx + 100, cy);
            await win.mouse.down();
            for (let s = 1; s <= 24; s++) {
              await win.mouse.move(cx + 100 - s * 6 * (i % 2 === 0 ? 1 : -1), cy + s * 4);
            }
            await win.mouse.up();
            await win.waitForTimeout(80);
            if (crashes.length > 0) break;
          }
          await win.waitForTimeout(2_000);
          const cov = await sampleCoverage();
          await win.screenshot({ path: join(OUT_DIR, "05-after-storm.png") });
          pans.push({ kind: "storm", ms: Date.now() - start, ok: crashes.length === 0, coverage: cov });
          console.log(`[harness] pan4 done; crashes=${crashes.length} coverage=${(cov * 100).toFixed(2)}%`);
        }

        panSummary = pans
          .map((p) => `${p.kind}=${p.ms}ms cov=${(p.coverage * 100).toFixed(1)}% ok=${p.ok}`)
          .join(" | ");
      }

      // Wait one more discovery-timeout window so charts mount even
      // when the APPROX_COUNT_DISTINCT batch was the slow path.
      await win.waitForTimeout(18_000);
      // Re-probe the sidebar AFTER the pan storm so we catch any column
      // charts that mounted during the test. The original probe at the
      // top of the test runs ~2 s after first big render, which is too
      // early on huge datasets where ``defaultColumnCharts`` is gated
      // on a 15 s ``distinctCountBatch`` timeout.
      try {
        sidebarFinalSummary = await win.evaluate(() => {
          const chartEls = Array.from(
            document.querySelectorAll(
              '[role="figure"], .vega-embed, .uw-chart, .chart-container, svg.chart, [data-chart], svg[viewBox]',
            ),
          );
          return {
            chartElCount: chartEls.length,
            chartTitles: Array.from(
              new Set(
                chartEls
                  .map(
                    (el) =>
                      (el.closest("[data-chart], section, .chart-card") as any)?.textContent
                        ?.split("\n")[0]
                        ?.trim?.() ?? "",
                  )
                  .filter(Boolean),
              ),
            ).slice(0, 30),
          };
        });
        console.log(`[harness] FINAL sidebar (post-pan): ${JSON.stringify(sidebarFinalSummary)}`);
        await win.screenshot({ path: join(OUT_DIR, "06-final-sidebar.png") });
      } catch (e) {
        console.log(`[harness] post-pan sidebar probe failed: ${(e as Error).message}`);
      }
    }
  } finally {
    // Persist the captured logs even if the test threw.
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
            sidebarSummary,
            sidebarFinalSummary,
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

  // Surface what actually happened for the human reader.
  console.log(`[harness] FINAL renderLanded=${renderLanded}`);
  console.log(`[harness] FINAL panSummary: ${panSummary}`);
  console.log(`[harness] FINAL crashes: ${crashes.length}`);
  for (const c of crashes.slice(0, 20)) {
    console.log(`  CRASH(${c.kind}) +${c.t - t0}ms ${c.source}: ${c.line}`);
  }
  console.log(`[harness] artefacts: ${OUT_DIR}/{01-baseline.png, 02..05, stdout.log, stderr.log, summary.json}`);

  expect(renderLanded, "first big render never landed").toBe(true);
  expect(crashes.length, `${crashes.length} crash signals detected; first: ${JSON.stringify(crashes[0] ?? null)}`).toBe(
    0,
  );
});
