/**
 * Diagnostic spec for the "empty map + empty side panel" regression
 * the user observes in the packaged desktop Electron app on the
 * eubucco file (322 M rows). The render-coverage.spec.ts test runs
 * Playwright's bundled Chromium, NOT the Electron build — so it
 * cannot catch Electron-specific regressions in the streaming
 * connector, WebGPU adapter selection, or the renderer-process
 * sandbox. This spec drives the actual ``Geospatial Atlas.app``
 * via Playwright's ``_electron`` API, captures every console line
 * and page error, and reports which queries / GPU events fired.
 *
 * Run::
 *
 *   DATASET=/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet \
 *     npx playwright test e2e/desktop-electron-eubucco-debug.spec.ts \
 *     --project=desktop-electron --workers=1
 */

import { test, expect, _electron as electron, type ConsoleMessage } from "@playwright/test";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { join } from "node:path";

const REPO = join(__dirname, "..");
const DESKTOP = join(REPO, "apps/desktop");
const DATASET = process.env.DATASET ?? "/Users/dome/work/general/eubucco/eubucco_lat_lon.parquet";

// Prefer the PACKAGED .app binary over node_modules/electron — the
// regression we're chasing only manifests in the packaged bundle, not
// the dev shell. Falls back to dev electron if the packaged build is
// missing.
const PACKAGED_BIN = join(DESKTOP, "release/mac-arm64/Geospatial Atlas.app/Contents/MacOS/Geospatial Atlas");
const ELECTRON_BIN: string = (() => {
  if (existsSync(PACKAGED_BIN)) return PACKAGED_BIN;
  const req = createRequire(join(DESKTOP, "package.json"));
  return req("electron") as unknown as string;
})();

test("desktop electron eubucco diagnostic", async () => {
  test.setTimeout(15 * 60 * 1000);

  expect(existsSync(DATASET), `dataset missing at ${DATASET}`).toBe(true);
  const packagedApp = join(
    DESKTOP,
    "release/mac-arm64/Geospatial Atlas.app/Contents/Resources/sidecar/geospatial-atlas-sidecar",
  );
  const hasPackaged = existsSync(packagedApp);
  console.log(`[diag] packaged sidecar present: ${hasPackaged}`);
  // Drive the dev shell — same renderer process model + same bundled
  // viewer once the navigation lands. We can't drive the .app
  // directly because Playwright's ``_electron`` requires
  // ``electron`` as a launchable binary, not a packaged .app.
  expect(
    existsSync(join(DESKTOP, "resources/sidecar/geospatial-atlas-sidecar")),
    "PyInstaller sidecar binary not built",
  ).toBe(true);

  const isPackaged = ELECTRON_BIN === PACKAGED_BIN;
  console.log(`[diag] launching ${isPackaged ? "PACKAGED .app" : "dev electron"} bin: ${ELECTRON_BIN}`);
  const app = await electron.launch({
    cwd: DESKTOP,
    // For the packaged .app, ``app.isPackaged === true`` so the entry
    // dir is implicit and we pass the dataset as a positional arg
    // (matching the user's manual launch). For dev electron we still
    // need to point it at apps/desktop and let
    // ``GEOSPATIAL_ATLAS_INITIAL_DATASET`` carry the dataset.
    args: isPackaged ? [DATASET] : ["."],
    executablePath: ELECTRON_BIN,
    env: {
      ...process.env,
      GEOSPATIAL_ATLAS_INITIAL_DATASET: DATASET,
    },
  });

  const consoleLines: { type: string; text: string; t: number }[] = [];
  const pageErrors: string[] = [];
  const requestFailures: { url: string; failure: string }[] = [];
  const slowRequests: { url: string; ms: number; status: number; size: number }[] = [];

  try {
    const window = await app.firstWindow();

    window.on("console", (msg: ConsoleMessage) => {
      const t = Date.now();
      const text = msg.text();
      consoleLines.push({ type: msg.type(), text, t });
    });
    window.on("pageerror", (e) => {
      pageErrors.push(`${e.name}: ${e.message}`);
    });
    window.on("requestfailed", (req) => {
      requestFailures.push({
        url: req.url(),
        failure: req.failure()?.errorText ?? "unknown",
      });
    });
    // Key by Playwright Request object — distinct per request, not per URL.
    const reqMeta = new WeakMap<
      ReturnType<typeof window.context>["pages"] extends () => infer X ? any : never,
      { start: number; body: string }
    >();
    let queryIdx = 0;
    window.on("request", (req) => {
      if (!req.url().includes("/data/query")) return;
      const body = req.postData() ?? "";
      // Also log immediately so we have a chronological view.
      const idx = queryIdx;
      console.log(`[diag] REQ Q${idx} body=${body.slice(0, 350)}`);
      reqMeta.set(req as any, { start: Date.now(), body });
    });
    window.on("response", async (res) => {
      const url = res.url();
      if (!url.includes("/data/query")) return;
      const meta = reqMeta.get(res.request() as any);
      const ms = meta ? Date.now() - meta.start : -1;
      const body = meta?.body ?? "<no body>";
      let size = 0;
      try {
        size = (await res.body()).byteLength;
      } catch {}
      slowRequests.push({ url, ms, status: res.status(), size });
      const idx = queryIdx++;
      console.log(`[diag] RES Q${idx} ${res.status()} ${size}B ${ms}ms body=${body.slice(0, 350)}`);
      if (res.status() >= 400) {
        let errResp = "";
        try {
          errResp = (await res.text()).slice(0, 500);
        } catch {}
        console.log(`[diag] Q${idx}   ERROR: ${errResp}`);
      }
    });

    // Wait for navigation off the boot URL into the sidecar.
    const sidecarDeadline = Date.now() + 240_000;
    let onSidecar = false;
    while (Date.now() < sidecarDeadline) {
      await window.waitForTimeout(1500);
      const url = window.url();
      if (/127\.0\.0\.1:\d+/.test(url) && !url.includes(":1420")) {
        onSidecar = true;
        console.log(`[diag] navigated to sidecar URL: ${url}`);
        break;
      }
    }
    expect(onSidecar, `window never navigated to sidecar URL (still ${window.url()})`).toBe(true);

    // Wait up to 90 s for first big render. Long enough to see the
    // failing queries, short enough that iteration is fast.
    let renderLanded = false;
    try {
      await window.waitForFunction(() => (window as any).__atlasFirstBigRenderGpuLogged === true, null, {
        timeout: 90_000,
        polling: 250,
      });
      renderLanded = true;
      console.log(`[diag] first big render flag SET at t=${Date.now()}`);
    } catch (e) {
      console.log(`[diag] first big render flag NEVER SET (90 s): ${(e as Error).message}`);
    }

    // Quiesce so any deferred work fires.
    await window.waitForTimeout(5_000);

    // Capture screenshot.
    await window.screenshot({
      path: "e2e/test-results/desktop-eubucco-diagnostic.png",
      fullPage: false,
    });

    // Window-side state probe.
    const probe = await window.evaluate(() => {
      const w = window as any;
      const canvases = Array.from(document.querySelectorAll("canvas")).map((c) => {
        const r = (c as HTMLCanvasElement).getBoundingClientRect();
        return {
          w: r.width,
          h: r.height,
          cw: (c as HTMLCanvasElement).width,
          ch: (c as HTMLCanvasElement).height,
        };
      });
      return {
        firstBigRender: w.__atlasFirstBigRenderGpuLogged ?? null,
        firstBigRenderMs: w.__atlasFirstBigRenderGpuMs ?? null,
        atlasStage: w.__atlasStageMarks ?? null,
        canvases,
        scatterPoints: document.querySelector("[aria-label='points']")?.textContent ?? null,
      };
    });
    console.log(`[diag] probe: ${JSON.stringify(probe, null, 2)}`);

    // Pan-crash regression check. The Apr 26 2026 report was "one pan
    // crashes the window" on the 322 M eubucco scatter — the renderer
    // bundled accumulate + 3 downsample passes + draw + gamma into one
    // GPUCommandBuffer, which on M-series adapters occasionally
    // exceeded Metal's 5 s watchdog and tripped
    // ``kIOGPUCommandBufferCallbackErrorTimeout``. The split fix shipped
    // in this build submits each downsample compute pass in its own
    // command buffer for count > 50 M. To verify, drag the scatter
    // canvas through several pan gestures and assert the renderer:
    //   1. logs no GPU errors (uncaptured / device-lost)
    //   2. eventually re-paints the scatter canvas (pixel coverage > 0)
    let panGpuErrors: any[] = [];
    let postPanCoverage = -1;
    if (renderLanded && probe.canvases.length > 0) {
      const biggest = probe.canvases.reduce((a, b) => (a.w * a.h > b.w * b.h ? a : b));
      console.log(`[diag] panning biggest canvas (${biggest.w}x${biggest.h})`);
      const PAN_COUNT = 6;
      for (let i = 0; i < PAN_COUNT; i++) {
        const cx = biggest.w / 2;
        const cy = biggest.h / 2;
        const dx = (i % 2 === 0 ? 1 : -1) * 220;
        const dy = (i % 3 === 0 ? 1 : -1) * 140;
        try {
          await window.mouse.move(cx, cy);
          await window.mouse.down();
          for (let s = 1; s <= 8; s++) {
            await window.mouse.move(cx + (dx * s) / 8, cy + (dy * s) / 8, { steps: 1 });
            await window.waitForTimeout(15);
          }
          await window.mouse.up();
        } catch (e) {
          console.log(`[diag] pan ${i} mouse op threw: ${(e as Error).message}`);
          break;
        }
        // Short settle so the next pan's first move isn't coalesced
        // with this pan's last move into a flick gesture.
        await window.waitForTimeout(400);
        const errs = await window
          .evaluate(() => {
            const w = window as any;
            return (w.__atlasGpuErrors ?? []).map((r: any) => ({
              kind: r.kind,
              message: String(r.message ?? "").slice(0, 200),
              reason: r.reason ?? null,
            }));
          })
          .catch(() => null);
        console.log(`[diag] pan ${i} done; gpuErrors=${errs ? errs.length : "probe-failed"}`);
        if (errs && errs.length > 0) {
          for (const e of errs) console.log(`  GPUERR: ${e.kind} ${e.reason ?? ""} ${e.message}`);
          panGpuErrors = errs;
          break;
        }
      }
      // Wait for the canvas to repaint after the slow data queries
      // (Q6 / column-discovery) settle. Up to 60 s — generous because
      // the side-panel queries can take 47 s on 322 M × 20 cols.
      console.log(`[diag] waiting for post-pan repaint (max 60s)…`);
      const repaintDeadline = Date.now() + 60_000;
      const sample = async (): Promise<number> => {
        const png = await window.screenshot({ fullPage: false });
        return await window.evaluate(
          async (b64) => {
            const img = new Image();
            await new Promise<void>((res, rej) => {
              img.onload = () => res();
              img.onerror = () => rej(new Error("img load failed"));
              img.src = b64;
            });
            const cv = document.createElement("canvas");
            cv.width = 800;
            cv.height = 600;
            const ctx = cv.getContext("2d", { willReadFrequently: true });
            if (!ctx) return -1;
            ctx.drawImage(img, 0, 0, 800, 600);
            const data = ctx.getImageData(0, 0, 800, 600).data;
            const total = 800 * 600;
            // Count pixels darker than near-white background.
            let nonWhite = 0;
            for (let p = 0; p < data.length; p += 4) {
              if (data[p] < 240 || data[p + 1] < 240 || data[p + 2] < 240) nonWhite++;
            }
            return nonWhite / total;
          },
          `data:image/png;base64,${png.toString("base64")}`,
        );
      };
      while (Date.now() < repaintDeadline) {
        postPanCoverage = await sample().catch(() => -1);
        console.log(`[diag] post-pan coverage probe = ${(postPanCoverage * 100).toFixed(2)}%`);
        if (postPanCoverage > 0.05) break;
        await window.waitForTimeout(2_000);
      }
      try {
        await window.screenshot({
          path: "e2e/test-results/desktop-eubucco-after-pan.png",
          fullPage: false,
        });
      } catch (e) {
        console.log(`[diag] post-pan screenshot failed: ${(e as Error).message}`);
      }
    }

    // Surface logs ALWAYS (whether passed or not).
    const errs = consoleLines.filter((l) => l.type === "error");
    const warns = consoleLines.filter((l) => l.type === "warning");
    console.log(`[diag] console errors: ${errs.length}`);
    for (const e of errs.slice(0, 30)) console.log(`  ERR: ${e.text.slice(0, 400)}`);
    console.log(`[diag] console warnings: ${warns.length}`);
    for (const w of warns.slice(0, 10)) console.log(`  WARN: ${w.text.slice(0, 300)}`);
    console.log(`[diag] page errors: ${pageErrors.length}`);
    for (const e of pageErrors.slice(0, 20)) console.log(`  PAGEERR: ${e.slice(0, 400)}`);
    console.log(`[diag] request failures: ${requestFailures.length}`);
    for (const f of requestFailures.slice(0, 20)) console.log(`  REQFAIL: ${f.url.slice(0, 200)} -> ${f.failure}`);
    console.log(`[diag] /data/query responses: ${slowRequests.length}`);
    for (const r of slowRequests) console.log(`  QUERY: ${r.status} ${r.size}B ${r.ms}ms`);

    console.log(
      `[diag] FINAL: renderLanded=${renderLanded} panGpuErrors=${panGpuErrors.length} postPanCoverage=${(postPanCoverage * 100).toFixed(2)}%`,
    );

    expect(renderLanded, "first big render never landed in Electron").toBe(true);
    expect(panGpuErrors.length, `pan storm produced GPU errors: ${JSON.stringify(panGpuErrors).slice(0, 400)}`).toBe(0);
    expect(
      postPanCoverage,
      `scatter never repainted after pan (final coverage ${(postPanCoverage * 100).toFixed(2)}%) — likely Metal watchdog crash`,
    ).toBeGreaterThan(0.05);
  } finally {
    await app.close();
  }
});
