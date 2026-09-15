/**
 * Shared test helpers for Geospatial Atlas E2E tests.
 *
 * Provides:
 *  - Automatic test-data download (once, then cached locally)
 *  - Process lifecycle management (start / wait / teardown)
 *  - Reusable page-level assertions
 */

import { type ChildProcess, spawn } from "child_process";
import { type Page } from "@playwright/test";
import { existsSync, mkdirSync } from "fs";
import { get } from "https";
import { createWriteStream } from "fs";
import path from "path";
import { E2E_CONSTANTS } from "../playwright.config.js";

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const DATA_DIR = path.resolve(__dirname, ".data");
const DATA_FILE = path.join(DATA_DIR, "dataset_test.parquet");
const DATA_URL =
  "https://github.com/do-me/geospatial-atlas-apps/raw/refs/heads/main/GISCO_Education/data/dataset_0.parquet";

/**
 * Return the path to the test parquet file, downloading it on first use.
 *
 * Override the default URL/path via E2E_PARQUET_FILE env var.
 */
export async function getTestDataPath(): Promise<string> {
  const override = process.env.E2E_PARQUET_FILE;
  if (override) return override;

  if (existsSync(DATA_FILE)) return DATA_FILE;

  mkdirSync(DATA_DIR, { recursive: true });
  console.log(`Downloading test data to ${DATA_FILE} …`);
  await download(DATA_URL, DATA_FILE);
  console.log("Download complete.");
  return DATA_FILE;
}

/** Follow redirects and write the final response body to `dest`. */
function download(url: string, dest: string, maxRedirects = 5): Promise<void> {
  return new Promise((resolve, reject) => {
    const attempt = (currentUrl: string, redirectsLeft: number) => {
      const mod = currentUrl.startsWith("https") ? require("https") : require("http");
      mod
        .get(currentUrl, (res: any) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            if (redirectsLeft <= 0) return reject(new Error("Too many redirects"));
            return attempt(res.headers.location, redirectsLeft - 1);
          }
          if (res.statusCode !== 200) {
            return reject(new Error(`Download failed: HTTP ${res.statusCode}`));
          }
          const file = createWriteStream(dest);
          res.pipe(file);
          file.on("finish", () => file.close(() => resolve()));
          file.on("error", reject);
        })
        .on("error", reject);
    };
    attempt(url, maxRedirects);
  });
}

// ---------------------------------------------------------------------------
// Process lifecycle
// ---------------------------------------------------------------------------

/** Poll `url` until it responds with 2xx, or throw after `timeoutMs`. */
export async function waitForServer(url: string, timeoutMs = E2E_CONSTANTS.SERVER_STARTUP_TIMEOUT): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      // server not up yet
    }
    await new Promise((r) => setTimeout(r, E2E_CONSTANTS.POLL_INTERVAL));
  }
  throw new Error(`Server at ${url} did not become ready within ${timeoutMs}ms`);
}

/**
 * Start the Python backend via `uv run geospatial-atlas`.
 *
 * The CLI auto-detects lon/lat as GIS columns (`find_gis_columns`).
 * `--disable-projection` skips expensive embedding/UMAP computation
 * while keeping the pre-existing coordinate columns.
 */
export function startBackendServer(parquetFile: string): ChildProcess {
  const { BACKEND_DIR, STATIC_DIR, SERVER_PORT } = E2E_CONSTANTS;
  return spawn(
    "uv",
    [
      "run",
      "--directory",
      BACKEND_DIR,
      "geospatial-atlas",
      parquetFile,
      "--port",
      String(SERVER_PORT),
      "--no-auto-port",
      "--static",
      STATIC_DIR,
      "--disable-projection",
    ],
    { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env } },
  );
}

/** Start the Vite dev server for the viewer package. */
export function startDevServer(): ChildProcess {
  return spawn("npm", ["run", "dev", "--", "--port", String(E2E_CONSTANTS.DEV_PORT)], {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: path.resolve(__dirname, "../packages/viewer"),
    env: { ...process.env },
  });
}

/** Gracefully shut down a child process (SIGTERM, then SIGKILL fallback). */
export async function teardown(proc: ChildProcess | undefined): Promise<void> {
  if (!proc) return;
  proc.kill("SIGTERM");
  await new Promise((r) => setTimeout(r, 1_000));
  if (!proc.killed) proc.kill("SIGKILL");
}

// ---------------------------------------------------------------------------
// Page-level helpers
// ---------------------------------------------------------------------------

/** Wait until the app renders at least one `<canvas>` element. */
export async function waitForCanvas(page: Page, timeout = 30_000): Promise<void> {
  await page.waitForSelector("canvas", { timeout });
}

/** Wait for the scatter plot to finish its initial render cycle. */
export async function waitForDataRender(page: Page): Promise<void> {
  await waitForCanvas(page);
  // The point count badge appears once data is loaded (e.g. "333,843 points").
  // Fall back to a short settle time if the badge isn't found.
  try {
    await page.locator("text=/\\d[\\d,]* points/").first().waitFor({ timeout: 15_000 });
  } catch {
    await page.waitForTimeout(3_000);
  }
}

// ---------------------------------------------------------------------------
// Basemap alignment helpers
// ---------------------------------------------------------------------------

/**
 * Reference points for basemap alignment verification.
 *
 * These are well-known European locations that must sit on land
 * in the GISCO schools dataset. If scatter points and the MapLibre
 * basemap disagree on where these coordinates fall, the Mercator
 * projection round-trip is broken.
 */
export const ALIGNMENT_REFERENCE_POINTS = [
  { name: "Paris", lon: 2.35, lat: 48.86 },
  { name: "Berlin", lon: 13.38, lat: 52.52 },
  { name: "Rome", lon: 12.5, lat: 41.9 },
  { name: "Madrid", lon: -3.7, lat: 40.42 },
  { name: "Warsaw", lon: 21.01, lat: 52.23 },
] as const;

/**
 * Web Mercator latitude projection (same formula as viewport_utils.ts).
 * Needed to verify scatter ↔ MapLibre coordinate agreement.
 */
export function projectLat(lat: number): number {
  const latRad = (lat * Math.PI) / 180;
  return (Math.log(Math.tan(Math.PI / 4 + latRad / 2)) * 180) / Math.PI;
}

// ---------------------------------------------------------------------------
// Frontend file-upload flow helpers
// ---------------------------------------------------------------------------

/**
 * Upload a parquet file in the frontend FileViewer, wait for auto-detection,
 * confirm settings, and wait for the full app (canvas) to render.
 *
 * Returns true if the app rendered successfully, false if DuckDB WASM
 * failed to load the file (e.g. parquet extension unsigned in dev mode).
 */
export async function uploadFileAndRender(page: Page, baseUrl: string, filePath: string): Promise<boolean> {
  await page.goto(`${baseUrl}/#/file`);
  const { expect } = await import("@playwright/test");

  // Wait for the drop zone
  await expect(page.locator("text=Drag & drop")).toBeVisible({ timeout: 10_000 });

  // Upload file
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);

  // Wait for drop zone to disappear (loading started)
  await expect(page.locator("text=Drag & drop")).not.toBeVisible({ timeout: 30_000 });

  // Wait for either Confirm button (settings view) or error
  const confirmBtn = page.locator("button:has-text('Confirm')");
  const errorMsg = page.locator("text=Extension Autoloading Error");

  await Promise.race([
    confirmBtn.waitFor({ timeout: 30_000 }).catch(() => {}),
    errorMsg.waitFor({ timeout: 30_000 }).catch(() => {}),
  ]);

  if (await errorMsg.isVisible()) return false;
  if (!(await confirmBtn.isVisible())) return false;

  // Click Confirm to load the visualization
  await confirmBtn.click();

  // Wait for full render — canvas should appear
  try {
    await waitForCanvas(page, 60_000);
    await waitForDataRender(page);
  } catch {
    return false;
  }

  return true;
}
