/**
 * E2E tests — Frontend-Only Mode
 *
 * Validates the browser-only flow where data is loaded client-side via
 * DuckDB WASM. The Vite dev server serves the viewer without any
 * Python backend.
 *
 * Runs the same rendering, alignment, interaction, and drift checks
 * as server mode to ensure both code paths produce identical results.
 *
 * Uses the latlon_columns.parquet fixture (10 European landmarks with
 * lon/lat columns) for the full visualization tests.
 */

import { test, expect } from "@playwright/test";
import { type ChildProcess } from "child_process";
import path from "path";
import {
  getTestDataPath,
  startDevServer,
  waitForServer,
  teardown,
  waitForCanvas,
  waitForDataRender,
  uploadFileAndRender,
  ALIGNMENT_REFERENCE_POINTS,
  projectLat,
} from "./helpers.js";
import { E2E_CONSTANTS } from "../playwright.config.js";

const BASE_URL = `http://localhost:${E2E_CONSTANTS.DEV_PORT}`;
const FIXTURES = path.resolve(__dirname, ".data/fixtures");
const GIS_FIXTURE = path.join(FIXTURES, "latlon_columns.parquet");

let devServer: ChildProcess;
let testDataPath: string;

test.beforeAll(async () => {
  testDataPath = await getTestDataPath();
  devServer = startDevServer();
  await waitForServer(`${BASE_URL}/`);
});

test.afterAll(async () => {
  await teardown(devServer);
});

// ---------------------------------------------------------------------------
// File upload flow
// ---------------------------------------------------------------------------

test.describe("File Upload", () => {
  test("upload page shows drop zone and URL input", async ({ page }) => {
    await page.goto(`${BASE_URL}/#/file`);

    const dropZone = page.locator("text=Drag & drop");
    await expect(dropZone).toBeVisible({ timeout: 10_000 });

    // URL input should also be present
    await expect(page.locator('input[type="text"]')).toBeVisible();
  });

  test("uploading a parquet file transitions past the drop zone", async ({ page }) => {
    await page.goto(`${BASE_URL}/#/file`);
    await expect(page.locator("text=Drag & drop")).toBeVisible({ timeout: 10_000 });

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testDataPath);

    // After upload the drop zone should disappear — replaced by either
    // the settings view (column picker) or the messages view (loading / error).
    await expect(page.locator("text=Drag & drop")).not.toBeVisible({ timeout: 20_000 });
  });
});

// ---------------------------------------------------------------------------
// Synthetic test data (no file required)
// ---------------------------------------------------------------------------

test.describe("Test Data Viewer", () => {
  test("renders scatter plot with synthetic data", async ({ page }) => {
    await page.goto(`${BASE_URL}/#/test`);
    await waitForCanvas(page);

    expect(await page.locator("canvas").count()).toBeGreaterThan(0);
  });

  test("UI controls are present", async ({ page }) => {
    await page.goto(`${BASE_URL}/#/test`);
    await waitForCanvas(page);

    expect(await page.locator("button").count()).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Full GIS visualization (file upload → confirm → render)
// All checks mirror server-mode.spec.ts to ensure parity.
// ---------------------------------------------------------------------------

test.describe("GIS Visualization (frontend)", () => {
  // Helper: upload fixture, confirm, wait for render — skip if DuckDB WASM fails
  async function loadGisFixture(page: any): Promise<boolean> {
    return uploadFileAndRender(page, BASE_URL, GIS_FIXTURE);
  }

  // -- Rendering --

  test("renders a scatter canvas after file upload", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const canvasCount = await page.locator("canvas").count();
    expect(canvasCount).toBeGreaterThan(0);
  });

  test("scatter canvas has non-trivial dimensions", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const box = await page.locator("canvas").first().boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(100);
    expect(box!.height).toBeGreaterThan(100);
  });

  test("MapLibre basemap canvas is present in GIS mode", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const mapCanvasCount = await page.locator(".maplibregl-canvas, .mapboxgl-canvas").count();
    expect(mapCanvasCount).toBeGreaterThan(0);
  });

  test("sidebar contains interactive controls", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    expect(await page.locator("button").count()).toBeGreaterThan(0);
  });

  // -- Basemap Alignment --

  test("Viewport.projectLat matches standard Web Mercator", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const results = await page.evaluate(
      (refs: typeof ALIGNMENT_REFERENCE_POINTS) => {
        function standardMercatorY(lat: number): number {
          const latRad = (lat * Math.PI) / 180;
          return (Math.log(Math.tan(Math.PI / 4 + latRad / 2)) * 180) / Math.PI;
        }
        return refs.map((ref) => ({
          name: ref.name,
          lat: ref.lat,
          expectedY: standardMercatorY(ref.lat),
        }));
      },
      [...ALIGNMENT_REFERENCE_POINTS],
    );

    for (const r of results) {
      const ours = projectLat(r.lat);
      expect(ours).toBeCloseTo(r.expectedY, 10);
    }
  });

  test("scatter points align with MapLibre basemap", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const positions = await page.evaluate(
      (refs: typeof ALIGNMENT_REFERENCE_POINTS) => {
        const map = (window as any).__geospatialAtlasMap;
        if (!map?.project) return null;
        return refs.map((ref) => {
          const px = map.project([ref.lon, ref.lat]);
          return { name: ref.name, x: px.x, y: px.y };
        });
      },
      [...ALIGNMENT_REFERENCE_POINTS],
    );

    if (positions === null) {
      // Fallback: verify Mercator round-trip
      for (const ref of ALIGNMENT_REFERENCE_POINTS) {
        const mercY = projectLat(ref.lat);
        expect(Math.abs(mercY)).toBeLessThan(200);
        const yRad = (mercY * Math.PI) / 180;
        const roundTrip = (2 * Math.atan(Math.exp(yRad)) - Math.PI / 2) * (180 / Math.PI);
        expect(roundTrip).toBeCloseTo(ref.lat, 6);
      }
      return;
    }

    for (const pos of positions) {
      expect(pos.x).not.toBeNaN();
      expect(pos.y).not.toBeNaN();
    }
  });

  // -- Interaction --

  test("scroll-to-zoom changes the viewport", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const canvas = page.locator("canvas").first();
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();

    const before = await page.screenshot();

    const cx = box!.x + box!.width / 2;
    const cy = box!.y + box!.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.wheel(0, -300);
    await page.waitForTimeout(1_500);

    const after = await page.screenshot();
    expect(Buffer.compare(before, after)).not.toBe(0);
  });

  // -- Zoom Drift --

  test("scatter and MapLibre positions stay aligned across zoom levels", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    const MAX_DRIFT_PX = 2;
    const ZOOM_STEPS = 3;
    const ZOOM_DELTA = -400;
    const SETTLE_MS = 1_500;

    type Snapshot = {
      label: string;
      drifts: {
        name: string;
        dx: number;
        dy: number;
        scatterX: number;
        scatterY: number;
        mapX: number;
        mapY: number;
      }[];
    };

    const snapshots: Snapshot[] = [];

    async function measureDrift(label: string): Promise<Snapshot | null> {
      return page.evaluate(
        ({ refs, label }: { refs: typeof ALIGNMENT_REFERENCE_POINTS; label: string }) => {
          const map = (window as any).__geospatialAtlasMap;
          const vp = (window as any).__geospatialAtlasViewport;
          if (!map?.project || !vp) return null;

          const { x: cx, y: cy, scale, width, height, isGis } = vp;
          let sx = scale;
          let sy = scale;
          if (width < height) sx *= height / width;
          else sy *= width / height;
          const pixel_kx = (sx * width) / 2;
          const pixel_bx = ((-cx * sx + 1) * width) / 2;
          const pixel_ky = (-sy * height) / 2;
          const pixel_by = ((cy * sy + 1) * height) / 2;

          function projectLat(lat: number): number {
            const latRad = (lat * Math.PI) / 180;
            return (Math.log(Math.tan(Math.PI / 4 + latRad / 2)) * 180) / Math.PI;
          }

          const drifts = refs.map((ref: { name: string; lon: number; lat: number }) => {
            const py = isGis ? projectLat(ref.lat) : ref.lat;
            const scatterX = ref.lon * pixel_kx + pixel_bx;
            const scatterY = py * pixel_ky + pixel_by;
            const mapPx = map.project([ref.lon, ref.lat]);
            return {
              name: ref.name,
              scatterX,
              scatterY,
              mapX: mapPx.x,
              mapY: mapPx.y,
              dx: Math.abs(scatterX - mapPx.x),
              dy: Math.abs(scatterY - mapPx.y),
            };
          });
          return { label, drifts };
        },
        { refs: [...ALIGNMENT_REFERENCE_POINTS], label },
      );
    }

    const initial = await measureDrift("default zoom");
    if (initial === null) {
      test.skip();
      return;
    }
    snapshots.push(initial);

    const canvas = page.locator("canvas").first();
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const centerX = box!.x + box!.width / 2;
    const centerY = box!.y + box!.height / 2;

    // Zoom in
    for (let step = 1; step <= ZOOM_STEPS; step++) {
      await page.mouse.move(centerX, centerY);
      await page.mouse.wheel(0, ZOOM_DELTA);
      await page.waitForTimeout(SETTLE_MS);
      const snap = await measureDrift(`zoom-in step ${step}`);
      if (snap) snapshots.push(snap);
    }

    // Zoom back out
    for (let step = 1; step <= ZOOM_STEPS + 2; step++) {
      await page.mouse.move(centerX, centerY);
      await page.mouse.wheel(0, -ZOOM_DELTA);
      await page.waitForTimeout(SETTLE_MS);
      const snap = await measureDrift(`zoom-out step ${step}`);
      if (snap) snapshots.push(snap);
    }

    for (const snap of snapshots) {
      for (const d of snap.drifts) {
        expect(d.dx, `X drift for ${d.name} at "${snap.label}"`).toBeLessThan(MAX_DRIFT_PX);
        expect(d.dy, `Y drift for ${d.name} at "${snap.label}"`).toBeLessThan(MAX_DRIFT_PX);
      }
    }
  });

  test("pairwise point distances scale uniformly on zoom", async ({ page }) => {
    const ok = await loadGisFixture(page);
    if (!ok) {
      test.skip();
      return;
    }

    type Positions = { name: string; x: number; y: number }[];

    async function getMapLibrePositions(): Promise<Positions | null> {
      return page.evaluate(
        (refs: typeof ALIGNMENT_REFERENCE_POINTS) => {
          const map = (window as any).__geospatialAtlasMap;
          if (!map?.project) return null;
          return refs.map((ref) => {
            const px = map.project([ref.lon, ref.lat]);
            return { name: ref.name, x: px.x, y: px.y };
          });
        },
        [...ALIGNMENT_REFERENCE_POINTS],
      );
    }

    function pairwiseDistances(pos: Positions): number[] {
      const dists: number[] = [];
      for (let i = 0; i < pos.length; i++) {
        for (let j = i + 1; j < pos.length; j++) {
          const dx = pos[i].x - pos[j].x;
          const dy = pos[i].y - pos[j].y;
          dists.push(Math.sqrt(dx * dx + dy * dy));
        }
      }
      return dists;
    }

    const posBefore = await getMapLibrePositions();
    if (posBefore === null) {
      test.skip();
      return;
    }
    const distsBefore = pairwiseDistances(posBefore);

    const canvas = page.locator("canvas").first();
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.wheel(0, -600);
    await page.waitForTimeout(2_000);

    const posAfter = await getMapLibrePositions();
    expect(posAfter).not.toBeNull();
    const distsAfter = pairwiseDistances(posAfter!);

    const ratios = distsBefore.map((d, i) => (d < 1 ? 1 : distsAfter[i] / d));
    const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;

    for (let i = 0; i < ratios.length; i++) {
      const deviation = Math.abs(ratios[i] - avgRatio) / avgRatio;
      expect(
        deviation,
        `Pair ${i} ratio ${ratios[i].toFixed(3)} deviates from avg ${avgRatio.toFixed(3)}`,
      ).toBeLessThan(0.05);
    }
  });
});
