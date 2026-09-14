/**
 * E2E tests — GIS Detection & GeoParquet
 *
 * Tests the auto-detection of GIS columns and geoparquet geometry
 * extraction in both server mode and frontend-only mode.
 */

import { test, expect } from "@playwright/test";
import { type ChildProcess, spawn } from "child_process";
import path from "path";
import { inflateRawSync } from "zlib";
import {
  waitForServer,
  teardown,
  waitForCanvas,
  waitForDataRender,
  projectLat,
} from "./helpers.js";
import { E2E_CONSTANTS } from "../playwright.config.js";

const FIXTURES = path.resolve(__dirname, ".data/fixtures");
const SERVER_PORT = 5089; // use a different port from the main tests
const BASE_URL = `http://localhost:${SERVER_PORT}`;

const PHOTON_RESULTS = [
  {
    geometry: { coordinates: [45.3182, 2.0469] },
    properties: { name: "First Result", city: "Mogadishu", country: "Somalia", osm_value: "city" },
  },
  {
    geometry: { coordinates: [12.5, 41.9] },
    properties: { name: "Second Result", city: "Rome", country: "Italy", osm_value: "city" },
  },
  {
    geometry: { coordinates: [2.35, 48.85] },
    properties: { name: "Third Result", city: "Paris", country: "France", osm_value: "city" },
  },
  {
    geometry: { coordinates: [13.4, 52.5] },
    properties: { name: "Fourth Result", city: "Berlin", country: "Germany", osm_value: "city" },
  },
  {
    geometry: { coordinates: [-0.1, 51.5] },
    properties: { name: "Fifth Result", city: "London", country: "UK", osm_value: "city" },
  },
];

function stateFromURL(url: string): any {
  const state = new URLSearchParams(new URL(url).hash.split("?").at(-1)).get("state");
  if (state == null) {
    throw new Error("URL does not contain serialized state");
  }
  return JSON.parse(inflateRawSync(Buffer.from(state, "base64url")).toString());
}

function startServerWithFixture(parquetFile: string): ChildProcess {
  return spawn(
    "uv",
    [
      "run",
      "--directory",
      E2E_CONSTANTS.BACKEND_DIR,
      "geospatial-atlas",
      parquetFile,
      "--port",
      String(SERVER_PORT),
      "--no-auto-port",
      "--static",
      E2E_CONSTANTS.STATIC_DIR,
      "--disable-projection",
    ],
    { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env } },
  );
}

// ---------------------------------------------------------------------------
// GeoParquet server mode
// ---------------------------------------------------------------------------

test.describe.serial("GeoParquet Server Mode", () => {
  let server: ChildProcess;

  test.afterEach(async () => {
    await teardown(server);
    // Wait for port to be fully released
    await new Promise((r) => setTimeout(r, 2_000));
  });

  test("detects geometry column and extracts lon/lat from geoparquet", async ({
    request,
  }) => {
    server = startServerWithFixture(path.join(FIXTURES, "geoparquet_points.parquet"));
    await waitForServer(`${BASE_URL}/data/metadata.json`);

    const res = await request.get(`${BASE_URL}/data/metadata.json`);
    expect(res.ok()).toBeTruthy();

    const metadata = await res.json();
    const projection = metadata.props?.data?.projection;
    expect(projection).toBeDefined();
    expect(projection.x).toBe("lon");
    expect(projection.y).toBe("lat");
    expect(projection.isGis).toBe(true);
  });

  test("geoparquet app renders with scatter canvas", async ({ page }) => {
    server = startServerWithFixture(path.join(FIXTURES, "geoparquet_points.parquet"));
    await waitForServer(`${BASE_URL}/data/metadata.json`);

    await page.goto(BASE_URL);
    await waitForCanvas(page);

    const canvasCount = await page.locator("canvas").count();
    expect(canvasCount).toBeGreaterThan(0);
  });

  test("lon/lat parquet detected without geometry column", async ({ request }) => {
    server = startServerWithFixture(path.join(FIXTURES, "latlon_columns.parquet"));
    await waitForServer(`${BASE_URL}/data/metadata.json`);

    const res = await request.get(`${BASE_URL}/data/metadata.json`);
    const metadata = await res.json();
    const projection = metadata.props?.data?.projection;
    expect(projection).toBeDefined();
    expect(projection.x).toBe("lon");
    expect(projection.y).toBe("lat");
    expect(projection.isGis).toBe(true);
  });

  test("Enter selects the first geocoder result unless keyboard navigation selects another", async ({ page }) => {
    server = startServerWithFixture(path.join(FIXTURES, "latlon_columns.parquet"));
    await waitForServer(`${BASE_URL}/data/metadata.json`);
    await page.route("https://photon.komoot.io/api/**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ features: PHOTON_RESULTS }),
      }),
    );

    await page.goto(BASE_URL);
    const input = page.getByPlaceholder("Go to place...");
    await input.fill("result");
    const resultButtons = page.locator("ul li button");
    await expect(resultButtons).toHaveCount(5);

    // A pointer resting over the menu must not change the keyboard default.
    await resultButtons.last().hover();
    await input.press("Enter");
    await expect(input).toHaveValue("First Result, Mogadishu, Somalia");

    await expect
      .poll(() => stateFromURL(page.url()).chartStates?.["1"]?.viewport)
      .toMatchObject({
        x: 45.3182,
        y: projectLat(2.0469),
      });

    // Explicit arrow-key navigation remains available.
    await input.fill("result");
    await expect(resultButtons).toHaveCount(5);
    await input.press("ArrowDown");
    await input.press("ArrowDown");
    await input.press("Enter");
    await expect(input).toHaveValue("Second Result, Rome, Italy");
  });

  test("longitude/latitude columns detected", async ({ request }) => {
    server = startServerWithFixture(
      path.join(FIXTURES, "longitude_latitude_columns.parquet"),
    );
    await waitForServer(`${BASE_URL}/data/metadata.json`);

    const res = await request.get(`${BASE_URL}/data/metadata.json`);
    const metadata = await res.json();
    const projection = metadata.props?.data?.projection;
    expect(projection).toBeDefined();
    // find_gis_columns checks "longitude"/"latitude" first in its priority list
    expect(["longitude", "lon"]).toContain(projection.x);
    expect(["latitude", "lat"]).toContain(projection.y);
    expect(projection.isGis).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Frontend auto-detection (SettingsView)
// ---------------------------------------------------------------------------

test.describe("Frontend Auto GIS Detection", () => {
  let devServer: ChildProcess;

  test.beforeAll(async () => {
    devServer = spawn(
      "npm",
      ["run", "dev", "--", "--port", String(E2E_CONSTANTS.DEV_PORT)],
      {
        stdio: ["ignore", "pipe", "pipe"],
        cwd: path.resolve(__dirname, "../packages/viewer"),
        env: { ...process.env },
      },
    );
    await waitForServer(`http://localhost:${E2E_CONSTANTS.DEV_PORT}/`);
  });

  test.afterAll(async () => {
    await teardown(devServer);
  });

  const DEV_URL = `http://localhost:${E2E_CONSTANTS.DEV_PORT}`;

  /**
   * Helper: upload a fixture file and wait for the settings view.
   * DuckDB WASM init can be slow, so we give it generous timeouts.
   * If the parquet extension fails (unsigned in dev), the page shows
   * an error message — we detect that and skip gracefully.
   */
  async function uploadAndWaitForSettings(page: any, filePath: string) {
    await page.goto(`${DEV_URL}/#/file`);
    await expect(page.locator("text=Drag & drop")).toBeVisible({ timeout: 10_000 });

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(filePath);

    // Wait for the drop zone to disappear (data loading started)
    await expect(page.locator("text=Drag & drop")).not.toBeVisible({ timeout: 30_000 });

    // Wait for either: settings view (Confirm button) or error message
    const confirmBtn = page.locator("text=Confirm");
    const errorMsg = page.locator("text=Extension Autoloading Error");
    await Promise.race([
      confirmBtn.waitFor({ timeout: 30_000 }).catch(() => {}),
      errorMsg.waitFor({ timeout: 30_000 }).catch(() => {}),
    ]);

    // If DuckDB parquet extension failed, skip the test
    if (await errorMsg.isVisible()) {
      return false;
    }
    return true;
  }

  test("auto-detects lon/lat columns and pre-fills settings", async ({ page }) => {
    const ok = await uploadAndWaitForSettings(
      page,
      path.join(FIXTURES, "latlon_columns.parquet"),
    );
    if (!ok) {
      test.skip();
      return;
    }

    await expect(
      page.locator("text=Auto-detected GIS columns"),
    ).toBeVisible({ timeout: 5_000 });
  });

  test("auto-detects longitude/latitude columns", async ({ page }) => {
    const ok = await uploadAndWaitForSettings(
      page,
      path.join(FIXTURES, "longitude_latitude_columns.parquet"),
    );
    if (!ok) {
      test.skip();
      return;
    }

    await expect(
      page.locator("text=Auto-detected GIS columns"),
    ).toBeVisible({ timeout: 5_000 });
  });

  test("auto-detects fuzzy lon/lat column names", async ({ page }) => {
    const ok = await uploadAndWaitForSettings(
      page,
      path.join(FIXTURES, "fuzzy_latlon_columns.parquet"),
    );
    if (!ok) {
      test.skip();
      return;
    }

    await expect(
      page.locator("text=Auto-detected GIS columns"),
    ).toBeVisible({ timeout: 5_000 });
  });

  test("no detection banner for non-GIS data", async ({ page }) => {
    const ok = await uploadAndWaitForSettings(
      page,
      path.join(FIXTURES, "no_gis_columns.parquet"),
    );
    if (!ok) {
      test.skip();
      return;
    }

    // No auto-detection banner should appear
    await expect(page.locator("text=Auto-detected")).not.toBeVisible({ timeout: 3_000 });
  });

  test("auto-detects geometry column in geoparquet", async ({ page }) => {
    const ok = await uploadAndWaitForSettings(
      page,
      path.join(FIXTURES, "geoparquet_points.parquet"),
    );
    if (!ok) {
      test.skip();
      return;
    }

    await expect(
      page.locator("text=geometry column"),
    ).toBeVisible({ timeout: 5_000 });
  });
});
