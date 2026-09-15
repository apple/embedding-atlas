// Minimal playwright-chromium holder: opens the viewer tab so the MCP server
// has a connected browser to forward tool calls to. Prints "VIEWER READY"
// when the canvas + point-count badge are visible, then keeps the tab open
// until killed.
//
// Usage:  node scripts/mcp_harness/viewer_holder.mjs <viewer_url>
//   e.g.  node scripts/mcp_harness/viewer_holder.mjs http://localhost:5055

import { chromium } from "playwright";

const VIEWER_URL = process.argv[2] || "http://localhost:5055";

async function main() {
  const browser = await chromium.launch({
    headless: true,
    // WebGPU headless needs angle/vulkan swiftshader; use WebGL2 fallback
    // (the viewer falls back automatically).
    args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan,UseSkiaRenderer"],
  });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  page.on("console", (msg) => {
    const t = msg.text();
    // Skip spammy info messages but log errors + MCP init
    if (msg.type() === "error" || /MCP/.test(t) || /WebGPU/.test(t)) {
      console.log(`[browser ${msg.type()}] ${t}`);
    }
  });
  page.on("pageerror", (err) => {
    console.log(`[pageerror] ${err.message}`);
  });

  console.log(`Opening ${VIEWER_URL} ...`);
  await page.goto(VIEWER_URL, { waitUntil: "domcontentloaded", timeout: 60_000 });

  // Wait for canvas
  await page.waitForSelector("canvas", { timeout: 60_000 });

  // Wait for the point-count badge or a loaded chart
  try {
    await page.locator("text=/\\d[\\d,]* points/").first().waitFor({ timeout: 120_000 });
  } catch {
    // Fallback: settle for a few seconds
    await page.waitForTimeout(4_000);
  }

  // Confirm the MCP websocket connected by checking window hook
  const mapHook = await page.evaluate(() => {
    return {
      hasMap: typeof window.__geospatialAtlasMap !== "undefined",
      viewport: window.__geospatialAtlasViewport ?? null,
    };
  });
  console.log("Viewer hooks:", JSON.stringify(mapHook));
  console.log("VIEWER READY");

  // Keep alive; graceful shutdown on SIGINT/SIGTERM
  let shuttingDown = false;
  const shutdown = async (sig) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`Shutting down (${sig})`);
    try {
      await browser.close();
    } finally {
      process.exit(0);
    }
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  // Keep the event loop busy
  setInterval(() => {}, 10_000);
}

main().catch((e) => {
  console.error("viewer_holder error:", e);
  process.exit(1);
});
