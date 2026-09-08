// Copyright (c) 2025 Geospatial Atlas contributors. MIT License.
//
// Geospatial extensions to the MCP tool set: fly-to, viewport inspection,
// bbox selection, basemap styling, and composite fly+screenshot.
//
// These tools run inside the viewer JS (same process as the scatter chart +
// MapLibre basemap), so they can both READ from the live viewport window
// hook (`__geospatialAtlasViewport`) and WRITE by setting chart state —
// which the Svelte effect in EmbeddingViewImpl mirrors into MapLibre.
//
// Design notes
// ------------
// Driving MapLibre directly (flyTo on __geospatialAtlasMap) would
// desynchronize the scatter overlay, so all writes go through chart state.
// The chart viewport uses {x, y, scale} where in GIS mode x is longitude,
// y is Web-Mercator-projected latitude, and scale determines zoom per:
//   zoom = log2(360 * scale * width / 1024)
//
// Bbox-to-viewport math (landscape, i.e. canvas width >= height):
//   visible_x_deg = 2 / scale
//   visible_y_proj_deg = 2 * height / (scale * width)
//   -> scale_fit = min(2/target_w, 2*height/(width*target_h))

import type { MCPTool, ToolResponse } from "../app/mcp_server.js";
import type { EmbeddingAtlasStore } from "../stores/embedding_atlas_store.js";
import type { ModelContextDelegate } from "./model_context.js";
import { screenshot } from "../utils/screenshot.js";
import { get } from "svelte/store";

// Mirror of Viewport.projectLat / unprojectLat — duplicated here to avoid
// a cross-package import cycle (this module is in viewer, those live in
// @embedding-atlas/component).
function projectLat(lat: number): number {
  const rad = (lat * Math.PI) / 180;
  return (Math.log(Math.tan(Math.PI / 4 + rad / 2)) * 180) / Math.PI;
}
function unprojectLat(y: number): number {
  const rad = (y * Math.PI) / 180;
  return (2 * Math.atan(Math.exp(rad)) - Math.PI / 2) * (180 / Math.PI);
}

interface ViewportHook {
  x: number;
  y: number; // Mercator-projected latitude (GIS mode)
  scale: number;
  width: number;
  height: number;
  isGis: boolean;
}

function readViewportHook(): ViewportHook | null {
  const w = (window as any).__geospatialAtlasViewport;
  if (!w || typeof w.scale !== "number") return null;
  return {
    x: w.x,
    y: w.y,
    scale: w.scale,
    width: w.width,
    height: w.height,
    isGis: !!w.isGis,
  };
}

function scaleToZoom(scale: number, width: number): number {
  return Math.log2((360 * scale * width) / 1024);
}
function zoomToScale(zoom: number, width: number): number {
  return (1024 * Math.pow(2, zoom)) / (360 * width);
}

/** Compute visible bbox (geographic) from current viewport hook. */
function viewportBbox(v: ViewportHook): {
  west: number;
  east: number;
  south: number;
  north: number;
} {
  let sx = v.scale;
  let sy = v.scale;
  if (v.width < v.height) sx *= v.height / v.width;
  else sy *= v.width / v.height;
  const halfX = 1 / sx;
  const halfY = 1 / sy;
  const west = v.x - halfX;
  const east = v.x + halfX;
  const projS = v.y - halfY;
  const projN = v.y + halfY;
  return {
    west,
    east,
    south: v.isGis ? unprojectLat(projS) : projS,
    north: v.isGis ? unprojectLat(projN) : projN,
  };
}

/**
 * Compute {x, y, scale} that fits a geographic bbox inside (width, height)
 * with optional padding (fraction of the smallest side, default 0.05).
 */
function bboxToViewport(
  west: number,
  south: number,
  east: number,
  north: number,
  width: number,
  height: number,
  isGis: boolean,
  padding: number = 0.05,
): { x: number; y: number; scale: number } {
  const projS = isGis ? projectLat(south) : south;
  const projN = isGis ? projectLat(north) : north;
  const targetW = Math.max(1e-9, east - west);
  const targetH = Math.max(1e-9, projN - projS);
  const pad = 1 + Math.max(0, padding) * 2;
  let scaleX: number, scaleY: number;
  if (width >= height) {
    scaleX = 2 / (targetW * pad);
    scaleY = (2 * height) / (width * targetH * pad);
  } else {
    scaleX = (2 * width) / (height * targetW * pad);
    scaleY = 2 / (targetH * pad);
  }
  return {
    x: (west + east) / 2,
    y: (projS + projN) / 2,
    scale: Math.min(scaleX, scaleY),
  };
}

/** Find the GIS embedding chart id; null if none. */
function findGisChartId(store: EmbeddingAtlasStore): string | null {
  const charts = get(store.charts);
  for (const [id, spec] of Object.entries(charts) as [string, any][]) {
    if (spec?.type === "embedding" && spec?.data?.isGis) return id;
  }
  // Fall back: any embedding chart
  for (const [id, spec] of Object.entries(charts) as [string, any][]) {
    if (spec?.type === "embedding") return id;
  }
  return null;
}

/** Merge a partial state update into an existing chart state. */
function updateChartState(store: EmbeddingAtlasStore, id: string, patch: Record<string, any>) {
  store.updateChartState(id, (draft) => Object.assign(draft, patch));
}

/** Wait ~2 frames so the Svelte $effect that mirrors state → MapLibre fires. */
async function nextRender(): Promise<void> {
  await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
}

/**
 * Wait for MapLibre to finish loading tiles and become idle. Resolves
 * immediately if the map isn't present. Uses a timeout so a never-idle map
 * (e.g. network flap) doesn't hang the tool call.
 */
async function waitForMapIdle(timeoutMs: number = 6000): Promise<void> {
  const map = (window as any).__geospatialAtlasMap;
  if (!map || typeof map.once !== "function") return;
  // If already idle (no pending tiles + no ongoing animations) and rendered,
  // resolve fast — triggerRepaint() below still guarantees a fresh frame.
  await new Promise<void>((resolve) => {
    let resolved = false;
    const done = () => {
      if (resolved) return;
      resolved = true;
      resolve();
    };
    map.once("idle", done);
    map.triggerRepaint?.();
    setTimeout(done, timeoutMs);
  });
}

/**
 * Find the common parent of the MapLibre basemap and the scatter overlay
 * canvas so a screenshot captures *both*. DOM layout from
 * EmbeddingViewImpl.svelte:
 *   <div style="position:relative">
 *     <div class="maplibregl-map" style="position:absolute; z:0">…</div>
 *     <canvas style="position:absolute; z:1">…</canvas>
 *     <svg>…tooltips, selection, lasso…</svg>
 *   </div>
 * Taking only the `.maplibregl-map` gives tiles without points.
 */
function findMapContainer(delegate: ModelContextDelegate): HTMLElement | null {
  const cont = delegate.container;
  const mapEl = cont.querySelector<HTMLElement>(".maplibregl-map");
  if (mapEl && mapEl.parentElement instanceof HTMLElement) {
    return mapEl.parentElement;
  }
  return mapEl;
}

function textResponse(t: string): ToolResponse {
  return { content: [{ type: "text", text: t }] };
}
function jsonResponse(v: any): ToolResponse {
  return textResponse(JSON.stringify(v));
}

/** Error helper — returns a tool response with `isError: true`. */
function errorResponse(msg: string): ToolResponse {
  return { content: [{ type: "text", text: msg }], isError: true };
}

// Screenshot shared options (same as model_context.ts)
const SHOT_OPTS = { maxWidth: 1568, maxHeight: 1568, pixelRatio: 2 };

export function geoTools(store: EmbeddingAtlasStore, delegate: ModelContextDelegate): MCPTool[] {
  return [
    {
      name: "get_map_viewport",
      description:
        "Get the current map viewport: center lat/lon, MapLibre zoom, geographic bounding box (west/south/east/north), and canvas size in CSS pixels. Works only while a GIS embedding chart is present. Screenshots reflect this viewport.",
      inputSchema: { type: "object", additionalProperties: false },
      execute: async () => {
        const v = readViewportHook();
        if (!v) return errorResponse("No viewport available — is a GIS chart loaded?");
        const bbox = viewportBbox(v);
        const chartId = findGisChartId(store);
        return jsonResponse({
          chart_id: chartId,
          center: {
            lon: v.x,
            lat: v.isGis ? unprojectLat(v.y) : v.y,
          },
          zoom: scaleToZoom(v.scale, v.width),
          scale: v.scale,
          bbox,
          canvas: { width: v.width, height: v.height },
          is_gis: v.isGis,
        });
      },
    },

    {
      name: "fly_to_point",
      description:
        "Move the map centre to the given (lon, lat) at the given MapLibre zoom. Use higher zoom values for closer views: 3=continent, 6=country, 10=city, 14=street, 17=building. Sets the GIS embedding chart's viewport; the MapLibre basemap and scatter overlay follow automatically.",
      inputSchema: {
        type: "object",
        properties: {
          lon: { type: "number", description: "Longitude in degrees (-180..180)" },
          lat: { type: "number", description: "Latitude in degrees (-85..85)" },
          zoom: {
            type: "number",
            description: "MapLibre zoom (0 = world, 14 = street). Default: 10. Clamped to [1, 19].",
          },
        },
        required: ["lon", "lat"],
        additionalProperties: false,
      },
      execute: async (params: { lon: number; lat: number; zoom?: number }) => {
        const v = readViewportHook();
        if (!v) return errorResponse("No viewport available.");
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart found.");
        const zoom = Math.min(19, Math.max(1, params.zoom ?? 10));
        const scale = zoomToScale(zoom, v.width);
        const x = params.lon;
        const y = v.isGis ? projectLat(params.lat) : params.lat;
        updateChartState(store, chartId, { viewport: { x, y, scale } });
        await nextRender();
        return jsonResponse({
          ok: true,
          viewport: { x, y, scale },
          center: { lon: params.lon, lat: params.lat },
          zoom,
        });
      },
    },

    {
      name: "fly_to_bbox",
      description:
        "Fit the map to the given geographic bounding box. Padding is a fraction of the smallest side (default 0.05 = 5%). Use this to zoom to a region of interest.",
      inputSchema: {
        type: "object",
        properties: {
          west: { type: "number", description: "Min longitude" },
          south: { type: "number", description: "Min latitude" },
          east: { type: "number", description: "Max longitude" },
          north: { type: "number", description: "Max latitude" },
          padding: {
            type: "number",
            description: "Fraction of visible area left as padding (default 0.05).",
          },
        },
        required: ["west", "south", "east", "north"],
        additionalProperties: false,
      },
      execute: async (params: { west: number; south: number; east: number; north: number; padding?: number }) => {
        const v = readViewportHook();
        if (!v) return errorResponse("No viewport available.");
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart found.");
        const vp = bboxToViewport(
          params.west,
          params.south,
          params.east,
          params.north,
          v.width,
          v.height,
          v.isGis,
          params.padding ?? 0.05,
        );
        updateChartState(store, chartId, { viewport: vp });
        await nextRender();
        return jsonResponse({
          ok: true,
          viewport: vp,
          zoom: scaleToZoom(vp.scale, v.width),
          center: {
            lon: vp.x,
            lat: v.isGis ? unprojectLat(vp.y) : vp.y,
          },
        });
      },
    },

    {
      name: "get_map_screenshot",
      description:
        "PNG screenshot of only the map canvas (scatter overlay + MapLibre basemap), without the surrounding UI chrome, legend, or other panels. Preferred for visual inspection of geospatial patterns.",
      inputSchema: { type: "object", additionalProperties: false },
      execute: async () => {
        const el = findMapContainer(delegate);
        if (!el) return errorResponse("Map container not found. Is a GIS embedding chart rendered?");
        await nextRender();
        await waitForMapIdle();
        const img = await screenshot(el, SHOT_OPTS);
        return imageResponse(img);
      },
    },

    {
      name: "get_map_screenshot_at",
      description:
        "Composite: fly to a point or bbox, then capture a map-only screenshot. Provide EITHER {lon, lat, zoom?} for a point OR {west, south, east, north, padding?} for a bbox. Returns the screenshot plus the resulting viewport info.",
      inputSchema: {
        type: "object",
        properties: {
          lon: { type: "number" },
          lat: { type: "number" },
          zoom: { type: "number" },
          west: { type: "number" },
          south: { type: "number" },
          east: { type: "number" },
          north: { type: "number" },
          padding: { type: "number" },
          settle_ms: {
            type: "number",
            description: "Extra wait after render before capturing (default 400ms)",
          },
        },
        additionalProperties: false,
      },
      execute: async (params: any) => {
        const v = readViewportHook();
        if (!v) return errorResponse("No viewport available.");
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart found.");
        let vp: { x: number; y: number; scale: number };
        if (params.lon != null && params.lat != null) {
          const zoom = Math.min(19, Math.max(1, params.zoom ?? 10));
          vp = {
            x: params.lon,
            y: v.isGis ? projectLat(params.lat) : params.lat,
            scale: zoomToScale(zoom, v.width),
          };
        } else if (params.west != null && params.south != null && params.east != null && params.north != null) {
          vp = bboxToViewport(
            params.west,
            params.south,
            params.east,
            params.north,
            v.width,
            v.height,
            v.isGis,
            params.padding ?? 0.05,
          );
        } else {
          return errorResponse("Provide either {lon, lat, zoom?} or {west, south, east, north, padding?}");
        }
        updateChartState(store, chartId, { viewport: vp });
        await nextRender();
        await waitForMapIdle();
        // Extra settle time only if caller asked for it (default 0).
        if (params.settle_ms) {
          await new Promise((r) => setTimeout(r, params.settle_ms));
        }
        const el = findMapContainer(delegate);
        if (!el) return errorResponse("Map container not found.");
        const img = await screenshot(el, SHOT_OPTS);
        const bbox = viewportBbox({ ...v, x: vp.x, y: vp.y, scale: vp.scale });
        return {
          content: [
            ...imageResponse(img).content,
            {
              type: "text",
              text: JSON.stringify({
                center: { lon: vp.x, lat: v.isGis ? unprojectLat(vp.y) : vp.y },
                zoom: scaleToZoom(vp.scale, v.width),
                bbox,
              }),
            },
          ],
        };
      },
    },

    {
      name: "select_bbox",
      description:
        "Cross-filter the whole viewer to rows whose (lon, lat) fall inside the given bbox. Sets the GIS embedding chart's range selection (`brush`). Returns a JSON object with {applied, row_count, matched_count} — the matched_count is from a SQL COUNT of rows in the bbox. Use {} to clear the selection.",
      inputSchema: {
        type: "object",
        properties: {
          west: { type: "number" },
          south: { type: "number" },
          east: { type: "number" },
          north: { type: "number" },
        },
        additionalProperties: false,
      },
      execute: async (params: any) => {
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart found.");
        const spec = get(store.charts)[chartId];
        const xCol = spec?.data?.x;
        const yCol = spec?.data?.y;

        if (params.west == null || params.south == null || params.east == null || params.north == null) {
          updateChartState(store, chartId, { brush: undefined });
          return jsonResponse({ applied: "clear" });
        }
        // Chart brush is stored in projected coords (matching internalDataY)
        const isGis = !!spec?.data?.isGis;
        const projS = isGis ? projectLat(params.south) : params.south;
        const projN = isGis ? projectLat(params.north) : params.north;
        const brush = {
          xMin: params.west,
          xMax: params.east,
          yMin: projS,
          yMax: projN,
        };
        updateChartState(store, chartId, { brush });

        let matched: number | null = null;
        if (xCol && yCol) {
          const q = await store.coordinator.query(
            `SELECT COUNT(*)::BIGINT AS c FROM "${store.props.data.table}"
             WHERE "${xCol}" BETWEEN ${params.west} AND ${params.east}
               AND "${yCol}" BETWEEN ${params.south} AND ${params.north}`,
          );
          matched = Number(q.toArray()[0].c);
        }
        await nextRender();
        return jsonResponse({ applied: "bbox", brush, matched_count: matched });
      },
    },

    {
      name: "clear_selection",
      description: "Clear the GIS cross-filter brush (equivalent to select_bbox with no args).",
      inputSchema: { type: "object", additionalProperties: false },
      execute: async () => {
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart found.");
        updateChartState(store, chartId, { brush: undefined });
        return textResponse("ok");
      },
    },

    {
      name: "count_in_bbox",
      description:
        "Read-only variant of `select_bbox`: returns just the number of rows whose (lon, lat) falls inside the bbox, without touching the current cross-filter. Handy for quick anomaly probes.",
      inputSchema: {
        type: "object",
        properties: {
          west: { type: "number" },
          south: { type: "number" },
          east: { type: "number" },
          north: { type: "number" },
        },
        required: ["west", "south", "east", "north"],
        additionalProperties: false,
      },
      execute: async (params: { west: number; south: number; east: number; north: number }) => {
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart.");
        const spec = get(store.charts)[chartId];
        const xCol = spec?.data?.x;
        const yCol = spec?.data?.y;
        if (!xCol || !yCol) return errorResponse("chart has no x/y columns.");
        const q = await store.coordinator.query(
          `SELECT COUNT(*)::BIGINT c FROM "${store.props.data.table}"
             WHERE "${xCol}" BETWEEN ${params.west} AND ${params.east}
               AND "${yCol}" BETWEEN ${params.south} AND ${params.north}`,
        );
        const c = Number(q.toArray()[0].c);
        return jsonResponse({ count: c, bbox: params });
      },
    },

    {
      name: "find_nearby",
      description:
        "Return the N places closest to (lon, lat) within `radius_km` kilometres. Uses great-circle distance on the stored lon/lat columns; excludes rows with NULL coords. Default: limit=20, radius_km=1.",
      inputSchema: {
        type: "object",
        properties: {
          lon: { type: "number" },
          lat: { type: "number" },
          radius_km: {
            type: "number",
            description: "Search radius in kilometres (default 1)",
          },
          limit: {
            type: "number",
            description: "Max number of results (default 20)",
          },
          columns: {
            type: "array",
            items: { type: "string" },
            description:
              "Optional list of additional columns to include in the result. If omitted, returns id, lon, lat, distance_m.",
          },
          where: {
            type: "string",
            description:
              "Optional SQL WHERE clause (without the keyword) to filter further, e.g. `source_dataset = 'meta'`.",
          },
        },
        required: ["lon", "lat"],
        additionalProperties: false,
      },
      execute: async (params: {
        lon: number;
        lat: number;
        radius_km?: number;
        limit?: number;
        columns?: string[];
        where?: string;
      }) => {
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart.");
        const spec = get(store.charts)[chartId];
        const xCol = spec?.data?.x;
        const yCol = spec?.data?.y;
        if (!xCol || !yCol) return errorResponse("chart has no x/y columns.");
        const r = Math.max(0.0001, params.radius_km ?? 1);
        const n = Math.max(1, Math.min(500, params.limit ?? 20));
        // Coarse pre-filter with a bbox in degrees (1° ≈ 111 km lat,
        // cos(lat) * 111 km lon). Keeps the Haversine cost down on 75 M rows.
        const dLat = r / 111;
        const dLon = r / (111 * Math.max(0.01, Math.cos((params.lat * Math.PI) / 180)));
        const where = params.where ? `AND (${params.where})` : "";
        const extra = (params.columns ?? [])
          .filter((c) => /^[a-zA-Z0-9_]+$/.test(c))
          .map((c) => `, "${c}"`)
          .join("");
        const sql = `
            WITH candidates AS (
              SELECT "${xCol}" AS lon, "${yCol}" AS lat${extra}
              FROM "${store.props.data.table}"
              WHERE "${xCol}" BETWEEN ${params.lon - dLon} AND ${params.lon + dLon}
                AND "${yCol}" BETWEEN ${params.lat - dLat} AND ${params.lat + dLat}
                ${where}
            )
            SELECT *,
              (6371000 * 2 * asin(sqrt(
                pow(sin(radians(lat - ${params.lat}) / 2), 2) +
                cos(radians(${params.lat})) * cos(radians(lat))
                * pow(sin(radians(lon - ${params.lon}) / 2), 2)
              )))::DOUBLE AS distance_m
            FROM candidates
            WHERE distance_m <= ${r * 1000}
            ORDER BY distance_m
            LIMIT ${n}`;
        const q = await store.coordinator.query(sql);
        return jsonResponse({
          center: { lon: params.lon, lat: params.lat },
          radius_km: r,
          results: q.toArray(),
        });
      },
    },

    {
      name: "density_grid",
      description:
        "Bin the rows inside a bbox into an `nx` × `ny` grid and return the count in each cell. Useful for spotting dense anomalies inside a region. Grid cells are reported as (cx, cy, count); grid corners use the bbox exactly.",
      inputSchema: {
        type: "object",
        properties: {
          west: { type: "number" },
          south: { type: "number" },
          east: { type: "number" },
          north: { type: "number" },
          nx: { type: "number", description: "Number of columns (default 20, max 200)" },
          ny: { type: "number", description: "Number of rows (default 20, max 200)" },
          top_k: {
            type: "number",
            description: "Return only the top-k densest cells (default: all).",
          },
        },
        required: ["west", "south", "east", "north"],
        additionalProperties: false,
      },
      execute: async (params: any) => {
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart.");
        const spec = get(store.charts)[chartId];
        const xCol = spec?.data?.x;
        const yCol = spec?.data?.y;
        if (!xCol || !yCol) return errorResponse("chart has no x/y columns.");
        const nx = Math.max(2, Math.min(200, params.nx ?? 20));
        const ny = Math.max(2, Math.min(200, params.ny ?? 20));
        const w = Math.max(1e-9, params.east - params.west);
        const h = Math.max(1e-9, params.north - params.south);
        const topK = params.top_k ? `LIMIT ${Math.max(1, Math.floor(params.top_k))}` : "";
        const sql = `
            SELECT
              LEAST(${nx - 1}, GREATEST(0, CAST(floor(("${xCol}" - ${params.west}) / ${w / nx}) AS INT))) AS ix,
              LEAST(${ny - 1}, GREATEST(0, CAST(floor(("${yCol}" - ${params.south}) / ${h / ny}) AS INT))) AS iy,
              COUNT(*)::BIGINT AS n
            FROM "${store.props.data.table}"
            WHERE "${xCol}" BETWEEN ${params.west} AND ${params.east}
              AND "${yCol}" BETWEEN ${params.south} AND ${params.north}
            GROUP BY 1, 2
            ORDER BY n DESC
            ${topK}`;
        const q = await store.coordinator.query(sql);
        const rows = q.toArray() as { ix: number; iy: number; n: number | bigint }[];
        const cellW = w / nx;
        const cellH = h / ny;
        const cells = rows.map((r) => ({
          ix: Number(r.ix),
          iy: Number(r.iy),
          count: Number(r.n),
          cx: params.west + (Number(r.ix) + 0.5) * cellW,
          cy: params.south + (Number(r.iy) + 0.5) * cellH,
        }));
        return jsonResponse({
          bbox: { west: params.west, south: params.south, east: params.east, north: params.north },
          grid: { nx, ny, cell_w_deg: cellW, cell_h_deg: cellH },
          cells,
        });
      },
    },

    {
      name: "highlight_points",
      description:
        "Draw temporary circular markers at the given (lon, lat) coordinates so they stand out in the next `get_map_screenshot` call. Pass an empty list to clear markers. Markers persist across calls but are purely client-side — they don't modify chart data.",
      inputSchema: {
        type: "object",
        properties: {
          points: {
            type: "array",
            items: {
              type: "object",
              properties: {
                lon: { type: "number" },
                lat: { type: "number" },
                label: { type: "string" },
                color: { type: "string", description: "CSS colour (default #ff3b30)" },
                radius: { type: "number", description: "Marker radius in px (default 10)" },
              },
              required: ["lon", "lat"],
              additionalProperties: false,
            },
          },
        },
        required: ["points"],
        additionalProperties: false,
      },
      execute: async (params: { points: any[] }) => {
        const cont = delegate.container;
        const map = (window as any).__geospatialAtlasMap;
        const view = readViewportHook();
        if (!view) return errorResponse("No viewport.");
        // Remove any pre-existing overlay from a previous highlight call.
        const existing = cont.querySelector<HTMLElement>("[data-gsa-highlight-layer]");
        if (existing) existing.remove();
        if (!params.points || params.points.length === 0) {
          return jsonResponse({ ok: true, cleared: true });
        }
        // Find the anchor — parent of the MapLibre container — we position
        // absolute inside it so markers land on the right pixels without
        // worrying about transforms.
        const mapEl = cont.querySelector<HTMLElement>(".maplibregl-map");
        const parent = (mapEl?.parentElement as HTMLElement) ?? cont;
        const rect = mapEl?.getBoundingClientRect() ?? { width: view.width, height: view.height };
        const layer = document.createElement("div");
        layer.setAttribute("data-gsa-highlight-layer", "");
        layer.style.cssText = `position:absolute;inset:0;pointer-events:none;z-index:5;`;
        parent.appendChild(layer);
        // Project using maplibre so we stay in sync with the basemap
        for (const p of params.points) {
          if (!map || typeof map.project !== "function") continue;
          const { x, y } = map.project([p.lon, p.lat]);
          const r = Math.max(2, Math.min(80, Number(p.radius) || 10));
          const color = p.color || "#ff3b30";
          const dot = document.createElement("div");
          dot.style.cssText =
            `position:absolute;left:${x - r}px;top:${y - r}px;` +
            `width:${2 * r}px;height:${2 * r}px;border-radius:50%;` +
            `border:2px solid ${color};background:${color}33;` +
            `box-shadow:0 0 6px ${color};`;
          layer.appendChild(dot);
          if (p.label) {
            const lbl = document.createElement("div");
            lbl.textContent = String(p.label);
            lbl.style.cssText =
              `position:absolute;left:${x + r + 4}px;top:${y - 10}px;` +
              `color:#111;background:rgba(255,255,255,0.85);` +
              `padding:2px 6px;border-radius:3px;font:11px/1.2 -apple-system,sans-serif;` +
              `border:1px solid ${color};white-space:nowrap;`;
            layer.appendChild(lbl);
          }
        }
        return jsonResponse({ ok: true, drawn: params.points.length });
      },
    },

    {
      name: "set_basemap_style",
      description:
        "Switch the MapLibre basemap. Accepts a MapLibre style URL, a built-in name (`openfreemap-liberty`, `openfreemap-positron`, `openfreemap-bright`, `osm-raster`, `none`), or `null` / 'none' to hide the basemap.",
      inputSchema: {
        type: "object",
        properties: {
          style: {
            type: "string",
            description: "Style URL or built-in name",
          },
        },
        required: ["style"],
        additionalProperties: false,
      },
      execute: async (params: { style: string | null }) => {
        const chartId = findGisChartId(store);
        if (!chartId) return errorResponse("No GIS embedding chart found.");
        const known: Record<string, string | null> = {
          none: null,
          null: null,
          "openfreemap-liberty": "https://tiles.openfreemap.org/styles/liberty",
          "openfreemap-positron": "https://tiles.openfreemap.org/styles/positron",
          "openfreemap-bright": "https://tiles.openfreemap.org/styles/bright",
          "osm-raster":
            // Inline raster style — no keys required.
            "data:application/json;base64," +
            btoa(
              JSON.stringify({
                version: 8,
                sources: {
                  osm: {
                    type: "raster",
                    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                    tileSize: 256,
                    attribution: "© OpenStreetMap contributors",
                  },
                },
                layers: [{ id: "osm", type: "raster", source: "osm" }],
              }),
            ),
        };
        let styleValue: string | null;
        if (params.style == null) styleValue = null;
        else if (params.style in known) styleValue = known[params.style];
        else styleValue = params.style;
        store.updateChart(chartId, (draft) => {
          (draft as any).mapStyle = styleValue;
        });
        await nextRender();
        return jsonResponse({ ok: true, mapStyle: styleValue });
      },
    },
  ];
}

function imageResponse(dataUrl: string): ToolResponse {
  const comma = dataUrl.indexOf(",");
  if (!dataUrl.startsWith("data:") || comma < 0) return textResponse("failed to take screenshot");
  const meta = dataUrl.substring(5, comma);
  const b64 = dataUrl.substring(comma + 1);
  const mimeType = meta.replace(";base64", "");
  if (!mimeType.startsWith("image/")) return textResponse("bad mime");
  return { content: [{ type: "image", data: b64, mimeType }] };
}
