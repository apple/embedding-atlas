# Connecting an LLM agent to Geospatial Atlas via MCP

Geospatial Atlas ships a **Model Context Protocol** server so that any MCP-capable LLM
(Claude Desktop, Claude Code, Cursor, Continue, your own SDK client, …) can drive the
viewer in real time: run SQL against your data, list/add/update/delete charts, change
column styles, capture screenshots.

The transport is **Streamable HTTP** — no bridge scripts, no Node.js, no stdio shim.

## Setup (3 steps)

### 1. Start the server with MCP enabled

**Via the Python CLI:**

```bash
uv run geospatial-atlas /path/to/your.parquet --mcp
```

**Via the desktop app:** MCP is enabled by default. When you load a
dataset, the idle-state launch form has an "Expose MCP endpoint"
checkbox; the port is picked at launch time and shown (with a
copy button) in the status panel. Uncheck to run without MCP. The
preference is per-launch — closing the app returns to the default
on/off you last set.

The URL banner printed on launch is where the viewer _and_ MCP clients connect:

```
  ➜ URL: http://localhost:5055
  ➜ MCP server: http://localhost:5055/mcp
```

### 2. Open the viewer in a browser

```
http://localhost:5055
```

The viewer is where the tool handlers actually execute (chart state, Mosaic coordinator,
canvases all live in the webview). Keep the tab open while you're chatting with the LLM.

**Two supported paths:**

1. **Interactive — real Chrome/Safari tab.** What you want ~99 % of
   the time: full WebGPU, fast tile loading, no headless quirks. Just
   open the URL in your normal browser and leave the tab open.
2. **Headless — Playwright chromium.** Useful for autonomous runs,
   CI, or any time you don't want a visible window. Script:

   ```bash
   node scripts/mcp_harness/viewer_holder.mjs http://localhost:5055
   ```

   The harness prints `VIEWER READY` when the map hook is live and
   stays up until killed. See `scripts/mcp_harness/mcp.sh` for a
   companion curl wrapper (`list`, `call`, `sql`, `raw`).

Either path exposes the exact same tool set to the MCP endpoint —
clients can't tell which one is driving.

### 3. Point your LLM at `/mcp`

**Claude Desktop** — edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "geospatial-atlas": {
      "url": "http://localhost:5055/mcp"
    }
  }
}
```

Fully quit Claude Desktop (⌘Q) and reopen. You should see 19 tools show up in the
tool-picker.

**Claude Code / Cursor / Continue / any other client**: same — they all accept a URL
entry for MCP servers. Check their docs for the exact config path.

**Programmatic / raw** — any HTTP client:

```bash
curl -s -X POST http://localhost:5055/mcp/ \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | grep '^data: ' | head -1 | cut -c7-
```

That returns the live tools list as JSON.

## The tools

### Data

- `get_data_schema` — table name + column list (names, types)
- `run_sql_query` — readonly SQL against the loaded DuckDB

### Charts

- `list_charts`, `add_chart`, `delete_chart`
- `get_chart_spec`, `set_chart_spec`
- `get_chart_state`, `set_chart_state`, `clear_chart_state`
- `get_chart_screenshot` — PNG of one chart

### Layout

- `get_layout_type`, `set_layout_type` (`"list"` / `"dashboard"`)
- `get_layout_state`, `set_layout_state`
- `get_full_screenshot` — PNG of the entire viewer

### Rendering

- `list_renderers` — column renderer types (text, number, timestamp, bar, …)
- `get_column_styles`, `set_column_style`

### Geospatial (GIS datasets only)

The geo tools operate on whichever embedding chart has `data.isGis = true`
— they locate it by scanning `charts` and drive the MapLibre basemap

- scatter overlay in lockstep by setting chart state. Screenshots only
  capture the rendered basemap after calling `map.once('idle')`, so tiles
  are guaranteed loaded before the PNG is produced.

* `get_map_viewport` — current center (lon/lat), MapLibre zoom, bbox
  (west/south/east/north), canvas size in CSS pixels, and the chart id.
  Always the first tool to call when doing anything geographic.
* `fly_to_point` — `{lon, lat, zoom?}`; zoom defaults to 10, clamped to
  `[1, 19]`. Uses `jumpTo` internally so it's instantaneous.
* `fly_to_bbox` — fit a `{west, south, east, north, padding?}` region.
  Honours the viewport aspect ratio; `padding` is a fraction of the
  smaller side.
* `get_map_screenshot` — PNG of _just_ the map canvas (scatter overlay +
  basemap + markers), without the sidebar / charts / etc.
* `get_map_screenshot_at` — composite: set viewport (either `{lon, lat,
zoom?}` or `{west, south, east, north, padding?}`), wait for the map
  to become idle, then screenshot. Accepts an optional `settle_ms`
  that adds fixed wait after tile idle (default 0).
* `select_bbox` — cross-filter the whole viewer to rows whose `(lon, lat)`
  falls in the bbox; sets the GIS chart's `brush`. Returns
  `{applied, brush, matched_count}` where `matched_count` is a SQL
  `COUNT(*)` inside the bbox (not the downsampled render count).
* `clear_selection` — clear the brush.
* `count_in_bbox` — read-only: just the row count inside a bbox. Does
  _not_ touch the brush. Use this when you're probing many candidate
  regions.
* `find_nearby` — `{lon, lat, radius_km?, limit?, columns?, where?}`.
  Returns the nearest rows with great-circle distance. Pre-filters with
  a degree-sized bbox so it's fast even on 75 M rows; requires an ad-hoc
  SQL `where` fragment for extra filtering.
* `density_grid` — `{west, south, east, north, nx?, ny?, top_k?}` —
  bucket rows into an `nx × ny` grid and return per-cell counts. Great
  for finding density outliers inside a specific region.
* `highlight_points` — draw temporary circular markers (with optional
  colour + label) at `{points: [{lon, lat, label?, color?, radius?}]}`.
  Appears on the next `get_map_screenshot`. Pass `{points: []}` to clear.
* `set_basemap_style` — `{style}` accepts a MapLibre style URL or one of
  the built-in keys: `"openfreemap-liberty"` (default),
  `"openfreemap-positron"`, `"openfreemap-bright"`, `"osm-raster"`,
  `"none"`.

**Viewport mechanics.** The GIS embedding chart stores its viewport as
`{x, y, scale}` where `x = lon`, `y = projectLat(lat)`, and the derived
MapLibre zoom is `log2(360 · scale · canvas_width / 1024)`. The geo
tools translate to and from lat/lon + zoom automatically so clients
shouldn't need to think about it. If for some reason you want to set
viewport manually, `set_chart_state` with the GIS chart id still works.

## Architecture

```
Claude Desktop / Cursor / etc.   ─── Streamable HTTP ──▶  /mcp
                                                            │
                                                            │ JSON-RPC forward
                                                            ▼
                                                 WebSocketHandler
                                                            │
                                                            │ ws://…/data/mcp_websocket
                                                            ▼
                                                 Viewer JavaScript
                                                            │
                                                            │ executes
                                                            ▼
                                                 DuckDB / charts / canvas
```

The Python side (`packages/backend/embedding_atlas/mcp_bridge.py`) is a pure
protocol adapter. The tool implementations themselves live in the viewer
(`packages/viewer/src/model_context/model_context.ts`) — a tool added there
becomes available to MCP clients automatically with zero Python changes.

## Caveats

- **Viewer browser tab must be open.** The tools run in the webview — close the tab and
  `tools/call` returns an error. On `tools/list`, a warning logs and the cached
  tool list (if any) is returned.
- **One active viewer per server.** Opening a second tab disconnects the first. The
  last-connected viewer owns the MCP session.
- **Read-only SQL enforced server-side.** The backend DuckDB connection has
  `enable_external_access = false` and `lock_configuration = true`. Attempts to
  mutate data will error out, not succeed silently.
- **MCP needs to be opted in.** The CLI ships it as `--mcp`
  (default off). The native desktop app ships it default **on**, with a
  checkbox on the dataset picker to disable per-launch; once a dataset
  is loaded the port is shown and the full `/mcp` URL can be copied
  directly from the status panel.

## Example prompt

> I just loaded an Overture Places parquet file. Use the geospatial-atlas tools to:
>
> 1. Describe the schema.
> 2. Count rows per `primary_category` and show me the top 10.
> 3. Add a density plot coloured by category.
> 4. Switch the layout to dashboard and take a screenshot.

Claude will chain `get_data_schema` → `run_sql_query` → `add_chart` (density spec)
→ `set_column_style` → `set_layout_type` → `get_full_screenshot`. You'll see each
step happen live in the browser.

## Legacy endpoint

The pre-standard `POST /mcp` endpoint (simple JSON-RPC forwarder, no session, no SSE)
is still available at `/mcp_legacy` for backwards compatibility. New integrations
should use the standard `/mcp` endpoint.
