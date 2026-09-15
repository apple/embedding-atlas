# Geospatial Atlas — desktop app

An Electron wrapper that ships the FastAPI backend and the Svelte viewer as
a single native application for macOS, Linux, and Windows. The shell uses
Chromium, so WebGPU and V8 performance match the user's Chrome browser
exactly — important because the viewer is a WebGPU-accelerated scatter
renderer that is noticeably slower under WebKit-based webviews.

## Architecture

```
┌──────────────────────── .app / installer ──────────────────────┐
│ (macOS)  Contents/MacOS/Geospatial Atlas  ← Electron main        │
│ (macOS)  Contents/Resources/app.asar      ← main.js + preload.js │
│                                              + built Svelte UI   │
│ (macOS)  Contents/Resources/sidecar/      ← PyInstaller --onedir │
│             geospatial-atlas-sidecar      ← real Python binary   │
│             _internal/                    ← duckdb, pyarrow, etc │
│               embedding_atlas/static/     ← prebuilt Svelte view │
└──────────────────────────────────────────────────────────────────┘
```

Lifecycle:

1. Electron main launches the `BrowserWindow` and loads the Svelte
   bootstrap UI (`src/App.svelte`) via the preload bridge.
2. User picks a dataset (file picker, drag-drop, or argv /
   `GEOSPATIAL_ATLAS_INITIAL_DATASET`).
3. Main process picks a free TCP port, spawns the PyInstaller sidecar
   with `GEOSPATIAL_ATLAS_HOST`, `GEOSPATIAL_ATLAS_PORT`, and
   `GEOSPATIAL_ATLAS_PARENT_PID` in the environment.
4. Sidecar (PyInstaller `--onedir`) loads the dataset, auto-detects GIS
   columns, starts FastAPI + uvicorn.
5. Main polls `/data/metadata.json`, emits `sidecar-ready`; the Svelte UI
   redirects the webview to `http://127.0.0.1:<PORT>`.
6. On quit the main process kills the sidecar. Belt-and-suspenders: the
   sidecar also runs three parallel watchdogs (stdin-EOF, `kqueue
NOTE_EXIT`, and a getppid poll).

## Prerequisites

- macOS 11+, Linux (Ubuntu 22.04+ equivalent), or Windows 10+
- Node 22+
- `uv` (for the Python backend venv)
- Rust toolchain (`rustup`) with `wasm32-unknown-unknown` — needed for
  the viewer's density-clustering + UMAP WASM crates, **not** for the
  shell

## Build from source

```bash
cd apps/desktop

# 1. Install JS deps
npm install

# 2. Build the Svelte viewer for the backend (required — provides
#    embedding_atlas/static bundled into the sidecar)
cd ../../packages/backend && ./build.sh && cd ../../apps/desktop

# 3. Build the PyInstaller sidecar for the host arch
npm run build:sidecar

# 4. Build the Electron app + platform installer
npm run dist              # auto-detect host platform
# or force a specific target:
npm run dist:mac          # .dmg
npm run dist:linux        # .deb + .rpm
npm run dist:win          # .msi + NSIS .exe
```

Artifacts land in `apps/desktop/release/`.

## Running

**From Finder / file manager:** double-click the installer, then launch
the installed app. The picker opens; choose any Parquet/CSV/GeoParquet
file with `lon`/`lat` columns or a WKB `geometry` column.

**From the command line with a dataset pre-loaded:**

```bash
# macOS
open -a "Geospatial Atlas" --args /path/to/your.parquet

# or invoke the binary directly (works on Linux/Windows too)
"release/mac-arm64/Geospatial Atlas.app/Contents/MacOS/Geospatial Atlas" /path/to/your.parquet
```

A sample test dataset is bundled in the repo at
`e2e/.data/dataset_test.parquet` (333 843 European school records).

## Dev loop (hot reload)

```bash
# Make sure the sidecar binary exists at least once
npm run build:sidecar

# Live-reload the Svelte UI + auto-rebuild the main process on change
npm run dev
```

The `dev` script runs Vite + Electron in parallel via `concurrently`;
Electron loads `http://127.0.0.1:1420` where Vite serves the renderer
with HMR.

## Signing / notarization

Current builds ship **unsigned**. macOS Gatekeeper shows "Geospatial
Atlas.app is damaged" on first launch; the fix is:

```bash
xattr -cr "/Applications/Geospatial Atlas.app"
```

or System Settings → Privacy & Security → Open Anyway.

For production distribution, wire the following secrets and set
`identity` in `electron-builder.yml`:

- `CSC_LINK` + `CSC_KEY_PASSWORD` — Developer ID certificate (macOS)
- `APPLE_ID` + `APPLE_APP_SPECIFIC_PASSWORD` + `APPLE_TEAM_ID` — notarization
- `WINDOWS_CERTIFICATE_FILE` + `WINDOWS_CERTIFICATE_PASSWORD` — EV codesign

`entitlements.mac.plist` already contains the flags required for the
bundled Python interpreter under hardened runtime:

- `com.apple.security.cs.allow-jit`
- `com.apple.security.cs.allow-unsigned-executable-memory`
- `com.apple.security.cs.disable-library-validation`

## Known limitations

- First launch takes ~8–12 s while PyInstaller unpacks native libs and
  Python cold-imports pyarrow + DuckDB. Subsequent launches are ~2–3 s.
- macOS Intel (`x86_64`) pre-built bundles were dropped starting v0.0.2
  because `macos-13` GitHub runners sit in queue for hours. Intel-Mac
  users should use the `uv run geospatial-atlas ...` CLI path.
- Embedding projection (UMAP) is _not_ bundled — this build is a GIS
  viewer. Install the Python CLI (`uv tool install embedding-atlas`) if
  you need that pipeline.
