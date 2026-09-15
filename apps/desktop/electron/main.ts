// Electron main process — Tauri-shell replacement.
//
// Feature parity with the original apps/desktop/src-tauri/src/lib.rs:
// spawns the PyInstaller sidecar, polls for readiness, owns persistent
// per-dataset viewer state, handles drag-drop of datasets anywhere on
// the window, injects the "back to picker" home button on viewer
// pages, and persists the MCP on/off toggle.

import { app, BrowserWindow, dialog, ipcMain, shell, type IpcMainInvokeEvent } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createServer } from "node:net";
import { pathToFileURL } from "node:url";

// ---------- constants ----------

const SUPPORTED_EXTENSIONS = ["parquet", "geoparquet", "csv", "tsv", "json", "jsonl", "arrow", "feather"];

const isDev = !app.isPackaged;
// 30 min — large parquets (tens of GB / 100M+ rows) cold-load well past the
// old 2 min budget. The user can always quit the window if something is
// genuinely stuck; a too-tight timeout forces a restart mid-load.
const SIDECAR_READY_TIMEOUT_MS = 1_800_000;
const SIDECAR_HEALTH_POLL_MS = 300;

// ---------- sidecar state (module-level, single window) ----------

interface RunningSidecar {
  child: ChildProcess;
  intentionalKill: { value: boolean };
}

const state = {
  running: null as RunningSidecar | null,
  currentDataset: null as string | null,
  bootstrapUrl: null as string | null,
  mcpEnabled: true,
  mcpUrl: "",
};

let mainWindow: BrowserWindow | null = null;

// ---------- helpers ----------

function pickFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      if (typeof addr === "object" && addr) {
        const port = addr.port;
        srv.close(() => resolve(port));
      } else {
        reject(new Error("no address from listener"));
      }
    });
  });
}

function sidecarBinaryPath(): string {
  // Packaged: <resources>/sidecar/<bin>. Dev: apps/desktop/resources/sidecar/<bin>.
  const binName = process.platform === "win32" ? "geospatial-atlas-sidecar.exe" : "geospatial-atlas-sidecar";
  if (app.isPackaged) {
    return join(process.resourcesPath, "sidecar", binName);
  }
  return join(__dirname, "..", "..", "resources", "sidecar", binName);
}

function bootstrapUrl(): string {
  if (isDev) {
    return "http://127.0.0.1:1420";
  }
  // Packaged layout (inside app.asar):
  //   /electron/dist/main.js  ← __dirname
  //   /dist/index.html
  // So go up two levels: electron/dist → electron → app.asar, then dist/.
  return pathToFileURL(join(__dirname, "..", "..", "dist", "index.html")).toString();
}

function emit(channel: string, payload: unknown) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(`evt:${channel}`, payload);
  }
}

function isSupportedDataset(path: string): boolean {
  const lower = path.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith("." + ext));
}

// ---------- state file (per-dataset viewer state) ----------

function stateFilePath(): string {
  const dir = app.getPath("userData");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  return join(dir, "viewer-state.json");
}

function loadStateMap(): Record<string, string> {
  try {
    const data = readFileSync(stateFilePath(), "utf8");
    return JSON.parse(data) as Record<string, string>;
  } catch {
    return {};
  }
}

function saveStateMap(map: Record<string, string>) {
  try {
    writeFileSync(stateFilePath(), JSON.stringify(map, null, 2), "utf8");
  } catch {
    // best-effort
  }
}

// ---------- sidecar lifecycle ----------

function killRunning() {
  const rs = state.running;
  state.running = null;
  if (rs) {
    rs.intentionalKill.value = true;
    try {
      rs.child.kill();
    } catch {
      // process may already be dead
    }
  }
  state.mcpUrl = "";
}

function handleSidecarLine(line: string) {
  if (!line) return;
  if (line.startsWith("GSA_PROGRESS ")) {
    try {
      const progress = JSON.parse(line.slice("GSA_PROGRESS ".length));
      emit("sidecar-progress", progress);
      return;
    } catch {
      // fall through to log
    }
  }
  emit("sidecar-log", { line });
}

function attachStreamReader(stream: NodeJS.ReadableStream) {
  let buf = "";
  stream.on("data", (chunk: Buffer | string) => {
    buf += chunk.toString("utf8");
    let nl = buf.indexOf("\n");
    while (nl >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, "");
      buf = buf.slice(nl + 1);
      handleSidecarLine(line);
      nl = buf.indexOf("\n");
    }
  });
  stream.on("end", () => {
    if (buf) handleSidecarLine(buf);
  });
}

async function waitForSidecarReady(url: string): Promise<void> {
  const deadline = Date.now() + SIDECAR_READY_TIMEOUT_MS;
  const health = `${url}/data/metadata.json`;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(health, { signal: AbortSignal.timeout(800) });
      if (res.ok) return;
    } catch {
      // keep polling
    }
    await new Promise((r) => setTimeout(r, SIDECAR_HEALTH_POLL_MS));
  }
  throw new Error("timed out waiting for sidecar to become ready");
}

async function launchSidecar(dataset: string, limit: number, text: string): Promise<void> {
  killRunning();
  state.currentDataset = dataset;

  const port = await pickFreePort();
  const host = "127.0.0.1";
  const url = `http://${host}:${port}`;
  const binPath = sidecarBinaryPath();

  if (!existsSync(binPath)) {
    throw new Error(`sidecar binary not found at ${binPath} — did you run the python-sidecar build?`);
  }

  const mcpOn = state.mcpEnabled;
  const child = spawn(binPath, [dataset, String(limit), text], {
    env: {
      ...process.env,
      GEOSPATIAL_ATLAS_HOST: host,
      GEOSPATIAL_ATLAS_PORT: String(port),
      GEOSPATIAL_ATLAS_PARENT_PID: String(process.pid),
      GEOSPATIAL_ATLAS_MCP: mcpOn ? "1" : "0",
    },
    stdio: ["pipe", "pipe", "pipe"],
    // Windows: detach so the console window doesn't flash.
    windowsHide: true,
  });

  const intentionalKill = { value: false };
  state.running = { child, intentionalKill };

  if (child.stdout) attachStreamReader(child.stdout);
  if (child.stderr) attachStreamReader(child.stderr);

  child.on("exit", () => {
    if (!intentionalKill.value) {
      emit("sidecar-error", { message: "sidecar exited unexpectedly" });
    }
  });
  child.on("error", (err) => {
    emit("sidecar-error", { message: `failed to spawn sidecar: ${err.message}` });
  });

  // Poll for readiness off the main thread.
  void waitForSidecarReady(url).then(
    () => {
      const mcpUrl = mcpOn ? `${url}/mcp` : "";
      state.mcpUrl = mcpUrl;
      emit("sidecar-ready", { url, mcp_url: mcpUrl });
    },
    (err) => {
      emit("sidecar-error", { message: String(err.message ?? err) });
    },
  );
}

function returnHome() {
  killRunning();
  state.currentDataset = null;
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const target = state.bootstrapUrl ?? bootstrapUrl();
  void mainWindow.loadURL(target);
}

// ---------- dropped file handling ----------

async function handleDroppedFile(path: string) {
  if (!isSupportedDataset(path)) {
    emit("sidecar-error", {
      message: "Dropped file is not a supported dataset (.parquet, .csv, .json, .arrow …)",
    });
    return;
  }
  // Match the Tauri flow: navigate back to bootstrap so the progress UI is
  // visible immediately, then launch the new sidecar.
  returnHome();
  try {
    await launchSidecar(path, 0, "");
  } catch (e) {
    emit("sidecar-error", { message: String((e as Error).message ?? e) });
  }
}

// ---------- IPC handlers (Tauri-command equivalents) ----------

function registerIpcHandlers() {
  ipcMain.handle(
    "cmd:launch_sidecar",
    async (_e: IpcMainInvokeEvent, args: { dataset: string; limit: number; text: string }) => {
      await launchSidecar(args.dataset, args.limit, args.text);
    },
  );

  ipcMain.handle("cmd:return_home", () => returnHome());

  ipcMain.handle("cmd:get_current_dataset", () => state.currentDataset);

  ipcMain.handle("cmd:load_viewer_state", () => {
    if (!state.currentDataset) return null;
    const map = loadStateMap();
    return map[state.currentDataset] ?? null;
  });

  ipcMain.handle("cmd:save_viewer_state", (_e: IpcMainInvokeEvent, args: { hash: string }) => {
    if (!state.currentDataset) return;
    const map = loadStateMap();
    const cleaned = (args.hash ?? "").replace(/^#/, "");
    if (cleaned === "") {
      delete map[state.currentDataset];
    } else {
      map[state.currentDataset] = cleaned;
    }
    saveStateMap(map);
  });

  ipcMain.handle("cmd:get_mcp_enabled", () => state.mcpEnabled);
  ipcMain.handle("cmd:set_mcp_enabled", (_e: IpcMainInvokeEvent, args: { enabled: boolean }) => {
    state.mcpEnabled = !!args.enabled;
    return state.mcpEnabled;
  });
  ipcMain.handle("cmd:get_mcp_url", () => state.mcpUrl);

  // File-picker dialog (Tauri plugin-dialog equivalent).
  ipcMain.handle(
    "dialog:open",
    async (
      _e: IpcMainInvokeEvent,
      options: {
        multiple?: boolean;
        directory?: boolean;
        filters?: Array<{ name: string; extensions: string[] }>;
      },
    ) => {
      if (!mainWindow) return null;
      const result = await dialog.showOpenDialog(mainWindow, {
        properties: [
          options.directory ? "openDirectory" : "openFile",
          ...(options.multiple ? (["multiSelections"] as const) : []),
        ],
        filters: options.filters,
      });
      if (result.canceled || result.filePaths.length === 0) return null;
      return options.multiple ? result.filePaths : result.filePaths[0];
    },
  );

  // Called from the renderer when a file is dropped on the window.
  ipcMain.handle("cmd:dropped_file", async (_e: IpcMainInvokeEvent, args: { path: string }) => {
    await handleDroppedFile(args.path);
  });
}

// ---------- CLI / argv bootstrap ----------

function initialDatasetFromArgv(): string | null {
  // Packaged apps get the dataset path as a positional arg (if any).
  // argv[0] is the app binary; argv[1..] may include flags or a file.
  // Require both existsSync AND a supported extension so launching the
  // dev shell with `.` as the entry doesn't get mis-detected as a
  // dataset (and so a future user dragging a non-data file via OS
  // "Open With" gets a clear "unsupported" error instead of a confused
  // sidecar crash).
  const args = app.isPackaged ? process.argv.slice(1) : process.argv.slice(2);
  for (const a of args) {
    if (a.startsWith("-")) continue;
    if (existsSync(a) && isSupportedDataset(a)) return a;
  }
  return process.env["GEOSPATIAL_ATLAS_INITIAL_DATASET"] ?? null;
}

// ---------- window + app lifecycle ----------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "Geospatial Atlas",
    backgroundColor: "#111418",
    webPreferences: {
      preload: join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // WebGPU is on by default in current Electron/Chromium — no flag needed.
    },
  });

  state.bootstrapUrl = bootstrapUrl();
  void mainWindow.loadURL(state.bootstrapUrl);

  // Open external http(s) links in the user's default browser, not in-app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http://127.0.0.1") || url.startsWith("file://")) {
      return { action: "allow" };
    }
    void shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
    killRunning();
  });

  // ---- crash + console observability ----
  // The packaged app silently absorbs renderer console messages and
  // renderer-process deaths, so when WebGPU exhausts the Metal watchdog
  // or the renderer OOMs there is no visible signal at all. These hooks
  // forward everything to the main-process stdout/stderr (which the
  // desktop-electron-real-pan.spec.ts harness captures via Playwright's
  // _electron API). The user can also tail them by launching from a
  // terminal: ``Geospatial\ Atlas.app/Contents/MacOS/Geospatial\ Atlas``.
  const wc = mainWindow.webContents;
  wc.on("console-message", (_e, level, message, line, src) => {
    const tag = ["v", "i", "w", "e"][level] ?? "?";
    process.stdout.write(`[renderer-console][${tag}] ${src || "?"}:${line}: ${message}\n`);
  });
  wc.on("render-process-gone", (_e, details) => {
    const line = `[renderer-gone] reason=${details.reason} exitCode=${details.exitCode}\n`;
    process.stdout.write(line);
    process.stderr.write(line);
  });
  wc.on("unresponsive", () => {
    process.stdout.write(`[renderer-unresponsive]\n`);
  });
  wc.on("responsive", () => {
    process.stdout.write(`[renderer-responsive]\n`);
  });
  wc.on("did-fail-load", (_e, errorCode, errorDescription, url) => {
    process.stdout.write(`[renderer-load-fail] code=${errorCode} desc=${errorDescription} url=${url}\n`);
  });
  wc.on("preload-error", (_e, preloadPath, error) => {
    process.stdout.write(`[preload-error] path=${preloadPath} ${error.message}\n`);
  });
}

// Pin the userData directory to the bundle identifier so per-dataset
// state written by the old Tauri build ({appData}/io.github.do-me.geospatial-atlas/
// viewer-state.json) carries over transparently. Without this override
// Electron would default to the package.json `name` ("geospatial-atlas-desktop")
// and the migration from Tauri would silently drop every saved viewport.
const BUNDLE_ID = "io.github.do-me.geospatial-atlas";
app.setPath("userData", join(app.getPath("appData"), BUNDLE_ID));

// Hardening + perf flags applied before app.whenReady.
// UseOzonePlatform is Linux-specific (Wayland/X11 abstraction). On Windows
// it disables the native GPU backend and drops Chromium into software
// rendering — `supportsDx12: false`, `webgl: disabled_off`, and WebGPU
// falls back to WebGL. Scope the switch to Linux.
if (process.platform === "linux") {
  app.commandLine.appendSwitch("enable-features", "UseOzonePlatform");
}
// Force the discrete GPU on multi-GPU macOS systems (big perf win for
// WebGPU/WebGL scatter rendering on MBPs with integrated + discrete).
app.commandLine.appendSwitch("force-high-performance-gpu");

// Optional: enable Chrome DevTools Protocol over a TCP port so an
// external harness (Playwright via chromium.connectOverCDP, or a
// browser at http://localhost:<port>) can attach. Off by default —
// only opens when ``GEOSPATIAL_ATLAS_DEBUG_PORT`` is set, which the
// e2e/desktop-electron-real-pan.spec.ts harness sets to expose
// renderer-side metrics, network, and JS heap during pan storms.
const debugPort = process.env["GEOSPATIAL_ATLAS_DEBUG_PORT"];
if (debugPort && /^\d+$/.test(debugPort)) {
  app.commandLine.appendSwitch("remote-debugging-port", debugPort);
  app.commandLine.appendSwitch("remote-allow-origins", "*");
  process.stdout.write(`[debug] CDP listening on port ${debugPort}\n`);
}
// Chromium blocklists WebGPU on a range of older NVIDIA and Intel GPUs that
// are nonetheless perfectly capable of running Dawn on D3D12 (e.g. GeForce
// MX150, Intel UHD 620). `--enable-unsafe-webgpu` bypasses that blocklist.
// The "unsafe" name is historical — WebGPU is stable in Chromium; this flag
// now just undoes a conservative hardware blocklist. We ship unsigned and
// users already trust the binary, so this is the right tradeoff for a
// scatter-renderer whose whole value prop is WebGPU.
app.commandLine.appendSwitch("enable-unsafe-webgpu");

// V8 tuning for the 322 M-row scatter:
//
//   --max-old-space-size=16384
//     Each axis ships as a 1.29 GB Uint32Array; toArray() materialises
//     a fresh contiguous copy and the previous render's arrays only
//     drop their refs at the next setProps microtask, so peak heap
//     reaches ~5.8 GB during a full-extent re-fetch — and color-by
//     adds a 322 MB category buffer on top. Default V8 budget (~4 GB)
//     OOMs immediately at this scale. 16 GB gives 2-3 full peaks of
//     headroom before fragmentation forces the next major GC; 12 GB
//     was empirically tight when 9 minutes of accumulated sidebar +
//     mosaic state stacked under the next scatter peak. V8 *budget*
//     is not resident memory: the OS only commits pages we touch.
//
//   --expose-gc
//     Lets EmbeddingViewMosaic.queryResult call ``globalThis.gc()``
//     synchronously between the null-out of the prior render's
//     buffers and the new toArray(). Without that, V8 only runs major
//     GC under pressure — and "pressure" surfaces as an alloc failure
//     (the RangeError we saw mid-session). Forcing GC turns "alloc
//     fails because old buffers haven't been swept yet" into "alloc
//     succeeds against compacted heap".
app.commandLine.appendSwitch("js-flags", "--max-old-space-size=16384 --expose-gc");

// Single-instance: subsequent launches focus the existing window and
// forward the dataset arg.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", (_e, argv) => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
      const dataset = argv.slice(1).find((a) => !a.startsWith("-") && existsSync(a));
      if (dataset) void handleDroppedFile(dataset);
    }
  });

  // App-wide crash hooks. ``child-process-gone`` fires for the GPU
  // process / utility processes, etc.; ``gpu-info-update`` fires when
  // Chromium's GPU info changes (which it does on a Metal device-lost,
  // because the GPU process re-initialises). Forward both so the test
  // harness sees the SAME signals a real "system almost crashed" looks
  // like at the OS level.
  app.on("child-process-gone", (_e, details) => {
    const line = `[child-gone] type=${details.type} reason=${details.reason} exitCode=${details.exitCode} name=${details.name ?? ""}\n`;
    process.stdout.write(line);
    process.stderr.write(line);
  });
  app.on("gpu-info-update", () => {
    process.stdout.write(`[gpu-info-update]\n`);
  });
  // Optional periodic CPU/RAM dump for every Electron process. Off by
  // default; the e2e harness enables it via env var so we can correlate
  // a "system almost crashed" report with which process actually
  // ballooned (renderer? GPU? sidecar started by Electron? duckdb?).
  const metricsMs = Number(process.env["GEOSPATIAL_ATLAS_METRICS_INTERVAL"] ?? "0");
  if (Number.isFinite(metricsMs) && metricsMs > 0) {
    setInterval(() => {
      try {
        const m = app.getAppMetrics();
        const cumCpu = m.reduce((s, p) => s + (p.cpu?.percentCPUUsage ?? 0), 0);
        const cumRssMib = Math.round(m.reduce((s, p) => s + (p.memory?.workingSetSize ?? 0), 0) / 1024);
        const detail = m
          .map(
            (p) =>
              `${p.type}${p.serviceName ? ":" + p.serviceName : ""}=${(p.cpu?.percentCPUUsage ?? 0).toFixed(1)}%/${Math.round((p.memory?.workingSetSize ?? 0) / 1024)}MiB`,
          )
          .join(" ");
        process.stdout.write(`[metrics] cumCpu=${cumCpu.toFixed(1)}% cumRssMiB=${cumRssMib} ${detail}\n`);
      } catch (e) {
        process.stdout.write(`[metrics-err] ${(e as Error).message}\n`);
      }
    }, metricsMs);
  }

  app.whenReady().then(() => {
    registerIpcHandlers();
    createWindow();

    const initial = initialDatasetFromArgv();
    if (initial) {
      // Defer so the window is ready to receive events.
      setTimeout(() => {
        void launchSidecar(initial, 0, "").catch((e) => {
          emit("sidecar-error", { message: String((e as Error).message ?? e) });
        });
      }, 300);
    }

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("window-all-closed", () => {
    killRunning();
    if (process.platform !== "darwin") app.quit();
  });

  app.on("before-quit", () => {
    killRunning();
  });
}
