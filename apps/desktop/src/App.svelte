<script lang="ts">
  import { onMount } from "svelte";
  import { invoke, listen, open } from "./bridge";

  type Status =
    | { kind: "idle" }
    | { kind: "loading"; dataset: string }
    | { kind: "ready"; url: string; dataset: string }
    | { kind: "error"; message: string };

  type WebGPU = { state: "probing" } | { state: "ok"; adapter: string } | { state: "missing"; reason: string };

  type Progress = { stage: string; percent: number; detail: string };

  let status: Status = $state({ kind: "idle" });
  let log: string = $state("");
  let webgpu: WebGPU = $state({ state: "probing" });
  let progress: Progress | null = $state(null);
  let topN: string = $state("");
  let textCol: string = $state("");
  let mcpEnabled: boolean = $state(true);
  let mcpUrl: string = $state("");
  let mcpCopied: boolean = $state(false);

  const STAGE_LABEL: Record<string, string> = {
    analyze: "Opening file",
    spatial: "Loading GIS extension",
    load: "Loading data",
    bounds: "Computing bounds",
    ready: "Ready",
  };

  async function checkWebGPU() {
    const gpu = (navigator as any).gpu;
    if (gpu == null || gpu.requestAdapter == null) {
      webgpu = { state: "missing", reason: "navigator.gpu unavailable" };
      return;
    }
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter == null) {
        webgpu = { state: "missing", reason: "no suitable GPU adapter" };
        return;
      }
      const info: any = (await adapter.requestAdapterInfo?.()) ?? {};
      webgpu = {
        state: "ok",
        adapter: info.description || info.device || info.vendor || "",
      };
    } catch (e) {
      webgpu = { state: "missing", reason: String(e) };
    }
  }

  onMount(() => {
    checkWebGPU();

    // Load the persisted MCP preference (default: enabled).
    invoke<boolean>("get_mcp_enabled")
      .then((v) => {
        mcpEnabled = v;
      })
      .catch(() => {});

    const unlistenReady = listen<{ url: string; mcp_url: string }>("sidecar-ready", async (event) => {
      const prev = status;
      const dataset = prev.kind === "loading" ? prev.dataset : "";
      const savedHash = await invoke<string | null>("load_viewer_state").catch(() => null);
      const target = event.payload.url + (savedHash ? `#${savedHash}` : "");
      mcpUrl = event.payload.mcp_url ?? "";
      status = { kind: "ready", url: target, dataset };
      window.location.replace(target);
    });

    const unlistenError = listen<{ message: string }>("sidecar-error", (event) => {
      status = { kind: "error", message: event.payload.message };
    });

    const unlistenLog = listen<{ line: string }>("sidecar-log", (event) => {
      log = (log + event.payload.line + "\n").slice(-4000);
    });

    const unlistenProgress = listen<Progress>("sidecar-progress", (event) => {
      progress = event.payload;
    });

    return () => {
      unlistenReady.then((f) => f());
      unlistenError.then((f) => f());
      unlistenLog.then((f) => f());
      unlistenProgress.then((f) => f());
    };
  });

  async function pickDataset() {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: "Tabular / Geospatial",
          extensions: ["parquet", "geoparquet", "csv", "tsv", "json", "jsonl", "arrow", "feather"],
        },
      ],
    });
    if (!selected || typeof selected !== "string") return;
    status = { kind: "loading", dataset: selected };
    log = "";
    progress = null;
    const parsed = Number.parseInt(topN, 10);
    const limit = Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
    const text = textCol.trim();
    try {
      await invoke("launch_sidecar", { dataset: selected, limit, text });
    } catch (e) {
      status = { kind: "error", message: String(e) };
    }
  }
</script>

<main>
  <header>
    <h1>Geospatial Atlas</h1>
    <p class="tagline">WebGPU-enabled viewer for geospatial &amp; embedding data</p>
  </header>

  <div class="gpu-row">
    {#if webgpu.state === "probing"}
      <span class="gpu probing">Checking GPU…</span>
    {:else if webgpu.state === "ok"}
      <span class="gpu ok">✓ WebGPU{webgpu.adapter ? ` · ${webgpu.adapter}` : ""}</span>
    {:else}
      <span class="gpu warn">⚠ WebGPU unavailable — {webgpu.reason}. Rendering will fall back to WebGL.</span>
    {/if}
  </div>

  {#if status.kind === "idle"}
    <section class="cta">
      <div class="topn-row">
        <label class="topn-label" for="topn">Row limit (optional)</label>
        <input
          id="topn"
          class="topn-input"
          type="number"
          min="1"
          step="1"
          inputmode="numeric"
          placeholder="all rows"
          bind:value={topN}
        />
      </div>
      <div class="topn-row">
        <label class="topn-label" for="text-col">Text column (optional)</label>
        <input
          id="text-col"
          class="topn-input topn-input--wide"
          type="text"
          placeholder="e.g. name, description"
          bind:value={textCol}
        />
      </div>
      <div class="topn-row mcp-row">
        <label class="topn-label" for="mcp-toggle"> Expose MCP (Model Context Protocol) endpoint </label>
        <input
          id="mcp-toggle"
          type="checkbox"
          bind:checked={mcpEnabled}
          onchange={() => {
            invoke("set_mcp_enabled", { enabled: mcpEnabled }).catch(() => {});
          }}
        />
      </div>
      <p class="hint mcp-hint">
        When enabled, Claude Desktop / Cursor / Continue can connect to
        <code>http://127.0.0.1:&lt;port&gt;/mcp</code>. The port is picked fresh each launch; it will be shown below
        once the viewer is up.
      </p>
      <button onclick={pickDataset}>Open dataset…</button>
      <p class="hint">
        Choose a Parquet, GeoParquet, CSV, or Arrow file containing
        <code>lon</code>/<code>lat</code> (or a WKB/native <code>geometry</code>) column.
        <br />
        Set a <em>row limit</em> for a quick glimpse of a large file — DuckDB reads only the first&nbsp;N rows
        (SQL&nbsp;<code>LIMIT</code>).
        <br />
        Set a <em>text column</em> (mirrors <code>--text</code>) to pick which column feeds tooltips and search.
        <br />
        You can also drag &amp; drop a file anywhere on the window. The view state is saved per dataset and restored on next
        open.
      </p>
    </section>
  {:else if status.kind === "loading"}
    <section class="status">
      <p class="filename">{status.dataset}</p>
      {#if progress}
        <div class="progress-wrap">
          <div class="progress-row">
            <span class="progress-stage">{STAGE_LABEL[progress.stage] ?? progress.stage}</span>
            <span class="progress-pct">{Math.max(0, Math.min(100, progress.percent)).toFixed(0)}%</span>
          </div>
          <div class="progress-bar">
            <div
              class="progress-bar-fill"
              style:width="{Math.max(2, Math.min(100, progress.percent))}%"
              class:indeterminate={progress.percent < 0}
            ></div>
          </div>
          {#if progress.detail}
            <p class="progress-detail">{progress.detail}</p>
          {/if}
        </div>
      {:else}
        <div class="spinner"></div>
        <p>Starting server…</p>
      {/if}
      {#if log}
        <details class="log-details">
          <summary>Details</summary>
          <pre class="log">{log}</pre>
        </details>
      {/if}
    </section>
  {:else if status.kind === "error"}
    <section class="status">
      <p class="error">Server failed to start.</p>
      <pre class="log">{status.message}</pre>
      {#if log}<pre class="log">{log}</pre>{/if}
      <button onclick={pickDataset}>Try another dataset</button>
    </section>
  {:else if status.kind === "ready"}
    <section class="status">
      <p>Loading viewer at {status.url}…</p>
      {#if mcpUrl}
        <div class="mcp-panel">
          <span class="mcp-label">MCP endpoint</span>
          <code class="mcp-url">{mcpUrl}</code>
          <button
            class="mcp-copy"
            onclick={async () => {
              try {
                await navigator.clipboard.writeText(mcpUrl);
                mcpCopied = true;
                setTimeout(() => (mcpCopied = false), 1500);
              } catch {}
            }}
          >
            {mcpCopied ? "Copied" : "Copy"}
          </button>
        </div>
      {/if}
    </section>
  {/if}
</main>

<style>
  :global(html, body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: #111418;
    color: #f0f0f0;
    height: 100%;
  }
  :global(#app) {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
  }
  main {
    max-width: 640px;
    padding: 2rem;
    text-align: center;
  }
  header h1 {
    margin: 0 0 0.25rem;
    font-size: 1.75rem;
  }
  .tagline {
    margin: 0 0 1rem;
    color: #a0a4ab;
  }
  .gpu-row {
    margin-bottom: 1.5rem;
    min-height: 1.5rem;
  }
  .gpu {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    line-height: 1.2;
  }
  .gpu.probing {
    background: #1f232a;
    color: #a0a4ab;
  }
  .gpu.ok {
    background: #064e3b;
    color: #86efac;
  }
  .gpu.warn {
    background: #422006;
    color: #fde68a;
    max-width: 560px;
    text-align: left;
    white-space: normal;
  }
  button {
    background: #2563eb;
    border: none;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
  }
  button:hover {
    background: #1d4ed8;
  }
  .hint {
    color: #a0a4ab;
    font-size: 0.85rem;
    margin-top: 1rem;
    line-height: 1.5;
  }
  code {
    background: #1f232a;
    padding: 1px 5px;
    border-radius: 3px;
  }
  .filename {
    color: #a0a4ab;
    font-size: 0.8rem;
    word-break: break-all;
  }
  .error {
    color: #f87171;
  }
  .log {
    background: #0b0d10;
    color: #c6cad1;
    padding: 0.75rem;
    text-align: left;
    font-size: 0.72rem;
    max-height: 260px;
    overflow: auto;
    border-radius: 6px;
    white-space: pre-wrap;
  }
  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #2a2f38;
    border-top-color: #2563eb;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
    margin: 0 auto 1rem;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .progress-wrap {
    width: 100%;
    max-width: 480px;
    margin: 1.5rem auto 0;
  }
  .progress-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    margin-bottom: 0.35rem;
    color: #d0d4db;
  }
  .progress-stage {
    font-weight: 500;
  }
  .progress-pct {
    color: #a0a4ab;
    font-variant-numeric: tabular-nums;
  }
  .progress-bar {
    height: 8px;
    background: #1f232a;
    border-radius: 999px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    border-radius: 999px;
    transition: width 0.25s ease-out;
  }
  .progress-bar-fill.indeterminate {
    width: 40% !important;
    animation: indeterminate 1.2s ease-in-out infinite;
  }
  @keyframes indeterminate {
    0% {
      transform: translateX(-120%);
    }
    100% {
      transform: translateX(320%);
    }
  }
  .progress-detail {
    font-size: 0.75rem;
    color: #a0a4ab;
    margin: 0.5rem 0 0;
    min-height: 1rem;
  }
  .log-details {
    margin-top: 1rem;
    font-size: 0.72rem;
    color: #a0a4ab;
    text-align: left;
    max-width: 480px;
    margin-inline: auto;
  }
  .log-details summary {
    cursor: pointer;
    user-select: none;
  }
  .topn-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  .topn-label {
    color: #a0a4ab;
    font-size: 0.85rem;
  }
  .topn-input {
    width: 110px;
    padding: 0.35rem 0.6rem;
    border-radius: 6px;
    border: 1px solid rgba(100, 116, 139, 0.35);
    background: #0b0d10;
    color: inherit;
    font: inherit;
    text-align: right;
  }
  .topn-input--wide {
    width: 220px;
    text-align: left;
  }
  .mcp-row {
    gap: 0.75rem;
  }
  .mcp-hint {
    margin-top: -0.5rem;
    margin-bottom: 1rem;
    color: #9097a3;
  }
  .mcp-panel {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.75rem;
    padding: 0.4rem 0.7rem;
    border-radius: 6px;
    background: #1a1e25;
    border: 1px solid rgba(100, 116, 139, 0.25);
    font-size: 0.8rem;
  }
  .mcp-label {
    color: #a0a4ab;
  }
  .mcp-url {
    color: #60a5fa;
    font-family: ui-monospace, SFMono-Regular, monospace;
    user-select: all;
  }
  .mcp-copy {
    margin-left: auto;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    border: 1px solid rgba(100, 116, 139, 0.35);
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 0.72rem;
  }
  .mcp-copy:hover {
    background: rgba(100, 116, 139, 0.2);
  }
  .topn-input::placeholder {
    color: #565d6a;
  }
</style>
