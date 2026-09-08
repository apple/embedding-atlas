<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Router from "svelte-spa-router";

  import BackendViewer from "./BackendViewer.svelte";
  import FileViewer from "./FileViewer.svelte";
  import TestDataViewer from "./TestDataViewer.svelte";
  import WebGPUBanner from "./components/WebGPUBanner.svelte";

  import { resolveAppConfig, detectHome } from "./app_config.js";

  const config = resolveAppConfig();

  // If no explicit config was set, auto-detect by probing for the backend.
  let routes: any = $state(null);

  async function init() {
    let home = config.home;
    if (typeof window !== "undefined" && (window as any).EMBEDDING_ATLAS_CONFIG == null) {
      home = await detectHome();
    }

    const r: any = {
      "/": home === "file-viewer" ? FileViewer : BackendViewer,
    };

    if (import.meta.env.DEV) {
      r["/test"] = TestDataViewer;
      r["/file"] = FileViewer;
    }

    // Always expose /file in production too (for standalone static deploys)
    if (home === "file-viewer") {
      r["/file"] = FileViewer;
    }

    routes = r;
  }

  init();
</script>

<WebGPUBanner />

{#if routes}
  <Router routes={routes} />
{/if}
