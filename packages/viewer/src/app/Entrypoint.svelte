<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Router from "svelte-spa-router";

  import BackendViewer from "./BackendViewer.svelte";
  import FileViewer from "./FileViewer.svelte";
  import TestDataViewer from "./TestDataViewer.svelte";

  import { resolveAppConfig } from "./app_config.js";

  const config = resolveAppConfig();

  const routes: any = {
    "/": config.home == "file-viewer" ? FileViewer : BackendViewer,
  };

  if (import.meta.env.DEV) {
    routes["/test"] = TestDataViewer;
    routes["/file"] = FileViewer;
  }
</script>

<Router routes={routes} />
