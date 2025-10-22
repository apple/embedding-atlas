<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Router from "svelte-spa-router";

  import AdhocViewer from "./AdhocViewer.svelte";
  import Home from "./Home.svelte";
  import Test from "./Test.svelte";

  const routes = {
    "/": (window as any)?.EMBEDDING_ATLAS_HOME == "upload" ? AdhocViewer : Home,
    "/test": Test,
    "/upload": AdhocViewer,
  };
</script>

<Router routes={routes} />
