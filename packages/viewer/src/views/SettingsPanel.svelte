<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Coordinator } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import PanelContainer from "./PanelContainer.svelte";
  import PanelSection from "./PanelSection.svelte";

  import { EMBEDDING_ATLAS_VERSION } from "../constants.js";

  interface Props {
    coordinator: Coordinator;
    mcpStatus?: string;
  }

  let { coordinator, mcpStatus }: Props = $props();

  let duckdbVersion = $state<string | null>(null);

  onMount(async () => {
    try {
      let result = await coordinator.query("SELECT version() AS version");
      duckdbVersion = result.get(0)?.version ?? null;
    } catch {
      // ignore
    }
  });
</script>

<PanelContainer title="Status / Settings" class="p-2 flex flex-col gap-2 overflow-hidden">
  {#if mcpStatus}
    <PanelSection title="MCP (Model Context Protocol)">
      <div class="flex flex-none gap-2 select-none items-center">
        {#if mcpStatus == "connecting"}
          <div class="w-3 h-3 rounded-full bg-orange-500 animate-pulse"></div>
          Connecting...
        {:else if mcpStatus == "connected"}
          <div class="w-3 h-3 rounded-full bg-green-500"></div>
          Connected
        {:else if mcpStatus == "closed" || mcpStatus == "error"}
          <div class="w-3 h-3 rounded-full bg-red-500"></div>
          Error or server closed connection
        {/if}
      </div>
    </PanelSection>
  {/if}

  <PanelSection title="Versions">
    <div>Embedding Atlas, v{EMBEDDING_ATLAS_VERSION}</div>
    {#if duckdbVersion}
      <div>DuckDB, {duckdbVersion}</div>
    {/if}
  </PanelSection>
</PanelContainer>
