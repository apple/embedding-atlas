<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Coordinator } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import DisclosureButton from "../widgets/DisclosureButton.svelte";
  import ModelNameInput from "./ModelNameInput.svelte";
  import PanelContainer from "./PanelContainer.svelte";
  import PanelSection from "./PanelSection.svelte";
  import ProviderConfigForm from "./ProviderConfigForm.svelte";

  import { IconReset } from "../assets/icons.js";

  import { EMBEDDING_ATLAS_VERSION } from "../constants.js";
  import {
    DEFAULT_MODELS,
    DEFAULT_PROVIDER_CONFIGS,
    defaultModels,
    providerConfigs,
  } from "../inference/model_config_store.js";

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

  function resetProviders() {
    providerConfigs.update((c) => ({
      ...c,
      "transformers.js": DEFAULT_PROVIDER_CONFIGS["transformers.js"],
      openai: DEFAULT_PROVIDER_CONFIGS.openai,
    }));
  }
</script>

{#snippet resetButton(onClick: () => void, tooltip: string)}
  <button
    type="button"
    class="text-slate-400 hover:text-slate-700 dark:text-slate-500 dark:hover:text-slate-300"
    title={tooltip}
    onclick={onClick}
  >
    <IconReset class="w-4 h-4" />
  </button>
{/snippet}

<PanelContainer title="Status / Settings" class="p-2 flex flex-col gap-2 overflow-hidden">
  <PanelSection title="Versions">
    <div>Embedding Atlas, v{EMBEDDING_ATLAS_VERSION}</div>
    {#if duckdbVersion}
      <div>DuckDB, {duckdbVersion}</div>
    {/if}
  </PanelSection>
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

  <PanelSection title="Default Models">
    <!-- Embedding model for projection -->
    <div class="flex items-center justify-between">
      <div class="text-slate-500 dark:text-slate-400">Embedding Model for Projection</div>
      {@render resetButton(
        () => defaultModels.update((d) => ({ ...d, embedding: DEFAULT_MODELS.embedding })),
        "Reset to default",
      )}
    </div>
    <ModelNameInput
      bind:value={() => $defaultModels.embedding, (v) => defaultModels.update((d) => ({ ...d, embedding: v }))}
    />
    <!-- Embedding model for highlight -->
    <div class="flex items-center justify-between mt-2">
      <div class="text-slate-500 dark:text-slate-400">Embedding Model for Highlight</div>
      {@render resetButton(
        () => defaultModels.update((d) => ({ ...d, highlight: DEFAULT_MODELS.highlight })),
        "Reset to default",
      )}
    </div>
    <ModelNameInput
      bind:value={() => $defaultModels.highlight, (v) => defaultModels.update((d) => ({ ...d, highlight: v }))}
      modality="text"
    />
    <p class="text-sm text-slate-400 dark:text-slate-600">Reload page for changes to take effect.</p>
  </PanelSection>

  <PanelSection title="Model Provider Configs">
    {#snippet actions()}
      {@render resetButton(resetProviders, "Reset to defaults")}
    {/snippet}
    <DisclosureButton label="Transformers.js (in-browser)">
      <ProviderConfigForm providerType="transformers.js" />
    </DisclosureButton>
    <DisclosureButton label="OpenAI-compatible">
      <ProviderConfigForm providerType="openai" />
    </DisclosureButton>
  </PanelSection>
</PanelContainer>
