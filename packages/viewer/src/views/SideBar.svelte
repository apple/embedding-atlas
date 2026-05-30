<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  import SideBarButton from "./SideBarButton.svelte";

  import type { Writable } from "svelte/store";
  import {
    IconDarkMode,
    IconImportExportPanel,
    IconLightMode,
    IconSchemaPanel,
    IconSearchPanel,
    IconSettingsPanel,
  } from "../assets/icons.js";

  interface Props {
    activePanel: Writable<string | undefined>;
    colorScheme: "light" | "dark";
    onChangeColorScheme: (value: "light" | "dark") => void;

    searchPanel: Snippet;
    schemaPanel: Snippet;
    importExportPanel: Snippet;
    settingsPanel: Snippet;
  }

  let {
    colorScheme,
    onChangeColorScheme,
    activePanel,
    settingsPanel,
    searchPanel,
    schemaPanel,
    importExportPanel,
  }: Props = $props();

  function togglePanel(name: string) {
    return () => {
      if ($activePanel == name) {
        $activePanel = undefined;
      } else {
        $activePanel = name;
      }
    };
  }

  let currentPanel = $derived(
    $activePanel == "settings"
      ? settingsPanel
      : $activePanel == "schema"
        ? schemaPanel
        : $activePanel == "search"
          ? searchPanel
          : $activePanel == "import-export"
            ? importExportPanel
            : undefined,
  );
</script>

<div class="flex flex-col gap-2">
  <!-- Search panel -->
  <SideBarButton
    icon={IconSearchPanel}
    title="Search"
    onClick={togglePanel("search")}
    active={$activePanel == "search"}
    open={$activePanel == "search"}
  />

  <SideBarButton
    icon={IconSchemaPanel}
    title="Data schema and column styles"
    onClick={togglePanel("schema")}
    active={$activePanel == "schema"}
    open={$activePanel == "schema"}
  />

  <SideBarButton
    icon={IconImportExportPanel}
    title="Import / Export"
    onClick={togglePanel("import-export")}
    active={$activePanel == "import-export"}
    open={$activePanel == "import-export"}
  />

  <!-- Bottom -->
  <div class="flex-1"></div>

  <SideBarButton
    icon={colorScheme == "dark" ? IconLightMode : IconDarkMode}
    title="Toggle light / dark mode"
    onClick={() => {
      onChangeColorScheme(colorScheme == "light" ? "dark" : "light");
    }}
  />

  <SideBarButton
    icon={IconSettingsPanel}
    title="Status / Settings"
    onClick={togglePanel("settings")}
    active={$activePanel == "settings"}
    open={$activePanel == "settings"}
  />
</div>

{#if currentPanel != undefined}
  {#key currentPanel}
    <svelte:boundary>
      <div class="bg-slate-100 dark:bg-slate-900 rounded-md h-full">
        {@render currentPanel()}
      </div>
      {#snippet failed(error, reset)}
        <button onclick={reset}> An error occurred with this panel. Click to retry. </button>
      {/snippet}
    </svelte:boundary>
  {/key}
{/if}
