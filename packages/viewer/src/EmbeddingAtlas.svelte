<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";
  import { get } from "svelte/store";

  import LayoutOptionsView from "./layouts/LayoutOptionsView.svelte";
  import LayoutView from "./layouts/LayoutView.svelte";
  import AppIcon from "./views/AppIcon.svelte";
  import ImportExportPanel from "./views/ImportExportPanel.svelte";
  import LayoutTabs from "./views/LayoutTabs.svelte";
  import SchemaPanel from "./views/SchemaPanel.svelte";
  import SearchPanel from "./views/SearchPanel.svelte";
  import SelectionBar from "./views/SelectionBar.svelte";
  import SettingsPanel from "./views/SettingsPanel.svelte";
  import SideBar from "./views/SideBar.svelte";
  import BorderlessButton from "./widgets/BorderlessButton.svelte";

  import { IconRedo, IconUndo } from "./assets/icons.js";

  import type { EmbeddingAtlasProps } from "./api.js";
  import { provideModelContext } from "./model_context/model_context.js";
  import { EmbeddingAtlasStore, setStoreContext } from "./stores/embedding_atlas_store.js";

  let {
    coordinator,
    data,
    initialState,
    searcher: specifiedSearcher,
    defaultChartsConfig,
    embeddingViewConfig = null,
    embeddingViewLabels = null,
    chartTheme,
    colorScheme: colorSchemeProp,
    onExportApplication,
    onExportSelection,
    onStateChange,
    onPredicateChange,
    modelContext,
    cache,
  }: EmbeddingAtlasProps = $props();

  // svelte-ignore state_referenced_locally
  let store = new EmbeddingAtlasStore({
    coordinator,
    data,
    initialState,
    searcher: specifiedSearcher,
    defaultChartsConfig,
    embeddingViewConfig,
    embeddingViewLabels,
    chartTheme,
    onExportApplication,
    onExportSelection,
    onStateChange,
    onPredicateChange,
    modelContext,
    cache,
  });

  // Set the store context
  setStoreContext(store);

  let { colorScheme, userColorScheme, chartContext, activePanel, currentLayout, canUndo, canRedo } = store;

  $effect.pre(() => {
    $userColorScheme = colorSchemeProp;
  });

  let container: HTMLDivElement;

  function onWindowKeydown(e: KeyboardEvent) {
    // ESC key to clear selection
    if (e.key == "Escape") {
      store.resetFilter();
      e.preventDefault();
      try {
        let active: any = document.activeElement;
        active?.blur?.();
      } catch (e) {}
    }

    // Undo / redo key
    if ((e.ctrlKey || e.metaKey) && e.key == "z") {
      const target = e.target as HTMLElement;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target.isContentEditable) {
        return;
      }
      e.preventDefault();
      if (e.shiftKey) {
        store.redo();
      } else {
        store.undo();
      }
    }
  }

  $effect.pre(() => {
    if (chartTheme !== get(store.chartTheme)) {
      store.chartTheme.set(chartTheme ?? undefined);
    }
  });

  let mcpStatus = $state.raw<string | undefined>(undefined);

  onMount(() => {
    if (modelContext) {
      provideModelContext(modelContext, store, {
        get container() {
          return container;
        },
      });

      $effect(() => {
        let subs = modelContext.connectionStatus?.subscribe((value) => {
          mcpStatus = value;
        });
        return () => {
          subs?.();
        };
      });
    }
  });
</script>

<div class="embedding-atlas-root" style:width="100%" style:height="100%" bind:this={container}>
  <div
    class="w-full h-full flex flex-col text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
    class:dark={$colorScheme == "dark"}
    style:color-scheme={$colorScheme}
  >
    {#await store.ready then}
      <!-- Toolbar -->
      <div class="m-2 flex flex-row items-center gap-2 flex-wrap">
        <AppIcon colorScheme={$colorScheme} />

        <div class="flex gap-0.5">
          <BorderlessButton icon={IconUndo} title="Undo" onClick={() => store.undo()} disabled={!$canUndo} />
          <BorderlessButton icon={IconRedo} title="Redo" onClick={() => store.redo()} disabled={!$canRedo} />
        </div>

        <LayoutTabs />

        <div class="flex-1"></div>

        <!-- Right side -->
        <SelectionBar />
        {#key $currentLayout}
          <LayoutOptionsView />
        {/key}
      </div>
      <div class="flex-1 overflow-hidden h-full ml-2 mb-2 mr-2 gap-2 flex flex-row">
        <!-- Side bar (navigation bar and panel) -->
        <SideBar
          colorScheme={$colorScheme}
          onChangeColorScheme={(v) => {
            $userColorScheme = v;
          }}
          activePanel={activePanel}
        >
          {#snippet searchPanel()}
            <SearchPanel />
          {/snippet}
          {#snippet schemaPanel()}
            <SchemaPanel />
          {/snippet}
          {#snippet importExportPanel()}
            <ImportExportPanel />
          {/snippet}
          {#snippet settingsPanel()}
            <SettingsPanel mcpStatus={mcpStatus} coordinator={coordinator} />
          {/snippet}
        </SideBar>
        <!-- Main Content -->
        <div class="flex-1 overflow-hidden h-full">
          <LayoutView />
        </div>
      </div>
    {:catch error}
      <div class="flex flex-col gap-2 justify-center items-center w-full h-full">
        <div class="text-red-500">Error initializing Embedding Atlas</div>
      </div>
    {/await}
  </div>
</div>
<svelte:window onkeydown={onWindowKeydown} />
