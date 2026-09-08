<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { debounce } from "@embedding-atlas/utils";
  import { coordinator as defaultCoordinator } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import EmbeddingAtlas from "../EmbeddingAtlas.svelte";
  import MessagesView from "./components/MessagesView.svelte";

  import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "../api.js";
  import { systemColorScheme } from "../utils/color_scheme.js";
  import type { ExportFormat } from "../utils/mosaic_exporter.js";
  import { getQueryPayload, setQueryPayload } from "../utils/query_payload.js";
  import type { DataSource } from "./data_source.js";

  const coordinator = defaultCoordinator();
  // Disable Mosaic's LRU query cache. Defaults are max=1000 entries, ttl=3 h,
  // which pins every Arrow Table result for hours — at 322 M each scatter
  // result is ~2.5 GB of Arrow Vector chunks, so a few filter changes
  // exhaust V8's ArrayBuffer pool and the next ``toArray`` OOMs (heap
  // snapshot showed 4093 small Vector buffers totalling 8+ GiB held by
  // ``QueryManager.clientCache``). Histogram/count queries are tiny —
  // letting them re-fetch on filter change is cheap; pinning a 2.5 GB
  // scatter is not.
  coordinator.manager.cache(false);

  interface Props {
    dataSource: DataSource;
  }

  let { dataSource }: Props = $props();

  let ready = $state(false);
  let error = $state(false);
  let status = $state("Loading...");
  let initialState: any | null = $state.raw(null);
  let config: Partial<EmbeddingAtlasProps> | null = $state.raw(null);

  onMount(async () => {
    try {
      initialState = await getQueryPayload("state");
      status = "Initializing database...";
      config = await dataSource.initializeCoordinator(coordinator, "dataset", (s) => {
        status = s;
      });
      ready = true;
    } catch (e: any) {
      error = true;
      status = e.message;
      return;
    }
  });

  async function onExportSelection(predicate: string | null, format: ExportFormat) {
    if (dataSource.downloadSelection) {
      await dataSource.downloadSelection(predicate, format);
    }
  }

  async function onDownloadArchive() {
    if (dataSource.downloadArchive) {
      await dataSource.downloadArchive();
    }
  }

  function onStateChange(state: EmbeddingAtlasState) {
    setQueryPayload("state", { ...state, predicate: undefined });
  }
</script>

<div class="fixed left-0 right-0 top-0 bottom-0">
  {#if ready && config != null}
    <EmbeddingAtlas
      coordinator={coordinator}
      data={{
        ...(config.data ?? { id: "id" }),
        // table is loaded with the name "dataset" above.
        table: "dataset",
      }}
      additionalTables={config.additionalTables}
      embeddingViewConfig={config.embeddingViewConfig}
      embeddingViewLabels={config.embeddingViewLabels}
      defaultChartsConfig={config.defaultChartsConfig}
      chartTheme={config.chartTheme}
      stylesheet={config.stylesheet}
      initialState={initialState ?? config.initialState}
      modelContext={config.modelContext}
      onExportApplication={dataSource.downloadArchive ? onDownloadArchive : null}
      onExportSelection={dataSource.downloadSelection ? onExportSelection : null}
      onStateChange={debounce(onStateChange, 200)}
      cache={dataSource.cache}
    />
  {:else}
    <div
      class="w-full h-full grid place-content-center select-none text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
      class:dark={$systemColorScheme == "dark"}
    >
      <MessagesView messages={[{ text: status, error: error }]} />
    </div>
  {/if}
</div>
