<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { debounce } from "@embedding-atlas/utils";
  import { coordinator as defaultCoordinator, DuckDBWASMConnector } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import EmbeddingAtlas from "../EmbeddingAtlas.svelte";
  import DatasetSetupView, { type Settings } from "./components/DatasetSetupView.svelte";
  import MessagesView, { appendedMessages, type Message } from "./components/MessagesView.svelte";

  import { IconClose } from "../assets/icons.js";

  import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "../api.js";
  import { computeEmbedding } from "../embedding/index.js";
  import { systemColorScheme } from "../utils/color_scheme.js";
  import { initializeDatabase } from "../utils/database.js";
  import { downloadBuffer } from "../utils/download.js";
  import { exportMosaicSelection, type ExportFormat } from "../utils/mosaic_exporter.js";
  import { getQueryPayload, setQueryPayload } from "../utils/query_payload.js";
  import { importDataTable } from "./import_data.js";

  const coordinator = defaultCoordinator();
  const databaseInitialized = initializeDatabase(coordinator, "wasm", null);

  let stage: "setup" | "ready" | "messages" = $state.raw("setup");
  let messages = $state.raw<Message[]>([]);
  let props = $state<Omit<EmbeddingAtlasProps, "coordinator"> | undefined>(undefined);
  let describe: { column_name: string; column_type: string }[] = $state.raw([]);
  let hashParams = $state.raw<{ data?: string; settings?: any; state?: any }>({});
  let dataLoaded = $state(false);

  function log(text: string, progress?: number, progressText?: string) {
    messages = appendedMessages(messages, { text: text, progress: progress, progressText: progressText });
  }

  function logError(text: string) {
    messages = appendedMessages(messages, { text: text, error: true });
  }

  async function loadHashParams() {
    hashParams = {
      data: await getQueryPayload("data", "text"),
      settings: await getQueryPayload("settings"),
      state: await getQueryPayload("state"),
    };
  }

  function clearHashParams() {
    setQueryPayload("data", undefined, "text");
    setQueryPayload("state", undefined);
    setQueryPayload("settings", undefined);
    hashParams = {};
  }

  // Load existing state from URL if available
  onMount(async () => {
    await loadHashParams();
    if (hashParams.data != undefined && typeof hashParams.data == "string") {
      await loadData([{ url: hashParams.data }]);
    }
  });

  /** Load data from inputs (list of files or urls) */
  async function loadData(inputs: (File | { url: string })[]) {
    stage = "messages";
    dataLoaded = false;
    try {
      log("Initializing database...");
      await databaseInitialized;

      let db = await (coordinator.databaseConnector()! as DuckDBWASMConnector).getDuckDB();
      await importDataTable(inputs, db, coordinator, "dataset", log);

      let describeResult = await coordinator.query(`DESCRIBE TABLE dataset`);
      describe = Array.from(describeResult) as typeof describe;

      // Create the __row_index__ column to use as row id
      await coordinator.exec(`
        CREATE OR REPLACE SEQUENCE __row_index_sequence__ MINVALUE 0 START 0;
        ALTER TABLE dataset ADD COLUMN IF NOT EXISTS __row_index__ INTEGER DEFAULT nextval('__row_index_sequence__');
      `);
    } catch (e: any) {
      stage = "setup";
      logError(e.message);
      return;
    }

    if (inputs.length == 1 && "url" in inputs[0]) {
      setQueryPayload("data", inputs[0].url, "text");
    }

    dataLoaded = true;
    stage = "setup";

    if (hashParams.settings != undefined) {
      await loadSettings(hashParams.settings);
    }
  }

  async function loadSettings(spec: Settings) {
    stage = "messages";

    try {
      let projectionColumns: { x: string; y: string; neighbors?: string } | undefined;
      let neighborsColumn: string | undefined;

      if (spec.embedding != null && "precomputed" in spec.embedding) {
        projectionColumns = { x: spec.embedding.precomputed.x, y: spec.embedding.precomputed.y };
        if (spec.embedding.precomputed.neighbors != undefined) {
          neighborsColumn = spec.embedding.precomputed.neighbors;
        }
      }

      if (spec.embedding != null && "compute" in spec.embedding) {
        let input = spec.embedding.compute.column;
        let type = spec.embedding.compute.type;
        let model = spec.embedding.compute.model;
        let x = input + "_proj_x";
        let y = input + "_proj_y";
        await computeEmbedding({
          coordinator: coordinator,
          table: "dataset",
          idColumn: "__row_index__",
          dataColumn: input,
          type: type,
          xColumn: x,
          yColumn: y,
          model: model,
          callback: (message, progress) => {
            log(`Embedding: ${message}`, progress);
          },
        });
        projectionColumns = { x, y };
      }

      props = {
        data: {
          table: "dataset",
          id: "__row_index__",
          text: spec.text,
          projection: projectionColumns,
          neighbors: neighborsColumn,
        },
        initialState: hashParams.state,
      };
    } catch (e: any) {
      logError(e.message);
      return;
    }

    await setQueryPayload("settings", spec);

    stage = "ready";
  }

  function onStateChange(state: EmbeddingAtlasState) {
    setQueryPayload("state", { ...state, predicate: undefined });
  }

  async function onExportSelection(predicate: string | null, format: ExportFormat) {
    let [bytes, name] = await exportMosaicSelection(coordinator, "dataset", predicate, format);
    downloadBuffer(bytes, name);
  }

  function handleChangeDataset() {
    dataLoaded = false;
    describe = [];
  }
</script>

<div class="fixed left-0 right-0 top-0 bottom-0">
  {#if stage == "ready" && props !== undefined}
    <EmbeddingAtlas
      coordinator={coordinator}
      onStateChange={debounce(onStateChange, 200)}
      onExportSelection={onExportSelection}
      {...props}
    />
  {:else}
    <div
      class="w-full h-full grid place-content-center select-none text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
      class:dark={$systemColorScheme == "dark"}
    >
      {#if stage == "setup"}
        <div class="flex flex-col gap-2 items-center">
          <DatasetSetupView
            columns={describe}
            onLoadData={loadData}
            onConfirm={loadSettings}
            onChangeDataset={handleChangeDataset}
            dataLoaded={dataLoaded}
          />
          {#if !dataLoaded && (hashParams.settings != undefined || hashParams.state != undefined)}
            <div
              class="max-w-4xl text-slate-600 dark:text-slate-300 mt-4 flex flex-col items-start gap-1 border-l-2 pl-2 border-slate-400 dark:border-slate-600"
            >
              <div>A saved view is available in the URL. It will be restored after the data loads.</div>
              <button
                class="flex gap-1 items-center border bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 dark:text-slate-400 rounded-md pl-1 pr-2"
                onclick={clearHashParams}
              >
                <IconClose class="w-4 h-4" />
                Clear Saved View
              </button>
            </div>
          {/if}
        </div>
      {:else if stage == "messages"}
        <MessagesView messages={messages} />
      {/if}
    </div>
  {/if}
</div>
