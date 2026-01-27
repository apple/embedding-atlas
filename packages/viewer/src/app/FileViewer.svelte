<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { debounce } from "@embedding-atlas/utils";
  import { coordinator as defaultCoordinator, DuckDBWASMConnector } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import EmbeddingAtlas from "../EmbeddingAtlas.svelte";
  import FileUpload from "./components/FileUpload.svelte";
  import MessagesView from "./components/MessagesView.svelte";
  import SettingsView, { type Settings } from "./components/SettingsView.svelte";
  import URLInput from "./components/URLInput.svelte";

  import { IconClose } from "../assets/icons.js";

  import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "../api.js";
  import { computeEmbedding } from "../embedding/index.js";
  import { systemColorScheme } from "../utils/color_scheme.js";
  import { initializeDatabase } from "../utils/database.js";
  import { downloadBuffer } from "../utils/download.js";
  import { exportMosaicSelection, type ExportFormat } from "../utils/mosaic_exporter.js";
  import { getQueryPayload, setQueryPayload } from "../utils/query_payload.js";
  import { importDataTable } from "./import_data.js";
  import { Logger } from "./logging.js";

  const coordinator = defaultCoordinator();
  const databaseInitialized = initializeDatabase(coordinator, "wasm", null);

  let stage: "load-data" | "columns" | "ready" | "messages" = $state.raw("load-data");
  let logger = new Logger();
  let messages = logger.messages;

  let props = $state<Omit<EmbeddingAtlasProps, "coordinator"> | undefined>(undefined);
  let describe: { column_name: string; column_type: string }[] = $state.raw([]);
  let hashParams = $state.raw<{ data?: string; settings?: any; state?: any }>({});

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
    try {
      logger.info("Initializing database...");
      await databaseInitialized;

      let db = await (coordinator.databaseConnector()! as DuckDBWASMConnector).getDuckDB();
      let conn = await (coordinator.databaseConnector()! as DuckDBWASMConnector).getConnection();
      await importDataTable(inputs, db, conn, "dataset", logger);

      let describeResult = await coordinator.query(`DESCRIBE TABLE dataset`);
      describe = Array.from(describeResult) as typeof describe;

      // Create the __row_index__ column to use as row id
      await coordinator.exec(`
        CREATE OR REPLACE SEQUENCE __row_index_sequence__ MINVALUE 0 START 0;
        ALTER TABLE dataset ADD COLUMN IF NOT EXISTS __row_index__ INTEGER DEFAULT nextval('__row_index_sequence__');
      `);
    } catch (e: unknown) {
      stage = "messages";
      logger.exception(e);
      return;
    }

    if (inputs.length == 1 && "url" in inputs[0]) {
      setQueryPayload("data", inputs[0].url, "text");
    }

    stage = "columns";

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
        let msg = logger.info(`Embedding: Initialize`);
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
            msg.update({ text: `Embedding: ${message}`, progress: progress });
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
    } catch (e: unknown) {
      logger.exception(e);
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
      {#if stage == "load-data"}
        <div class="w-[40rem] flex flex-col gap-2">
          <FileUpload extensions={[".csv", ".parquet", ".json", ".jsonl"]} multiple={true} onUpload={loadData} />
          <div class="w-full text-center text-slate-400 dark:text-slate-500">&mdash; or &mdash;</div>
          <URLInput onConfirm={(url) => loadData([{ url: url }])} />
          {#if hashParams.settings != undefined || hashParams.state != undefined}
            <div
              class="text-slate-600 dark:text-slate-300 mt-4 flex flex-col items-start gap-1 border-l-2 pl-2 border-slate-400 dark:border-slate-600"
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
          <div class="text-slate-400 dark:text-slate-500 mt-4">
            All data remains confined to the browser and is not transmitted elsewhere.
          </div>
        </div>
      {:else if stage == "columns"}
        <SettingsView columns={describe} onConfirm={(settings) => loadSettings(settings)} />
      {:else if stage == "messages"}
        <MessagesView messages={$messages} />
      {/if}
    </div>
  {/if}
</div>
