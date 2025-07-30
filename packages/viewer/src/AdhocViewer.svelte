<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator } from "@uwdata/mosaic-core";
  import { literal } from "@uwdata/mosaic-sql";

  import type { Value as ColumnsPickerValue } from "./ColumnsPicker.svelte";
  import ColumnsPicker from "./ColumnsPicker.svelte";
  import FileUpload from "./FileUpload.svelte";
  import EmbeddingAtlas from "./lib/EmbeddingAtlas.svelte";

  import { computeEmbedding } from "./embedding/index.js";
  import { systemDarkMode } from "./lib/dark_mode_store.js";
  import { initializeDatabase } from "./lib/database_utils.js";
  import { exportMosaicSelection, type ExportFormat } from "./lib/mosaic_exporter.js";
  import { downloadBuffer } from "./lib/utils.js";
  import MessagesView, { appendedMessages, type Message } from "./MessagesView.svelte";

  const coordinator = defaultCoordinator();
  const databaseInitialized = initializeDatabase(coordinator, "wasm", null);

  let stage: "upload" | "columns" | "ready" | "messages" = $state.raw("upload");
  let messages = $state.raw<Message[]>([]);
  let initialState: any | null = $state.raw(null);
  let textColumn: string | null = $state.raw(null);
  let projectionColumns: { x: string; y: string } | null = $state.raw(null);

  let describe: { column_name: string; column_type: string }[] = $state.raw([]);

  function log(text: string, progress?: number) {
    messages = appendedMessages(messages, { text: text, progress: progress });
  }

  function logError(text: string) {
    messages = appendedMessages(messages, { text: text, error: true });
  }

  async function onUploadDataset(file: File) {
    stage = "messages";
    try {
      log("Initializing database...");
      await databaseInitialized;
      log("Loading data...");
      let db = await coordinator.databaseConnector().getDuckDB();
      await db.registerFileBuffer(file.name, new Uint8Array(await file.arrayBuffer()));
      await coordinator.exec(`CREATE TABLE dataset AS SELECT * FROM ${literal(file.name)}`);
      describe = Array.from(await coordinator.query(`DESCRIBE TABLE dataset`));
      await coordinator.exec(`
        ALTER TABLE dataset DROP COLUMN IF EXISTS __row_index__;
        ALTER TABLE dataset ADD COLUMN __row_index__ INTEGER;
        CREATE SEQUENCE __row_index_sequence__;
        UPDATE dataset SET __row_index__ = nextval('__row_index_sequence__');
        DROP SEQUENCE __row_index_sequence__;
      `);
    } catch (e: any) {
      stage = "messages";
      logError(e.message);
      return;
    }
    stage = "columns";
  }

  async function onConfirmColumns(spec: ColumnsPickerValue) {
    stage = "messages";

    try {
      textColumn = spec.text;

      if (spec.embedding != null && "precomputed" in spec.embedding) {
        projectionColumns = { x: spec.embedding.precomputed.x, y: spec.embedding.precomputed.y };
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
    } catch (e: any) {
      logError(e.message);
      return;
    }

    stage = "ready";
  }

  async function onExportSelection(predicate: string | null, format: ExportFormat) {
    let [bytes, name] = await exportMosaicSelection(coordinator, "dataset", predicate, format);
    downloadBuffer(bytes, name);
  }
</script>

<div class="fixed left-0 right-0 top-0 bottom-0">
  {#if stage == "ready"}
    <EmbeddingAtlas
      coordinator={coordinator}
      table="dataset"
      initialState={initialState}
      idColumn="__row_index__"
      textColumn={textColumn}
      projectionColumns={projectionColumns}
      automaticLabels={true}
      onExportSelection={onExportSelection}
    />
  {:else}
    <div
      class="w-full h-full grid place-content-center select-none text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
      class:dark={$systemDarkMode}
    >
      {#if stage == "upload"}
        <FileUpload extensions={[".csv", ".parquet", ".json", ".jsonl"]} onUpload={onUploadDataset} />
        <p class="text-slate-400 dark:text-slate-500 mt-4">
          * All data remains confined to the browser and is not transmitted elsewhere.
        </p>
      {:else if stage == "columns"}
        <ColumnsPicker columns={describe} onConfirm={onConfirmColumns} />
      {:else if stage == "messages"}
        <MessagesView messages={messages} />
      {/if}
    </div>
  {/if}
</div>
