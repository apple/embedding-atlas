<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator, Selection, wasmConnector } from "@uwdata/mosaic-core";
  import { column, isBetween, loadParquet } from "@uwdata/mosaic-sql";
  import { onMount } from "svelte";

  import { createCustomCellClass, type CustomCellsConfig } from "../lib/api/custom-cells.js";
  import {
    createAdditionalHeaderContentClass,
    type AdditionalHeaderContentsConfig,
  } from "../lib/api/custom-headers.js";
  import type { Theme } from "../lib/api/style.js";
  import type { ColumnConfigChangeCallback, ColumnConfigs } from "../lib/context/config.svelte.js";
  import Table from "../lib/Table.svelte";
  import AdditionalHeaderContentExample from "./AdditionalHeaderContentExample.svelte";
  import CustomCellExample from "./CustomCellExample.svelte";

  let ready = $state(false);

  const indexColumn = "index";
  const dotColumn = "a.b.c";
  const bigIntColumn = "bigintcol";

  onMount(async () => {
    const wasm = await wasmConnector();
    coordinator().databaseConnector(wasm);
    let rootURL = `${window.location.protocol}//${window.location.host}`;
    let parquetURL = `https://huggingface.co/datasets/polygraf-ai/ACL-abstracts/resolve/main/data/train-00000-of-00001.parquet`;
    await coordinator().exec(loadParquet("data_table", parquetURL, {}));
    await coordinator().exec(`
      ALTER TABLE data_table ADD COLUMN ${indexColumn} INTEGER;
      CREATE SEQUENCE __row_index_sequence__;
      UPDATE data_table SET ${indexColumn} = nextval('__row_index_sequence__');
      DROP SEQUENCE __row_index_sequence__;
    `);
    await coordinator().exec(`
      ALTER TABLE data_table ADD COLUMN "${dotColumn}" INTEGER;
      UPDATE data_table SET "${dotColumn}" = 0;
    `);
    await coordinator().exec(`
      ALTER TABLE data_table ADD COLUMN "${bigIntColumn}" BIGINT;
      UPDATE data_table SET "${bigIntColumn}" = 9223372036854775807;
    `);
    ready = true;
  });

  let columns = [
    indexColumn,
    bigIntColumn,
    "entry_type",
    "title",
    "abstract",
    "citation_key",
    "editor",
    "month",
    "year",
    "address",
    "publisher",
    "url",
    "author",
    "booktitle",
    "pages",
    "journal",
    "volume",
    "doi",
    dotColumn,
  ];
  let columnConfigs: ColumnConfigs = {
    url: { width: 200 },
    numcitedby: { title: "cites" },
  };
  let onColumnConfigsChange: ColumnConfigChangeCallback = (column: string, newConfigs: ColumnConfigs) => {
    console.log("config change:", column, newConfigs);
  };

  let filterBy = Selection.single();
  let scrollTo: number | null = $state(null);

  let colorScheme: "light" | "dark" = $state("light");
  let theme: Theme = $state({});
  let customCellsConfig: CustomCellsConfig = $state({});
  let additionalHeaderContentsConfig: AdditionalHeaderContentsConfig = $state({});
  let showRowNumber: boolean = $state(true);
  let headerHeight: number | null = $state(null);
  let highlightedRows: number[] | null = $state([]);
</script>

<div class="app">
  <div class="controls">
    <button
      onclick={() => {
        filterBy.update({
          source: null,
          predicate: isBetween(column(indexColumn), [20, 30]),
          value: [0, 20],
        });
      }}
    >
      Filter
    </button>
    <button
      onclick={() => {
        scrollTo = 52;
      }}
    >
      Scroll To
    </button>
    <button
      onclick={() => {
        colorScheme = "dark";
      }}
    >
      Color Scheme
    </button>
    <button
      onclick={() => {
        showRowNumber = false;
      }}
    >
      Hide Row Number
    </button>
    <button
      onclick={() => {
        theme = {
          light: {
            primaryTextColor: "steelblue",
            secondaryTextColor: "steelblue",
            tertiaryTextColor: "lightblue",
            fontFamily: "serif",
            fontSize: "12px",
            primaryBackgroundColor: "lightblue",
            secondaryBackgroundColor: "lightcyan",
            hoverBackgroundColor: "lightcyan",
            headerFontFamily: "monospace",
            headerFontSize: "14px",
            cellFontFamily: "serif",
            cellFontSize: "12px",
            scrollbarBackgroundColor: "lightblue",
            scrollbarPillColor: "steelblue",
            scrollbarLabelBackgroundColor: "lightcyan",
            outlineColor: "pink",
          },
        };
      }}
    >
      Theme
    </button>
    <button
      onclick={() => {
        customCellsConfig["entry_type"] = createCustomCellClass(CustomCellExample);
      }}
    >
      Custom Cell
    </button>
    <button
      onclick={() => {
        additionalHeaderContentsConfig["entry_type"] =
          createAdditionalHeaderContentClass(AdditionalHeaderContentExample);
      }}
    >
      Additional Header Content
    </button>
    <button
      onclick={() => {
        headerHeight = 32;
      }}
    >
      Header Height
    </button>
    <button
      onclick={() => {
        highlightedRows = [3, 34];
      }}
    >
      Highlight Rows
    </button>
  </div>
  <div class="client">
    {#if ready}
      <Table
        table="data_table"
        rowKey={indexColumn}
        columns={columns}
        columnConfigs={columnConfigs}
        onColumnConfigsChange={onColumnConfigsChange}
        filter={filterBy}
        scrollTo={scrollTo}
        colorScheme={colorScheme}
        theme={theme}
        lineHeight={20}
        numLines={1}
        customCells={customCellsConfig}
        additionalHeaderContents={additionalHeaderContentsConfig}
        headerHeight={headerHeight}
        onRowClick={(rowId) => {
          console.log("clicked row:", rowId);
        }}
        highlightedRows={highlightedRows}
        highlightHoveredRow={true}
      />
    {/if}
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
  }

  .app {
    width: 100vw;
    height: 100vh;
    position: relative;
    background-color: white;
  }

  @media (prefers-color-scheme: dark) {
    .app {
      background-color: black;
    }
  }

  .client {
    width: 100%;
    height: calc(100% - 48px);
  }

  .controls {
    box-sizing: border-box;
    height: 48px;
    padding: 12px;
  }
</style>
