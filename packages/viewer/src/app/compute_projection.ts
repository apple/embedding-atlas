// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import { get } from "svelte/store";

import { createEmbeddingProjector } from "../embedding/index.js";
import { providerConfigs } from "../inference/model_config_store.js";
import { inferProvider } from "../inference/resolve.js";

/**
 * SQL page size — caps the working set extracted from DuckDB at a time, not
 * the embedding batch size (the model layer chunks internally to provider
 * limits). Sized so that on typical OpenAI/transformers settings each page
 * yields a handful of progress ticks: too small and we pay per-page RPC
 * overhead, too large and small datasets only see one or two ticks.
 */
const BATCH_SIZE = 2048;

async function* inputBatches(
  coordinator: Coordinator,
  table: string,
  idColumn: string,
  valueColumn: string,
): AsyncGenerator<any> {
  // Keyset pagination on the id column: a stable cross-page order (so every
  // row appears in exactly one page — no morsel-scan gaps or duplicates) and
  // O(n) total cost instead of OFFSET's repeated scan-from-start. Relies on
  // idColumn being unique, which is already assumed by the WHERE id = t1.id
  // update in setResultColumns.
  let lastId: unknown = undefined;
  while (true) {
    let query = SQL.Query.from(table)
      .select({ id: SQL.column(idColumn), value: SQL.column(valueColumn) })
      .orderby(SQL.column(idColumn))
      .limit(BATCH_SIZE);
    if (lastId !== undefined) {
      query = query.where(SQL.gt(SQL.column(idColumn), SQL.literal(lastId)));
    }
    let data = await coordinator.query(query);
    if (data.numRows === 0) {
      break;
    }
    lastId = data.get(data.numRows - 1).id;
    yield data;
    if (data.numRows < BATCH_SIZE) {
      break;
    }
  }
}

async function setResultColumns(
  coordinator: Coordinator,
  table: string,
  idColumn: string,
  xColumn: string,
  yColumn: string,
  zColumn: string | null,
  allIDs: any[][],
  coordinates: Float32Array,
) {
  let offset = 0;
  let outputDim = zColumn != null ? 3 : 2;

  await coordinator.exec(`
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(xColumn)} DOUBLE DEFAULT 0;
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(yColumn)} DOUBLE DEFAULT 0;
    ${zColumn != null ? `ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(zColumn)} DOUBLE DEFAULT 0;` : ""}
  `);

  for (let ids of allIDs) {
    let coords = coordinates.subarray(offset, offset + ids.length * outputDim);

    await coordinator.exec(`
        WITH t1 AS (
            SELECT
                UNNEST([${ids.map((x) => SQL.literal(x)).join(",")}]) AS id,
                UNNEST([${ids.map((_, i) => SQL.literal(coords[i * outputDim])).join(",")}]) AS x,
                UNNEST([${ids.map((_, i) => SQL.literal(coords[i * outputDim + 1])).join(",")}]) AS y
                ${zColumn != null ? `, UNNEST([${ids.map((_, i) => SQL.literal(coords[i * outputDim + 2])).join(",")}]) AS z` : ""}
        )
        UPDATE ${table}
            SET ${SQL.column(xColumn)} = t1.x, ${SQL.column(yColumn)} = t1.y
                ${zColumn != null ? `, ${SQL.column(zColumn)} = t1.z` : ""}
            FROM t1 WHERE ${SQL.column(idColumn, table)} = t1.id
    `);

    offset += ids.length * outputDim;
  }
}

export async function computeProjection(options: {
  coordinator: Coordinator;
  table: string;
  idColumn: string;
  dataColumn: string;
  xColumn: string;
  yColumn: string;
  /** Column to write a 3rd UMAP dimension to. When set, UMAP computes a 3D projection
   *  (`n_components: 3`) instead of 2D, and the embedding view renders in 3D. */
  zColumn?: string | null;
  type: "text" | "image";
  model: string;
  umapOptions?: { minDist?: number; nNeighbors?: number; gpu?: boolean };
  callback?: (message: string, progress?: number) => void;
}) {
  function progress(message: string, progress?: number) {
    options.callback?.(message, progress);
  }

  progress(`Loading ${options.model}...`);

  // Count up front so the projector can pre-allocate its output buffer in one
  // go — avoids the peak-2x-memory window of the previous accumulate-then-concat
  // shape.
  let countResult = await options.coordinator.query(SQL.Query.from(options.table).select({ count: SQL.count() }));
  let total: number = countResult.get(0).count;

  let config = get(providerConfigs)[inferProvider(options.model)] ?? {};
  let projector = await createEmbeddingProjector({
    model: options.model,
    config,
    modality: options.type,
    total,
  });

  let allIDs: any[][] = [];
  let idsCount = 0;

  for await (const data of inputBatches(options.coordinator, options.table, options.idColumn, options.dataColumn)) {
    progress("Embedding...", (idsCount / total) * 100);

    let ids = Array.from(data.getChild("id"));
    let values = Array.from(data.getChild("value"));
    await projector.batch(values);
    allIDs.push(ids);
    idsCount += ids.length;
  }

  progress("UMAP...");

  let outputDim = options.zColumn != null ? 3 : 2;
  let coordinates = await projector.umap(
    {
      ...options.umapOptions,
      progress: (p: number, stage: string) => {
        progress(`UMAP: ${stage}...`, p * 100);
      },
    },
    outputDim,
  );

  await setResultColumns(
    options.coordinator,
    options.table,
    options.idColumn,
    options.xColumn,
    options.yColumn,
    options.zColumn ?? null,
    allIDs,
    coordinates,
  );

  await projector.destroy();
}
