// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { WorkerRPC } from "./worker_helper.js";

let _rpc: Promise<(name: string, ...args: any[]) => Promise<any>> | null = null;
function connect() {
  if (_rpc == null) {
    let worker = new Worker(new URL("./embedding.worker.js", import.meta.url), { type: "module" });
    _rpc = WorkerRPC.connect(worker);
  }
  return _rpc;
}

async function* inputBatches(
  coordinator: Coordinator,
  table: string,
  idColumn: string,
  valueColumn: string,
  batchSize: number,
): AsyncGenerator<any> {
  let r = await coordinator.query(SQL.Query.from(table).select({ count: SQL.count() }));
  let count = r.get(0).count;
  let start = 0;
  while (start < count) {
    let range0 = start;
    let range1 = start + batchSize;
    if (range1 > count) {
      range1 = count;
    }
    let data = await coordinator.query(
      SQL.Query.from(table)
        .select({ id: SQL.column(idColumn), value: SQL.column(valueColumn) })
        .offset(start)
        .limit(range1 - range0),
    );
    yield { total: count, data: data };
    start = range1;
  }
}

async function setResultColumns(
  coordinator: Coordinator,
  table: string,
  idColumn: string,
  xColumn: string,
  yColumn: string,
  allIDs: any[][],
  coordinates: Float32Array,
) {
  let offset = 0;

  await coordinator.exec(`
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(xColumn)} DOUBLE DEFAULT 0;
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(yColumn)} DOUBLE DEFAULT 0;
  `);

  for (let ids of allIDs) {
    let xy = coordinates.subarray(offset, offset + ids.length * 2);

    await coordinator.exec(`
        WITH t1 AS (
            SELECT
                UNNEST([${ids.map((x) => SQL.literal(x)).join(",")}]) AS id,
                UNNEST([${ids.map((_, i) => xy[i * 2]).join(",")}]) AS x,
                UNNEST([${ids.map((_, i) => xy[i * 2 + 1]).join(",")}]) AS y
        )
        UPDATE ${table}
            SET ${SQL.column(xColumn)} = t1.x, ${SQL.column(yColumn)} = t1.y
            FROM t1 WHERE ${SQL.column(idColumn, table)} = t1.id
    `);

    offset += ids.length * 2;
  }
}

export async function computeEmbedding(options: {
  coordinator: Coordinator;
  table: string;
  idColumn: string;
  dataColumn: string;
  xColumn: string;
  yColumn: string;
  type: "text" | "image";
  model: string;
  callback?: (message: string, progress?: number) => void;
}) {
  function progress(message: string, progress?: number) {
    options.callback?.(message, progress);
  }

  progress(`Loading ${options.model}...`);

  let rpc = await connect();

  let instance = await rpc("embedding.new", { type: options.type, model: options.model });

  let allIDs: any[][] = [];
  let idsCount = 0;

  for await (const { total, data } of inputBatches(
    options.coordinator,
    options.table,
    options.idColumn,
    options.dataColumn,
    options.type == "text" ? 64 : 16,
  )) {
    progress("Processing Batches...", (idsCount / total) * 100);

    let ids = Array.from(data.getChild("id"));
    let values = Array.from(data.getChild("value"));
    await rpc("embedding.batch", instance, values);
    allIDs.push(ids);
    idsCount += ids.length;
  }

  progress("UMAP Projection...");

  let coordinates: Float32Array = await rpc("embedding.finalize", instance);

  await setResultColumns(
    options.coordinator,
    options.table,
    options.idColumn,
    options.xColumn,
    options.yColumn,
    allIDs,
    coordinates,
  );
}
