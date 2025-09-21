// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as duckdb from "@duckdb/duckdb-wasm";

// See https://duckdb.org/docs/api/wasm/instantiation

import duckdb_worker_mvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdb_wasm_mvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";

import duckdb_worker_eh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdb_wasm_eh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";

// COI version is not working yet.
// import duckdb_pthread_worker_coi from "@duckdb/duckdb-wasm/dist/duckdb-browser-coi.pthread.worker.js?url";
// import duckdb_worker_coi from "@duckdb/duckdb-wasm/dist/duckdb-browser-coi.worker.js?url";
// import duckdb_wasm_coi from "@duckdb/duckdb-wasm/dist/duckdb-coi.wasm?url";

function extensionRepositoryUrl() {
  let pageUrl = window.location.href;
  pageUrl = pageUrl.replace(/#.*$/, "");
  pageUrl = pageUrl.replace(/\/[^/]*$/, "/");
  if (!pageUrl.endsWith("/")) {
    pageUrl += "/";
  }
  return pageUrl + "duckdb-wasm";
}

export async function createDuckDB({ log = false } = {}) {
  const MANUAL_BUNDLES: duckdb.DuckDBBundles = {
    mvp: {
      mainModule: duckdb_wasm_mvp,
      mainWorker: duckdb_worker_mvp,
    },
    eh: {
      mainModule: duckdb_wasm_eh,
      mainWorker: duckdb_worker_eh,
    },
    // coi: {
    //   mainModule: duckdb_wasm_coi,
    //   mainWorker: duckdb_worker_coi,
    //   pthreadWorker: duckdb_pthread_worker_coi,
    // },
  };
  // Select a bundle based on browser checks
  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
  // Instantiate the asynchronous version of DuckDB-wasm
  const worker = new Worker(bundle.mainWorker!);
  const logger = log ? new duckdb.ConsoleLogger() : new duckdb.VoidLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  await db.open({
    filesystem: {
      // Suppresses the use of HTTP range requests for DuckDB-WASM running in the browser.
      forceFullHTTPReads: true,
    },
  });
  const connection = await db.connect();
  // Use a local extension repository to avoid CSP issues
  await connection.query(`SET custom_extension_repository = '${extensionRepositoryUrl()}';`);
  return { duckdb: db, connection: connection };
}
