// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { AsyncDuckDB, AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import { literal } from "@uwdata/mosaic-sql";

import { resolveAppConfig } from "./app_config.js";
import { LoggableError, type Logger } from "./logging.js";

export async function importDataTable(
  inputs: (File | { url: string })[],
  db: AsyncDuckDB,
  connection: AsyncDuckDBConnection,
  table: string,
  logger?: Logger,
) {
  let config = resolveAppConfig();
  for (let [index, input] of inputs.entries()) {
    if (input instanceof File) {
      // File
      logger?.info("Loading data from file...");
      let filename = input.name;
      let data = new Uint8Array(await input.arrayBuffer());
      await importFileIntoTable(index, inputs.length, data, filename, table, db, connection);
    } else if ("url" in input) {
      // Try invoke the function from the config.
      let result = await config.loadDataFromUrl?.(input.url, {
        db: db,
        connection: connection,
        table: table,
        fetch: async (url) => {
          let contents = await fetchWithProgress(url, { referrerPolicy: "no-referrer" }, logger);
          return new Uint8Array(await contents.arrayBuffer());
        },
        logger: logger,
      });

      if (result === true) {
        // Data already imported into table.
        continue;
      }

      let data: Uint8Array<ArrayBuffer>;
      let filename: string;

      if (result != undefined) {
        if ("data" in result) {
          data = result.data;
          filename = result.filename ?? input.url;
        } else if ("url" in result) {
          let fileContents = await fetchWithProgress(result.url, { referrerPolicy: "no-referrer" }, logger);
          data = new Uint8Array(await fileContents.arrayBuffer());
          filename = result.filename ?? input.url;
        } else {
          throw new Error("invalid result from loadDataFromUrl");
        }
      } else {
        let fileContents = await fetchWithProgress(input.url, { referrerPolicy: "no-referrer" }, logger);
        data = new Uint8Array(await fileContents.arrayBuffer());
        filename = input.url;
      }
      await importFileIntoTable(index, inputs.length, data, filename, table, db, connection);
    } else {
      throw new Error("invalid input type");
    }
  }
}

async function importFileIntoTable(
  index: number,
  count: number,
  data: Uint8Array<ArrayBuffer>,
  filename: string,
  table: string,
  db: AsyncDuckDB,
  connection: AsyncDuckDBConnection,
) {
  // Register the data as a temporary file
  let tempFile = `data-${index}` + extensionFromURL(filename);
  await db.registerFileBuffer(tempFile, data);

  // Load data into the table, if multiple inputs are used, add a filename column for the input name.
  if (count == 1) {
    await connection.query(`CREATE TABLE ${table} AS SELECT * FROM ${literal(tempFile)}`);
  } else {
    if (index == 0) {
      await connection.query(
        `CREATE TABLE ${table} AS SELECT *, ${literal(filename)} AS filename FROM ${literal(tempFile)}`,
      );
    } else {
      await connection.query(
        `INSERT INTO ${table} SELECT *, ${literal(filename)} AS filename FROM ${literal(tempFile)}`,
      );
    }
  }

  // Delete the temporary file
  await db.dropFile(tempFile);
}

async function fetchWithProgress(url: string, init?: RequestInit, logger?: Logger): Promise<Blob> {
  let res: Response;

  let msg = logger?.info("Loading data from URL...");

  try {
    res = await fetch(url, init);
  } catch (error) {
    throw new LoggableError(
      `Failed to fetch data from URL: This may be due to a network issue or the server blocking [cross-origin requests (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS). Check that the URL is valid and configured to allow access from this site.`,
      { markdown: true },
    );
  }

  if (!res.ok) {
    throw new Error(
      `Failed to fetch data from URL: ${httpErrorStatusText(res.status)}. Please check if the URL is accessible and the server is responding correctly.`,
    );
  }

  if (!res.body) {
    throw new Error(
      `Failed to fetch data from URL: Server response has no body content. This may indicate a server configuration issue or the resource may be empty.`,
    );
  }

  const UPDATE_INTERVAL_MS = 200;

  try {
    let reader = res.body.getReader();

    let chunks: Uint8Array<ArrayBuffer>[] = [];
    let bytesLoaded = 0;
    let lastUpdateTime = 0;

    while (true) {
      let { done, value } = await reader.read();
      if (done) break;

      if (value != null) {
        chunks.push(value);
        bytesLoaded += value.length;
      }

      let now = new Date().getTime();
      if (now - lastUpdateTime >= UPDATE_INTERVAL_MS) {
        msg?.update({ progressText: formatFileSize(bytesLoaded) });
        lastUpdateTime = now;
      }
    }

    msg?.update({ progressText: formatFileSize(bytesLoaded) });

    return new Blob(chunks);
  } catch (_) {
    throw new Error(`Failed to fetch data from URL: Error while reading data.`);
  }
}

function extensionFromMimeType(mimeType: string): string {
  let normalizedMimeType = mimeType.toLowerCase();

  // Handle other common MIME types
  let typeMap: Record<string, string> = {
    // JSON
    "application/json": ".json",
    "text/json": ".json",

    // JSON Lines / NDJSON
    "application/x-ndjson": ".jsonl",
    "application/ndjson": ".jsonl",
    "application/jsonlines": ".jsonl",
    "application/json-seq": ".jsonl",

    // Apache Parquet
    "application/vnd.apache.parquet": ".parquet",
    "application/x-parquet": ".parquet",
  };

  return typeMap[normalizedMimeType] || ".bin";
}

function extensionFromURL(url: string) {
  if (url.startsWith("data:")) {
    let mimeTypeMatch = url.match(/^data:([^;,]+)/);
    if (mimeTypeMatch) {
      let extension = extensionFromMimeType(mimeTypeMatch[1]);
      return extension;
    }
    // Fallback for malformed data URLs
    return ".bin";
  }

  // Strip hash and query
  let clean = url.split("#", 1)[0].split("?", 1)[0];

  // Get last path segment
  let lastSlash = clean.lastIndexOf("/");
  let filename = lastSlash >= 0 ? clean.slice(lastSlash + 1) : clean;

  // Ignore hidden files like ".gitignore"
  if (!filename || (filename[0] === "." && filename.indexOf(".", 1) === -1)) {
    return null;
  }

  let dot = filename.lastIndexOf(".");
  return dot > 0 ? filename.slice(dot) : null;
}

function formatFileSize(bytes: number, decimals = 1): string {
  let units: [string, number][] = [
    ["B", 0],
    ["KB", 0],
    ["MB", 1],
    ["GB", 2],
  ];

  let base = 1000;

  let value = bytes;
  let unitIndex = 0;

  while (value >= base && unitIndex < units.length - 1) {
    value /= base;
    unitIndex++;
  }

  return `${value.toFixed(units[unitIndex][1])} ${units[unitIndex][0]}`;
}

function httpErrorStatusText(code: number): string {
  let map: Record<string, string> = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    409: "Conflict",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
  };
  if (map[code]) {
    return `HTTP ${code} - ${map[code]}`;
  } else {
    return `HTTP ${code}`;
  }
}
