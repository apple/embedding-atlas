// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { AsyncDuckDB } from "@duckdb/duckdb-wasm";
import type { Coordinator } from "@uwdata/mosaic-core";
import { literal } from "@uwdata/mosaic-sql";

function fetchableUrl(url: string) {
  return url;
}

export async function importDataTable(
  inputs: (File | { url: string })[],
  db: AsyncDuckDB,
  coordinator: Coordinator,
  table: string,
  onProgress?: (text: string, progress?: number, progressText?: string, markdown?: boolean) => void,
) {
  let index = 0;
  for (let input of inputs) {
    let data: ArrayBuffer;
    let filename: string;
    if (input instanceof File) {
      onProgress?.("Loading data from file...");
      filename = input.name;
      data = await input.arrayBuffer();
    } else if ("url" in input) {
      onProgress?.("Loading data from URL...");
      filename = input.url;
      let fileContents = await fetchWithProgress(
        fetchableUrl(input.url),
        { referrerPolicy: "no-referrer" },
        (n: number) => {
          onProgress?.("Loading data from URL...", undefined, formatFileSize(n));
        },
        (text: string, markdown?: boolean) => {
          onProgress?.(text, undefined, undefined, markdown);
        },
      );
      data = await fileContents.arrayBuffer();
    } else {
      throw new Error("invalid input type");
    }

    // Register the data as a temporary file
    let tempFile = `data-${index}` + extensionFromURL(filename);
    await db.registerFileBuffer(tempFile, new Uint8Array(data));

    // Load data into the table, if multiple inputs are used, add a filename column for the input name.
    if (inputs.length == 1) {
      await coordinator.query(`CREATE TABLE ${table} AS SELECT * FROM ${literal(tempFile)}`);
    } else {
      if (index == 0) {
        await coordinator.query(
          `CREATE TABLE ${table} AS SELECT *, ${literal(filename)} AS filename FROM ${literal(tempFile)}`,
        );
      } else {
        await coordinator.query(
          `INSERT INTO ${table} SELECT *, ${literal(filename)} AS filename FROM ${literal(tempFile)}`,
        );
      }
    }

    // Delete the temporary file
    await db.dropFile(tempFile);

    index += 1;
  }
}

async function fetchWithProgress(
  url: string,
  init?: RequestInit,
  onProgress?: (bytesLoaded: number) => void,
  onInfo?: (text: string, markdown?: boolean) => void,
): Promise<Blob> {
  let res: Response;

  try {
    res = await fetch(url, init);
  } catch (error) {
    onInfo?.(
      "Could not fetch data from URL, which may be due to a network issue or the server blocking [cross-origin requests (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS). Check that the URL is valid and configured to allow access from this site.",
      true,
    );
    throw new Error("Failed to fetch data from URL.");
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

  try {
    let reader = res.body.getReader();

    let chunks: Uint8Array<ArrayBuffer>[] = [];
    let bytesLoaded = 0;

    while (true) {
      let { done, value } = await reader.read();
      if (done) break;

      if (value != null) {
        chunks.push(value);
        bytesLoaded += value.length;
      }

      onProgress?.(bytesLoaded);
    }

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

function formatFileSize(bytes: number, decimals = 2): string {
  let units = ["B", "KB", "MB", "GB"];
  let base = 1000;

  let value = bytes;
  let unitIndex = 0;

  while (value >= base && unitIndex < units.length - 1) {
    value /= base;
    unitIndex++;
  }

  let formatted = unitIndex === 0 ? value.toString() : value.toFixed(decimals);

  return `${formatted} ${units[unitIndex]}`;
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
