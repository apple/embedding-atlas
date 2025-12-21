// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import type { EmbeddingAtlasProps } from "../api.js";
import { initializeDatabase } from "../utils/database.js";
import { downloadBuffer } from "../utils/download.js";
import { exportMosaicSelection, filenameForSelection, type ExportFormat } from "../utils/mosaic_exporter.js";
import type { DataSource } from "./data_source.js";

function joinUrl(a: string, b: string) {
  if (b.startsWith(".")) {
    b = b.slice(1);
  }
  if (a.endsWith("/") && b.startsWith("/")) {
    return a + b.slice(1);
  } else if (!a.endsWith("/") && !b.startsWith("/")) {
    return a + "/" + b;
  } else {
    return a + b;
  }
}

interface Metadata {
  props: Partial<EmbeddingAtlasProps>;

  isStatic?: boolean;
  database?: {
    type: "wasm" | "socket" | "rest";
    uri?: string;
    load?: boolean;
  };
}

export class BackendDataSource implements DataSource {
  private serverUrl: string;
  downloadArchive: (() => Promise<void>) | undefined = undefined;
  downloadSelection: ((predicate: string | null, format: ExportFormat) => Promise<void>) | undefined = undefined;

  constructor(serverUrl: string) {
    if (serverUrl.startsWith("http")) {
      this.serverUrl = serverUrl;
    } else {
      let pageUrl = window.location.origin + window.location.pathname;
      pageUrl = pageUrl.replace(/\/[^/]*$/, "/");
      this.serverUrl = joinUrl(pageUrl, serverUrl);
    }
  }

  async initializeCoordinator(
    coordinator: Coordinator,
    table: string,
    onStatus: (message: string) => void,
  ): Promise<Partial<EmbeddingAtlasProps>> {
    let metadata = await this.metadata();

    onStatus("Initializing database...");
    let dbType = metadata.database?.type ?? "wasm";
    await initializeDatabase(coordinator, dbType, metadata.database?.uri ?? joinUrl(this.serverUrl, "query"));

    if (metadata.database?.load) {
      onStatus("Loading data...");
      let datasetUrl = joinUrl(this.serverUrl, "dataset.parquet");
      await coordinator.exec(`
        CREATE OR REPLACE TABLE ${table} AS (SELECT * FROM read_parquet(${SQL.literal(datasetUrl)}));
      `);
    }

    if (!metadata.isStatic) {
      this.downloadArchive = async () => {
        let resp = await this.fetchEndpoint("archive.zip");
        let data = await resp.arrayBuffer();
        downloadBuffer(data, "embedding-atlas.zip");
      };
    }

    if (dbType == "wasm") {
      this.downloadSelection = async (predicate, format) => {
        let [bytes, name] = await exportMosaicSelection(coordinator, table, predicate, format);
        downloadBuffer(bytes, name);
      };
    } else if (!metadata.isStatic) {
      this.downloadSelection = async (predicate, format) => {
        let name = filenameForSelection(format);
        let resp = await this.fetchEndpoint("selection", {
          method: "POST",
          body: JSON.stringify({ predicate: predicate, format: format }),
        });
        let data = await resp.arrayBuffer();
        downloadBuffer(data, name);
      };
    }

    return metadata.props;
  }

  private async fetchEndpoint(endpoint: string, init?: RequestInit) {
    let resp = await fetch(joinUrl(this.serverUrl, endpoint), init);
    if (resp.status != 200) {
      throw new Error("ERROR FETCH");
    }
    return resp;
  }

  private async metadata(): Promise<Metadata> {
    try {
      return await this.fetchEndpoint("metadata.json").then((x) => x.json());
    } catch (e) {
      throw new Error("Network Error: Failed to fetch dataset metadata");
    }
  }

  async cacheGet(key: string) {
    try {
      return await this.fetchEndpoint("cache/" + key).then((x) => x.json());
    } catch (e) {
      return null;
    }
  }

  async cacheSet(key: string, value: any) {
    try {
      await this.fetchEndpoint("cache/" + key, {
        method: "POST",
        body: JSON.stringify(value),
      });
    } catch (e) {
      // Ignore set cache errors.
    }
  }

  cache = {
    get: (key: string) => this.cacheGet(key),
    set: (key: string, value: any) => this.cacheSet(key, value),
  };
}
