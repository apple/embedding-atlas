// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

export interface BackendCapabilities {
  embedding_computation: boolean;
  supported_models: {
    text: string[];
    image: string[];
  };
}

export interface BackendEmbeddingResult {
  projection: number[][];
  knn_indices: number[][];
  knn_distances: number[][];
}

let cachedCapabilities: BackendCapabilities | null = null;
let capabilitiesChecked = false;

/**
 * Check if backend supports embedding computation
 */
export async function checkBackendCapabilities(): Promise<BackendCapabilities | null> {
  if (capabilitiesChecked) {
    return cachedCapabilities;
  }

  try {
    const response = await fetch("/api/capabilities");
    if (response.ok) {
      cachedCapabilities = await response.json();
      capabilitiesChecked = true;
      return cachedCapabilities;
    }
  } catch (error) {
    // Backend not available or doesn't support capabilities endpoint
    console.log("Backend capabilities check failed:", error);
  }

  capabilitiesChecked = true;
  cachedCapabilities = null;
  return null;
}

/**
 * Check if a model requires backend computation
 */
export function requiresBackend(model: string, type: "text" | "image"): boolean {
  // I-JEPA models always require backend
  if (model.includes("ijepa") || model.includes("facebook/ijepa")) {
    return true;
  }

  // Models with full HuggingFace paths (not Xenova/) likely need backend
  if (type === "image" && model.includes("/") && !model.startsWith("Xenova/")) {
    return true;
  }

  return false;
}

/**
 * Compute embeddings using the Python backend
 */
export async function computeEmbeddingBackend(options: {
  coordinator: Coordinator;
  table: string;
  idColumn: string;
  dataColumn: string;
  xColumn: string;
  yColumn: string;
  type: "text" | "image";
  model: string;
  callback?: (message: string, progress?: number) => void;
}): Promise<void> {
  const { coordinator, table, idColumn, dataColumn, xColumn, yColumn, type, model, callback } = options;

  function progress(message: string, progressValue?: number) {
    callback?.(message, progressValue);
  }

  progress(`Loading data for ${model}...`);

  // Fetch all data from the database
  const result = await coordinator.query(
    SQL.Query.from(table)
      .select({ id: SQL.column(idColumn), value: SQL.column(dataColumn) })
      .orderby(idColumn)
  );

  const ids = Array.from(result.getChild("id"));
  const values = Array.from(result.getChild("value"));

  progress(`Sending data to backend for embedding...`);

  // Send to backend for processing
  const response = await fetch("/api/compute-embedding", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: values,
      type: type,
      model: model,
      batch_size: type === "text" ? 32 : 16,
      umap_args: {
        metric: "cosine",
        n_neighbors: 15,
      },
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Backend embedding computation failed: ${error.error || response.statusText}`);
  }

  progress("Receiving results from backend...");

  const embeddingResult: BackendEmbeddingResult = await response.json();

  progress("Storing results in database...");

  // Create columns if they don't exist
  await coordinator.exec(`
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(xColumn)} DOUBLE DEFAULT 0;
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(yColumn)} DOUBLE DEFAULT 0;
  `);

  // Update the database with the projection results
  // We'll do this in batches to avoid SQL size limits
  const batchSize = 1000;
  for (let i = 0; i < ids.length; i += batchSize) {
    const batchIds = ids.slice(i, i + batchSize);
    const batchProjection = embeddingResult.projection.slice(i, i + batchSize);

    await coordinator.exec(`
      WITH t1 AS (
        SELECT
          UNNEST([${batchIds.map((id) => SQL.literal(id)).join(",")}]) AS id,
          UNNEST([${batchProjection.map((p) => p[0]).join(",")}]) AS x,
          UNNEST([${batchProjection.map((p) => p[1]).join(",")}]) AS y
      )
      UPDATE ${table}
        SET ${SQL.column(xColumn)} = t1.x, ${SQL.column(yColumn)} = t1.y
        FROM t1 WHERE ${SQL.column(idColumn, table)} = t1.id
    `);

    const progressPercent = ((i + batchSize) / ids.length) * 100;
    progress(`Updating database... ${Math.min(100, Math.round(progressPercent))}%`);
  }

  progress("Embedding computation complete!");
}
