// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Data export utilities for qualitative coding tool.
 * Exports data with coding annotations in various formats.
 */

import type { CodingState, Code, Memo } from "./types.js";

export type ExportFormat = "csv" | "json" | "jsonl";

export interface ExportOptions {
  /** Include coding annotations */
  includeCoding?: boolean;
  /** Include memos */
  includeMemos?: boolean;
  /** Include code metadata (color, level, description) */
  includeCodeMetadata?: boolean;
  /** Columns to export (empty = all) */
  columns?: string[];
  /** Export only selected rows */
  selectedRows?: Set<number | string>;
}

export const defaultExportOptions: ExportOptions = {
  includeCoding: true,
  includeMemos: true,
  includeCodeMetadata: false,
  columns: [],
  selectedRows: undefined,
};

export interface DataRow {
  [key: string]: any;
}

/**
 * Export data with coding annotations to CSV format
 */
export function exportToCSV(
  rows: DataRow[],
  columns: string[],
  codingState: CodingState | null,
  options: ExportOptions = defaultExportOptions
): string {
  const exportColumns = options.columns?.length ? options.columns : columns;
  const allColumns = [...exportColumns];

  // Add coding columns if enabled
  if (options.includeCoding && codingState) {
    allColumns.push("codes");
    if (options.includeCodeMetadata) {
      allColumns.push("code_ids", "code_colors", "code_levels");
    }
  }

  // Add memo column if enabled
  if (options.includeMemos && codingState) {
    allColumns.push("memos");
  }

  // Build CSV
  const csvRows: string[] = [];

  // Header
  csvRows.push(allColumns.map(escapeCSVValue).join(","));

  // Data rows
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const rowId = row.__row_index__ ?? row.id ?? i;

    // Skip if not in selected rows
    if (options.selectedRows && !options.selectedRows.has(rowId)) {
      continue;
    }

    const values: string[] = [];

    // Original columns
    for (const col of exportColumns) {
      values.push(escapeCSVValue(row[col]));
    }

    // Coding columns
    if (options.includeCoding && codingState) {
      const codeIds = codingState.codeApplications[String(rowId)] || [];
      const codes = codeIds
        .map((id: string) => codingState.codes[id])
        .filter((c): c is Code => c !== undefined);

      values.push(escapeCSVValue(codes.map((c: Code) => c.name).join("; ")));

      if (options.includeCodeMetadata) {
        values.push(escapeCSVValue(codeIds.join("; ")));
        values.push(escapeCSVValue(codes.map((c: Code) => c.color).join("; ")));
        values.push(escapeCSVValue(codes.map((c: Code) => c.level).join("; ")));
      }
    }

    // Memos column
    if (options.includeMemos && codingState) {
      const memos = Object.values(codingState.memos).filter(
        (m) => m.linkedDataPoints.includes(rowId)
      );
      values.push(escapeCSVValue(memos.map((m) => m.content).join(" | ")));
    }

    csvRows.push(values.join(","));
  }

  return csvRows.join("\n");
}

/**
 * Export data with coding annotations to JSON format
 */
export function exportToJSON(
  rows: DataRow[],
  columns: string[],
  codingState: CodingState | null,
  options: ExportOptions = defaultExportOptions
): string {
  const exportColumns = options.columns?.length ? options.columns : columns;
  const result: any[] = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const rowId = row.__row_index__ ?? row.id ?? i;

    // Skip if not in selected rows
    if (options.selectedRows && !options.selectedRows.has(rowId)) {
      continue;
    }

    const exportRow: any = {};

    // Original columns
    for (const col of exportColumns) {
      exportRow[col] = row[col];
    }

    // Coding data
    if (options.includeCoding && codingState) {
      const codeIds = codingState.codeApplications[String(rowId)] || [];
      const codes = codeIds
        .map((id: string) => codingState.codes[id])
        .filter((c): c is Code => c !== undefined);

      if (options.includeCodeMetadata) {
        exportRow._codes = codes.map((c: Code) => ({
          id: c.id,
          name: c.name,
          color: c.color,
          level: c.level,
          description: c.description,
        }));
      } else {
        exportRow._codes = codes.map((c: Code) => c.name);
      }
    }

    // Memos
    if (options.includeMemos && codingState) {
      const memos = Object.values(codingState.memos).filter(
        (m) => m.linkedDataPoints.includes(rowId)
      );
      exportRow._memos = memos.map((m) => ({
        content: m.content,
        memoType: m.memoType,
        createdAt: m.createdAt,
      }));
    }

    result.push(exportRow);
  }

  return JSON.stringify(result, null, 2);
}

/**
 * Export data with coding annotations to JSON Lines format
 */
export function exportToJSONL(
  rows: DataRow[],
  columns: string[],
  codingState: CodingState | null,
  options: ExportOptions = defaultExportOptions
): string {
  const exportColumns = options.columns?.length ? options.columns : columns;
  const lines: string[] = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const rowId = row.__row_index__ ?? row.id ?? i;

    // Skip if not in selected rows
    if (options.selectedRows && !options.selectedRows.has(rowId)) {
      continue;
    }

    const exportRow: any = {};

    // Original columns
    for (const col of exportColumns) {
      exportRow[col] = row[col];
    }

    // Coding data
    if (options.includeCoding && codingState) {
      const codeIds = codingState.codeApplications[String(rowId)] || [];
      const codes = codeIds
        .map((id: string) => codingState.codes[id])
        .filter((c): c is Code => c !== undefined);

      exportRow._codes = codes.map((c: Code) => c.name);
    }

    // Memos
    if (options.includeMemos && codingState) {
      const memos = Object.values(codingState.memos).filter(
        (m) => m.linkedDataPoints.includes(rowId)
      );
      if (memos.length > 0) {
        exportRow._memos = memos.map((m) => m.content);
      }
    }

    lines.push(JSON.stringify(exportRow));
  }

  return lines.join("\n");
}

/**
 * Export coding state separately (codes, memos, events)
 */
export function exportCodingState(codingState: CodingState): string {
  return JSON.stringify(codingState, null, 2);
}

/**
 * Export a codebook (list of codes with descriptions)
 */
export function exportCodebook(codingState: CodingState): string {
  const codes = Object.values(codingState.codes);
  const codebook = codes.map((code) => ({
    name: code.name,
    description: code.description || "",
    level: code.level,
    color: code.color,
    parent: code.parentId
      ? codingState.codes[code.parentId]?.name || null
      : null,
    frequency: code.frequency,
    createdAt: code.createdAt,
  }));

  return JSON.stringify(codebook, null, 2);
}

/**
 * Export memos separately
 */
export function exportMemos(codingState: CodingState): string {
  const memos = Object.values(codingState.memos);
  return JSON.stringify(memos, null, 2);
}

/**
 * Generate a summary report of coding
 */
export function generateCodingSummary(
  codingState: CodingState,
  totalRows: number
): string {
  const codes = Object.values(codingState.codes);
  const memos = Object.values(codingState.memos);

  const codedRows = new Set<string | number>();
  for (const [rowId, codeIds] of Object.entries(codingState.codeApplications)) {
    if (codeIds.length > 0) {
      codedRows.add(rowId);
    }
  }

  const summary = {
    totalDataPoints: totalRows,
    codedDataPoints: codedRows.size,
    codingCoverage: ((codedRows.size / totalRows) * 100).toFixed(1) + "%",
    totalCodes: codes.length,
    codesByLevel: {
      level1_open: codes.filter((c) => c.level === 1).length,
      level2_axial: codes.filter((c) => c.level === 2).length,
      level3_selective: codes.filter((c) => c.level === 3).length,
    },
    totalMemos: memos.length,
    memosByType: {
      theoretical: memos.filter((m) => m.memoType === "theoretical").length,
      methodological: memos.filter((m) => m.memoType === "methodological").length,
      observational: memos.filter((m) => m.memoType === "observational").length,
    },
    topCodes: codes
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10)
      .map((c) => ({ name: c.name, frequency: c.frequency })),
  };

  return JSON.stringify(summary, null, 2);
}

/**
 * Escape a value for CSV format
 */
function escapeCSVValue(value: any): string {
  if (value === null || value === undefined) {
    return "";
  }

  const str = String(value);

  // Check if escaping is needed
  if (str.includes(",") || str.includes('"') || str.includes("\n") || str.includes("\r")) {
    return '"' + str.replace(/"/g, '""') + '"';
  }

  return str;
}

/**
 * Download data as a file
 */
export function downloadAsFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export data in the specified format and trigger download
 */
export function exportAndDownload(
  rows: DataRow[],
  columns: string[],
  codingState: CodingState | null,
  format: ExportFormat,
  filename: string,
  options: ExportOptions = defaultExportOptions
): void {
  let content: string;
  let mimeType: string;
  let ext: string;

  switch (format) {
    case "csv":
      content = exportToCSV(rows, columns, codingState, options);
      mimeType = "text/csv";
      ext = ".csv";
      break;
    case "json":
      content = exportToJSON(rows, columns, codingState, options);
      mimeType = "application/json";
      ext = ".json";
      break;
    case "jsonl":
      content = exportToJSONL(rows, columns, codingState, options);
      mimeType = "application/x-ndjson";
      ext = ".jsonl";
      break;
  }

  // Ensure filename has correct extension
  if (!filename.endsWith(ext)) {
    filename += ext;
  }

  downloadAsFile(content, filename, mimeType);
}
