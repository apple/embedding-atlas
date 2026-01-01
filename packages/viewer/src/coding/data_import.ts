// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Data import utilities for qualitative coding tool.
 * Handles CSV/JSON import with media URL detection and processing.
 */

import { detectMediaType, isYouTubeUrl, isVimeoUrl, getYouTubeThumbnail } from "../media/index.js";

export interface ImportedRow {
  [key: string]: any;
}

export interface MediaColumn {
  name: string;
  type: "image" | "video" | "mixed";
  count: number;
}

export interface ImportResult {
  rows: ImportedRow[];
  columns: string[];
  mediaColumns: MediaColumn[];
  errors: string[];
}

export interface FolderImportOptions {
  /** Extract metadata from filename patterns */
  extractMetadata?: boolean;
  /** Pattern to extract metadata from filenames (regex with named groups) */
  filenamePattern?: RegExp;
  /** Generate thumbnails for videos */
  generateThumbnails?: boolean;
  /** Maximum file size in bytes (default 50MB) */
  maxFileSize?: number;
}

export const defaultFolderImportOptions: FolderImportOptions = {
  extractMetadata: true,
  filenamePattern: /^(?<id>\d+)?_?(?<name>[^.]+)\.(?<ext>\w+)$/,
  generateThumbnails: true,
  maxFileSize: 50 * 1024 * 1024,
};

/**
 * Parse CSV text into rows
 */
export function parseCSV(text: string): ImportResult {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length === 0) {
    return { rows: [], columns: [], mediaColumns: [], errors: ["Empty CSV file"] };
  }

  // Parse header
  const headers = parseCSVLine(lines[0]);
  const rows: ImportedRow[] = [];
  const errors: string[] = [];

  // Parse data rows
  for (let i = 1; i < lines.length; i++) {
    try {
      const values = parseCSVLine(lines[i]);
      const row: ImportedRow = {};
      headers.forEach((header, idx) => {
        row[header] = values[idx] ?? "";
      });
      rows.push(row);
    } catch (e) {
      errors.push(`Error parsing line ${i + 1}: ${e}`);
    }
  }

  // Detect media columns
  const mediaColumns = detectMediaColumns(rows, headers);

  return { rows, columns: headers, mediaColumns, errors };
}

/**
 * Parse a single CSV line handling quoted values
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const nextChar = line[i + 1];

    if (inQuotes) {
      if (char === '"' && nextChar === '"') {
        current += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        current += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ",") {
        result.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
  }

  result.push(current.trim());
  return result;
}

/**
 * Detect columns that contain media URLs
 */
function detectMediaColumns(rows: ImportedRow[], columns: string[]): MediaColumn[] {
  const mediaColumns: MediaColumn[] = [];

  for (const column of columns) {
    let imageCount = 0;
    let videoCount = 0;

    for (const row of rows) {
      const value = row[column];
      if (typeof value === "string" && value.trim()) {
        const mediaType = detectMediaType(value);
        if (mediaType === "image") {
          imageCount++;
        } else if (mediaType === "video") {
          videoCount++;
        }
      }
    }

    const total = imageCount + videoCount;
    if (total > 0 && total >= rows.length * 0.5) {
      // At least 50% of rows have media
      let type: "image" | "video" | "mixed";
      if (imageCount > 0 && videoCount > 0) {
        type = "mixed";
      } else if (imageCount > 0) {
        type = "image";
      } else {
        type = "video";
      }
      mediaColumns.push({ name: column, type, count: total });
    }
  }

  return mediaColumns;
}

/**
 * Import files from a folder (via File API)
 */
export async function importFolder(
  files: FileList | File[],
  options: FolderImportOptions = defaultFolderImportOptions
): Promise<ImportResult> {
  const rows: ImportedRow[] = [];
  const errors: string[] = [];
  const fileArray = Array.from(files);

  // Filter and sort files
  const validFiles = fileArray.filter((file) => {
    if (options.maxFileSize && file.size > options.maxFileSize) {
      errors.push(`Skipped ${file.name}: exceeds max file size`);
      return false;
    }
    return isMediaFile(file);
  });

  for (const file of validFiles) {
    const row: ImportedRow = {
      filename: file.name,
      path: file.webkitRelativePath || file.name,
      size: file.size,
      type: file.type,
      lastModified: new Date(file.lastModified).toISOString(),
    };

    // Create object URL for the file
    row.media_url = URL.createObjectURL(file);

    // Extract metadata from filename if enabled
    if (options.extractMetadata && options.filenamePattern) {
      const match = file.name.match(options.filenamePattern);
      if (match?.groups) {
        for (const [key, value] of Object.entries(match.groups)) {
          if (value && key !== "ext") {
            row[key] = value;
          }
        }
      }
    }

    // Detect media type
    const mediaType = getMediaTypeFromFile(file);
    row.media_type = mediaType;

    rows.push(row);
  }

  // Determine columns from first row or defaults
  const columns = rows.length > 0 ? Object.keys(rows[0]) : ["filename", "path", "media_url", "media_type"];

  // Detect media columns
  const mediaColumns = detectMediaColumns(rows, columns);

  return { rows, columns, mediaColumns, errors };
}

/**
 * Check if a file is a supported media file
 */
function isMediaFile(file: File): boolean {
  const imageTypes = ["image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml"];
  const videoTypes = ["video/mp4", "video/webm", "video/ogg", "video/quicktime"];

  return imageTypes.includes(file.type) || videoTypes.includes(file.type);
}

/**
 * Get media type from a File object
 */
function getMediaTypeFromFile(file: File): "image" | "video" | "unknown" {
  if (file.type.startsWith("image/")) {
    return "image";
  } else if (file.type.startsWith("video/")) {
    return "video";
  }
  return "unknown";
}

/**
 * Fetch thumbnails for video URLs (YouTube, Vimeo)
 */
export async function fetchVideoThumbnails(
  rows: ImportedRow[],
  videoColumn: string,
  thumbnailColumn: string = "thumbnail_url"
): Promise<ImportedRow[]> {
  const result: ImportedRow[] = [];

  for (const row of rows) {
    const newRow = { ...row };
    const videoUrl = row[videoColumn];

    if (typeof videoUrl === "string") {
      if (isYouTubeUrl(videoUrl)) {
        // YouTube thumbnails are easy to get
        const thumbnail = getYouTubeThumbnail(videoUrl);
        if (thumbnail) {
          newRow[thumbnailColumn] = thumbnail;
        }
      } else if (isVimeoUrl(videoUrl)) {
        // For Vimeo, we'd need to use their API (requires authentication)
        // For now, leave empty - could be enhanced later
        newRow[thumbnailColumn] = "";
      }
    }

    result.push(newRow);
  }

  return result;
}

/**
 * Add a new column to all rows
 */
export function addColumn(
  rows: ImportedRow[],
  columnName: string,
  defaultValue: any = ""
): ImportedRow[] {
  return rows.map((row) => ({
    ...row,
    [columnName]: defaultValue,
  }));
}

/**
 * Remove a column from all rows
 */
export function removeColumn(rows: ImportedRow[], columnName: string): ImportedRow[] {
  return rows.map((row) => {
    const newRow = { ...row };
    delete newRow[columnName];
    return newRow;
  });
}

/**
 * Rename a column in all rows
 */
export function renameColumn(
  rows: ImportedRow[],
  oldName: string,
  newName: string
): ImportedRow[] {
  return rows.map((row) => {
    const newRow = { ...row };
    if (oldName in newRow) {
      newRow[newName] = newRow[oldName];
      delete newRow[oldName];
    }
    return newRow;
  });
}

/**
 * Update a cell value
 */
export function updateCell(
  rows: ImportedRow[],
  rowIndex: number,
  columnName: string,
  value: any
): ImportedRow[] {
  return rows.map((row, idx) => {
    if (idx === rowIndex) {
      return { ...row, [columnName]: value };
    }
    return row;
  });
}
