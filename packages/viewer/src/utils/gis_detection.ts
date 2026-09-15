/**
 * Auto-detection of GIS columns in uploaded datasets.
 *
 * Implements a priority hierarchy:
 *   1. Exact lon/lat or longitude/latitude column pairs
 *   2. Fuzzy match — first columns whose names contain "lat"/"lon"
 *   3. Geometry column (GeoParquet, WKB, or GeoJSON) — requires coordinate extraction
 *
 * The detection result tells the SettingsView which columns to pre-fill
 * and whether geometry extraction is needed before the data can be viewed.
 */

export type GisDetectionResult =
  | {
      type: "geometry";
      /** Name of the geometry column. */
      geometryColumn: string;
      /** Columns to create (lon/lat will be extracted from geometry). */
      xColumn: string;
      yColumn: string;
    }
  | {
      type: "columns";
      xColumn: string;
      yColumn: string;
    }
  | null;

interface ColumnInfo {
  column_name: string;
  column_type: string;
}

/**
 * Detect GIS columns from a list of DuckDB DESCRIBE results.
 *
 * Returns a detection result or null if no GIS columns found.
 */
export function detectGisColumns(columns: ColumnInfo[]): GisDetectionResult {
  const names = columns.map((c) => c.column_name);
  const nameSet = new Set(names.map((n) => n.toLowerCase()));
  const typeMap = new Map(columns.map((c) => [c.column_name.toLowerCase(), c.column_type]));

  // --- Priority 1: Exact lon/lat pair ---
  const exactPairs: [string, string][] = [
    ["lon", "lat"],
    ["longitude", "latitude"],
    ["lng", "lat"],
  ];
  for (const [xCand, yCand] of exactPairs) {
    const xMatch = findNumericColumn(columns, xCand);
    const yMatch = findNumericColumn(columns, yCand);
    if (xMatch && yMatch) {
      return { type: "columns", xColumn: xMatch, yColumn: yMatch };
    }
  }

  // --- Priority 2: Fuzzy match — first columns containing "lon"/"lat" ---
  const lonPatterns = ["lon", "longitude", "lng"];
  const latPatterns = ["lat", "latitude"];

  let fuzzyX: string | null = null;
  let fuzzyY: string | null = null;

  for (const col of columns) {
    const lower = col.column_name.toLowerCase();
    const isNumeric = isNumericType(col.column_type);
    if (!isNumeric) continue;

    if (!fuzzyX && lonPatterns.some((p) => lower.includes(p))) {
      fuzzyX = col.column_name;
    }
    if (!fuzzyY && latPatterns.some((p) => lower.includes(p))) {
      fuzzyY = col.column_name;
    }
  }

  if (fuzzyX && fuzzyY) {
    return { type: "columns", xColumn: fuzzyX, yColumn: fuzzyY };
  }

  // --- Priority 3: Geometry column ---
  // Checked last because if named lon/lat columns exist, those are preferred.
  const geometryCandidates = ["geometry", "geom", "wkb_geometry", "the_geom", "geo"];
  for (const candidate of geometryCandidates) {
    if (nameSet.has(candidate)) {
      const colType = typeMap.get(candidate) ?? "";
      if (isSupportedGeometryType(colType)) {
        const originalName = names.find((n) => n.toLowerCase() === candidate)!;
        const xCol = findUniqueName(names, "lon");
        const yCol = findUniqueName(names, "lat");
        return { type: "geometry", geometryColumn: originalName, xColumn: xCol, yColumn: yCol };
      }
    }
  }

  return null;
}

/** Types used for Point geometry by DuckDB's Parquet and JSON readers. */
function isSupportedGeometryType(dbType: string): boolean {
  const type = dbType.toUpperCase();
  const compact = type.replace(/\s/g, "");

  if (
    type.includes("GEOMETRY") ||
    type.includes("BLOB") ||
    type.includes("BYTEA") ||
    type.includes("VARBINARY") ||
    type === "JSON" ||
    /^VARCHAR(?:\(\d+\))?$/.test(type)
  ) {
    return true;
  }

  // Some editors serialize WKB bytes as a Parquet list instead of a binary column.
  if (/^(U?TINYINT|U?SMALLINT|U?INTEGER|U?BIGINT)\[(\d+)?\]$/.test(compact)) {
    return true;
  }

  // DuckDB materializes a GeoJSON object as a struct when reading JSON/Parquet.
  return type.includes("STRUCT") && type.includes("TYPE") && type.includes("COORDINATES");
}

/** Find an exact case-insensitive match among numeric columns. */
function findNumericColumn(columns: ColumnInfo[], name: string): string | null {
  for (const col of columns) {
    if (col.column_name.toLowerCase() === name && isNumericType(col.column_type)) {
      return col.column_name;
    }
  }
  return null;
}

function isNumericType(dbType: string): boolean {
  const t = dbType.toUpperCase();
  return (
    t.includes("INT") ||
    t.includes("FLOAT") ||
    t.includes("DOUBLE") ||
    t.includes("DECIMAL") ||
    t.includes("NUMERIC") ||
    t.includes("REAL") ||
    t.includes("BIGINT") ||
    t.includes("SMALLINT") ||
    t.includes("TINYINT") ||
    t.includes("HUGEINT")
  );
}

function findUniqueName(existing: string[], base: string): string {
  const set = new Set(existing.map((n) => n.toLowerCase()));
  if (!set.has(base)) return base;
  let i = 1;
  while (set.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}
