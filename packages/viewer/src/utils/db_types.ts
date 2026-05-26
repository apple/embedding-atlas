// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// Pure helpers for classifying DuckDB column types. Kept free of any database
// or browser dependencies so the logic can be reused and unit tested directly.

/** Column description */
export interface ColumnDesc {
  name: string;
  type: string;
  jsType: JSType | null;
}

export type JSType = "string" | "number" | "string[]" | "Date";

export function jsTypeFromDBType(dbType: string): JSType | null {
  if (numberTypes.has(dbType) || decimalTypeRegex.test(dbType)) {
    return "number";
  } else if (stringTypes.has(dbType)) {
    return "string";
  } else if (dateTypes.has(dbType)) {
    return "Date";
  } else if (dbType.match(/^(VARCHAR|TEXT)\[\d*\]$/)) {
    return "string[]";
  } else {
    return null;
  }
}

/**
 * Returns true if the DuckDB type is a floating-point or decimal type (as opposed
 * to an integer type). Used to decide whether a low-cardinality numeric column
 * should be treated as continuous (binned) rather than categorical. DuckDB reports
 * DECIMAL types with precision and scale, e.g. `DECIMAL(18,3)`.
 */
export function isFloatingPointDBType(dbType: string): boolean {
  return floatTypes.has(dbType) || decimalTypeRegex.test(dbType);
}

// DuckDB reports decimals as DECIMAL(p) or DECIMAL(p,s), with NUMERIC as an alias.
// Anchored at both ends so array types such as DECIMAL(18,3)[] are not matched: we
// only classify plain numeric scalars here, not arrays of them.
const decimalTypeRegex = /^(DECIMAL|NUMERIC)\(\d+(\s*,\s*\d+)?\)$/;

const floatTypes = new Set(["REAL", "FLOAT4", "FLOAT8", "FLOAT", "DOUBLE"]);

const numberTypes = new Set([
  "REAL",
  "FLOAT4",
  "FLOAT8",
  "FLOAT",
  "DOUBLE",
  "INT",
  "TINYINT",
  "INT1",
  "SMALLINT",
  "INT2",
  "SHORT",
  "INTEGER",
  "INT4",
  "INT",
  "SIGNED",
  "INT8",
  "LONG",
  "BIGINT",
  "UTINYINT",
  "USMALLINT",
  "UINTEGER",
  "UBIGINT",
  "UHUGEINT",
]);

const stringTypes = new Set(["BOOLEAN", "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING"]);

const dateTypes = new Set([
  "DATE",
  "TIME",
  "DATETIME",
  "TIMESTAMP",
  "TIMESTAMPTZ",
  "TIMESTAMP WITH TIME ZONE",
  "TIMESTAMP WITHOUT TIME ZONE",
]);
