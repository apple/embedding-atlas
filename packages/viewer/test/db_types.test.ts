// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { isFloatingPointDBType, jsTypeFromDBType } from "./db_types.js";

describe("isFloatingPointDBType", () => {
  test("returns true for floating-point types", () => {
    // These are the type strings DuckDB reports for floating-point columns.
    // DuckDB normalizes REAL to FLOAT, but the legacy aliases are handled too.
    for (let t of ["DOUBLE", "FLOAT", "REAL", "FLOAT4", "FLOAT8"]) {
      expect(isFloatingPointDBType(t), t).toBe(true);
    }
  });

  test("returns true for decimal types with precision and scale", () => {
    // DuckDB reports DECIMAL columns as e.g. DECIMAL(18,3).
    expect(isFloatingPointDBType("DECIMAL(18,3)")).toBe(true);
    expect(isFloatingPointDBType("DECIMAL(10,0)")).toBe(true);
    expect(isFloatingPointDBType("NUMERIC(5,2)")).toBe(true);
  });

  test("returns false for integer types", () => {
    for (let t of ["INTEGER", "BIGINT", "TINYINT", "SMALLINT", "HUGEINT", "UINTEGER", "UBIGINT"]) {
      expect(isFloatingPointDBType(t), t).toBe(false);
    }
  });

  test("returns false for non-numeric types", () => {
    for (let t of ["VARCHAR", "TEXT", "BOOLEAN", "DATE", "TIMESTAMP", "VARCHAR[3]"]) {
      expect(isFloatingPointDBType(t), t).toBe(false);
    }
  });

  test("returns false for arrays of decimals", () => {
    // DECIMAL(18,3)[] is an array column, not a plain numeric scalar, so it must
    // not be treated as a continuous floating-point type.
    expect(isFloatingPointDBType("DECIMAL(18,3)[]")).toBe(false);
    expect(isFloatingPointDBType("NUMERIC(5,2)[]")).toBe(false);
  });
});

describe("jsTypeFromDBType", () => {
  test("classifies floating-point and integer types as number", () => {
    expect(jsTypeFromDBType("DOUBLE")).toBe("number");
    expect(jsTypeFromDBType("FLOAT")).toBe("number");
    expect(jsTypeFromDBType("INTEGER")).toBe("number");
    expect(jsTypeFromDBType("BIGINT")).toBe("number");
    expect(jsTypeFromDBType("DECIMAL(18,3)")).toBe("number");
    expect(jsTypeFromDBType("NUMERIC(5,2)")).toBe("number");
  });

  test("classifies string, list, and date types", () => {
    expect(jsTypeFromDBType("VARCHAR")).toBe("string");
    expect(jsTypeFromDBType("BOOLEAN")).toBe("string");
    expect(jsTypeFromDBType("VARCHAR[3]")).toBe("string[]");
    expect(jsTypeFromDBType("TEXT[]")).toBe("string[]");
    expect(jsTypeFromDBType("TIMESTAMP")).toBe("Date");
  });

  test("returns null for unknown types", () => {
    expect(jsTypeFromDBType("BLOB")).toBe(null);
    // Arrays of decimals are not plain numeric scalars and have no JS array type here.
    expect(jsTypeFromDBType("DECIMAL(18,3)[]")).toBe(null);
  });
});
