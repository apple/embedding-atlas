// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { isReadonlyQuery } from "../src/model_context/readonly_query.js";

describe("isReadonlyQuery", () => {
  it("allows plain reads", () => {
    expect(isReadonlyQuery("SELECT * FROM dataset")).toBe(true);
    expect(isReadonlyQuery("  with t AS (SELECT 1) SELECT * FROM t")).toBe(true);
    expect(isReadonlyQuery("VALUES (1), (2);")).toBe(true);
    expect(isReadonlyQuery("DESCRIBE dataset")).toBe(true);
    expect(isReadonlyQuery("SHOW TABLES")).toBe(true);
    expect(isReadonlyQuery("EXPLAIN SELECT 1")).toBe(true);
    expect(isReadonlyQuery("-- comment\nSELECT version() AS version")).toBe(true);
  });

  it("rejects writes and stacked statements", () => {
    expect(isReadonlyQuery("DROP TABLE dataset")).toBe(false);
    expect(isReadonlyQuery("DELETE FROM dataset")).toBe(false);
    expect(isReadonlyQuery("INSERT INTO dataset VALUES (1)")).toBe(false);
    expect(isReadonlyQuery("UPDATE dataset SET x = 1")).toBe(false);
    expect(isReadonlyQuery("CREATE TABLE t (x INT)")).toBe(false);
    expect(isReadonlyQuery("COPY dataset TO 'out.parquet'")).toBe(false);
    expect(isReadonlyQuery("ATTACH 'other.db'")).toBe(false);
    expect(isReadonlyQuery("SELECT * FROM dataset; DROP TABLE dataset")).toBe(false);
    expect(isReadonlyQuery("")).toBe(false);
  });

  it("ignores keywords inside strings and comments", () => {
    expect(isReadonlyQuery("SELECT 'DROP TABLE not_a_table' AS x")).toBe(true);
    expect(isReadonlyQuery("SELECT * FROM t -- DELETE")).toBe(true);
  });
});
