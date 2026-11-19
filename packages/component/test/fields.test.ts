import * as SQL from "@uwdata/mosaic-sql";
import { describe, expect, test } from "vitest";

import {
  categoryIndexExpr,
  DEFAULT_NULL_COLOR,
  DEFAULT_OTHER_COLOR,
  fieldExpr,
  fieldKey,
  scaleColors,
} from "../src/lib/embedding_view/data/fields.js";

describe("fields", () => {
  test("fieldExpr and fieldKey", () => {
    expect(String(fieldExpr("x"))).toBe('"x"');
    expect(String(fieldExpr({ sql: "a + b" }))).toBe("(a + b)");
    expect(fieldKey("x")).toBe("column:x");
    expect(fieldKey({ sql: "x" })).toBe("sql:x");
    expect(fieldKey(null)).toBe(null);
  });
});

describe("categoryIndexExpr", () => {
  test("without domain or thresholds, values are indices", () => {
    expect(String(categoryIndexExpr(SQL.column("c"), null))).toBe('"c"');
    expect(String(categoryIndexExpr(SQL.column("c"), { range: ["red"] }))).toBe('"c"');
  });

  test("domain values compare as text; other and null (and NaN) come last", () => {
    let e = String(categoryIndexExpr(SQL.column("c"), { domain: ["a", 1, true] }));
    expect(e).toBe(
      `(CASE WHEN "c" IS NULL OR (typeof("c") IN ('FLOAT', 'DOUBLE') AND isnan(TRY_CAST("c" AS DOUBLE))) THEN 4 WHEN ("c")::TEXT = 'a' THEN 0 WHEN ("c")::TEXT = '1' THEN 1 WHEN ("c")::TEXT = 'true' THEN 2 ELSE 3 END)`,
    );
  });

  test("thresholds bin numbers; outside is other; null and NaN are null", () => {
    let e = String(categoryIndexExpr(SQL.column("v"), { thresholds: [0, 10, 20] }));
    expect(e).toContain(`WHEN ("v")::DOUBLE IS NULL OR isnan(("v")::DOUBLE) THEN 3`);
    expect(e).toContain(`WHEN ("v")::DOUBLE < 0 OR ("v")::DOUBLE > 20 THEN 2`);
    expect(e).toContain(`WHEN ("v")::DOUBLE < 10 THEN 0`);
    expect(e).toContain(`ELSE 1 END`);
  });

  test("time thresholds compare epoch milliseconds", () => {
    let e = String(categoryIndexExpr(SQL.column("t"), { thresholds: [0, 1000], type: "time" }));
    expect(e).toContain("epoch_ms");
  });
});

describe("scaleColors", () => {
  test("categories, then other and null", () => {
    expect(scaleColors({ domain: ["a", "b"], range: ["red", "blue"] })).toEqual([
      "red",
      "blue",
      DEFAULT_OTHER_COLOR,
      DEFAULT_NULL_COLOR,
    ]);
    expect(scaleColors({ domain: ["a"], other: "#111", null: "#222" })!.slice(1)).toEqual(["#111", "#222"]);
    expect(scaleColors({ range: ["red"] })).toBe(null);
    expect(scaleColors({ thresholds: [0, 1, 2] })!.length).toBe(4);
  });
});
