// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import { describe, expect, test } from "vitest";

import { instancesDisplayQuery } from "../src/charts/instances/display_query.js";

describe("instancesDisplayQuery", () => {
  test("casts wide integers after numeric sorting and pagination", () => {
    let page = SQL.Query.from("dataset")
      .select({
        __id__: SQL.column("__row_index__"),
        candidate_h3_9: SQL.column("candidate_h3_9"),
        name: SQL.column("name"),
      })
      .orderby(SQL.asc(SQL.column("candidate_h3_9")))
      .limit(50)
      .offset(100);

    let query = instancesDisplayQuery(
      page,
      ["__id__", "candidate_h3_9", "name"],
      new Map([
        ["__id__", "INTEGER"],
        ["candidate_h3_9", "UBIGINT"],
        ["name", "VARCHAR"],
      ]),
    );

    expect(String(query)).toBe(
      'SELECT "__id__", ("candidate_h3_9")::VARCHAR AS "candidate_h3_9", "name" FROM ' +
        '(SELECT "__row_index__" AS "__id__", "candidate_h3_9", "name" FROM "dataset" ' +
        'ORDER BY "candidate_h3_9" ASC LIMIT 50 OFFSET 100)',
    );
  });

  test("does not wrap queries without wide integers", () => {
    let page = SQL.Query.from("dataset").select("name").limit(10);

    expect(instancesDisplayQuery(page, ["name"], new Map([["name", "VARCHAR"]]))).toBe(page);
  });
});
