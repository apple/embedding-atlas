// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Coordinator, restConnector, socketConnector, wasmConnector, type Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import { format } from "d3-format";

import { inferBinning } from "./charts/binning.js";
import { defaultCategoryColors, defaultOrdinalColors } from "./colors.js";
import { createDuckDB } from "./duckdb.js";
import { plotUniqueId, type Plot } from "./plots/plot.js";
import { makeCountPlot, makeHistogram } from "./plots/specs.js";

export async function initializeDatabase(
  coordinator: Coordinator,
  type: "wasm" | "socket" | "rest",
  uri: string | null | undefined = undefined,
) {
  const db = await createDuckDB();
  if (type == "wasm") {
    const conn = await wasmConnector({ duckdb: db.duckdb, connection: db.connection });
    coordinator.databaseConnector(conn);
  } else if (type == "socket") {
    const conn = await socketConnector({ uri: uri ?? "" });
    coordinator.databaseConnector(conn);
  } else if (type == "rest") {
    const conn = await restConnector({ uri: uri ?? "" });
    coordinator.databaseConnector(conn);
  }
}

export function predicateToString(predicate: ReturnType<Selection["predicate"]>): string | null {
  if (predicate == null) {
    return null;
  }
  if (predicate instanceof Array) {
    if (predicate.length == 0) {
      return null;
    }
    return SQL.and(predicate).toString().trim();
  }
  if (typeof predicate == "string") {
    return predicate.trim();
  }
  if (typeof predicate == "boolean") {
    return SQL.literal(predicate).toString();
  }
  return predicate.toString().trim();
}

export interface ColumnDesc {
  name: string;
  type: string;
  jsType: JSColumnType | null;
  distinctCount: number;
}

export interface EmbeddingLegend {
  indexColumn: string;
  legend: {
    label: string;
    color: string;
    predicate: any;
    count: number;
  }[];
}

export class TableInfo {
  table: string;
  coordinator: Coordinator;

  constructor(coordinator: Coordinator, table: string) {
    this.table = table;
    this.coordinator = coordinator;
  }

  async exec(query: any) {
    await this.coordinator.exec(query);
  }

  async query<T = any>(query: any): Promise<T[]> {
    let result = await this.coordinator.query(query);
    return result.toArray().map((x: any) => ({ ...x })) as T[];
  }

  async queryOne<T = any>(query: any): Promise<T> {
    let result = await this.coordinator.query(query);
    return { ...result.get(0) } as T;
  }

  async describe() {
    return await this.query<{ column_name: string; column_type: string }>(SQL.sql`DESCRIBE ${this.table}`);
  }

  async distinctCount(column: string): Promise<number> {
    let r = await this.queryOne(SQL.sql`SELECT COUNT(DISTINCT ${SQL.column(column)}) AS count FROM ${this.table}`);
    return r.count;
  }

  async columnDescriptions(): Promise<ColumnDesc[]> {
    let columns = await this.describe();
    let result: ColumnDesc[] = [];
    for (let column of columns) {
      result.push({
        name: column.column_name,
        type: column.column_type,
        jsType: jsTypeFromDBType(column.column_type),
        distinctCount: await this.distinctCount(column.column_name),
      });
    }
    return result;
  }

  async defaultViewportScale(x: string, y: string): Promise<number> {
    let { stdX, stdY } = await this.queryOne(
      SQL.Query.from(this.table).select({
        stdX: SQL.sql`STDDEV(${SQL.column(x)})::FLOAT`,
        stdY: SQL.sql`STDDEV(${SQL.column(y)})::FLOAT`,
      }),
    );
    let scale = 1.0 / (Math.max(stdX, stdY, 1e-3) * 3);
    return scale;
  }

  defaultPlots(columns: ColumnDesc[]): Plot[] {
    let plots: Plot[] = [
      { id: plotUniqueId(), title: "Named Selections", spec: { component: "SelectionList", props: {} } },
    ];

    for (let item of columns) {
      if (item.jsType == null) {
        continue;
      }
      // Skip the column if there's only a single unique value.
      if (item.distinctCount <= 1) {
        continue;
      }
      switch (item.jsType) {
        case "string":
          if (item.distinctCount <= 1000) {
            plots.push({
              id: plotUniqueId(),
              title: item.name,
              spec: makeCountPlot({ name: item.name, type: "discrete" }),
            });
          }
          break;
        case "string[]":
          plots.push({
            id: plotUniqueId(),
            title: item.name,
            spec: makeCountPlot({ name: item.name, type: "discrete[]" }),
          });
          break;
        case "number":
          if (item.distinctCount <= 10) {
            plots.push({
              id: plotUniqueId(),
              title: item.name,
              spec: makeCountPlot({ name: item.name, type: "discrete" }),
            });
          } else {
            plots.push({
              id: plotUniqueId(),
              title: item.name,
              spec: makeHistogram({ name: item.name, type: "continuous" }),
            });
          }
          break;
      }
    }
    return plots;
  }

  async makeCategoryColumn(column: string, maxCategories: number): Promise<EmbeddingLegend> {
    let indexColumnName = `_ev_${column}_id`;
    let values: { value: string; count: number }[] = await this.query(
      SQL.Query.from(this.table)
        .select({ value: SQL.cast(SQL.column(column), "TEXT"), count: SQL.count() })
        .where(SQL.not(SQL.isNull(SQL.cast(SQL.column(column), "TEXT"))))
        .groupby(SQL.cast(SQL.column(column), "TEXT"))
        .orderby(SQL.desc(SQL.count()))
        .limit(maxCategories),
    );

    let otherIndex = values.length;
    let nullIndex = values.length + 1;

    // Add the index column.
    await this.exec(SQL.sql`
      ALTER TABLE ${this.table} ADD COLUMN IF NOT EXISTS ${SQL.column(indexColumnName)} INTEGER DEFAULT 0;
      UPDATE ${this.table}
      SET ${SQL.column(indexColumnName)} = CASE ${SQL.column(column)}::TEXT
        ${values.map(({ value }, i) => SQL.sql`WHEN ${SQL.literal(value)} THEN ${SQL.literal(i)}`).join(" ")}
      ELSE (CASE WHEN ${SQL.column(column)} IS NULL THEN ${SQL.literal(nullIndex)} ELSE ${SQL.literal(otherIndex)} END) END
    `);

    // Count by index.
    let counts = await this.query(SQL.sql`
      SELECT ${SQL.column(indexColumnName)} AS index, COUNT(*)::INT AS count
      FROM ${this.table}
      GROUP BY ${SQL.column(indexColumnName)}
    `);
    let countMap = new Map<number, number>();
    for (let item of counts) {
      countMap.set(item.index, item.count);
    }
    let otherCount = countMap.get(otherIndex) ?? 0;
    let nullCount = countMap.get(nullIndex) ?? 0;

    let colors = defaultCategoryColors(values.length);

    let legend: EmbeddingLegend["legend"] = values.map(({ value }, i) => ({
      label: value,
      color: colors[i],
      predicate: SQL.eq(SQL.cast(SQL.column(column), "TEXT"), SQL.literal(value)),
      count: countMap.get(i) ?? 0,
    }));

    if (otherCount > 0) {
      let { otherCategoryCount } = await this.queryOne(SQL.sql`
        SELECT COUNT(DISTINCT(${SQL.column(column)}::TEXT)) AS otherCategoryCount
        FROM ${this.table}
        WHERE ${SQL.column(indexColumnName)} = ${SQL.literal(otherIndex)} AND ${SQL.column(column)} IS NOT NULL
      `);
      legend.push({
        label: `(other ${otherCategoryCount.toLocaleString()})`,
        color: "#9eabc2",
        predicate:
          values.length > 0
            ? SQL.sql`${SQL.column(column)} IS NOT NULL AND ${SQL.column(column)}::TEXT NOT IN (${values.map((x) => SQL.literal(x.value)).join(",")})`
            : SQL.sql`${SQL.column(column)} IS NOT NULL`,
        count: otherCount,
      });
    }
    if (nullCount > 0) {
      if (otherCount <= 0) {
        // If there is no other, reduce null index by 1 before we add the null item.
        await this.exec(`
          UPDATE ${this.table}
          SET ${SQL.column(indexColumnName)} = ${SQL.column(indexColumnName)} - 1 WHERE ${SQL.column(indexColumnName)} = ${SQL.literal(nullIndex)}
        `);
        nullIndex -= 1;
      }
      legend.push({
        label: "(null)",
        color: "#aaaaaa",
        predicate: SQL.isNull(SQL.column(column)),
        count: nullCount,
      });
    }

    return {
      indexColumn: indexColumnName,
      legend: legend,
    };
  }

  async makeBinnedNumericColumn(column: string): Promise<EmbeddingLegend> {
    let stats = await this.queryOne(
      SQL.Query.from(this.table)
        .select({
          count: SQL.count(),
          min: SQL.min(SQL.column(column)),
          max: SQL.max(SQL.column(column)),
          mean: SQL.avg(SQL.column(column)),
          median: SQL.median(SQL.column(column)),
        })
        .where(SQL.isFinite(SQL.column(column))),
    );
    let binning = inferBinning(stats);
    let indexColumnName = `_ev_${column}_id`;

    let expr: SQL.ExprNode = SQL.cast(SQL.column(column), "DOUBLE");
    expr = binning.scale.expr(expr, binning.scale.constant ?? 0);
    let binIndexExpr = SQL.cond(
      SQL.isFinite(SQL.cast(SQL.column(column), "DOUBLE")),
      SQL.floor(SQL.mul(SQL.sub(expr, binning.binStart), 1 / binning.binSize)),
      SQL.literal(null),
    );

    await this.exec(SQL.sql`
      ALTER TABLE ${this.table} ADD COLUMN IF NOT EXISTS ${SQL.column(indexColumnName)} INTEGER DEFAULT 0;
      UPDATE ${this.table}
      SET ${SQL.column(indexColumnName)} = ${binIndexExpr}
    `);

    // Count by index.
    let counts = await this.query(SQL.sql`
      SELECT ${SQL.column(indexColumnName)} AS index, COUNT(*)::INT AS count
      FROM ${this.table}
      GROUP BY ${SQL.column(indexColumnName)}
      ORDER BY ${SQL.column(indexColumnName)} ASC
    `);

    let minIndex = null;
    let maxIndex = null;
    let index2Count = new Map<number | null, number>();

    let reverse = (x: number) => binning.scale.reverse(x, binning.scale.constant ?? 0);
    for (let { index, count } of counts as { index: number | null; count: number }[]) {
      if (index != null) {
        if (minIndex == null || index < minIndex) {
          minIndex = index;
        }
        if (maxIndex == null || index > maxIndex) {
          maxIndex = index;
        }
      }
      index2Count.set(index, count);
    }

    let legend: EmbeddingLegend["legend"] = [];

    let fmt = format(".6");

    if (minIndex != null && maxIndex != null) {
      let colors = defaultOrdinalColors(maxIndex - minIndex + 1);
      for (let index = minIndex; index <= maxIndex; index++) {
        let lowerBound = reverse(index * binning.binSize + binning.binStart);
        let upperBound = reverse((index + 1) * binning.binSize + binning.binStart);
        legend.push({
          label: `[${fmt(lowerBound)}, ${fmt(upperBound)})`,
          color: colors[index - minIndex],
          predicate: SQL.eq(binIndexExpr, SQL.literal(index)),
          count: index2Count.get(index) ?? 0,
        });
      }
    }

    if (index2Count.has(null)) {
      let nullIndex = legend.length;
      await this.exec(`
        UPDATE ${this.table}
        SET ${SQL.column(indexColumnName)} = ${SQL.literal(nullIndex)}
        WHERE ${SQL.column(indexColumnName)} IS NULL
      `);
      legend.push({
        label: "(null / nan / inf)",
        color: "#aaaaaa",
        predicate: SQL.isNull(binIndexExpr),
        count: index2Count.get(null) ?? 0,
      });
    }

    return {
      indexColumn: indexColumnName,
      legend: legend,
    };
  }
}

export type JSColumnType = "string" | "number" | "string[]";

export function jsTypeFromDBType(dbType: string): JSColumnType | null {
  if (numberTypes.has(dbType)) {
    return "number";
  } else if (stringTypes.has(dbType)) {
    return "string";
  } else if (dbType.match(/^(VARCHAR|TEXT)\[\d*\]$/)) {
    return "string[]";
  } else {
    return null;
  }
}

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

const stringTypes = new Set(["BOOLEAN", "DATE", "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING"]);
