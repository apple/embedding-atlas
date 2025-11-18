// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { defaultCategoryColors } from "@embedding-atlas/component";
import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import * as d3 from "d3";

import { distinctCount, jsTypeFromDBType } from "../../utils/database.js";
import { inferBinning } from "../common/binning.js";
import { defaultOrdinalColors } from "../common/theme.js";

export interface EmbeddingLegend {
  indexColumn: string;
  legend: {
    label: string;
    color: string;
    predicate: any;
    count: number;
  }[];
}

export async function makeCategoryColumn(
  coordinator: Coordinator,
  table: string,
  column: string | null | undefined,
): Promise<EmbeddingLegend | null> {
  if (column == null) {
    return null;
  }
  let [desc] = Array.from(await coordinator.query(SQL.Query.describe(SQL.Query.from(table).select(column))));
  if (desc == null) {
    return null;
  }
  let jsType = jsTypeFromDBType(desc.column_type);
  if (jsType == "string") {
    return await makeDiscreteCategoryColumn(coordinator, table, column, 10);
  } else if (jsType == "number") {
    let distinct = await distinctCount(coordinator, table, column);
    if (distinct <= 10) {
      return await makeDiscreteCategoryColumn(coordinator, table, column, 10);
    } else {
      return await makeBinnedNumericColumn(coordinator, table, column);
    }
  }
  return null;
}

async function makeDiscreteCategoryColumn(
  coordinator: Coordinator,
  table: string,
  column: string,
  maxCategories: number,
): Promise<EmbeddingLegend> {
  let indexColumnName = `_ev_${column}_id`;
  let values = Array.from(
    await coordinator.query(
      SQL.Query.from(table)
        .select({ value: SQL.cast(SQL.column(column), "TEXT"), count: SQL.count() })
        .where(SQL.not(SQL.isNull(SQL.cast(SQL.column(column), "TEXT"))))
        .groupby(SQL.cast(SQL.column(column), "TEXT"))
        .orderby(SQL.desc(SQL.count()))
        .limit(maxCategories),
    ),
  ) as { value: string; count: number }[];

  let otherIndex = values.length;
  let nullIndex = values.length + 1;

  // Add the index column.
  await coordinator.exec(`
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(indexColumnName)} INTEGER DEFAULT 0;
    UPDATE ${table}
    SET ${SQL.column(indexColumnName)} = CASE ${SQL.column(column)}::TEXT
          ${values.map(({ value }, i) => SQL.sql`WHEN ${SQL.literal(value)} THEN ${SQL.literal(i)}`).join(" ")}
          ELSE (CASE WHEN ${SQL.column(column)} IS NULL THEN ${SQL.literal(nullIndex)} ELSE ${SQL.literal(otherIndex)} END) END
  `);

  // Count by index.
  let counts = Array.from(
    await coordinator.query(
      SQL.Query.from(table)
        .select({ index: SQL.column(indexColumnName), count: SQL.cast(SQL.count(), "INT") })
        .groupby(SQL.column(indexColumnName)),
    ),
  );
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
    let { otherCategoryCount } = (
      await coordinator.query(`
        SELECT COUNT(DISTINCT(${SQL.column(column)}::TEXT)) AS otherCategoryCount
        FROM ${table}
        WHERE ${SQL.column(indexColumnName)} = ${SQL.literal(otherIndex)} AND ${SQL.column(column)} IS NOT NULL
    `)
    ).get(0);
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
      await coordinator.exec(`
          UPDATE ${table}
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

async function makeBinnedNumericColumn(
  coordinator: Coordinator,
  table: string,
  column: string,
): Promise<EmbeddingLegend> {
  let stats = (
    await coordinator.query(
      SQL.Query.from(table)
        .select({
          count: SQL.count(),
          min: SQL.min(SQL.column(column)),
          max: SQL.max(SQL.column(column)),
          mean: SQL.avg(SQL.column(column)),
          median: SQL.median(SQL.column(column)),
        })
        .where(SQL.isFinite(SQL.column(column))),
    )
  ).get(0) as any;
  let binning = inferBinning(stats);
  let indexColumnName = `_ev_${column}_id`;

  let expr: SQL.ExprNode = SQL.cast(SQL.column(column), "DOUBLE");
  expr = binning.scale.expr(expr, binning.scale.constant ?? 0);
  let binIndexExpr = SQL.cond(
    SQL.isFinite(SQL.cast(SQL.column(column), "DOUBLE")),
    SQL.floor(SQL.mul(SQL.sub(expr, binning.binStart), 1 / binning.binSize)),
    SQL.literal(null),
  );

  await coordinator.exec(`
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(indexColumnName)} INTEGER DEFAULT 0;
    UPDATE ${table}
    SET ${SQL.column(indexColumnName)} = ${binIndexExpr}
  `);

  // Count by index.
  let counts = Array.from(
    await coordinator.query(`
      SELECT ${SQL.column(indexColumnName)} AS index, COUNT(*)::INT AS count
      FROM ${table}
      GROUP BY ${SQL.column(indexColumnName)}
      ORDER BY ${SQL.column(indexColumnName)} ASC
    `),
  );

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

  let fmt = d3.format(".6");

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
    await coordinator.exec(`
        UPDATE ${table}
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
