// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { boundingRect, type Point, type Rectangle } from "../utils.js";
import type { DataField, DataPoint, DataPointID } from "./types.js";

export function predicateForDataPoints(
  source: { x: string; y: string; z?: string | null; identifier?: string | null; category?: string | null },
  points: DataPoint[],
) {
  if (points.length == 0) {
    return SQL.literal(false);
  }
  // A configured identifier is the SEMANTIC key: stable across views, joins, reloads,
  // and other coordinated clients, so it always wins. rowid is only a physical,
  // table-local fallback for exact 3D-pick identity when NO identifier exists (so
  // duplicate/stacked points sharing x/y/z/category are not over-selected); it must
  // never be emitted as a cross-client predicate when an identifier is configured.
  if (source.identifier != null) {
    let identifier = source.identifier;
    return SQL.or(...points.map((p) => SQL.eq(SQL.column(identifier), SQL.literal(p.identifier))));
  } else if (points.every((p) => p.rowid != null)) {
    return SQL.or(...points.map((p) => SQL.sql`rowid = ${SQL.literal(p.rowid)}`));
  } else {
    let { x, y, z, category } = source;
    return SQL.or(
      ...points.map((p) => {
        let conditions = [
          SQL.eq(SQL.cast(SQL.column(x), "DOUBLE"), SQL.literal(p.x)),
          SQL.eq(SQL.cast(SQL.column(y), "DOUBLE"), SQL.literal(p.y)),
        ];
        // Include z so points that share (x, y, category) but differ in depth are
        // not conflated into the same coordinated selection.
        if (z != null && p.z != null) {
          conditions.push(SQL.eq(SQL.cast(SQL.column(z), "DOUBLE"), SQL.literal(p.z)));
        }
        if (category != null) {
          conditions.push(SQL.eq(SQL.cast(SQL.column(category), "INTEGER"), SQL.literal(p.category)));
        }
        return SQL.and(...conditions);
      }),
    );
  }
}

function pointInPolygonPredicate(x: SQL.ExprNode, y: SQL.ExprNode, polygon: Point[]) {
  // Equavilant algorithm:
  // let counter = 0;
  // for each edge (x1, y1) to (x2, y2)) {
  //   if ((y1 <= y) && (y < y2) || (y2 <= y) && (y < y1)) { // -> pred1
  //     if ((x2 - x1) * (y - y1) / (y2 - y1) + x1 < x) {    // -> pred2
  //       counter += 1;
  //     }
  //   }
  // }
  // return counter % 2 == 1;

  let parts: SQL.ExprNode[] = [];
  for (let i = 0; i < polygon.length; i++) {
    let j = (i + 1) % polygon.length;
    let { x: x1, y: y1 } = polygon[i];
    let { x: x2, y: y2 } = polygon[j];
    let pred1 =
      y1 < y2
        ? SQL.and(SQL.lte(SQL.literal(y1), y), SQL.lt(y, SQL.literal(y2)))
        : SQL.and(SQL.lte(SQL.literal(y2), y), SQL.lt(y, SQL.literal(y1)));
    let pred2 = (y1 < y2 ? SQL.lt : SQL.gt)(
      SQL.sub(SQL.mul(SQL.literal(x2 - x1), y), SQL.mul(SQL.literal(y2 - y1), x)),
      SQL.literal((x2 - x1) * y1 - (y2 - y1) * x1),
    );
    parts.push(SQL.cast(SQL.and(pred1, pred2), "INT"));
  }
  let sum = parts.reduce((a, b) => SQL.add(a, b));
  return SQL.eq(SQL.mod(sum, SQL.literal(2)), SQL.literal(1));
}

export function predicateForRangeSelection(source: { x: string; y: string }, range: Rectangle | Point[]) {
  if (range instanceof Array) {
    if (range.length < 3) {
      // Degenerate case.
      return SQL.literal(false);
    }
    let bounds = boundingRect(range);
    return SQL.and(
      SQL.isBetween(SQL.column(source.x), [bounds.xMin, bounds.xMax]),
      SQL.isBetween(SQL.column(source.y), [bounds.yMin, bounds.yMax]),
      pointInPolygonPredicate(SQL.column(source.x), SQL.column(source.y), range),
    );
  } else {
    return SQL.and(
      SQL.isBetween(SQL.column(source.x), [range.xMin, range.xMax]),
      SQL.isBetween(SQL.column(source.y), [range.yMin, range.yMax]),
    );
  }
}

export async function queryApproximateDensity(
  coordinator: Coordinator,
  source: { table: string; x: string; y: string; category?: string | null },
): Promise<{
  centerX: number;
  centerY: number;
  scaler: number;
  totalCount: number;
  categoryCount: number;
  maxDensity: number;
}> {
  let { x, y, table } = source;
  // Find the view transform that fits all data points in a square view.
  let r = await coordinator.query(
    SQL.Query.from(table).select({
      centerX: SQL.sql`MEDIAN(${SQL.column(x)})`,
      centerY: SQL.sql`MEDIAN(${SQL.column(y)})`,
      stdX: SQL.sql`STDDEV(${SQL.column(x)})`,
      stdY: SQL.sql`STDDEV(${SQL.column(y)})`,
      ...(source.category != null
        ? {
            maxCategory: SQL.sql`MAX(${SQL.column(source.category)}::UTINYINT)`,
          }
        : {}),
    }),
  );
  let { centerX, centerY, stdX, stdY, maxCategory } = r.get(0);
  let scaler = 1.0 / (Math.max(stdX, stdY, 1e-3) * 3);

  // Estimate maximum density.
  // This is the approximate max number of points per square unit in data dimensions.
  let binWidth = 0.1 / scaler;
  let xBinClause = SQL.sql`FLOOR((${SQL.column(x)} - ${centerX}) / ${binWidth})`;
  let yBinClause = SQL.sql`FLOOR((${SQL.column(y)} - ${centerY}) / ${binWidth})`;
  let categoryClause = source.category != null ? SQL.column(source.category) : null;
  let groupby = categoryClause != null ? [xBinClause, yBinClause, categoryClause] : [xBinClause, yBinClause];
  let q = SQL.Query.from(
    SQL.Query.from(table)
      .select({ count: SQL.sql`COUNT(*)` })
      .groupby(...groupby),
  ).select({
    totalCount: SQL.sql`SUM(count)::INT`,
    maxCount: SQL.sql`MAX(count)::INT`,
  });

  r = await coordinator.query(q);
  let { maxCount, totalCount } = r.get(0);
  let maxDensity = maxCount / (binWidth * binWidth);

  return {
    centerX,
    centerY,
    scaler,
    totalCount,
    categoryCount: (maxCategory ?? 0) + 1,
    maxDensity,
  };
}

export interface DataPointQuerySource {
  table: string;
  x: string;
  y: string;
  z?: string | null;
  category?: string | null;
  text?: string | null;
  identifier?: string | null;
  additionalFields?: Record<string, DataField> | null;
}

export class DataPointQuery {
  coordinator: Coordinator;
  source: DataPointQuerySource;
  lastDistance: number;
  selectParams: Record<string, SQL.ExprNode>;

  constructor(coordinator: Coordinator, source: DataPointQuerySource) {
    this.coordinator = coordinator;
    this.source = source;
    this.lastDistance = 0;

    let { x, y, z, category, text, identifier } = this.source;
    let fieldExpressions: Record<string, SQL.ExprNode> = {};
    let fields = source.additionalFields ?? {};
    for (let key in fields) {
      let spec = fields[key];
      if (typeof spec == "string") {
        fieldExpressions["field_" + key] = SQL.column(spec);
      } else {
        fieldExpressions["field_" + key] = SQL.sql`${spec.sql}`;
      }
    }
    this.selectParams = {
      x: SQL.sql`${SQL.column(x)}::DOUBLE`,
      y: SQL.sql`${SQL.column(y)}::DOUBLE`,
      ...(z != null ? { z: SQL.sql`${SQL.column(z)}::DOUBLE` } : {}),
      ...(category != null ? { category: SQL.sql`${SQL.column(category)}::INT` } : {}),
      ...(text != null ? { text: SQL.sql`${SQL.column(text)}` } : {}),
      ...(identifier != null ? { identifier: SQL.sql`${SQL.column(identifier)}` } : {}),
      ...fieldExpressions,
    };
  }

  _convertToDataPoint(row: any): DataPoint {
    let fields: Record<string, any> = {};
    for (let key in row) {
      if (key.startsWith("field_")) {
        fields[key.slice("field_".length)] = row[key];
      }
    }
    return {
      x: row.x,
      y: row.y,
      z: row.z,
      category: row.category,
      text: row.text,
      identifier: row.identifier,
      fields: fields,
    };
  }

  async queryClosestPoint(
    predicate: any | null,
    px: number,
    py: number,
    unitDistance: number,
  ): Promise<DataPoint | null> {
    let rMax = unitDistance * 12;
    let { x, y } = this.source;

    for (let r of [this.lastDistance, rMax]) {
      if (r == 0 || r > rMax) {
        continue;
      }
      let q = SQL.Query.from(this.source.table).select(this.selectParams);
      q = q.where(SQL.sql`${SQL.column(x)} BETWEEN ${px - r} AND ${px + r}`);
      q = q.where(SQL.sql`${SQL.column(y)} BETWEEN ${py - r} AND ${py + r}`);
      if (predicate) {
        q = q.where(predicate);
      }
      q = q.orderby(SQL.sql`(x - (${px}))**2 + (y - (${py}))**2`).limit(1);
      let result = await this.coordinator.query(q);
      let point = result.get(0);
      if (point) {
        this.lastDistance = Math.max(Math.abs(point.x - px), Math.abs(point.y - py)) * 4;
        return this._convertToDataPoint(point);
      }
    }
    return null;
  }

  /**
   * Like {@link queryClosestPoint} but disambiguates by z as well, so points that
   * stack at the same (x, y) but differ in depth resolve to the correct row.
   * Used as the 3D pick fallback when no stable identifier column is available.
   */
  async queryClosestPoint3D(
    predicate: any | null,
    px: number,
    py: number,
    pz: number,
    unitDistance: number,
  ): Promise<DataPoint | null> {
    let { x, y, z } = this.source;
    if (z == null) {
      return this.queryClosestPoint(predicate, px, py, unitDistance);
    }
    let rMax = unitDistance * 12;

    for (let r of [this.lastDistance, rMax]) {
      if (r == 0 || r > rMax) {
        continue;
      }
      let q = SQL.Query.from(this.source.table).select(this.selectParams);
      // px/py/pz are the RENDERED Float32 coordinates of the picked point. Compare
      // against the FLOAT-cast source columns on all three axes so the picked row
      // matches its rendered values exactly: the x/y pixel-derived radius `r` is not
      // a reliable tolerance when coordinates have large offsets / small spread (the
      // source DOUBLE can differ from the Float32 by more than r) or when z is on a
      // different scale. The orderby below still resolves the nearest in true 3D
      // distance among the candidates.
      // z gets a tolerance scaled to its OWN magnitude (not the x/y pixel radius `r`),
      // so a large-offset or differently-scaled z still brackets the picked row. This
      // only widens the candidate set — CAST(z AS FLOAT) equals pz exactly for the
      // picked row, and the orderby below still selects the nearest in 3D distance.
      let rz = Math.max(r, Math.abs(pz) * 1e-4 + 1e-4);
      q = q.where(SQL.sql`CAST(${SQL.column(x)} AS FLOAT) BETWEEN ${px - r} AND ${px + r}`);
      q = q.where(SQL.sql`CAST(${SQL.column(y)} AS FLOAT) BETWEEN ${py - r} AND ${py + r}`);
      q = q.where(SQL.sql`CAST(${SQL.column(z)} AS FLOAT) BETWEEN ${pz - rz} AND ${pz + rz}`);
      if (predicate) {
        q = q.where(predicate);
      }
      q = q.orderby(SQL.sql`(x - (${px}))**2 + (y - (${py}))**2 + (z - (${pz}))**2`).limit(1);
      let result = await this.coordinator.query(q);
      let point = result.get(0);
      if (point) {
        this.lastDistance = Math.max(Math.abs(point.x - px), Math.abs(point.y - py), Math.abs(point.z - pz)) * 4;
        return this._convertToDataPoint(point);
      }
    }
    return null;
  }

  async queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    let { table, identifier } = this.source;
    if (identifier == null) {
      return [];
    }
    let q = SQL.Query.from(table).select(this.selectParams);
    q = q.where(
      SQL.isIn(
        SQL.column(identifier),
        identifiers.map((x) => SQL.literal(x)),
      ),
    );
    let result = Array.from(await this.coordinator.query(q));
    return result.map((row) => this._convertToDataPoint(row));
  }

  /**
   * Resolves the exact row for a DuckDB `rowid` (used by 3D pick-by-index). Optionally
   * constrained to the active cross-filter so a row filtered out while the pick
   * resolved is not resurrected. Returns null if no such row matches.
   */
  async queryByRowId(rowid: bigint, predicate?: any | null): Promise<DataPoint | null> {
    let q = SQL.Query.from(this.source.table).select(this.selectParams);
    q = q.where(SQL.sql`rowid = ${SQL.literal(rowid)}`);
    if (predicate) {
      q = q.where(predicate);
    }
    let result = await this.coordinator.query(q.limit(1));
    let row = result.get(0);
    if (row == null) {
      return null;
    }
    let point = this._convertToDataPoint(row);
    // Carry the rowid so the coordinated-selection predicate can target this exact
    // row even without a user identifier (rowids fit within 2^53 -> Number is exact).
    point.rowid = Number(rowid);
    return point;
  }

  /**
   * Resolves the exact row for a configured identifier value (the 3D pick-by-index
   * fallback for rowid-less sources such as views/joins). Optionally constrained to the
   * active cross-filter. Returns null if no such row matches (the source has no
   * identifier, the row was filtered out, or the value is unknown).
   */
  async queryByIdentifier(id: DataPointID, predicate?: any | null): Promise<DataPoint | null> {
    let identifier = this.source.identifier;
    if (identifier == null) {
      return null;
    }
    let q = SQL.Query.from(this.source.table).select(this.selectParams);
    q = q.where(SQL.eq(SQL.column(identifier), SQL.literal(id)));
    if (predicate) {
      q = q.where(predicate);
    }
    let result = await this.coordinator.query(q.limit(1));
    let row = result.get(0);
    return row != null ? this._convertToDataPoint(row) : null;
  }
}
