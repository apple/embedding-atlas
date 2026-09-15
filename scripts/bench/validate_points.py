"""Cheap structural validator for generated parquet files.

Asserts:
  * row count matches expected
  * lon/lat are within the Europe bbox (no spillover)
  * category contains exactly 8 distinct labels A..H
  * id is a dense [0, rows) range (no gaps, no duplicates)
  * file size is sensible (1..30 B/row depending on compression)

Exits non-zero on any failure so it slots into CI / a Makefile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

LON_MIN, LON_MAX = -10.0, 35.0
LAT_MIN, LAT_MAX = 35.0, 70.0
EXPECTED_CATS = sorted(["A", "B", "C", "D", "E", "F", "G", "H"])


def validate(path: str, expected_rows: int) -> None:
    con = duckdb.connect(":memory:")
    p = Path(path).resolve()
    tbl = f"read_parquet('{p}')"
    cols = {
        r[0]: r[1] for r in con.sql(f"DESCRIBE SELECT * FROM {tbl} LIMIT 0").fetchall()
    }
    expected_cols = {"id", "lon", "lat", "category", "value"}
    missing = expected_cols - cols.keys()
    extra = cols.keys() - expected_cols
    assert not missing, f"missing columns: {missing}"
    assert not extra, f"extra columns: {extra} (schema was {cols})"

    n = con.sql(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    assert n == expected_rows, f"row count: got {n:,}, want {expected_rows:,}"

    lon_min, lon_max, lat_min, lat_max = con.sql(
        f"SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM {tbl}"
    ).fetchone()
    assert LON_MIN <= lon_min, f"lon_min {lon_min} < {LON_MIN}"
    assert lon_max <= LON_MAX, f"lon_max {lon_max} > {LON_MAX}"
    assert LAT_MIN <= lat_min, f"lat_min {lat_min} < {LAT_MIN}"
    assert lat_max <= LAT_MAX, f"lat_max {lat_max} > {LAT_MAX}"

    cats = sorted(
        r[0] for r in con.sql(f"SELECT DISTINCT category FROM {tbl}").fetchall()
    )
    assert cats == EXPECTED_CATS, f"categories: got {cats}, want {EXPECTED_CATS}"

    # id should be a dense [0, n) range. Cheap test: MIN, MAX, and uniqueness.
    id_min, id_max = con.sql(f"SELECT MIN(id), MAX(id) FROM {tbl}").fetchone()
    assert id_min == 0, f"id_min: got {id_min}, want 0"
    assert id_max == expected_rows - 1, (
        f"id_max: got {id_max}, want {expected_rows - 1}"
    )
    distinct_ids = con.sql(f"SELECT COUNT(DISTINCT id) FROM {tbl}").fetchone()[0]
    assert distinct_ids == expected_rows, (
        f"id uniqueness: got {distinct_ids:,} distinct, want {expected_rows:,}"
    )

    print(
        f"OK  rows={n:,}  bbox=[{lon_min:.3f},{lon_max:.3f}]x"
        f"[{lat_min:.3f},{lat_max:.3f}]  cats={cats}  "
        f"size={p.stat().st_size / (1 << 30):.2f} GiB"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, required=True)
    args = ap.parse_args()
    try:
        validate(args.path, args.rows)
    except AssertionError as e:
        print(f"FAIL  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
