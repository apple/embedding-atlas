"""Generate N random points uniform over a Europe bbox + benchmark backends.

Backends benchmarked:
  * DuckDB  — `random()` SQL + COPY TO parquet (zstd)
  * Polars  — `pl.int_range`/`pl.Series` random + write_parquet (zstd)

Usage::

    uv run --with polars python scripts/bench/gen_points.py \
        --rows 50_000_000 --backend duckdb --out /tmp/p_duck.parquet
    uv run --with polars python scripts/bench/gen_points.py \
        --rows 50_000_000 --backend polars --out /tmp/p_polars.parquet

Schema (kept identical across backends so downstream perf is comparable):
  id        BIGINT       sequential row index (matches the loader's __row_index__)
  lon       DOUBLE       uniform on [LON_MIN, LON_MAX]
  lat       DOUBLE       uniform on [LAT_MIN, LAT_MAX]
  category  VARCHAR      one of 8 letters (uniform); for color-by tests
  value     DOUBLE       uniform on [0, 100]; for histogram tests

Reports wall time, peak RSS (resident, in MB), and output file size.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

# Europe-ish bbox (mainland EU + UK/IS, conservative).
LON_MIN, LON_MAX = -10.0, 35.0
LAT_MIN, LAT_MAX = 35.0, 70.0
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def fmt_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    return f"{n / (1 << 10):.2f} KiB"


def peak_rss_mib() -> float:
    # ru_maxrss is in bytes on macOS, kB on Linux. Detect by magnitude.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024  # KB → MiB


def gen_duckdb(rows: int, out: str) -> None:
    import duckdb

    con = duckdb.connect(":memory:")
    # Use OS tmp dir for spilling — the loader does the same in production.
    con.sql("SET temp_directory = '/tmp/duckdb_bench'")
    con.sql("PRAGMA threads=8")
    sql = f"""
        COPY (
          SELECT
            i AS id,
            {LON_MIN} + random() * {LON_MAX - LON_MIN} AS lon,
            {LAT_MIN} + random() * {LAT_MAX - LAT_MIN} AS lat,
            ['A','B','C','D','E','F','G','H'][1 + (i % 8)] AS category,
            random() * 100.0 AS value
          FROM range({rows}) t(i)
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
    """
    con.sql(sql)


def gen_polars(rows: int, out: str) -> None:
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Chunked, streamed write keeps peak RSS bounded for the 300M case
    # (one chunk at 25M is ~750 MB total of ndarray buffers, freed each pass).
    chunk = 25_000_000
    rng = np.random.default_rng(seed=0xA71A5)  # deterministic
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("lon", pa.float64()),
            ("lat", pa.float64()),
            ("category", pa.string()),
            ("value", pa.float64()),
        ]
    )
    cats = np.array(CATEGORIES)
    writer = pq.ParquetWriter(out, schema, compression="zstd", use_dictionary=True)
    try:
        written = 0
        while written < rows:
            n = min(chunk, rows - written)
            ids = np.arange(written, written + n, dtype=np.int64)
            lons = rng.uniform(LON_MIN, LON_MAX, n)
            lats = rng.uniform(LAT_MIN, LAT_MAX, n)
            cat_idx = (ids % 8).astype(np.int64)
            categories = cats[cat_idx]  # vectorised string lookup
            values = rng.uniform(0.0, 100.0, n)
            df = pl.DataFrame(
                {
                    "id": ids,
                    "lon": lons,
                    "lat": lats,
                    "category": categories,
                    "value": values,
                }
            )
            tbl = df.to_arrow().cast(schema)
            writer.write_table(tbl, row_group_size=1_000_000)
            written += n
            del ids, lons, lats, cat_idx, categories, values, df, tbl
    finally:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--backend", choices=["duckdb", "polars"], required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = args.out
    if Path(out).exists():
        Path(out).unlink()

    t0 = time.perf_counter()
    if args.backend == "duckdb":
        gen_duckdb(args.rows, out)
    else:
        gen_polars(args.rows, out)
    elapsed = time.perf_counter() - t0

    size = Path(out).stat().st_size
    rss = peak_rss_mib()
    print(
        f"backend={args.backend:7s}  rows={args.rows:>12,}  "
        f"time={elapsed:6.2f}s  peakRSS={rss:7.1f} MiB  "
        f"file={fmt_size(size)}  ({size / args.rows:.2f} B/row)"
    )


if __name__ == "__main__":
    main()
