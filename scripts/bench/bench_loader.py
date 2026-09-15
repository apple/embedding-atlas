"""Time each stage of fast_load_parquet (and a few candidate alternatives).

Stages reported (matches the loader's progress callbacks):
  setup        connect, set memory_limit/temp_directory, install spatial
  describe     parse parquet footer (DESCRIBE)
  ctas         CREATE OR REPLACE TABLE … AS SELECT … (the heavy step)
  count        SELECT COUNT(*) FROM dataset
  bounds       4× MIN/MAX in one query

Alternatives tried for the materialisation step:
  ctas_default   what the loader does today (full TABLE materialisation)
  view           CREATE VIEW (cheap but breaks ALTER-on-VIEW; we still measure)
  ctas_lazy_id   TABLE materialisation but write file_row_number lazily as a
                 generated column rather than physical bytes (DuckDB rejects
                 generated cols on tables, so this falls back; benchmarked
                 only to confirm it isn't viable)

Usage:
  uv --directory packages/backend run python scripts/bench/bench_loader.py \
      --path /tmp/gsa_bench/europe_300m.parquet --variant ctas_default
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import duckdb

LON_MIN, LON_MAX = -10.0, 35.0
LAT_MIN, LAT_MAX = 35.0, 70.0


def fmt(s: float) -> str:
    return f"{s:6.2f}s"


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


class Timer:
    def __init__(self) -> None:
        self.stages: list[tuple[str, float]] = []
        self._t0 = 0.0
        self._cur = ""

    def start(self, name: str) -> None:
        self._cur = name
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        elapsed = time.perf_counter() - self._t0
        self.stages.append((self._cur, elapsed))

    def report(self, header: str = "") -> None:
        if header:
            print(header)
        total = sum(t for _, t in self.stages)
        for name, t in self.stages:
            print(f"  {name:14s} {fmt(t)}  ({100 * t / total:4.1f}%)")
        print(f"  {'TOTAL':14s} {fmt(total)}  peakRSS={peak_rss_mib():7.1f} MiB")


def make_con(threads: int = 8) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.sql(f"SET memory_limit = '32GB'")
    con.sql(f"SET temp_directory = '/tmp/duckdb_bench'")
    con.sql(f"PRAGMA threads={threads}")
    return con


def variant_ctas_default(con: duckdb.DuckDBPyConnection, path: str, t: Timer) -> None:
    t.start("describe")
    con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0").fetchall()
    t.stop()
    t.start("ctas")
    con.sql(
        f"CREATE OR REPLACE TABLE dataset AS "
        f"SELECT * EXCLUDE (file_row_number), "
        f"file_row_number AS __row_index__ "
        f"FROM read_parquet('{path}', file_row_number=true)"
    )
    t.stop()
    t.start("count")
    con.sql("SELECT COUNT(*) FROM dataset").fetchone()
    t.stop()
    t.start("bounds")
    con.sql("SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM dataset").fetchone()
    t.stop()


def variant_view(con: duckdb.DuckDBPyConnection, path: str, t: Timer) -> None:
    t.start("describe")
    con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0").fetchall()
    t.stop()
    t.start("ctas")
    con.sql(
        f"CREATE OR REPLACE VIEW dataset AS "
        f"SELECT * EXCLUDE (file_row_number), "
        f"file_row_number AS __row_index__ "
        f"FROM read_parquet('{path}', file_row_number=true)"
    )
    t.stop()
    t.start("count")
    con.sql("SELECT COUNT(*) FROM dataset").fetchone()
    t.stop()
    t.start("bounds")
    con.sql("SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM dataset").fetchone()
    t.stop()


def variant_ctas_no_id(con: duckdb.DuckDBPyConnection, path: str, t: Timer) -> None:
    """No row-id materialised. Strictly faster; the viewer must compute ids
    elsewhere — known to be needed for the row-pointer in the data view."""
    t.start("describe")
    con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0").fetchall()
    t.stop()
    t.start("ctas")
    con.sql(f"CREATE OR REPLACE TABLE dataset AS SELECT * FROM read_parquet('{path}')")
    t.stop()
    t.start("count")
    con.sql("SELECT COUNT(*) FROM dataset").fetchone()
    t.stop()
    t.start("bounds")
    con.sql("SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM dataset").fetchone()
    t.stop()


def variant_ctas_window_id(con: duckdb.DuckDBPyConnection, path: str, t: Timer) -> None:
    """Use ROW_NUMBER() OVER () instead of file_row_number — what the loader
    falls back to when the file has a column literally named file_row_number."""
    t.start("describe")
    con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0").fetchall()
    t.stop()
    t.start("ctas")
    con.sql(
        f"CREATE OR REPLACE TABLE dataset AS "
        f"SELECT *, ROW_NUMBER() OVER () AS __row_index__ "
        f"FROM read_parquet('{path}')"
    )
    t.stop()
    t.start("count")
    con.sql("SELECT COUNT(*) FROM dataset").fetchone()
    t.stop()
    t.start("bounds")
    con.sql("SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM dataset").fetchone()
    t.stop()


VARIANTS = {
    "ctas_default": variant_ctas_default,
    "view": variant_view,
    "ctas_no_id": variant_ctas_no_id,
    "ctas_window_id": variant_ctas_window_id,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), default="ctas_default")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    t = Timer()
    t.start("setup")
    con = make_con(args.threads)
    try:
        con.sql("INSTALL spatial; LOAD spatial")
    except Exception:
        pass
    t.stop()
    VARIANTS[args.variant](con, args.path, t)
    t.report(f"=== variant={args.variant}  path={args.path} ===")


if __name__ == "__main__":
    main()
