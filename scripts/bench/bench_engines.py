"""Head-to-head benchmark: DuckDB vs Polars vs SedonaDB.

Stages benchmarked (the heavy compute stages of the atlas pipeline):

  count       SELECT COUNT(*) FROM parquet      — scan throughput / metadata
  bounds      SELECT MIN/MAX(lon, lat)          — aggregation, may use stats
  cat_count   SELECT category, COUNT(*) GROUP   — hash-aggregation
  scatter     SELECT lon, lat → Arrow Table     — projection + materialise
  scatter_q   SELECT u16-quantised x, y         — what the wire layer does

Each (engine, stage) is run N=3 times in interleaved order — engine A trial 1,
engine B trial 1, engine C trial 1, engine A trial 2, ... — so OS-page-cache
warmup is fairly distributed across engines. Each trial runs in a fresh
subprocess so RSS is the engine's own peak, not the parent's.

Usage::

  scripts/bench/bench_engines.py --path /tmp/gsa_bench/europe_300m.parquet

Result on /tmp/gsa_bench/europe_300m.parquet (300 M rows, 6.6 GiB,
warm OS cache, M-series macOS, sedonadb 0.3.0 / duckdb 1.5.2 /
polars 1.40.1; median of 3 trials):

    stage          duckdb    polars    sedonadb   winner
    count           0.04s     0.06s      0.08s     duckdb
    bounds          0.03s     0.64s      0.08s     duckdb
    cat_count       0.12s     0.45s      0.31s     duckdb
    scatter         0.89s     0.58s      0.77s     polars
    scatter_q       1.78s     0.91s      0.82s     sedonadb (1.1x over polars,
                                                            2.2x over duckdb)

Verdict: DuckDB stays. SedonaDB only wins on the projection-heavy stage
that is wire-bottlenecked end-to-end (1 s SQL win behind a 9 s Arrow IPC
HTTP send), and loses 2-3x on aggregation stages that are interactive
(color-by `cat_count` would visibly regress 0.12 s -> 0.31 s). DuckDB
also has a much leaner RSS profile (50-2.8 GB vs Polars' 5-9 GB and
SedonaDB's 0.1-9.4 GB). SedonaDB's published wins are all spatial
(ST_*, KNN, spatial joins) — none of which the atlas pipeline uses.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
PYBIN = "/tmp/sedona_bench/.venv/bin/python"  # scratch venv with all 3 engines


# -------------------------------------------------------------- worker ----


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def worker(engine: str, stage: str, path: str) -> dict:
    """Run a single (engine, stage) trial. Returns timing + result digest."""
    t0 = time.perf_counter()
    rows = digest = None

    if engine == "duckdb":
        import duckdb

        con = duckdb.connect(":memory:")
        con.sql("SET memory_limit = '32GB'")
        con.sql("SET temp_directory = '/tmp/duckdb_bench_engines'")
        con.sql("PRAGMA threads=8")
        con.sql(
            f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM read_parquet('{path}')"
        )
        if stage == "count":
            rows = con.sql("SELECT COUNT(*) FROM dataset").fetchone()[0]
            digest = rows
        elif stage == "bounds":
            digest = con.sql(
                "SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM dataset"
            ).fetchone()
        elif stage == "cat_count":
            digest = sorted(
                con.sql(
                    "SELECT category, COUNT(*) FROM dataset GROUP BY 1 ORDER BY 1"
                ).fetchall()
            )
        elif stage == "scatter":
            tbl = con.sql("SELECT lon, lat FROM dataset").fetch_arrow_table()
            rows = tbl.num_rows
            digest = (rows, tbl.nbytes)
        elif stage == "scatter_q":
            tbl = con.sql(
                "SELECT "
                "GREATEST(0, LEAST(65535, ROUND((lon - (-10.0))/45.0 * 65535)))::USMALLINT AS x, "
                "GREATEST(0, LEAST(65535, ROUND((lat - 35.0)/35.0 * 65535)))::USMALLINT AS y "
                "FROM dataset"
            ).fetch_arrow_table()
            rows = tbl.num_rows
            digest = (rows, tbl.nbytes)
        else:
            raise SystemExit(f"unknown stage: {stage}")

    elif engine == "polars":
        import polars as pl

        if stage == "count":
            rows = pl.scan_parquet(path).select(pl.len()).collect().item()
            digest = rows
        elif stage == "bounds":
            df = (
                pl.scan_parquet(path)
                .select(
                    pl.col("lon").min().alias("lon_min"),
                    pl.col("lon").max().alias("lon_max"),
                    pl.col("lat").min().alias("lat_min"),
                    pl.col("lat").max().alias("lat_max"),
                )
                .collect()
            )
            digest = tuple(df.row(0))
        elif stage == "cat_count":
            df = (
                pl.scan_parquet(path)
                .group_by("category")
                .len()
                .sort("category")
                .collect()
            )
            digest = sorted([(r[0], r[1]) for r in df.iter_rows()])
        elif stage == "scatter":
            df = pl.scan_parquet(path).select("lon", "lat").collect()
            tbl = df.to_arrow()
            rows = tbl.num_rows
            digest = (rows, tbl.nbytes)
        elif stage == "scatter_q":
            # Polars u16-quantise via expressions.
            df = (
                pl.scan_parquet(path)
                .select(
                    (
                        ((pl.col("lon") - (-10.0)) / 45.0 * 65535)
                        .round()
                        .clip(0, 65535)
                        .cast(pl.UInt16)
                        .alias("x")
                    ),
                    (
                        ((pl.col("lat") - 35.0) / 35.0 * 65535)
                        .round()
                        .clip(0, 65535)
                        .cast(pl.UInt16)
                        .alias("y")
                    ),
                )
                .collect()
            )
            tbl = df.to_arrow()
            rows = tbl.num_rows
            digest = (rows, tbl.nbytes)
        else:
            raise SystemExit(f"unknown stage: {stage}")

    elif engine == "sedonadb":
        import sedonadb

        con = sedonadb.connect()
        df = con.read_parquet(path)
        df.to_view("dataset", overwrite=True)
        if stage == "count":
            tbl = con.sql("SELECT COUNT(*) AS n FROM dataset").to_arrow_table()
            rows = tbl.column(0)[0].as_py()
            digest = rows
        elif stage == "bounds":
            tbl = con.sql(
                "SELECT MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM dataset"
            ).to_arrow_table()
            digest = tuple(tbl.column(i)[0].as_py() for i in range(4))
        elif stage == "cat_count":
            tbl = con.sql(
                "SELECT category, COUNT(*) AS n FROM dataset GROUP BY 1 ORDER BY 1"
            ).to_arrow_table()
            digest = sorted(
                [
                    (tbl.column(0)[i].as_py(), tbl.column(1)[i].as_py())
                    for i in range(tbl.num_rows)
                ]
            )
        elif stage == "scatter":
            tbl = con.sql("SELECT lon, lat FROM dataset").to_arrow_table()
            rows = tbl.num_rows
            digest = (rows, tbl.nbytes)
        elif stage == "scatter_q":
            # DataFusion's SQL surface lacks a UInt16 cast (`CAST AS UInt16`
            # rejects, `CAST AS SMALLINT` is i16 and overflows). The
            # `arrow_cast(..., 'UInt16')` UDF is the supported route.
            tbl = con.sql(
                "SELECT "
                "arrow_cast(LEAST(65535, GREATEST(0, ROUND((lon - (-10.0))/45.0 * 65535))), 'UInt16') AS x, "
                "arrow_cast(LEAST(65535, GREATEST(0, ROUND((lat -  35.0)/35.0 * 65535))), 'UInt16') AS y "
                "FROM dataset"
            ).to_arrow_table()
            rows = tbl.num_rows
            digest = (rows, tbl.nbytes)
        else:
            raise SystemExit(f"unknown stage: {stage}")
    else:
        raise SystemExit(f"unknown engine: {engine}")

    elapsed = time.perf_counter() - t0
    return {
        "engine": engine,
        "stage": stage,
        "elapsed_s": elapsed,
        "rss_mib": peak_rss_mib(),
        "digest_repr": repr(digest)[:200],
    }


# -------------------------------------------------------------- driver ----


def run_subprocess(engine: str, stage: str, path: str) -> dict:
    cmd = [
        PYBIN,
        str(THIS),
        "--worker",
        "--engine",
        engine,
        "--stage",
        stage,
        "--path",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        return {
            "engine": engine,
            "stage": stage,
            "elapsed_s": float("nan"),
            "rss_mib": 0,
            "digest_repr": "",
            "error": (out.stderr or out.stdout)[-400:],
        }
    # Last line of stdout is the JSON result.
    last = out.stdout.strip().splitlines()[-1]
    return json.loads(last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--engine", choices=["duckdb", "polars", "sedonadb"])
    ap.add_argument(
        "--stage", choices=["count", "bounds", "cat_count", "scatter", "scatter_q"]
    )
    ap.add_argument("--path", required=False)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["count", "bounds", "cat_count", "scatter_q", "scatter"],
    )
    ap.add_argument("--engines", nargs="+", default=["duckdb", "polars", "sedonadb"])
    args = ap.parse_args()

    if args.worker:
        result = worker(args.engine, args.stage, args.path)
        print(json.dumps(result))
        return

    # Parent: interleave (trial, stage, engine).
    if not args.path:
        raise SystemExit("--path required for parent invocation")
    results: list[dict] = []
    for trial in range(args.trials):
        for stage in args.stages:
            for engine in args.engines:
                print(
                    f"  trial {trial + 1}/{args.trials}  {stage:<10s}  {engine:<8s} ...",
                    end=" ",
                    flush=True,
                )
                r = run_subprocess(engine, stage, args.path)
                r["trial"] = trial
                results.append(r)
                if "error" in r:
                    print(f"ERROR: {r['error'][:120]}")
                else:
                    print(f"{r['elapsed_s']:6.2f}s  rss={r['rss_mib']:7.1f} MiB")

    # Aggregate: median per (stage, engine).
    print("\n=== summary (median over trials, all times in seconds) ===")
    print(
        f"{'stage':<12s} {'engine':<10s} {'median':>8s} {'min':>8s} {'max':>8s} {'rss_mib':>10s}  digest_repr"
    )
    for stage in args.stages:
        for engine in args.engines:
            triplet = [
                r
                for r in results
                if r["stage"] == stage and r["engine"] == engine and "error" not in r
            ]
            if not triplet:
                err = next(
                    (
                        r["error"]
                        for r in results
                        if r["stage"] == stage and r["engine"] == engine
                    ),
                    "no data",
                )
                print(f"{stage:<12s} {engine:<10s} {'ERR':>8s} -- {err[:60]}")
                continue
            elapsed = sorted(r["elapsed_s"] for r in triplet)
            med = elapsed[len(elapsed) // 2]
            rss = max(r["rss_mib"] for r in triplet)
            digest = triplet[0]["digest_repr"]
            print(
                f"{stage:<12s} {engine:<10s} {med:8.2f} {min(elapsed):8.2f} {max(elapsed):8.2f} {rss:10.1f}  {digest[:60]}"
            )

    # Pretty-print "winner per stage" table.
    print("\n=== winner per stage (lowest median) ===")
    for stage in args.stages:
        rows = []
        for engine in args.engines:
            triplet = [
                r
                for r in results
                if r["stage"] == stage and r["engine"] == engine and "error" not in r
            ]
            if triplet:
                elapsed = sorted(r["elapsed_s"] for r in triplet)
                med = elapsed[len(elapsed) // 2]
                rows.append((engine, med))
        if not rows:
            continue
        rows.sort(key=lambda x: x[1])
        win, runner = rows[0], rows[1] if len(rows) > 1 else (None, None)
        speedup = (runner[1] / win[1]) if (runner and runner[1]) else float("inf")
        rest = ", ".join(f"{e}={t:.2f}s" for e, t in rows[1:])
        print(
            f"  {stage:<12s} winner: {win[0]:<10s} {win[1]:6.2f}s  ({speedup:.2f}x faster than runner-up)  | others: {rest}"
        )


if __name__ == "__main__":
    main()
