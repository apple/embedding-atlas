"""Micro-bench candidate DuckDB optimisations against the 300M parquet.

Three families of experiments:

  A. scatter_q rewrites  — the per-render wire query (baseline 1.78s)
  B. cat_count rewrites   — per-color-switch top-N (baseline 0.12s)
  C. color-by indexing    — the per-color-switch ALTER+UPDATE...CASE pattern
                            currently used by category_column.ts (baseline ?s)

Each experiment runs in a fresh subprocess for clean RSS. 3 trials, median.
Threads pinned to 8 (matches loader). Constraint: no point may be dropped.

Usage::
  scripts/bench/bench_duckdb_opts.py --path /tmp/gsa_bench/europe_300m.parquet
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
PYBIN = "/tmp/sedona_bench/.venv/bin/python"


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def make_con(threads: int = 8):
    import duckdb

    con = duckdb.connect(":memory:")
    con.sql("SET memory_limit = '32GB'")
    con.sql("SET temp_directory = '/tmp/duckdb_bench_opts'")
    con.sql(f"PRAGMA threads={threads}")
    return con


# ============================================================== experiments


def setup_view(con, path: str) -> None:
    """The current loader path: VIEW, no row index, no pre-quantise."""
    con.sql(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM read_parquet('{path}')")


def setup_ctas(con, path: str) -> None:
    """Materialise into an in-memory table (the actual production loader path)."""
    con.sql(
        f"CREATE OR REPLACE TABLE dataset AS "
        f"SELECT * EXCLUDE (file_row_number), "
        f"file_row_number AS __row_index__ "
        f"FROM read_parquet('{path}', file_row_number=true)"
    )


def setup_ctas_pre_quantised(con, path: str) -> None:
    """CTAS that materialises u16 quantised x/y at load time.

    Bounds for the synthetic dataset are known: lon ∈ [-10, 35], lat ∈ [35, 70].
    Production would compute these in the bounds step then plug them in.
    """
    con.sql(
        f"CREATE OR REPLACE TABLE dataset AS "
        f"SELECT * EXCLUDE (file_row_number), "
        f"file_row_number AS __row_index__, "
        f"((lon - (-10.0)) * (65535.0/45.0))::USMALLINT AS __x_u16__, "
        f"((lat -  35.0)  * (65535.0/35.0))::USMALLINT AS __y_u16__ "
        f"FROM read_parquet('{path}', file_row_number=true)"
    )


def setup_ctas_enum(con, path: str) -> None:
    """CTAS with category column re-encoded as ENUM (8 cats, 1-byte ordinal)."""
    con.sql("CREATE OR REPLACE TYPE cat_enum AS ENUM ('A','B','C','D','E','F','G','H')")
    con.sql(
        f"CREATE OR REPLACE TABLE dataset AS "
        f"SELECT * EXCLUDE (category, file_row_number), "
        f"category::cat_enum AS category, "
        f"file_row_number AS __row_index__ "
        f"FROM read_parquet('{path}', file_row_number=true)"
    )


# --------------- A. scatter_q variants ----------------


SCATTER_Q_BASELINE = """
SELECT
  GREATEST(0, LEAST(65535, ROUND((lon - (-10.0))/45.0 * 65535)))::USMALLINT AS x,
  GREATEST(0, LEAST(65535, ROUND((lat -  35.0)/35.0 * 65535)))::USMALLINT AS y
FROM dataset
"""

SCATTER_Q_NO_CLAMP = """
SELECT
  ROUND((lon - (-10.0))/45.0 * 65535)::USMALLINT AS x,
  ROUND((lat -  35.0)/35.0 * 65535)::USMALLINT AS y
FROM dataset
"""

SCATTER_Q_FMA = """
SELECT
  ((lon + 10.0) * (65535.0/45.0))::USMALLINT AS x,
  ((lat - 35.0) * (65535.0/35.0))::USMALLINT AS y
FROM dataset
"""

SCATTER_Q_FLOOR = """
SELECT
  GREATEST(0, LEAST(65535, FLOOR((lon - (-10.0))/45.0 * 65535)))::USMALLINT AS x,
  GREATEST(0, LEAST(65535, FLOOR((lat -  35.0)/35.0 * 65535)))::USMALLINT AS y
FROM dataset
"""

SCATTER_Q_COMBINED = """
SELECT
  ((lon + 10.0) * 1456.333333333333)::USMALLINT AS x,
  ((lat - 35.0) * 1872.428571428571)::USMALLINT AS y
FROM dataset
"""

SCATTER_Q_FROM_PRECOMPUTED = """
SELECT __x_u16__ AS x, __y_u16__ AS y FROM dataset
"""


# --------------- B. cat_count variants ----------------


CAT_COUNT_VARCHAR = "SELECT category, COUNT(*) FROM dataset GROUP BY 1 ORDER BY 1"
CAT_COUNT_ENUM = "SELECT category, COUNT(*) FROM dataset GROUP BY 1 ORDER BY 1"


# --------------- C. color-by indexing ----------------

# Current viewer pattern, simulated:
#   (1) ALTER ADD COLUMN
#   (2) UPDATE ... CASE
#   (3) GROUP BY new col
# Replacement option: a VIEW that adds the index column lazily.


def color_by_alter_update(con):
    """Current viewer pattern: ALTER + UPDATE on 300M rows, then aggregate."""
    con.sql("ALTER TABLE dataset ADD COLUMN __cat_idx__ TINYINT")
    con.sql(
        "UPDATE dataset SET __cat_idx__ = CASE category "
        "WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 WHEN 'D' THEN 3 "
        "WHEN 'E' THEN 4 WHEN 'F' THEN 5 WHEN 'G' THEN 6 WHEN 'H' THEN 7 "
        "ELSE NULL END"
    )
    rows = con.sql("SELECT __cat_idx__, COUNT(*) FROM dataset GROUP BY 1").fetchall()
    return rows


def color_by_view_replacement(con):
    """A VIEW with the CASE expression. No UPDATE; aggregation is one scan."""
    con.sql("DROP VIEW IF EXISTS dataset_with_idx")
    con.sql(
        "CREATE VIEW dataset_with_idx AS "
        "SELECT *, CASE category "
        "WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 WHEN 'D' THEN 3 "
        "WHEN 'E' THEN 4 WHEN 'F' THEN 5 WHEN 'G' THEN 6 WHEN 'H' THEN 7 "
        "ELSE NULL END::TINYINT AS __cat_idx__ "
        "FROM dataset"
    )
    rows = con.sql(
        "SELECT __cat_idx__, COUNT(*) FROM dataset_with_idx GROUP BY 1"
    ).fetchall()
    return rows


def color_by_enum_ordinal(con):
    """If category is already ENUM (set up at CTAS time), the ordinal is 1 byte
    and GROUP BY is direct on that — no CASE, no UPDATE."""
    rows = con.sql(
        "SELECT category, COUNT(*) FROM dataset GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return rows


# ============================================================== worker


def worker(experiment: str, path: str) -> dict:
    t0_total = time.perf_counter()

    if experiment.startswith("scatter_q_"):
        con = make_con()
        if experiment == "scatter_q_pre":
            setup_ctas_pre_quantised(con, path)
            sql = SCATTER_Q_FROM_PRECOMPUTED
        else:
            setup_ctas(con, path)
            sql = {
                "scatter_q_baseline": SCATTER_Q_BASELINE,
                "scatter_q_noclamp": SCATTER_Q_NO_CLAMP,
                "scatter_q_fma": SCATTER_Q_FMA,
                "scatter_q_floor": SCATTER_Q_FLOOR,
                "scatter_q_combined": SCATTER_Q_COMBINED,
            }[experiment]
        # Time only the SQL execution, not the (one-time) CTAS.
        t = time.perf_counter()
        tbl = con.sql(sql).fetch_arrow_table()
        elapsed = time.perf_counter() - t
        digest = (tbl.num_rows, tbl.nbytes)

    elif experiment == "cat_count_varchar":
        con = make_con()
        setup_ctas(con, path)
        t = time.perf_counter()
        rows = con.sql(CAT_COUNT_VARCHAR).fetchall()
        elapsed = time.perf_counter() - t
        digest = sorted(rows)

    elif experiment == "cat_count_enum":
        con = make_con()
        setup_ctas_enum(con, path)
        t = time.perf_counter()
        rows = con.sql(CAT_COUNT_ENUM).fetchall()
        elapsed = time.perf_counter() - t
        digest = sorted([(str(r[0]), r[1]) for r in rows])

    elif experiment == "colorby_alter_update":
        con = make_con()
        setup_ctas(con, path)
        t = time.perf_counter()
        rows = color_by_alter_update(con)
        elapsed = time.perf_counter() - t
        digest = sorted(rows)

    elif experiment == "colorby_view":
        con = make_con()
        setup_ctas(con, path)
        t = time.perf_counter()
        rows = color_by_view_replacement(con)
        elapsed = time.perf_counter() - t
        digest = sorted(rows)

    elif experiment == "colorby_enum":
        con = make_con()
        setup_ctas_enum(con, path)
        t = time.perf_counter()
        rows = color_by_enum_ordinal(con)
        elapsed = time.perf_counter() - t
        digest = sorted([(str(r[0]), r[1]) for r in rows])

    elif experiment == "ctas_baseline":
        con = make_con()
        t = time.perf_counter()
        setup_ctas(con, path)
        elapsed = time.perf_counter() - t
        digest = "ctas_done"

    elif experiment == "ctas_pre_quantised":
        con = make_con()
        t = time.perf_counter()
        setup_ctas_pre_quantised(con, path)
        elapsed = time.perf_counter() - t
        digest = "ctas_done"

    elif experiment == "ctas_enum":
        con = make_con()
        t = time.perf_counter()
        setup_ctas_enum(con, path)
        elapsed = time.perf_counter() - t
        digest = "ctas_done"

    else:
        raise SystemExit(f"unknown experiment: {experiment}")

    return {
        "experiment": experiment,
        "elapsed_s": elapsed,
        "total_s": time.perf_counter() - t0_total,
        "rss_mib": peak_rss_mib(),
        "digest_repr": repr(digest)[:200],
    }


# ============================================================== driver


def run_subprocess(experiment: str, path: str) -> dict:
    cmd = [PYBIN, str(THIS), "--worker", "--experiment", experiment, "--path", path]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        return {
            "experiment": experiment,
            "elapsed_s": float("nan"),
            "rss_mib": 0,
            "digest_repr": "",
            "error": (out.stderr or out.stdout)[-400:],
        }
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--experiment")
    ap.add_argument("--path", required=False)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    if args.worker:
        print(json.dumps(worker(args.experiment, args.path)))
        return

    if not args.path:
        raise SystemExit("--path required")

    experiments = [
        # Group: CTAS variants (one-time loader cost)
        "ctas_baseline",
        "ctas_pre_quantised",
        "ctas_enum",
        # Group A: scatter_q rewrites
        "scatter_q_baseline",
        "scatter_q_noclamp",
        "scatter_q_fma",
        "scatter_q_floor",
        "scatter_q_combined",
        "scatter_q_pre",
        # Group B: cat_count
        "cat_count_varchar",
        "cat_count_enum",
        # Group C: colorby indexing
        "colorby_alter_update",
        "colorby_view",
        "colorby_enum",
    ]

    results: list[dict] = []
    for trial in range(args.trials):
        for exp in experiments:
            print(
                f"  trial {trial + 1}/{args.trials}  {exp:<24s} ...",
                end=" ",
                flush=True,
            )
            r = run_subprocess(exp, args.path)
            r["trial"] = trial
            results.append(r)
            if "error" in r:
                print(f"ERROR: {r['error'][:120]}")
            else:
                extra = f"  total={r.get('total_s', 0):5.2f}s" if "total_s" in r else ""
                print(f"{r['elapsed_s']:6.2f}s  rss={r['rss_mib']:7.1f} MiB{extra}")

    print("\n=== summary (median over trials) ===")
    print(
        f"{'experiment':<24s} {'median':>8s} {'min':>8s} {'max':>8s} {'rss_mib':>10s}  digest"
    )
    for exp in experiments:
        triplet = [r for r in results if r["experiment"] == exp and "error" not in r]
        if not triplet:
            print(f"{exp:<24s} ERR")
            continue
        elapsed = sorted(r["elapsed_s"] for r in triplet)
        med = elapsed[len(elapsed) // 2]
        rss = max(r["rss_mib"] for r in triplet)
        digest = triplet[0]["digest_repr"]
        print(
            f"{exp:<24s} {med:8.3f} {min(elapsed):8.3f} {max(elapsed):8.3f} {rss:10.1f}  {digest[:55]}"
        )

    # Speedup tables
    print("\n=== scatter_q deltas vs baseline ===")
    base = [
        r["elapsed_s"]
        for r in results
        if r["experiment"] == "scatter_q_baseline" and "error" not in r
    ]
    base_med = sorted(base)[len(base) // 2] if base else float("nan")
    print(f"  baseline: {base_med:.3f}s")
    for exp in [
        "scatter_q_noclamp",
        "scatter_q_fma",
        "scatter_q_floor",
        "scatter_q_combined",
        "scatter_q_pre",
    ]:
        t = [
            r["elapsed_s"]
            for r in results
            if r["experiment"] == exp and "error" not in r
        ]
        med = sorted(t)[len(t) // 2] if t else float("nan")
        speedup = base_med / med if med else float("inf")
        print(
            f"  {exp:<24s} {med:6.3f}s  ({speedup:.2f}x faster, saves {base_med - med:.3f}s)"
        )

    print("\n=== cat_count deltas vs varchar ===")
    base = [
        r["elapsed_s"]
        for r in results
        if r["experiment"] == "cat_count_varchar" and "error" not in r
    ]
    base_med = sorted(base)[len(base) // 2] if base else float("nan")
    print(f"  varchar: {base_med:.3f}s")
    t = [
        r["elapsed_s"]
        for r in results
        if r["experiment"] == "cat_count_enum" and "error" not in r
    ]
    med = sorted(t)[len(t) // 2] if t else float("nan")
    speedup = base_med / med if med else float("inf")
    print(
        f"  enum   : {med:6.3f}s  ({speedup:.2f}x faster, saves {base_med - med:.3f}s)"
    )

    print("\n=== colorby deltas vs ALTER+UPDATE ===")
    base = [
        r["elapsed_s"]
        for r in results
        if r["experiment"] == "colorby_alter_update" and "error" not in r
    ]
    base_med = sorted(base)[len(base) // 2] if base else float("nan")
    print(f"  alter_update: {base_med:.3f}s")
    for exp in ["colorby_view", "colorby_enum"]:
        t = [
            r["elapsed_s"]
            for r in results
            if r["experiment"] == exp and "error" not in r
        ]
        med = sorted(t)[len(t) // 2] if t else float("nan")
        speedup = base_med / med if med else float("inf")
        print(
            f"  {exp:<14s} {med:6.3f}s  ({speedup:.2f}x faster, saves {base_med - med:.3f}s)"
        )

    print("\n=== ctas variants (one-time loader cost) ===")
    for exp in ["ctas_baseline", "ctas_pre_quantised", "ctas_enum"]:
        t = [
            r["elapsed_s"]
            for r in results
            if r["experiment"] == exp and "error" not in r
        ]
        med = sorted(t)[len(t) // 2] if t else float("nan")
        print(f"  {exp:<22s} {med:6.3f}s")


if __name__ == "__main__":
    main()
