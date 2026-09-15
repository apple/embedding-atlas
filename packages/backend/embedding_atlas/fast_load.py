"""DuckDB-native fast-path loader for GIS parquet files.

For single-parquet inputs with either a lon/lat pair or a WKB geometry
column, this loader materialises the file into a DuckDB TABLE via a
single ``CREATE TABLE ... AS SELECT`` that pulls the rows in once and
projects the parquet reader's ``file_row_number`` virtual column under
``__row_index__`` — so the viewer never has to do its own
``ALTER TABLE ... ADD COLUMN`` + ``UPDATE`` rowid pass.

Why a TABLE and not a VIEW: the viewer's ``makeCategoryColumn`` (color
by) issues ``ALTER TABLE dataset ADD COLUMN ...`` + ``UPDATE`` on every
color-by click. DuckDB only allows ``ALTER VIEW`` on views, so a
view-backed dataset hard-fails the moment the user picks a color
column. A 322 M-row materialisation takes ~75 s on a 16-core box and
each subsequent ALTER+UPDATE is ~9 s — slower than a view at startup,
but functional through the full feature surface.

The previous CTAS path crashed on >~10 GB parquet because DuckDB's
defaults are ``memory_limit = 80% RAM`` (no headroom) and
``temp_directory = '.tmp'`` (a relative path that often can't be
created from the user's cwd). When the working set spilled past the
limit it had nowhere to go and the process OOM-killed mid-load. We now
set ``memory_limit`` to a generous-but-bounded fraction of system RAM
and pin ``temp_directory`` to the OS tmp dir so spilling always has a
home.

Used by:
  * ``apps/desktop`` sidecar — always.
  * ``packages/backend`` CLI (``geospatial-atlas``) — auto-selected when
    inputs are a single Parquet with no ``--query``/``--sample`` and
    no embedding generation is requested.
  * Frontend-only / static distros — unaffected; they use DuckDB-WASM
    directly in the browser.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

import duckdb


# DuckDB temp-directory cleanup
# -----------------------------
# Each invocation of ``_safe_memory_temp_settings`` creates a per-process
# spill directory under ``$TMPDIR`` with the prefix ``duckdb_gsa_``. At
# 322 M-row scale a single load can spill 100+ GB; without explicit
# cleanup these dirs leak until macOS's tmp-cleaner sweeps them (which is
# rare — typically only on reboot, sometimes after several days). One
# session at a time is fine; four crashed sessions later the user is
# down 400 GB of disk for no benefit.
#
# Strategy: track every dir we create, wipe on normal exit (atexit), wipe
# on SIGTERM/SIGINT (Electron sends SIGTERM via ``child.kill()``), and
# sweep stale orphans (>24 h old) from prior crashed sessions at
# startup. ``ignore_errors=True`` everywhere — cleanup is best-effort,
# never let it fail the process.

_TEMP_DIR_PREFIX = "duckdb_gsa_"
_REGISTERED_TEMP_DIRS: set[str] = set()
_TEMP_DIR_LOCK = threading.Lock()
_CLEANUP_INSTALLED = False
_ORPHAN_SWEEP_DONE = False
_ORPHAN_SWEEP_MAX_AGE_HOURS = 24.0


def _cleanup_registered_temp_dirs() -> None:
    """Remove every duckdb_gsa_* dir registered by this process.

    Safe to call multiple times and from signal handlers — dirs are
    discarded from the registry as they're removed and ``rmtree`` is
    invoked with ``ignore_errors=True`` so a partial wipe (e.g. DuckDB
    finaliser still has an open fd on Windows) doesn't raise.
    """
    with _TEMP_DIR_LOCK:
        dirs = list(_REGISTERED_TEMP_DIRS)
        _REGISTERED_TEMP_DIRS.clear()
    for d in dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            # Defensive: rmtree itself shouldn't raise with ignore_errors=True,
            # but in cleanup paths we never want to bubble.
            pass


def _purge_stale_orphan_temp_dirs(
    max_age_hours: float = _ORPHAN_SWEEP_MAX_AGE_HOURS,
) -> None:
    """Best-effort sweep of duckdb_gsa_* dirs older than ``max_age_hours``.

    Catches dirs left by sessions that died without atexit running
    (SIGKILL, segfault, power loss). The age cutoff prevents us from
    wiping a *parallel* sidecar's active dir — e.g. a CLI invocation
    while the desktop app is open. 24 h is conservatively long; nobody
    runs a single DuckDB session for a day, but the OS sweeper is
    happier reclaiming on its own schedule than us being aggressive.
    """
    global _ORPHAN_SWEEP_DONE
    if _ORPHAN_SWEEP_DONE:
        return
    _ORPHAN_SWEEP_DONE = True
    root = tempfile.gettempdir()
    cutoff = time.time() - max_age_hours * 3600.0
    try:
        entries = os.listdir(root)
    except OSError:
        return
    swept = 0
    for entry in entries:
        if not entry.startswith(_TEMP_DIR_PREFIX):
            continue
        full = os.path.join(root, entry)
        try:
            st = os.stat(full)
        except OSError:
            continue
        if st.st_mtime >= cutoff:
            continue  # younger than cutoff — could belong to a live process
        try:
            shutil.rmtree(full, ignore_errors=True)
            swept += 1
        except Exception:
            pass
    if swept:
        # Single line on stderr so sidecar logs surface the reclaim event
        # without spamming healthy startups (which find nothing).
        print(
            f"[fast_load] swept {swept} stale duckdb_gsa_* dir(s) older than "
            f"{max_age_hours:g}h",
            file=sys.stderr,
            flush=True,
        )


def _install_cleanup_hooks() -> None:
    """Register atexit + SIGTERM/SIGINT handlers exactly once.

    Signal handlers can only be installed from the main thread; uvicorn
    schedules requests on worker threads, so guard with a thread check
    and fall back to atexit-only on non-main threads (the main thread
    will install the signal handler when it imports this module from
    its own context).
    """
    global _CLEANUP_INSTALLED
    if _CLEANUP_INSTALLED:
        return
    _CLEANUP_INSTALLED = True

    atexit.register(_cleanup_registered_temp_dirs)

    if threading.current_thread() is not threading.main_thread():
        return

    def _signal_handler(signum, frame):
        _cleanup_registered_temp_dirs()
        # Restore the default disposition and re-raise so the process
        # exits as if we hadn't intercepted (correct exit codes,
        # correct shell behaviour, and parents like Electron see a
        # clean signal-driven termination).
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
        try:
            os.kill(os.getpid(), signum)
        except OSError:
            sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Not all platforms / embedding contexts allow this
            # (e.g. running inside a thread, or with a custom event
            # loop that has its own signal multiplexer). Best-effort.
            pass


def _safe_memory_temp_settings() -> tuple[str, str]:
    """Return ``(memory_limit, temp_directory)`` strings safe to apply to
    a fresh ``:memory:`` connection.

    The DuckDB defaults (``80% RAM`` limit, ``.tmp`` relative temp path)
    are the actual cause of large-file CREATE TABLE crashes: spilling
    has no usable scratch dir and no headroom over the OS.

    Side effects: registers the created temp dir for shutdown cleanup
    (atexit + SIGTERM/SIGINT) and, on first call, sweeps any
    duckdb_gsa_* dirs older than 24 h that are leftovers from
    previously-crashed sessions.
    """
    _install_cleanup_hooks()
    _purge_stale_orphan_temp_dirs()

    try:
        total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total_ram = 16 * 1024**3  # conservative fallback (16 GiB)
    # 50 % of system RAM, hard-capped at 64 GiB. The renderer process,
    # macOS, and any browsers the user is running need the other half —
    # a 96 GiB cap on a 128 GiB box once consumed enough swap that
    # macOS triggered "system has run out of application memory" and
    # took down the desktop. DuckDB *will* spill when it hits the cap,
    # which is fine: spilling at 64 GiB with adequate free disk is
    # healthy behavior. The 700+ GB of orphan /var temp that previously
    # accumulated was the result of leaked dirs from crashed sessions,
    # now plugged by the atexit/signal-handler cleanup above.
    limit_bytes = min(int(total_ram * 0.5), 64 * 1024**3)
    limit_str = f"{limit_bytes // (1024**3)}GB"
    # OS tmp dir always exists and has plenty of room; per-process subdir
    # so we never collide with another sidecar / CLI invocation.
    temp_dir = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
    with _TEMP_DIR_LOCK:
        _REGISTERED_TEMP_DIRS.add(temp_dir)
    return limit_str, temp_dir


ProgressCallback = Callable[[str, float, str], None]
"""Receives (stage, percent, detail). ``stage`` is a short slug; ``detail``
is a user-facing one-liner. ``percent`` is 0..100 or -1 if unknown."""


@dataclass
class FastLoadResult:
    connection: duckdb.DuckDBPyConnection
    table: str
    row_count: int
    x_column: str
    y_column: str
    # Stable per-row id, picked here so callers don't need a separate
    # ALTER TABLE + UPDATE pass — that pass would rewrite the whole
    # (multi-GB) view-backing parquet on every load.
    id_column: str
    columns: list[str]
    duration_seconds: float
    # Axis-aligned bounding box over (x_column, y_column). ``None`` if the
    # bounds query raised (e.g. spatial-extension edge cases on exotic
    # geometries). Consumers use these to quantize coordinates for the
    # wire — the frontend sends a MIN-MAX linear map to pack f32 → u32.
    x_bounds: tuple[float, float] | None = None
    y_bounds: tuple[float, float] | None = None
    # Pre-computed u32-quantised x/y column names (or ``None`` if bounds
    # weren't available at CTAS time). When present, the wire scatter
    # query is just ``SELECT __x_u32__, __y_u32__ FROM dataset`` — no
    # per-row arithmetic, no clamps, ~6× faster than the on-the-fly cast
    # at 300 M-row scale. u32 (vs the u16 first cut) gives 65 536× finer
    # quantisation — at the eubucco 40°-lon span that's 1.5 cm per step
    # vs 110 m per step, killing the visible street-level grid pattern
    # without losing the wire-pack performance win.
    quantised_x_column: str | None = None
    quantised_y_column: str | None = None
    # Mercator-projected y, also packed as u32 over [merc_y_min, merc_y_max].
    # Only populated when ``y`` is in geographic latitude space (-90..90 deg);
    # ``merc_y_bounds`` is the linear range to use for unpacking on the wire.
    # Lets the GIS fast-path skip the JS-side Mercator loop entirely (~5.9 s
    # on 322 M rows on Apple-Silicon Chrome).
    quantised_merc_y_column: str | None = None
    merc_y_bounds: tuple[float, float] | None = None
    # Low-cardinality VARCHAR columns that were ENUM-encoded at CTAS.
    # Maps original column name -> ordered list of distinct values
    # (the ENUM ordinal is the index into the list). Empty if no
    # columns qualified or auto-encoding was disabled.
    enum_columns: dict[str, list[str]] | None = None
    # True when ``dataset`` is currently exposed as a VIEW over the
    # source parquet (no full materialisation). The server promotes it
    # to a TABLE on the first ALTER/UPDATE so color-by clicks work.
    is_view: bool = False
    # Background-materialise thread: when ``is_view`` is True, a daemon
    # thread runs ``CREATE TABLE <materialise_table> AS SELECT * FROM
    # dataset`` immediately after the loader returns. The server's
    # ``_ensure_dataset_materialised`` joins this thread on the first
    # write — by which point the CTAS is usually already done, so the
    # color-by ALTER/UPDATE pays a near-zero swap (DROP VIEW + RENAME)
    # instead of a fresh ~5 s CTAS. Errors are captured in
    # ``materialise_error`` and the server falls back to synchronous CTAS.
    materialise_thread: "threading.Thread | None" = None
    materialise_table: str | None = None
    materialise_error: list | None = None


def _detect_columns(
    con: duckdb.DuckDBPyConnection, path: str
) -> tuple[Literal["xy", "geometry"], tuple[str, str] | str, list[str], dict[str, str]]:
    """Return (kind, info, all_columns, col_types).

    ``info`` is (x,y) column names or the geometry column name.
    ``col_types`` maps column name to its DuckDB type string.
    """
    # DESCRIBE doesn't require reading the full file, just the parquet footer.
    schema = con.sql(
        f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet({duckdb_literal(path)}) LIMIT 0)"
    ).fetchall()
    cols = [r[0] for r in schema]
    col_types = {r[0]: r[1] for r in schema}
    cols_lower = {c.lower(): c for c in cols}

    for xc, yc in [
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("lng", "lat"),
        ("x", "y"),
    ]:
        if xc in cols_lower and yc in cols_lower:
            return "xy", (cols_lower[xc], cols_lower[yc]), cols, col_types

    for cand in ["geometry", "geom", "wkb_geometry", "the_geom", "geo"]:
        if cand in cols_lower:
            return "geometry", cols_lower[cand], cols, col_types

    raise ValueError(
        "No GIS columns detected. Expected a lon/lat pair (e.g. "
        "longitude/latitude, lon/lat, x/y) or a WKB geometry / GEOMETRY "
        "column (geometry, geom, wkb_geometry, the_geom). Columns present: "
        + ", ".join(cols)
    )


def duckdb_literal(s: str) -> str:
    """Quote a path as a DuckDB string literal (escapes single quotes)."""
    return "'" + s.replace("'", "''") + "'"


def fast_load_parquet(
    path: str,
    *,
    table: str = "dataset",
    threads: int | None = None,
    enable_spatial: bool | None = None,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
    precompute_quantised: bool = True,
    enum_threshold: int = 1024,
    materialise: Literal["view", "table"] = "view",
    enum_sample_rows: int = 100_000,
) -> FastLoadResult:
    """Expose a parquet file as a DuckDB VIEW on a fresh ``:memory:`` connection.

    The view wraps ``read_parquet(path, file_row_number=true)`` so that
    DuckDB scans only the column chunks each query touches and never
    materialises the full table. This is what lets a 15.6 GB / 322 M-row
    file load instantly on a 16 GB machine — the previous
    ``CREATE TABLE AS SELECT *`` path needed 50–100 GB of RAM (or a
    tens-of-GB temp spill) for wide string-heavy schemas.

    The reader's virtual ``file_row_number`` column gives every row a
    stable id; we expose it under a unique alias so callers don't need a
    separate ``ALTER TABLE`` + ``UPDATE`` rewrite pass.

    If ``limit`` is set, the LIMIT lives inside the view definition.
    DuckDB pushes it down to the parquet reader, so glimpsing 1 000 rows
    of a 14 GB file takes milliseconds.

    Handles both legacy BLOB/WKB geometry columns and modern GeoParquet
    files where DuckDB surfaces the column as native ``GEOMETRY`` — the
    latter doesn't need an ``ST_GeomFromWKB`` wrapper.
    """
    t_start = time.perf_counter()
    fn = pathlib.Path(path)
    if not fn.is_file():
        raise FileNotFoundError(path)
    if limit is not None and limit <= 0:
        limit = None

    emit = progress or (lambda *_: None)

    con = duckdb.connect(":memory:")
    # Pin memory_limit + temp_directory before any heavy SQL runs so that
    # ``CREATE TABLE`` always has somewhere to spill on ≥10 GB inputs.
    mem_limit, temp_dir = _safe_memory_temp_settings()
    con.sql(f"SET memory_limit = '{mem_limit}'")
    con.sql(f"SET temp_directory = '{temp_dir}'")
    if threads:
        con.sql(f"PRAGMA threads={int(threads)}")
    # Required to make query_progress() return anything useful.
    con.sql("SET enable_progress_bar=true")
    con.sql("SET enable_progress_bar_print=false")

    # Load spatial first so that DESCRIBE surfaces GEOMETRY columns
    # correctly for GeoParquet 1.1 files. DuckDB 1.4+ surfaces the
    # column as ``GEOMETRY('OGC:CRS84')`` even without spatial loaded,
    # but ``ST_X`` / ``ST_Y`` obviously require the extension.
    spatial_loaded = False
    if enable_spatial is not False:
        try:
            emit("spatial", 5.0, "Loading DuckDB spatial extension")
            con.sql("INSTALL spatial")
            con.sql("LOAD spatial")
            spatial_loaded = True
        except Exception:
            if enable_spatial is True:
                raise

    emit("analyze", 8.0, f"Opening {fn.name}")
    kind, info, columns, col_types = _detect_columns(con, str(path))

    # Pick a stable id-column name that doesn't collide with the parquet
    # schema. We project the reader's ``file_row_number`` virtual column
    # under this alias.
    id_column = _unused_name(columns, "__row_index__")

    # If the parquet itself has a column literally named ``file_row_number``,
    # ``read_parquet(.., file_row_number=true)`` would collide on the
    # output schema. Fall back to a window-function row id in that case.
    has_frn_collision = any(c.lower() == "file_row_number" for c in columns)
    if has_frn_collision:
        read = f"read_parquet({duckdb_literal(str(path))})"
        id_expr = "ROW_NUMBER() OVER ()"
        passthrough = "*"
    else:
        read = f"read_parquet({duckdb_literal(str(path))}, file_row_number=true)"
        id_expr = "file_row_number"
        # Drop the virtual column from SELECT * so we don't surface it
        # twice (once raw, once aliased).
        passthrough = "* EXCLUDE (file_row_number)"

    limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""

    # Resolve x_out/y_out + the SELECT projection expressions. We need
    # x_out/y_out *before* the CTAS so the bounds pre-pass can target them
    # (when they come from ST_X/ST_Y of a geometry column there's no
    # column to bound on the parquet side, so we hold off in that case).
    if kind == "xy":
        x_col, y_col = info  # type: ignore[misc]
        x_out, y_out = x_col, y_col
        coord_select = ""  # x_out / y_out already exist in the source
        detail = f"Using columns {x_col} / {y_col}"
        coord_source_for_bounds = (x_col, y_col)
    else:
        geom = str(info)  # type: ignore[arg-type]
        geom_type = col_types.get(geom, "").upper()
        is_native_geometry = geom_type.startswith("GEOMETRY")
        if is_native_geometry and not spatial_loaded:
            raise RuntimeError(
                "File uses native GEOMETRY column but DuckDB spatial extension "
                "could not be loaded (is this machine offline on first run?)"
            )
        x_out = _unused_name(columns, "lon")
        y_out = _unused_name(columns + [x_out], "lat")
        geom_expr = (
            quote_ident(geom)
            if is_native_geometry
            else f"ST_GeomFromWKB({quote_ident(geom)})"
        )
        coord_select = (
            f"ST_X({geom_expr}) AS {quote_ident(x_out)}, "
            f"ST_Y({geom_expr}) AS {quote_ident(y_out)}, "
        )
        detail = (
            f"Extracting coordinates from {geom} "
            f"({'native GEOMETRY' if is_native_geometry else 'WKB'})"
        )
        # Geometry path: bounds need ST_X/ST_Y, no plain column to scan
        # on the parquet side. Compute bounds post-CTAS.
        coord_source_for_bounds = None

    # Pre-CTAS bounds pass (xy mode only). DuckDB reads parquet column
    # statistics from the file footer if the writer included them — turns
    # MIN/MAX into a footer read in tens of milliseconds, even on 300M
    # rows. If stats are missing we fall through to a streaming scan via
    # read_parquet, still cheap because we only touch x/y columns.
    x_bounds: tuple[float, float] | None = None
    y_bounds: tuple[float, float] | None = None
    if coord_source_for_bounds is not None:
        emit("bounds_pre", 9.0, "Reading bounds from parquet metadata")
        x_src, y_src = coord_source_for_bounds
        try:
            row = con.sql(
                f"SELECT MIN({quote_ident(x_src)}), MAX({quote_ident(x_src)}), "
                f"MIN({quote_ident(y_src)}), MAX({quote_ident(y_src)}) "
                f"FROM {read}"
            ).fetchone()
            if row is not None and all(v is not None for v in row):
                x_min, x_max, y_min, y_max = (float(v) for v in row)
                if x_max > x_min:
                    x_bounds = (x_min, x_max)
                if y_max > y_min:
                    y_bounds = (y_min, y_max)
        except Exception:
            pass

    # ENUM auto-detection on VARCHAR columns. Three-stage funnel — total
    # cost on a 75 M-row, 11-VARCHAR-col file drops from ~1.3 s (per-col
    # full-file HLL) to ~0.4 s.
    #
    #   1. Sample prefilter: APPROX_COUNT_DISTINCT on a LIMIT-N stub
    #      (~60 ms for 100 k rows, all 11 cols in one query). Anything
    #      with > 2× threshold distincts in 100 k samples cannot fit in
    #      ``enum_threshold`` globally — drop it.
    #   2. Full-file HLL on the survivors (one fused scan).
    #   3. Exact ``LIST(DISTINCT …)`` per column that crossed step 2,
    #      again fused into a single scan.
    #
    # Each pass reads strictly fewer columns than the previous, and
    # DuckDB's parquet reader column-prunes accordingly. Storage in the
    # final dataset drops from ~9 B/row VARCHAR to 1 B/row ENUM ordinal
    # for each encoded column.
    enum_columns: dict[str, list[str]] = {}
    if enum_threshold > 0:
        varchar_cols = [
            c
            for c, t in col_types.items()
            if t.upper() in ("VARCHAR", "STRING", "TEXT")
        ]
        if varchar_cols:
            try:
                emit(
                    "enum_prefilter",
                    9.5,
                    f"Sampling {len(varchar_cols)} VARCHAR columns",
                )
                # Stage 1: cheap LIMIT prefilter. 2× slack accounts for
                # cardinality under-estimation in a small sample.
                slack = max(2 * enum_threshold, enum_threshold + 64)
                sel = ", ".join(
                    f"APPROX_COUNT_DISTINCT({quote_ident(c)}) AS {quote_ident(c)}"
                    for c in varchar_cols
                )
                row = con.sql(
                    f"WITH s AS (SELECT * FROM {read} LIMIT {int(enum_sample_rows)}) "
                    f"SELECT {sel} FROM s"
                ).fetchone()
                survivors = (
                    [
                        c
                        for c, n in zip(varchar_cols, row)
                        if n is not None and n <= slack
                    ]
                    if row is not None
                    else []
                )

                if survivors:
                    # Stage 2: full-file HLL on survivors (one parquet scan,
                    # fused aggregates).
                    sel = ", ".join(
                        f"APPROX_COUNT_DISTINCT({quote_ident(c)}) AS {quote_ident(c)}"
                        for c in survivors
                    )
                    row = con.sql(f"SELECT {sel} FROM {read}").fetchone()
                    keepers = (
                        [
                            c
                            for c, n in zip(survivors, row)
                            if n is not None and n <= enum_threshold
                        ]
                        if row is not None
                        else []
                    )

                    if keepers:
                        # Stage 3: exact distinct values, fused.
                        sel = ", ".join(
                            f"LIST(DISTINCT {quote_ident(c)}) FILTER (WHERE {quote_ident(c)} IS NOT NULL) AS {quote_ident(c)}"
                            for c in keepers
                        )
                        listed = con.sql(f"SELECT {sel} FROM {read}").fetchone()
                        if listed is not None:
                            for col, values in zip(keepers, listed):
                                if values and 0 < len(values) <= enum_threshold:
                                    enum_columns[col] = sorted(values)
            except Exception:
                # ENUM detection is best-effort: an unusual collation or
                # blob-typed VARCHAR shouldn't fail the load.
                pass

    # Build the CTAS SELECT list with: passthrough columns (with ENUM
    # casts where applicable), synthetic coords (geometry path), pre-
    # quantised __x_u32__/__y_u32__ when we have bounds, and the row id.
    quant_x_col: str | None = None
    quant_y_col: str | None = None

    # GEOMETRY columns can't be serialised to Arrow IPC by DuckDB's spatial
    # extension — the moment Mosaic's instances table issues
    # ``SELECT * FROM dataset`` (or even ``DESC SELECT *``), the request
    # 500s with ``TransactionContext::ActiveTransaction called without
    # active transaction``. Cast each GEOMETRY column to TEXT (WKT) in the
    # view: same data, Arrow-IPC-safe, and the per-row WKT string only
    # materialises when a query actually selects the column.
    geom_columns = [c for c, t in col_types.items() if t.upper().startswith("GEOMETRY")]

    if enum_columns or geom_columns:
        # Create one ENUM type per encoded column. Names are scoped by
        # column to avoid collisions across multiple datasets in the same
        # process (CREATE OR REPLACE handles re-loads).
        for col, values in enum_columns.items():
            type_name = _enum_type_name(table, col)
            literal_list = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
            con.sql(
                f"CREATE OR REPLACE TYPE {quote_ident(type_name)} AS ENUM ({literal_list})"
            )
        # Replace the passthrough * with explicit projections so we can
        # cast just the ENUM and GEOMETRY cols. Anything else passes
        # through unchanged.
        rewritten = list(enum_columns.keys()) + geom_columns
        excludes = (
            (["file_row_number"] + rewritten) if not has_frn_collision else rewritten
        )
        passthrough = "* EXCLUDE (" + ", ".join(quote_ident(e) for e in excludes) + ")"
        casts = []
        for col in enum_columns:
            casts.append(
                f"{quote_ident(col)}::{quote_ident(_enum_type_name(table, col))} AS {quote_ident(col)}"
            )
        for col in geom_columns:
            casts.append(f"{quote_ident(col)}::TEXT AS {quote_ident(col)}")
        cast_clause = ", " + ", ".join(casts)
    else:
        cast_clause = ""

    quant_clause = ""
    quant_merc_y_col: str | None = None
    merc_y_bounds: tuple[float, float] | None = None
    if precompute_quantised and x_bounds is not None and y_bounds is not None:
        quant_x_col = _unused_name(columns + [id_column, x_out, y_out], "__x_u32__")
        quant_y_col = _unused_name(
            columns + [id_column, x_out, y_out, quant_x_col], "__y_u32__"
        )
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        # u32 quant: 4 294 967 295 = 2³² − 1 distinct buckets per axis.
        # Quant step at the eubucco lon span (~40°) is 4444 km / 2³² ≈
        # 1 mm — well below sub-pixel even at street zoom. (u16 was 110 m,
        # which produced a visible grid the moment a user zoomed past
        # ~city scale.)
        U32_MAX = 4_294_967_295
        x_scale = U32_MAX / (x_max - x_min)
        y_scale = U32_MAX / (y_max - y_min)
        # GREATEST/LEAST runs once at load — paranoid safety against
        # NULLs / edge bound rows (NULLs cast to NULL in DuckDB; clamps
        # squash any near-bound float drift). After this, the wire path
        # can be a clamp-free SELECT __x_u32__, __y_u32__.
        x_q_expr = (
            f"GREATEST(0, LEAST({U32_MAX}, "
            f"((COALESCE({quote_ident(x_out)}, {x_min}) - ({x_min!r})) * {x_scale!r}))"
            f")::UINTEGER"
        )
        y_q_expr = (
            f"GREATEST(0, LEAST({U32_MAX}, "
            f"((COALESCE({quote_ident(y_out)}, {y_min}) - ({y_min!r})) * {y_scale!r}))"
            f")::UINTEGER"
        )
        quant_clause = f", {x_q_expr} AS {quote_ident(quant_x_col)}, {y_q_expr} AS {quote_ident(quant_y_col)}"

        # Mercator-projected y, also packed as u32. For GIS datasets the
        # browser would otherwise have to project lat→Mercator on the JS
        # main thread for every row before paint — ~5.9 s on 322 M rows.
        # Pre-projecting at view-definition time pushes that into DuckDB's
        # vectorised C++ engine and out of the cold-load path entirely.
        # Bounds are valid only when ``y`` is in geographic latitude space
        # (-90..90); we range-check rather than relying on a user flag so
        # the same loader works for embedding (non-GIS) datasets without
        # spurious wraps.
        is_lat_like = -90.5 <= y_min <= y_max <= 90.5
        if is_lat_like:
            import math

            merc_y_min = (
                math.log(math.tan(math.pi / 4 + y_min * math.pi / 360)) * 180 / math.pi
            )
            merc_y_max = (
                math.log(math.tan(math.pi / 4 + y_max * math.pi / 360)) * 180 / math.pi
            )
            if merc_y_max > merc_y_min:
                merc_y_scale = U32_MAX / (merc_y_max - merc_y_min)
                quant_merc_y_col = _unused_name(
                    columns + [id_column, x_out, y_out, quant_x_col, quant_y_col],
                    "__y_merc_u32__",
                )
                # Inline mercator(lat) → linear-pack to u32. The CASE on
                # COALESCE keeps NULL → 0 (clamped to mid-band) without
                # exploding tan() near the poles for legitimate edge values.
                merc_q_expr = (
                    f"GREATEST(0, LEAST({U32_MAX}, "
                    f"(((LN(TAN(PI()/4 + COALESCE({quote_ident(y_out)}, {y_min}) * PI() / 360)) * 180 / PI()) - ({merc_y_min!r})) * {merc_y_scale!r})"
                    f"))::UINTEGER"
                )
                quant_clause += f", {merc_q_expr} AS {quote_ident(quant_merc_y_col)}"
                merc_y_bounds = (merc_y_min, merc_y_max)

    select = (
        f"SELECT {passthrough}, "
        f"{coord_select}"
        f"{id_expr} AS {quote_ident(id_column)}"
        f"{cast_clause}"
        f"{quant_clause} "
        f"FROM {read}{limit_clause}"
    )

    # Two materialisation strategies:
    #
    #   "view"  — default. The dataset is a VIEW over read_parquet(...),
    #             with quantised cols and ENUM casts inlined. First map
    #             render reads only x/y columns from parquet (≤ 1 s on
    #             75 M rows). The server lazily promotes the view to a
    #             TABLE on the first ALTER/UPDATE so color-by clicks
    #             work — saving 3.5 s of upfront wall time when the user
    #             never color-bys.
    #
    #   "table" — eager CREATE TABLE AS SELECT. Used by tests and any
    #             caller that wants the historic semantics (full table
    #             materialised before the function returns).
    is_view = materialise == "view"
    obj_kw = "VIEW" if is_view else "TABLE"
    sql = f"CREATE OR REPLACE {obj_kw} {quote_ident(table)} AS {select}"
    emit("load", 10.0, detail)
    if is_view:
        # CREATE VIEW is a metadata-only operation — no scan, < 10 ms.
        con.execute(sql)
    else:
        _run_with_progress(con, sql, emit)

    emit("count", 80.0, "Counting rows")
    row_count = con.sql(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0]

    if x_bounds is None or y_bounds is None:
        emit("bounds", 90.0, "Computing bounding box (post-CTAS fallback)")
        try:
            row = con.sql(
                f"SELECT MIN({quote_ident(x_out)}), MAX({quote_ident(x_out)}), "
                f"MIN({quote_ident(y_out)}), MAX({quote_ident(y_out)}) "
                f"FROM {quote_ident(table)}"
            ).fetchone()
            if row is not None and all(v is not None for v in row):
                x_min, x_max, y_min, y_max = (float(v) for v in row)
                if x_max > x_min:
                    x_bounds = (x_min, x_max)
                if y_max > y_min:
                    y_bounds = (y_min, y_max)
        except Exception:
            pass

    emit("ready", 100.0, f"Loaded {row_count:,} rows")
    extra_cols = [
        c
        for c in (x_out, y_out, id_column, quant_x_col, quant_y_col, quant_merc_y_col)
        if c is not None and c not in columns
    ]

    # Kick off background materialisation when we returned a VIEW. The
    # CTAS runs in parallel with the first read queries and the server's
    # write path joins this thread on the first ALTER, turning a 5 s
    # color-by surprise into a near-instant DROP VIEW + RENAME.
    bg_thread: threading.Thread | None = None
    bg_table: str | None = None
    bg_error: list = []
    if is_view:
        bg_table = _unused_name(columns + extra_cols + [table], f"__bg_mat_{table}__")
        bg_table_local = bg_table
        # Spatial sort at materialise time. With this in place, DuckDB's
        # per-row-group min/max statistics on x/y become tight, so the
        # ``WHERE x BETWEEN px-r AND px+r`` predicate the tooltip path
        # uses (mosaic_client.ts:queryClosestPoint) prunes >99 % of row
        # groups instead of scanning the whole table. Tooltip latency
        # at 322 M rows: ~10 s → ~50 ms.
        #
        # Sort key is the precomputed u32 cols, not raw f32: the u32
        # quantum is far below the typical tooltip radius in pixels,
        # so column-major bucketing on (x_u32, y_u32) gives the row
        # groups the tightest possible (x_min, x_max) bounds at any
        # zoom. When precompute_quantised was off (no bounds at load
        # time), we skip the sort — there's no cheap stable key to
        # sort on.
        if quant_x_col is not None and quant_y_col is not None:
            order_clause = (
                f" ORDER BY {quote_ident(quant_x_col)}, {quote_ident(quant_y_col)}"
            )
        else:
            order_clause = ""

        def _bg_materialise() -> None:
            try:
                cursor = con.cursor()
                try:
                    cursor.execute(
                        f"CREATE TABLE {quote_ident(bg_table_local)} AS "
                        f"SELECT * FROM {quote_ident(table)}{order_clause}"
                    )
                finally:
                    cursor.close()
            except BaseException as e:  # noqa: BLE001 — logged for the server
                bg_error.append(e)

        bg_thread = threading.Thread(
            target=_bg_materialise, name="fast-load-bg-mat", daemon=True
        )
        bg_thread.start()

    return FastLoadResult(
        connection=con,
        table=table,
        row_count=row_count,
        x_column=x_out,
        y_column=y_out,
        id_column=id_column,
        columns=columns + extra_cols,
        duration_seconds=time.perf_counter() - t_start,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        quantised_x_column=quant_x_col,
        quantised_y_column=quant_y_col,
        quantised_merc_y_column=quant_merc_y_col,
        merc_y_bounds=merc_y_bounds,
        enum_columns=enum_columns or None,
        is_view=is_view,
        materialise_thread=bg_thread,
        materialise_table=bg_table,
        materialise_error=bg_error,
    )


def _enum_type_name(table: str, col: str) -> str:
    """Build a per-(table, col) ENUM type name. Lower-cased + sanitised so
    we don't collide across runs of the same loader against different
    datasets in the same process."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in col).lower()
    return f"_gsa_enum_{table}_{safe}"


def _run_with_progress(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    emit: ProgressCallback,
) -> None:
    """Run ``sql`` on ``con`` in a worker thread and forward DuckDB's own
    progress percentage to ``emit`` every ~300 ms.

    DuckDB's ``query_progress()`` is safe to call from another thread
    while the connection is executing a query (verified with DuckDB 1.5).
    """
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            con.execute(sql)
        except BaseException as e:
            err.append(e)

    t = threading.Thread(target=_worker, name="fast-load-worker", daemon=True)
    t.start()
    last_pct = -1.0
    while t.is_alive():
        try:
            pct = con.query_progress()
        except Exception:
            pct = -1.0
        if pct is not None and pct >= 0 and abs(pct - last_pct) >= 0.5:
            emit("load", float(pct), "")
            last_pct = float(pct)
        time.sleep(0.25)
    t.join()
    if err:
        raise err[0]


def _unused_name(existing: list[str], candidate: str) -> str:
    lower = {c.lower() for c in existing}
    if candidate.lower() not in lower:
        return candidate
    i = 1
    while f"{candidate}_{i}".lower() in lower:
        i += 1
    return f"{candidate}_{i}"


def quote_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def progress_line(stage: str, percent: float, detail: str) -> str:
    """Serialize a progress event for pipe-based transport (Rust parses this)."""
    return "GSA_PROGRESS " + json.dumps(
        {"stage": stage, "percent": percent, "detail": detail}
    )
