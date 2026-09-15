"""Desktop sidecar entry point for Geospatial Atlas.

Launched by the Tauri Rust shell with:
  argv[1]                        = path to dataset file
  env GEOSPATIAL_ATLAS_HOST      = bind host (default 127.0.0.1)
  env GEOSPATIAL_ATLAS_PORT      = bind port
  env GEOSPATIAL_ATLAS_PARENT_PID = Tauri process pid (watchdog target)

Responsibilities:
  * Load the dataset (parquet / csv / arrow / geoparquet).
  * Auto-detect GIS columns (lat/lon or WKB geometry).
  * Mount the existing FastAPI server from `embedding_atlas.server`.
  * Serve uvicorn on the chosen port.
  * Exit as soon as the Tauri parent process disappears (belt + suspenders
    against zombie sidecars if Tauri is SIGKILLed).
"""

from __future__ import annotations

import logging
import os
import pathlib
import signal
import sys
import threading
import time
from typing import Optional


def _log(msg: str) -> None:
    print(f"[sidecar] {msg}", flush=True)


def _resolve_static_dir() -> str:
    """Locate the prebuilt Svelte viewer bundled inside the app.

    When running under PyInstaller we unpack ``embedding_atlas`` into
    ``sys._MEIPASS`` (for --onedir this is the binary's folder), which
    contains ``embedding_atlas/static``.
    """
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(pathlib.Path(meipass) / "embedding_atlas" / "static")
    here = pathlib.Path(__file__).resolve().parent
    candidates.append(here / "embedding_atlas" / "static")
    try:
        import embedding_atlas

        pkg_dir = pathlib.Path(embedding_atlas.__file__).resolve().parent
        candidates.append(pkg_dir / "static")
    except Exception:
        pass
    for c in candidates:
        if c.is_dir() and (c / "index.html").is_file():
            return str(c)
    raise RuntimeError(
        "Could not locate bundled static viewer; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _start_parent_watchdog(parent_pid_env: Optional[str]) -> None:
    """Kill this process as soon as the Tauri parent goes away.

    Uses three independent mechanisms so a starved thread or closed stdin
    alone cannot strand the process:

      1. macOS kqueue ``EVFILT_PROC NOTE_EXIT`` on the parent PID — a
         native, non-polling notification that never misses and is not
         starved by asyncio-heavy threads.
      2. stdin EOF watcher — Tauri pipes stdin, so closure is another
         reliable parent-died signal.
      3. PID / getppid poll at 200 ms — belt and braces.
    """

    # Only activate stdin/kqueue watchdogs when we know we're being run
    # from a Tauri shell (which always sets PARENT_PID). Standalone CLI
    # invocations (including tests via bash backgrounding) have stdin
    # redirected to /dev/null, which would false-trigger the EOF watcher.
    if not parent_pid_env:
        return

    # Stdin-EOF watcher — Electron pipes stdin; pipe closure is a reliable
    # parent-died signal on POSIX.
    #
    # Skipped on Windows: any blocking read on a piped stdin (either the
    # high-level ``sys.stdin.buffer.read`` or the low-level ``os.read``)
    # starves the main thread under PyInstaller — the sidecar hangs
    # immediately after its env dump with no further progress. The
    # ctypes-based pid watchdog below replaces this safeguard on Windows.
    if sys.platform != "win32":

        def _watch_stdin() -> None:
            try:
                while True:
                    chunk = sys.stdin.buffer.read(4096)
                    if not chunk:
                        break
            except Exception:
                pass
            _log("stdin closed, shutting down")
            os._exit(0)

        threading.Thread(
            target=_watch_stdin, name="stdin-watchdog", daemon=True
        ).start()
    try:
        parent_pid = int(parent_pid_env)
    except ValueError:
        return

    # kqueue NOTE_EXIT watcher — fires immediately when parent dies.
    try:
        import select

        if hasattr(select, "kqueue"):

            def _watch_kq() -> None:
                try:
                    kq = select.kqueue()
                    ev = select.kevent(
                        parent_pid,
                        filter=select.KQ_FILTER_PROC,
                        flags=select.KQ_EV_ADD
                        | select.KQ_EV_ENABLE
                        | select.KQ_EV_ONESHOT,
                        fflags=select.KQ_NOTE_EXIT,
                    )
                    kq.control([ev], 0, None)
                    events = kq.control(None, 1, None)
                    if events:
                        _log(f"kqueue NOTE_EXIT for parent {parent_pid}, shutting down")
                        os._exit(0)
                except Exception as e:
                    _log(f"kqueue watcher failed: {e}")

            threading.Thread(target=_watch_kq, name="kq-watchdog", daemon=True).start()
    except Exception:
        pass

    # Parent-alive probe. On Windows, CPython's ``os.kill(pid, 0)`` opens
    # the target with PROCESS_ALL_ACCESS, which Electron's main process
    # routinely denies — that caused a false "parent pid X gone" shutdown
    # right at startup. Use OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)
    # + GetExitCodeProcess on Windows, which any user can query.
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as _wt

        _kernel32 = ctypes.windll.kernel32
        _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        _STILL_ACTIVE = 259
        _kernel32.OpenProcess.restype = _wt.HANDLE
        _kernel32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
        _kernel32.GetExitCodeProcess.argtypes = [_wt.HANDLE, ctypes.POINTER(_wt.DWORD)]
        _kernel32.GetExitCodeProcess.restype = _wt.BOOL
        _kernel32.CloseHandle.argtypes = [_wt.HANDLE]

        def _parent_alive(pid: int) -> bool:
            h = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            try:
                code = _wt.DWORD()
                if not _kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return True  # transient failure → assume alive
                return code.value == _STILL_ACTIVE
            finally:
                _kernel32.CloseHandle(h)
    else:

        def _parent_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            return True

    def _watch_pid() -> None:
        while True:
            if not _parent_alive(parent_pid):
                _log(f"parent pid {parent_pid} gone, shutting down")
                os._exit(0)
            # POSIX reparent-to-init heuristic — Windows never reparents, so
            # skip there (the probe above is authoritative).
            if sys.platform != "win32" and os.getppid() == 1:
                _log("reparented to launchd, shutting down")
                os._exit(0)
            time.sleep(0.2)

    threading.Thread(target=_watch_pid, name="pid-watchdog", daemon=True).start()


def _install_signal_handlers() -> None:
    def _graceful(signum, _frame):
        _log(f"signal {signum} received, exiting")
        os._exit(0)

    # SIGHUP is POSIX-only; on Windows the `signal` module does not expose it,
    # so referencing signal.SIGHUP raises AttributeError at tuple-construction
    # time (before the per-signal try/except can catch it).
    sigs = [signal.SIGINT, signal.SIGTERM]
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is not None:
        sigs.append(sighup)
    for s in sigs:
        try:
            signal.signal(s, _graceful)
        except (OSError, ValueError):
            pass


def _emit_progress(stage: str, percent: float, detail: str) -> None:
    """Forward loader progress to the Tauri shell as a GSA_PROGRESS line.
    The Rust sidecar handler parses these and emits 'sidecar-progress'."""
    from embedding_atlas.fast_load import progress_line

    print(progress_line(stage, percent, detail), flush=True)


def _fast_load(
    dataset_path: str,
    limit: Optional[int] = None,
    text_column: Optional[str] = None,
):
    """DuckDB-native fast path. Returns (connection, DataSource, props, is_gis)."""
    from embedding_atlas.cache import sha256_hexdigest
    from embedding_atlas.data_source import DataSource
    from embedding_atlas.fast_load import fast_load_parquet
    from embedding_atlas.options import make_embedding_atlas_props
    from embedding_atlas.version import __version__

    result = fast_load_parquet(dataset_path, progress=_emit_progress, limit=limit)

    # ``fast_load_parquet`` returns a view-backed connection — every
    # query reads column chunks straight from the on-disk parquet via
    # DuckDB's ``read_parquet``. The reader's ``file_row_number``
    # virtual column is already projected under ``id_column`` for the
    # viewer; no ALTER TABLE / UPDATE pass needed (and ALTER TABLE
    # against a view would fail anyway).
    #
    # We deliberately do not flip ``enable_external_access`` off here —
    # the view depends on read_parquet being able to open the source
    # file on every query. The desktop sidecar binds 127.0.0.1 only and
    # is launched against a single user-chosen file, so the marginal
    # exposure (an SQL endpoint that can read other local files) is
    # bounded by the loopback bind.
    con = result.connection
    id_col = result.id_column
    existing_cols: list[str] = [
        r[1] for r in con.sql(f'PRAGMA table_info("{result.table}")').fetchall()
    ]

    # Validate the text column (if provided). Warn but don't fail if the
    # user typed a name that doesn't exist.
    resolved_text: Optional[str] = None
    if text_column:
        cols_lower = {c.lower(): c for c in existing_cols}
        if text_column in existing_cols:
            resolved_text = text_column
        elif text_column.lower() in cols_lower:
            resolved_text = cols_lower[text_column.lower()]
        else:
            _log(
                f"text column {text_column!r} not found in dataset "
                f"(available: {', '.join(existing_cols)}); ignoring"
            )

    props = make_embedding_atlas_props(
        row_id=id_col,
        x=result.x_column,
        y=result.y_column,
        neighbors=None,
        text=resolved_text,
        point_size=None,
        stop_words=None,
        labels=None,
        is_gis=True,
    )
    # Wire-packing strategy depends on row count — kept in lockstep with
    # the same logic in packages/backend/embedding_atlas/cli.py:
    # f32 wire (Path C) for ≤200M, u16 wire (Path A) for >200M to stay
    # under V8's ~2 GB ArrayBuffer cap that ``mosaic-core``'s
    # ``response.arrayBuffer()`` allocation depends on.
    LARGE_ROW_THRESHOLD = 200_000_000
    is_very_large = result.row_count > LARGE_ROW_THRESHOLD
    if (
        result.x_bounds is not None
        and result.y_bounds is not None
        and "data" in props
        and "projection" in props["data"]
    ):
        x_min, x_max = result.x_bounds
        y_min, y_max = result.y_bounds
        props["data"]["projection"]["viewportHint"] = {
            "centerX": 0.5 * (x_min + x_max),
            "centerY": 0.5 * (y_min + y_max),
            "rangeX": x_max - x_min,
            "rangeY": y_max - y_min,
            "rowCount": result.row_count,
        }
        if is_very_large:
            use_merc_y = (
                result.quantised_merc_y_column is not None
                and result.merc_y_bounds is not None
            )
            if use_merc_y:
                merc_y_min, merc_y_max = result.merc_y_bounds
                props["data"]["projection"]["bounds"] = {
                    "x": [x_min, x_max],
                    "y": [merc_y_min, merc_y_max],
                }
            else:
                props["data"]["projection"]["bounds"] = {
                    "x": [x_min, x_max],
                    "y": [y_min, y_max],
                }
            if (
                result.quantised_x_column is not None
                and result.quantised_y_column is not None
            ):
                y_u16_col = (
                    result.quantised_merc_y_column
                    if use_merc_y
                    else result.quantised_y_column
                )
                props["data"]["projection"]["precomputed"] = {
                    "x_u16": result.quantised_x_column,
                    "y_u16": y_u16_col,
                    "y_is_mercator": use_merc_y,
                }
            props["data"]["projection"]["viewportHint"]["skipDeferredRefine"] = True
    metadata = {"props": props}
    identifier = sha256_hexdigest(
        [__version__, [dataset_path], metadata], scope="DataSource"
    )
    data_source = DataSource(identifier, None, metadata)
    _log(
        f"fast load done: {result.row_count:,} rows in {result.duration_seconds:.2f}s "
        f"(x={result.x_column}, y={result.y_column}, text={resolved_text})"
    )
    # Build the prewarm SQL identical to what the viewer's Mosaic client
    # constructs — see ``EmbeddingViewMosaic.svelte`` Path C. A byte-
    # identical SQL string is required for the cache key to hit. Returned
    # alongside the connection so ``main()`` can pass it to make_server.
    is_gis = props.get("data", {}).get("projection", {}).get("isGis", False)
    if is_very_large and "precomputed" in props.get("data", {}).get("projection", {}):
        precomputed = props["data"]["projection"]["precomputed"]
        prewarm_sql = (
            f'SELECT "{precomputed["x_u16"]}" AS "x", '
            f'"{precomputed["y_u16"]}" AS "y" '
            f'FROM "{result.table}"'
        )
    elif is_very_large and "bounds" in props.get("data", {}).get("projection", {}):
        x_min, x_max = result.x_bounds
        y_min, y_max = result.y_bounds
        x_scale = 65535 / (x_max - x_min)
        y_scale = 65535 / (y_max - y_min)
        prewarm_sql = (
            f"SELECT "
            f'((COALESCE("{result.x_column}", {x_min}) - {x_min}) * {x_scale})::USMALLINT AS "x", '
            f'((COALESCE("{result.y_column}", {y_min}) - {y_min}) * {y_scale})::USMALLINT AS "y" '
            f'FROM "{result.table}"'
        )
    elif is_gis:
        prewarm_sql = (
            f'SELECT "{result.x_column}"::FLOAT AS "x", '
            f'(LN(TAN(PI()/4 + "{result.y_column}" * PI() / 360))*180/PI())::FLOAT AS "y" '
            f'FROM "{result.table}"'
        )
    else:
        prewarm_sql = (
            f'SELECT "{result.x_column}"::FLOAT AS "x", '
            f'"{result.y_column}"::FLOAT AS "y" '
            f'FROM "{result.table}"'
        )
    return con, data_source, metadata, True, prewarm_sql, result


def _add_shutdown_endpoint(app) -> None:
    from fastapi import Response

    @app.post("/_internal/shutdown")
    def shutdown() -> Response:
        def _exit() -> None:
            time.sleep(0.1)
            os._exit(0)

        threading.Thread(target=_exit, daemon=True).start()
        return Response(status_code=204)


def main() -> int:
    _install_signal_handlers()
    _start_parent_watchdog(os.environ.get("GEOSPATIAL_ATLAS_PARENT_PID"))
    # Log received env + argv to aid debugging when the app UI is opaque.
    env_dump = {
        k: v for k, v in os.environ.items() if k.startswith("GEOSPATIAL_ATLAS_")
    }
    _log(f"argv={sys.argv!r}")
    _log(f"env={env_dump!r}")

    if len(sys.argv) < 2:
        print("usage: sidecar <dataset-path>", file=sys.stderr)
        return 2

    dataset_path = sys.argv[1]
    host = os.environ.get("GEOSPATIAL_ATLAS_HOST", "127.0.0.1")
    port_str = os.environ.get("GEOSPATIAL_ATLAS_PORT")
    if not port_str:
        print("GEOSPATIAL_ATLAS_PORT env var required", file=sys.stderr)
        return 2
    port = int(port_str)

    from embedding_atlas.server import make_server

    # MCP (Model Context Protocol) is opt-in via GEOSPATIAL_ATLAS_MCP.
    # The Tauri shell sets this when the user enables the "Claude Desktop
    # integration" toggle; absent/0 -> MCP endpoints are not mounted.
    _mcp_env = os.environ.get("GEOSPATIAL_ATLAS_MCP", "0").strip().lower()
    enable_mcp = _mcp_env in ("1", "true", "yes", "on")
    _log(f"MCP: {'enabled' if enable_mcp else 'disabled'}")

    # argv[2] is the optional row limit (0 or absent means "no limit").
    limit: Optional[int] = None
    if len(sys.argv) >= 3:
        try:
            n = int(sys.argv[2])
            if n > 0:
                limit = n
        except ValueError:
            _log(f"invalid limit argv[2]={sys.argv[2]!r}, ignoring")
    # argv[3] is the optional text column name (empty string = none).
    text_column: Optional[str] = None
    if len(sys.argv) >= 4 and sys.argv[3].strip():
        text_column = sys.argv[3].strip()
    _log(f"using row limit: {limit if limit else 'none'}; text column: {text_column!r}")
    connection, data_source, _meta, _is_gis, _prewarm_sql, _result = _fast_load(
        dataset_path, limit=limit, text_column=text_column
    )
    static_path = _resolve_static_dir()
    _log(f"static dir: {static_path}")

    app = make_server(
        data_source,
        static_path=static_path,
        duckdb_uri="server",
        mcp=enable_mcp,
        cors=False,
        duckdb_connection=connection,
        dataset_is_view=_result.is_view,
        materialise_thread=_result.materialise_thread,
        materialise_table=_result.materialise_table,
        materialise_error=_result.materialise_error,
        prewarm_arrow_queries=[_prewarm_sql],
    )
    _add_shutdown_endpoint(app)

    if enable_mcp:
        _log(f"MCP endpoint: http://{host}:{port}/mcp")
    _log(f"listening on http://{host}:{port}")

    import uvicorn

    uvicorn.run(app, host=host, port=port, access_log=False, log_level=logging.WARNING)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        # Bypass Python's orderly shutdown — our daemon watchdog threads
        # may be blocked on stdin.read(), which makes interpreter
        # finalisation hang / emit an ugly
        # "could not acquire lock for <stdin> at interpreter shutdown"
        # warning. os._exit skips that.
        os._exit(1)
    else:
        os._exit(code or 0)
