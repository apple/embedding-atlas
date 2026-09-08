# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from io import BytesIO
from typing import Callable

import duckdb
import pyarrow as pa
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .data_source import DataSource
from .utils import arrow_to_bytes, to_parquet_bytes


# Read-only SQL command prefixes. Anything starting with one of these is
# safe to memoize (no side effects on `dataset`). The cache is invalidated
# by any other command (ALTER, UPDATE, INSERT, CREATE, DROP, …).
_READONLY_RE = re.compile(
    r"^\s*(?:WITH\b|SELECT\b|TABLE\b|VALUES\b|DESCRIBE\b|SUMMARIZE\b|PRAGMA\b)", re.I
)


class _ArrowResultCache:
    """LRU cache of (sql -> Arrow IPC bytes), bounded by total bytes.

    The dataset is immutable per-load — the same `scatter_q` request
    fired by the viewer after every gesture release returns the same
    1.2 GB Arrow body byte-for-byte. Caching kills the redundant CTAS
    scans entirely (≤ 1 ms hit vs. ~200 ms miss at 300 M).

    Any non-SELECT command (ALTER, UPDATE, INSERT, DROP) clears the
    cache via ``invalidate()`` — the dataset's columns/values may have
    changed, so all cached results are now stale.
    """

    def __init__(self, max_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        # Pinned entries (typically prewarm) bypass the LRU eviction loop
        # AND the per-entry size cap — they are always retained until
        # explicit ``invalidate()``. This is what lets a 2.58 GB scatter
        # prewarm survive on a default 2 GB cache and survive subsequent
        # smaller misses that would otherwise evict it under LRU.
        self._pinned: dict[str, bytes] = {}

    @staticmethod
    def _key(sql: str) -> str:
        return hashlib.sha1(sql.encode("utf-8")).hexdigest()

    def get(self, sql: str) -> bytes | None:
        k = self._key(sql)
        with self._lock:
            pinned = self._pinned.get(k)
            if pinned is not None:
                self.hits += 1
                return pinned
            buf = self._cache.get(k)
            if buf is not None:
                self._cache.move_to_end(k)
                self.hits += 1
                return buf
            self.misses += 1
            return None

    def put(self, sql: str, buf: bytes) -> None:
        if len(buf) > self.max_bytes:
            return  # single response larger than cache budget — skip
        k = self._key(sql)
        with self._lock:
            if k in self._cache:
                self._size -= len(self._cache.pop(k))
            self._cache[k] = buf
            self._size += len(buf)
            while self._size > self.max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._size -= len(evicted)

    def put_pinned(self, sql: str, buf: bytes) -> None:
        """Store ``buf`` as a pinned entry — always retained, ignores the
        per-entry size cap and LRU eviction. Use only for prewarmed
        results whose size is known and intentional."""
        k = self._key(sql)
        with self._lock:
            self._pinned[k] = buf

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._size = 0
            self._pinned.clear()


def make_server(
    data_source: DataSource,
    *,
    static_path: str,
    mcp: bool = False,
    cors: bool | list[str] = False,
    duckdb_uri: str | None = None,
    duckdb_connection: "duckdb.DuckDBPyConnection | None" = None,
    dataset_is_view: bool = False,
    dataset_table_name: str = "dataset",
    materialise_thread: "threading.Thread | None" = None,
    materialise_table: str | None = None,
    materialise_error: list | None = None,
    prewarm_arrow_queries: list[str] | None = None,
):
    """Creates a server for hosting Geospatial Atlas.

    If ``duckdb_connection`` is provided, it is used as-is for server-mode
    queries (skipping the pandas-DF-based ``make_duckdb_connection``). This
    lets callers build a DuckDB table directly from Parquet via
    ``read_parquet`` and skip the pandas materialization, which for large
    geospatial files (tens of millions of rows) turns a minutes-long load
    into a few seconds.
    """

    additional_tables = data_source.additional_tables

    app = FastAPI()

    if cors is not None:
        if isinstance(cors, bool) and cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["*"],
            )
        elif isinstance(cors, list):
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["*"],
            )

    def _dataset_parquet_bytes() -> bytes:
        if data_source.dataset is not None:
            return to_parquet_bytes(data_source.dataset)
        # Fast-path: dataset was loaded directly into DuckDB; materialize
        # the full table on demand. This endpoint is only hit for
        # archive export / wasm fallback paths.
        if duckdb_connection is not None:
            return to_parquet_bytes(duckdb_connection.sql("SELECT * FROM dataset").df())
        raise RuntimeError("no dataset available to serialize")

    mount_bytes(
        app,
        "/data/dataset.parquet",
        "application/octet-stream",
        _dataset_parquet_bytes,
    )

    if additional_tables:
        for name, df in additional_tables.items():
            mount_bytes(
                app,
                f"/data/tables/{name}.parquet",
                "application/octet-stream",
                partial(to_parquet_bytes, df),
            )

    @app.get("/data/metadata.json")
    async def get_metadata():
        meta = {}
        # Database
        if duckdb_uri is None or duckdb_uri == "wasm":
            db_meta: dict = {"type": "wasm", "load": True}
            if additional_tables:
                db_meta["additionalTables"] = [
                    {"name": name, "url": f"tables/{name}.parquet"}
                    for name in additional_tables
                ]
            meta["database"] = db_meta
        elif duckdb_uri == "server":
            # Point to the server itself.
            meta["database"] = {"type": "rest"}
        else:
            # Point to the given uri.
            if duckdb_uri.startswith("http"):
                meta["database"] = {
                    "type": "rest",
                    "uri": duckdb_uri,
                    "load": True,
                }
            elif duckdb_uri.startswith("ws"):
                meta["database"] = {
                    "type": "socket",
                    "uri": duckdb_uri,
                    "load": True,
                }
            else:
                raise ValueError("invalid DuckDB uri")
        # MCP
        if mcp:
            meta["mcp"] = {"type": "websocket"}

        body = data_source.metadata | meta
        if debug_sql:
            keys = sorted(
                body.get("props", {}).get("data", {}).get("projection", {}).keys()
            )
            print(f"[metadata fetch] projection keys: {keys}", flush=True)
        return JSONResponse(
            body,
            headers={
                # Force the browser to refetch on every reload — otherwise stale
                # bounds/precomputed hints from a prior server config can drive
                # the viewer to issue u32-quantised queries against a server
                # that no longer advertises (or has) the precomputed columns.
                "Cache-Control": "no-store, no-cache, must-revalidate",
            },
        )

    @app.post("/data/cache/{name}")
    async def post_cache(request: Request, name: str):
        data_source.cache_set(name, await request.json())

    @app.get("/data/cache/{name}")
    async def get_cache(name: str):
        obj = data_source.cache_get(name)
        if obj is None:
            return Response(status_code=404)
        return obj

    @app.get("/data/archive.zip")
    async def make_archive():
        data = data_source.make_archive(static_path)
        return Response(content=data, media_type="application/zip")

    if duckdb_uri == "server":
        if duckdb_connection is None:
            duckdb_connection = make_duckdb_connection(
                data_source.dataset, additional_tables=additional_tables
            )
    else:
        duckdb_connection = None

    debug_sql = os.environ.get("GSA_DEBUG_SQL", "").lower() in ("1", "true", "yes")
    # Above this many bytes, the Arrow payload is sent via StreamingResponse
    # in 4 MiB chunks. uvicorn's default response path holds the entire
    # bytes object in a single h11 send, which silently drops the
    # connection for multi-hundred-MB / GB-scale scatter pulls — the
    # client sees status=000 / ERR_EMPTY_RESPONSE. 256 MiB threshold keeps
    # small queries on the fast path.
    STREAM_THRESHOLD = 256 * 1024 * 1024
    STREAM_CHUNK = 4 * 1024 * 1024

    def _chunked(buf: bytes, chunk: int = STREAM_CHUNK):
        n = len(buf)
        i = 0
        while i < n:
            yield buf[i : i + chunk]
            i += chunk

    # Per-connection scatter cache. The dataset is immutable per-load,
    # so the same Arrow query returns the same bytes every time the
    # viewer issues it (after every pan/zoom release, on color-by toggle,
    # …). A simple LRU short-circuits the redundant DuckDB scans.
    # Disabled via env so users can A/B at runtime.
    cache_disabled = os.environ.get("GSA_DISABLE_QUERY_CACHE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    arrow_cache: _ArrowResultCache | None = (
        None if cache_disabled else _ArrowResultCache()
    )

    # Lazy view→table promotion. fast_load_parquet returns the dataset as
    # a VIEW (no materialisation, ~5 ms) so the first map render is fast.
    # The viewer's color-by path issues ALTER TABLE ADD COLUMN + UPDATE
    # against `dataset` — DuckDB rejects both against a view. We promote
    # on the first such write: ~3-5 s for a 75 M-row file, paid only when
    # the user actually clicks color-by.
    #
    # Auto-detect the initial state — when ``dataset_is_view`` defaults to
    # False, look at duckdb_views() so the desktop sidecar binary (whose
    # entry point can't be updated without a PyInstaller rebuild) gets the
    # promotion path for free as soon as it imports a newer fast_load.py.
    def _detect_is_view() -> bool:
        if duckdb_connection is None:
            return False
        if dataset_is_view:
            return True
        try:
            row = duckdb_connection.sql(
                f"SELECT 1 FROM duckdb_views() WHERE view_name = '{dataset_table_name}'"
            ).fetchone()
            return row is not None
        except Exception:
            return False

    _dataset_state = {"is_view": _detect_is_view()}
    _dataset_state_lock = threading.Lock()

    def _ensure_dataset_materialised() -> None:
        if not _dataset_state["is_view"] or duckdb_connection is None:
            return
        with _dataset_state_lock:
            if not _dataset_state["is_view"]:
                return
            cur = duckdb_connection.cursor()
            try:
                # Background-materialise fast path: fast_load_parquet may
                # have started a thread that's already CTAS'd the dataset
                # into ``materialise_table``. If so, joining is either
                # instant (bg done) or a few seconds at most (bg in flight)
                # — vs. a synchronous ~5 s CTAS that always pays full cost.
                # Falls back to synchronous when the bg thread errored or
                # wasn't started (legacy callers, eager materialise mode).
                bg_succeeded = False
                if materialise_thread is not None and materialise_table is not None:
                    t_join = time.perf_counter()
                    materialise_thread.join()
                    if debug_sql:
                        print(
                            f"[bg-mat join] waited {(time.perf_counter() - t_join) * 1000:.1f} ms",
                            flush=True,
                        )
                    if not (materialise_error and materialise_error[0]):
                        cur.execute(f'DROP VIEW "{dataset_table_name}"')
                        cur.execute(
                            f'ALTER TABLE "{materialise_table}" RENAME TO "{dataset_table_name}"'
                        )
                        bg_succeeded = True
                if not bg_succeeded:
                    tmp = f"__{dataset_table_name}_mat_tmp__"
                    cur.execute(
                        f'CREATE TABLE "{tmp}" AS SELECT * FROM "{dataset_table_name}"'
                    )
                    cur.execute(f'DROP VIEW "{dataset_table_name}"')
                    cur.execute(f'ALTER TABLE "{tmp}" RENAME TO "{dataset_table_name}"')
            finally:
                cur.close()
            _dataset_state["is_view"] = False
            if arrow_cache is not None:
                arrow_cache.invalidate()

    # Eager promote: as soon as the bg materialise finishes, swap the
    # view for the table. Without this, the view stays in place until
    # the first write query (e.g. color-by ALTER) — which means
    # tooltip / scatter reads keep going through ``read_parquet(...)``
    # even after the spatially-sorted bg-mat table is ready. At 322 M
    # the unsorted-view tooltip path is ~10 s; the sorted-table path
    # is ~50 ms (DuckDB row-group min/max pruning on x/y).
    if (
        _dataset_state["is_view"]
        and materialise_thread is not None
        and materialise_table is not None
    ):

        def _eager_swap_when_ready() -> None:
            try:
                materialise_thread.join()
            except BaseException:
                return
            try:
                _ensure_dataset_materialised()
                if debug_sql:
                    print("[bg-mat eager-swap] view → table promotion done", flush=True)
            except BaseException as e:
                if debug_sql:
                    print(f"[bg-mat eager-swap] error: {e}", flush=True)

        threading.Thread(
            target=_eager_swap_when_ready,
            name="bg-mat-eager-swap",
            daemon=True,
        ).start()

    def handle_query(query: dict):
        assert duckdb_connection is not None
        sql = query["sql"]
        command = query["type"]
        t_start = time.perf_counter() if debug_sql else 0.0
        if debug_sql:
            preview = sql.replace("\n", " ")
            if len(preview) > 400:
                preview = preview[:400] + " ..."
            print(f"[sql {command}] {preview}", flush=True)

        # Fast path: cached Arrow body for an immutable read-only query.
        if (
            command == "arrow"
            and arrow_cache is not None
            and _READONLY_RE.match(sql) is not None
        ):
            cached = arrow_cache.get(sql)
            if cached is not None:
                if debug_sql:
                    dt = (time.perf_counter() - t_start) * 1000
                    print(
                        f"[sql arrow CACHE HIT {dt:6.1f}ms] -> {len(cached):,} bytes",
                        flush=True,
                    )
                if len(cached) >= STREAM_THRESHOLD:
                    return StreamingResponse(
                        _chunked(cached),
                        media_type="application/octet-stream",
                        headers={"Content-Length": str(len(cached))},
                    )
                return Response(
                    cached, headers={"Content-Type": "application/octet-stream"}
                )

        # Any non-read-only command may have mutated `dataset` — invalidate
        # the cache before we run it (we re-cache on the next read), and
        # promote the dataset view to a table if it isn't one yet (color-by
        # ALTER+UPDATE rejects views; the first such query pays the
        # materialisation cost).
        is_write = _READONLY_RE.match(sql) is None
        if is_write and command != "arrow":
            if arrow_cache is not None:
                arrow_cache.invalidate()
            _ensure_dataset_materialised()

        try:
            with duckdb_connection.cursor() as cursor:
                if command == "arrow":
                    # ``cursor.sql(sql).fetch_arrow_table()`` is two things at once:
                    #   * ~10× faster than ``cursor.execute(...).fetch_arrow_table()``
                    #     on large results (340 ms vs. 3.3 s on a 75 M-row scatter)
                    #     — the relation-based path streams batches in C++.
                    #   * Cursor-isolated. Calling ``con.sql()`` on the shared
                    #     connection has a nasty failure mode: if a query fails
                    #     (e.g. ``DESC SELECT *`` on a table with a GEOMETRY column,
                    #     which DuckDB can't serialise to Arrow IPC), the
                    #     connection enters a "pending query result" state and
                    #     EVERY subsequent query returns
                    #     ``Attempting to execute an unsuccessful or closed
                    #     pending query result`` until restart. The cursor
                    #     contains the failure to that one request.
                    rel = cursor.sql(sql)
                    buf = arrow_to_bytes(rel.fetch_arrow_table())
                    if debug_sql:
                        dt = (time.perf_counter() - t_start) * 1000
                        print(
                            f"[sql arrow {dt:6.1f}ms] -> {len(buf):,} bytes", flush=True
                        )
                    if arrow_cache is not None and _READONLY_RE.match(sql) is not None:
                        arrow_cache.put(sql, buf)
                    if len(buf) >= STREAM_THRESHOLD:
                        return StreamingResponse(
                            _chunked(buf),
                            media_type="application/octet-stream",
                            headers={"Content-Length": str(len(buf))},
                        )
                    return Response(
                        buf, headers={"Content-Type": "application/octet-stream"}
                    )
                result = cursor.execute(sql)
                if command == "exec":
                    if arrow_cache is not None:
                        arrow_cache.invalidate()
                    if debug_sql:
                        dt = (time.perf_counter() - t_start) * 1000
                        print(f"[sql exec {dt:6.1f}ms] ok", flush=True)
                    return JSONResponse({})
                elif command == "json":
                    data = result.df().to_json(orient="records")
                    if debug_sql:
                        dt = (time.perf_counter() - t_start) * 1000
                        print(
                            f"[sql json {dt:6.1f}ms] -> {len(data):,} bytes", flush=True
                        )
                    return Response(data, headers={"Content-Type": "application/json"})
                else:
                    raise ValueError(f"Unknown command {command}")
        except Exception as e:
            if debug_sql:
                dt = (time.perf_counter() - t_start) * 1000
                print(f"[sql ERR {command} {dt:6.1f}ms] {e}", flush=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    def handle_selection(query: dict):
        assert duckdb_connection is not None
        predicate = query.get("predicate", None)
        format = query["format"]

        try:
            with duckdb_connection.cursor() as cursor:
                statement = (
                    f"SELECT * FROM dataset WHERE {predicate}"
                    if predicate is not None
                    else "SELECT * FROM dataset"
                )
                # Convert to DataFrame and save since we've disabled DuckDB filesystem access
                result = cursor.sql(statement).to_df()

                output = BytesIO()
                if format == "parquet":
                    result.to_parquet(output, index=False)
                elif format == "json":
                    result.to_json(output, orient="records", index=False)
                elif format == "jsonl":
                    result.to_json(output, orient="records", lines=True, index=False)
                elif format == "csv":
                    result.to_csv(output, index=False)
                else:
                    raise ValueError("invalid format")

                return Response(
                    output.getvalue(),
                    headers={"Content-Type": "application/octet-stream"},
                )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    executor = concurrent.futures.ThreadPoolExecutor()

    @app.get("/data/query")
    async def get_query(req: Request):
        data = json.loads(req.query_params["query"])
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_query(data)
        )

    @app.post("/data/query")
    async def post_query(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_query(data)
        )

    @app.post("/data/selection")
    async def post_selection(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_selection(data)
        )

    if mcp:
        make_mcp_proxy(app)

    # Pre-warm the arrow cache with the queries the viewer will issue on
    # cold load. The big scatter SELECT scans 600 MB of column data and
    # takes ~3-4 s on the first hit. We run the prewarm SYNCHRONOUSLY
    # here so the metadata endpoint (and therefore "server ready") only
    # responds after the scatter bytes are cached — when the browser
    # immediately requests them, the response is a cache hit (~0.3 s for
    # network transfer instead of ~4 s for scan + transfer). The wall-
    # clock work is identical, but the user-perceived latency between
    # URL load and first paint drops from ~5 s to ~1.5 s. Disable via
    # ``GSA_DISABLE_PREWARM=1`` if the load-time cost ever shows up
    # somewhere it shouldn't (e.g. one-shot CLI exports).
    prewarm_disabled = os.environ.get("GSA_DISABLE_PREWARM", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if (
        prewarm_arrow_queries
        and arrow_cache is not None
        and duckdb_connection is not None
        and not prewarm_disabled
    ):
        for sql in prewarm_arrow_queries or []:
            try:
                t = time.perf_counter()
                cursor = duckdb_connection.cursor()
                try:
                    rel = cursor.sql(sql)
                    if rel is None:
                        continue
                    tbl = rel.fetch_arrow_table()
                    if isinstance(tbl, pa.Table) and tbl.num_rows > 0:
                        tbl = tbl.combine_chunks()
                    buf = arrow_to_bytes(tbl)
                    # Pin so the entry is retained regardless of size and
                    # is never evicted by LRU pressure from runtime puts.
                    # The default 2 GiB cache cap silently dropped the
                    # 2.58 GiB scatter for 322 M-row datasets — every
                    # browser scatter then re-executed for ~90 s.
                    arrow_cache.put_pinned(sql, buf)
                    if debug_sql:
                        print(
                            f"[prewarm] pinned {len(buf):,} bytes in "
                            f"{(time.perf_counter() - t) * 1000:.1f} ms",
                            flush=True,
                        )
                finally:
                    cursor.close()
            except Exception as exc:  # noqa: BLE001 — best effort
                if debug_sql:
                    print(f"[prewarm] failed: {exc}", flush=True)

    # Static files for the frontend
    app.mount("/", StaticFiles(directory=static_path, html=True))

    return app


class WebSocketHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.pending_requests: dict[str, asyncio.Future] = {}
        self.is_connected = True

    async def handle_connection(self):
        try:
            while self.is_connected:
                data = await self.websocket.receive_text()
                await self._handle_message(data)
        except Exception as _:
            pass
        finally:
            await self._cleanup()

    async def _handle_message(self, data: str):
        try:
            response = json.loads(data)
            request_id = response.get("id")
            if request_id and request_id in self.pending_requests:
                future = self.pending_requests.pop(request_id)
                if not future.done():
                    future.set_result(response.get("response"))
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")

    async def _cleanup(self):
        self.is_connected = False
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()

    async def send_request(self, request: dict, timeout: float = 300.0) -> dict:
        """Send a request to the WebSocket and wait for response.

        Timeout defaults to 5 min so MCP tools that run heavy DuckDB scans
        over 75 M+ row datasets don't get guillotined at 30 s.
        """
        if not self.is_connected:
            raise HTTPException(status_code=503, detail="WebSocket disconnected")

        request_id = str(uuid.uuid4())
        payload = {"id": request_id, "request": request}

        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            await self.websocket.send_text(json.dumps(payload))
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            if not self.is_connected:
                raise HTTPException(status_code=503, detail="WebSocket disconnected")
            else:
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )

    async def send_close(self):
        try:
            await self.websocket.send_text(json.dumps({"control": "close"}))
        except Exception:
            pass


def make_mcp_proxy(app: FastAPI):
    """Expose the viewer's MCP tools via the standard Streamable HTTP transport.

    Adds three routes:
      * ``GET/POST/DELETE /mcp`` — Streamable HTTP endpoint, the format
        Claude Desktop and every other MCP client speak natively. No
        stdio bridge / Node shim required.
      * ``WS /data/mcp_websocket`` — the viewer JS connects here on load
        and registers as the tool executor.
      * ``POST /mcp_legacy`` — the old home-grown "plain POST forward"
        endpoint, kept for backwards compatibility until downstreams
        migrate. New integrations should use ``/mcp``.
    """
    from .mcp_bridge import MCPBridge

    last_handler: dict[str, WebSocketHandler | None] = {"handler": None}

    def get_handler() -> "WebSocketHandler | None":
        return last_handler["handler"]

    @app.websocket("/data/mcp_websocket")
    async def websocket_mcp_ws(websocket: WebSocket):
        await websocket.accept()
        handler = WebSocketHandler(websocket)
        if last_handler["handler"] is not None:
            await last_handler["handler"].send_close()
        last_handler["handler"] = handler
        await handler.handle_connection()
        if last_handler["handler"] == handler:
            last_handler["handler"] = None

    # Standards-compliant MCP endpoint (Streamable HTTP).
    bridge = MCPBridge(get_handler)
    bridge.mount(app, path="/mcp")

    # Legacy plain-POST endpoint — will be removed after users migrate.
    @app.post("/mcp_legacy")
    async def post_mcp_legacy(request: Request):
        handler = last_handler["handler"]
        if handler is None or not handler.is_connected:
            raise HTTPException(status_code=503, detail="No MCP WebSocket connected")
        return await handler.send_request(await request.json())


def make_duckdb_connection(df, additional_tables: dict | None = None):
    con = duckdb.connect(":memory:")
    _ = df  # used in the query
    con.sql("CREATE TABLE dataset AS (SELECT * FROM df)")
    if additional_tables is not None:
        for name, additional_df in additional_tables.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"invalid table name: {name!r}")
            _ = additional_df  # used in the query
            con.sql(f"CREATE TABLE {name} AS (SELECT * FROM additional_df)")
    con.sql("SET enable_external_access = false")
    con.sql("SET lock_configuration = true")
    return con


def parse_range_header(request: Request, content_length: int):
    value = request.headers.get("Range")
    if value is not None:
        m = re.match(r"^ *bytes *= *([0-9]+) *- *([0-9]+) *$", value)
        if m is not None:
            r0 = int(m.group(1))
            r1 = int(m.group(2)) + 1
            if r0 < r1 and r0 <= content_length and r1 <= content_length:
                return (r0, r1)
    return None


def mount_bytes(
    app: FastAPI, url: str, media_type: str, make_content: Callable[[], bytes]
):
    @lru_cache(maxsize=1)
    def get_content() -> bytes:
        return make_content()

    @app.head(url)
    async def head(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            length = len(content)
        else:
            length = bytes_range[1] - bytes_range[0]
        return Response(
            headers={
                "Content-Length": str(length),
                "Content-Type": media_type,
            }
        )

    @app.get(url)
    async def get(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            return Response(content=content)
        else:
            r0, r1 = bytes_range
            result = content[r0:r1]
            return Response(
                content=result,
                headers={
                    "Content-Length": str(r1 - r0),
                    "Content-Range": f"bytes {r0}-{r1 - 1}/{len(content)}",
                    "Content-Type": media_type,
                },
                media_type=media_type,
                status_code=206,
            )
