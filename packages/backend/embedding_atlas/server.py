# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import asyncio
import concurrent.futures
import json
import os
import re
import uuid
from functools import lru_cache
from typing import Callable

import duckdb
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .data_source import DataSource
from .projection import _projection_for_images, _projection_for_texts, _project_text_with_sentence_transformers
from .utils import arrow_to_bytes, to_parquet_bytes


def make_server(
    data_source: DataSource,
    static_path: str,
    duckdb_uri: str | None = None,
):
    """Creates a server for hosting Embedding Atlas"""

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    mount_bytes(
        app,
        "/data/dataset.parquet",
        "application/octet-stream",
        lambda: to_parquet_bytes(data_source.dataset),
    )

    @app.get("/data/metadata.json")
    async def get_metadata():
        if duckdb_uri is None or duckdb_uri == "wasm":
            db_meta = {"database": {"type": "wasm", "load": True}}
        elif duckdb_uri == "server":
            # Point to the server itself.
            db_meta = {"database": {"type": "rest"}}
        else:
            # Point to the given uri.
            if duckdb_uri.startswith("http"):
                db_meta = {
                    "database": {"type": "rest", "uri": duckdb_uri, "load": True}
                }
            elif duckdb_uri.startswith("ws"):
                db_meta = {
                    "database": {"type": "socket", "uri": duckdb_uri, "load": True}
                }
            else:
                raise ValueError("invalid DuckDB uri")
        return data_source.metadata | db_meta

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
        duckdb_connection = make_duckdb_connection(data_source.dataset)
    else:
        duckdb_connection = None

    def handle_query(query: dict):
        assert duckdb_connection is not None
        sql = query["sql"]
        command = query["type"]
        with duckdb_connection.cursor() as cursor:
            try:
                result = cursor.execute(sql)
                if command == "exec":
                    return JSONResponse({})
                elif command == "arrow":
                    buf = arrow_to_bytes(result.arrow())
                    return Response(
                        buf, headers={"Content-Type": "application/octet-stream"}
                    )
                elif command == "json":
                    data = result.df().to_json(orient="records")
                    return Response(data, headers={"Content-Type": "application/json"})
                else:
                    raise ValueError(f"Unknown command {command}")
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

    def handle_selection(query: dict):
        assert duckdb_connection is not None
        predicate = query.get("predicate", None)
        format = query["format"]
        formats = {
            "json": "(FORMAT JSON, ARRAY true)",
            "jsonl": "(FORMAT JSON)",
            "csv": "(FORMAT CSV)",
            "parquet": "(FORMAT parquet)",
        }
        with duckdb_connection.cursor() as cursor:
            filename = ".selection-" + str(uuid.uuid4()) + ".tmp"
            try:
                if predicate is not None:
                    cursor.execute(
                        f"COPY (SELECT * FROM dataset WHERE {predicate}) TO '{filename}' {formats[format]}"
                    )
                else:
                    cursor.execute(f"COPY dataset TO '{filename}' {formats[format]}")
                with open(filename, "rb") as f:
                    buffer = f.read()
                    return Response(
                        buffer, headers={"Content-Type": "application/octet-stream"}
                    )
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            finally:
                try:
                    os.unlink(filename)
                except Exception:
                    pass

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

    def handle_compute_embedding(request_data: dict):
        """
        Compute embeddings using Python backend.
        Request format:
        {
            "data": [...],  # list of strings (text) or image data (base64 or bytes)
            "type": "text" | "image",
            "model": "model-name",
            "batch_size": 32,  # optional
            "umap_args": {}  # optional UMAP parameters
        }
        """
        try:
            data_list = request_data["data"]
            embedding_type = request_data["type"]
            model = request_data["model"]
            batch_size = request_data.get("batch_size", None)
            umap_args = request_data.get("umap_args", {})

            if embedding_type == "text":
                # Use the text projection function
                proj = _projection_for_texts(
                    texts=data_list,
                    model=model,
                    batch_size=batch_size,
                    text_projector=_project_text_with_sentence_transformers,
                    umap_args=umap_args,
                )
            elif embedding_type == "image":
                # Convert base64 image data if needed
                from io import BytesIO
                import base64

                processed_images = []
                for img_data in data_list:
                    if isinstance(img_data, str) and img_data.startswith("data:image"):
                        # Extract base64 part
                        base64_data = img_data.split(",", 1)[1] if "," in img_data else img_data
                        img_bytes = base64.b64decode(base64_data)
                        processed_images.append({"bytes": img_bytes})
                    elif isinstance(img_data, dict) and "bytes" in img_data:
                        # Already in the right format
                        processed_images.append(img_data)
                    else:
                        # Assume it's raw bytes or string bytes
                        if isinstance(img_data, str):
                            img_bytes = base64.b64decode(img_data)
                        else:
                            img_bytes = img_data
                        processed_images.append({"bytes": img_bytes})

                proj = _projection_for_images(
                    images=processed_images,
                    model=model,
                    batch_size=batch_size,
                    umap_args=umap_args,
                )
            else:
                return JSONResponse(
                    {"error": f"Invalid embedding type: {embedding_type}"},
                    status_code=400
                )

            # Return projection results
            return JSONResponse({
                "projection": proj.projection.tolist(),
                "knn_indices": proj.knn_indices.tolist(),
                "knn_distances": proj.knn_distances.tolist(),
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                {"error": str(e), "traceback": traceback.format_exc()},
                status_code=500
            )

    @app.post("/api/compute-embedding")
    async def compute_embedding(req: Request):
        """Compute embeddings using the Python backend"""
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_compute_embedding(data)
        )

    @app.get("/api/capabilities")
    async def get_capabilities():
        """Return backend capabilities"""
        return JSONResponse({
            "embedding_computation": True,
            "supported_models": {
                "text": [
                    "all-MiniLM-L6-v2",
                    "paraphrase-multilingual-mpnet-base-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                ],
                "image": [
                    "facebook/ijepa_vith14_1k",
                    "facebook/dinov2-small",
                    "facebook/dinov2-base",
                    "facebook/dinov2-large",
                    "google/vit-base-patch16-384",
                    "openai/clip-vit-base-patch32",
                ],
            }
        })

    # Static files for the frontend
    app.mount("/", StaticFiles(directory=static_path, html=True))

    return app


def make_duckdb_connection(df):
    con = duckdb.connect(":memory:")
    _ = df  # used in the query
    con.sql("CREATE TABLE dataset AS (SELECT * FROM df)")
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
