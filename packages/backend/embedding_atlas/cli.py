# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Command line interface."""

import importlib
import json
import logging
import pathlib
import socket
from pathlib import Path

import click
import inquirer
import numpy as np
import pandas as pd
import uvicorn

from .cache import sha256_hexdigest
from .data_source import DataSource
from .options import make_embedding_atlas_props
from .server import make_server
from .utils import (
    apply_logging_config,
    load_huggingface_data,
    load_pandas_data,
    logger,
    parse_kv,
)
from .version import __version__


class JSONParamType(click.ParamType):
    """Accepts a JSON string or a path to a JSON file."""

    name = "JSON"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        try:
            if value.strip().startswith("{"):
                return json.loads(value)
            with open(value) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}", param, ctx)
        except (FileNotFoundError, OSError) as e:
            self.fail(f"Could not read file: {e}", param, ctx)


def find_column_name(existing_names, candidate):
    if candidate not in existing_names:
        return candidate
    else:
        index = 1
        while True:
            s = f"{candidate}_{index}"
            if s not in existing_names:
                return s
            index += 1


def determine_and_load_data(filename: str, splits: list[str] | None = None):
    hf_prefix = "hf://datasets/"
    if filename.startswith(hf_prefix):
        return load_huggingface_data(filename[len(hf_prefix) :], splits)

    # Remote URLs (http://, s3://, ...), pandas handle these.
    if "://" in filename:
        return load_pandas_data(filename)

    p = Path(filename).expanduser()
    if p.is_file():
        return load_pandas_data(filename)
    if p.is_dir():
        return load_huggingface_data(filename, splits)

    # Not on disk. Only fall through to HF if the input looks like a Hub
    # name (no extension and contains at most one '/'); otherwise surface the missing-file
    # error so typos don't quietly hit the network.
    if Path(filename).suffix == "" and len(filename.split("/")) <= 2:
        return load_huggingface_data(filename, splits)

    raise FileNotFoundError(f"No such file or directory: {filename!r}")


def query_dataframe(query: str, data: pd.DataFrame) -> pd.DataFrame:
    import duckdb

    _ = data  # used in query
    return duckdb.sql(query).df()


def load_datasets(
    inputs: list[str],
    splits: list[str] | None = None,
    query: str | None = None,
    sample: int | None = None,
) -> pd.DataFrame:
    existing_column_names = set()
    dataframes = []
    for fn in inputs:
        print("Loading data from " + fn)
        df = determine_and_load_data(fn, splits=splits)
        dataframes.append(df)
        for c in df.columns:
            existing_column_names.add(c)

    file_name_column = find_column_name(existing_column_names, "FILE_NAME")
    for df, fn in zip(dataframes, inputs):
        df[file_name_column] = fn

    df = pd.concat(dataframes)

    if query is not None:
        df = query_dataframe(query, df)

    if sample:
        df = df.sample(n=sample, axis=0, random_state=np.random.RandomState(42))

    return df


def prompt_for_column(df: pd.DataFrame, message: str) -> str | None:
    question = [
        inquirer.List(
            "arg",
            message=message,
            choices=sorted(["(none)"] + [str(c) for c in df.columns]),
        ),
    ]
    r = inquirer.prompt(question)
    if r is None:
        return None
    text = r["arg"]  # type: ignore
    if text == "(none)":
        text = None
    return text


def find_available_port(start_port: int, max_attempts: int = 10, host="localhost"):
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("No available ports found in the given range")


def find_gis_columns(columns):
    """Detect GIS coordinate columns using a priority hierarchy.

    Priority:
      1. Explicit lat/lon pairs: lon/lat, longitude/latitude, lng/lat
      2. Generic x/y (less reliable but common in GIS tools)

    Returns (x_column, y_column) or (None, None).
    """
    cols_map = {c.lower(): c for c in columns}
    pairs = [
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("lng", "lat"),
        ("x", "y"),
    ]
    for x_cand, y_cand in pairs:
        if x_cand in cols_map and y_cand in cols_map:
            return cols_map[x_cand], cols_map[y_cand]
    return None, None


def find_geometry_column(df):
    """Detect a WKB geometry column in the dataframe.

    Checks for geoparquet metadata first (authoritative), then falls back
    to common column names with binary dtype.

    Returns the column name or None.
    """
    # Check geoparquet metadata via pyarrow
    try:
        import json
        import pyarrow as pa

        table = pa.Table.from_pandas(df) if hasattr(df, "dtypes") else None
        # If loaded with read_parquet, the schema metadata may be on the df attrs
        schema_meta = getattr(df, "attrs", {}).get("_schema_metadata", None)
        if schema_meta is None and table is not None:
            schema_meta = table.schema.metadata
        if schema_meta and b"geo" in schema_meta:
            geo = json.loads(schema_meta[b"geo"])
            primary = geo.get("primary_column")
            if primary and primary in df.columns:
                return primary
    except Exception:
        pass

    # Fallback: look for common geometry column names with binary/object dtype
    for name in ["geometry", "geom", "wkb_geometry", "the_geom", "geo"]:
        for col in df.columns:
            if col.lower() == name and df[col].dtype == "object":
                # Check if first non-null value looks like WKB
                sample = df[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], (bytes, bytearray)):
                    return col
    return None


def extract_coordinates_from_geometry(df, geom_column):
    """Extract lon/lat columns from a WKB geometry column using struct.

    Only supports Point geometries (WKB type 1). Falls back to DuckDB
    spatial if available for other geometry types.
    """
    import struct

    def parse_wkb_point(wkb):
        """Parse a WKB Point and return (lon, lat)."""
        if wkb is None or not isinstance(wkb, (bytes, bytearray)):
            return (None, None)
        if len(wkb) < 21:
            return (None, None)
        byte_order = wkb[0]
        fmt_i = "<I" if byte_order == 1 else ">I"
        fmt_dd = "<dd" if byte_order == 1 else ">dd"
        geom_type = struct.unpack(fmt_i, wkb[1:5])[0]
        # Type 1 = Point, 0x80000001 = Point with SRID
        base_type = geom_type & 0xFF
        if base_type != 1:
            return (None, None)
        offset = 5
        if geom_type & 0x20000000:  # has SRID
            offset += 4
        if len(wkb) < offset + 16:
            return (None, None)
        x, y = struct.unpack(fmt_dd, wkb[offset : offset + 16])
        return (x, y)

    coords = df[geom_column].apply(parse_wkb_point)
    lon_col = find_column_name(set(df.columns), "lon")
    lat_col = find_column_name(set(df.columns), "lat")
    df[lon_col] = coords.apply(lambda c: c[0])
    df[lat_col] = coords.apply(lambda c: c[1])

    # Drop rows where extraction failed
    valid = df[lon_col].notna() & df[lat_col].notna()
    if not valid.all():
        n_dropped = (~valid).sum()
        logger.warning(f"Dropped {n_dropped} rows with non-point or invalid geometries")
        df = df[valid].reset_index(drop=True)

    return df, lon_col, lat_col


def import_modules(names: list[str]):
    """Import the given list of modules."""
    for name in names:
        importlib.import_module(name)


def _try_fast_load(
    *,
    inputs: list[str],
    splits: list[str],
    query: str | None,
    sample: int | None,
    text: str | None,
    image: str | None,
    audio: str | None,
    vector: str | None,
    x_column: str | None,
    y_column: str | None,
    duckdb_uri: str,
):
    """Return a populated DuckDB connection for the fast path, or ``None``
    if any condition rules it out. Conditions:

      * exactly one input
      * ``.parquet`` extension on disk
      * no HuggingFace splits, ``--query``, or ``--sample``
      * no embedding modality (text/image/audio/vector) — we'd need
        pandas to run the projection
      * ``--duckdb server`` (fast path only helps server-mode today)

    The fast path still respects ``x_column`` / ``y_column`` if set, but
    for the common case of auto-detection it uses the logic from
    :mod:`.fast_load`.
    """
    if len(inputs) != 1:
        return None
    if splits:
        return None
    if query is not None or sample is not None:
        return None
    if text or image or audio or vector:
        return None
    if duckdb_uri != "server":
        return None
    path = inputs[0]
    if path.startswith("hf://") or not pathlib.Path(path).is_file():
        return None
    if pathlib.Path(path).suffix.lower() != ".parquet":
        return None

    from .fast_load import fast_load_parquet

    def _print(stage: str, pct: float, detail: str) -> None:
        if stage == "load":
            # Overwriting progress bar would require carriage returns; keep it
            # simple and log at every ~10% step.
            if int(pct) % 10 == 0:
                print(f"  [{stage}] {pct:5.1f}%  {detail}")
        else:
            print(f"  [{stage}] {detail}")

    try:
        result = fast_load_parquet(path, progress=_print)
    except Exception as e:
        print(f"(fast loader unavailable: {e} — falling back to pandas)")
        return None

    # Honor ``--x`` / ``--y`` overrides without falling back to pandas —
    # the fast loader has already exposed every parquet column via the
    # view, so re-labelling x/y is just metadata. We re-compute bounds
    # for the chosen columns; that's a single column-stat read (~1 s
    # even on 15+ GB files).
    if x_column or y_column:
        cols_lower = {c.lower(): c for c in result.columns}
        if x_column and x_column.lower() not in cols_lower:
            print(
                f"(--x {x_column!r} not found in dataset; "
                f"falling back to pandas to honor the override)"
            )
            return None
        if y_column and y_column.lower() not in cols_lower:
            print(
                f"(--y {y_column!r} not found in dataset; "
                f"falling back to pandas to honor the override)"
            )
            return None
        new_x = cols_lower[x_column.lower()] if x_column else result.x_column
        new_y = cols_lower[y_column.lower()] if y_column else result.y_column
        if (new_x, new_y) != (result.x_column, result.y_column):
            print(f"  [override] x={new_x}, y={new_y}")
            result.x_column = new_x
            result.y_column = new_y
            # Bounds are tied to whichever columns are x/y, so refresh.
            result.x_bounds = None
            result.y_bounds = None
            try:
                from .fast_load import quote_ident

                row = result.connection.sql(
                    f"SELECT MIN({quote_ident(new_x)}), MAX({quote_ident(new_x)}), "
                    f"MIN({quote_ident(new_y)}), MAX({quote_ident(new_y)}) "
                    f"FROM {quote_ident(result.table)}"
                ).fetchone()
                if row is not None and all(v is not None for v in row):
                    x_min, x_max, y_min, y_max = (float(v) for v in row)
                    if x_max > x_min:
                        result.x_bounds = (x_min, x_max)
                    if y_max > y_min:
                        result.y_bounds = (y_min, y_max)
            except Exception:
                pass
    return result


def _run_fast_path(
    *,
    fast_connection,
    static: str | None,
    host: str,
    port: int,
    enable_auto_port: bool,
    enable_mcp: bool,
    cors,
    duckdb_uri: str,
):
    """Serve the pre-populated DuckDB connection via FastAPI+uvicorn.

    Mirrors the relevant tail end of :func:`main` without the pandas
    bookkeeping.
    """
    from .cache import sha256_hexdigest
    from .data_source import DataSource
    from .options import make_embedding_atlas_props
    from .server import make_server
    from .version import __version__

    con = fast_connection.connection
    # ``fast_load_parquet`` already provisioned a stable per-row id by
    # projecting the parquet reader's ``file_row_number`` virtual column.
    # No ALTER TABLE / UPDATE needed — and importantly cannot be done on
    # a view, which is what the loader now produces.
    id_col = fast_connection.id_column

    props = make_embedding_atlas_props(
        row_id=id_col,
        x=fast_connection.x_column,
        y=fast_connection.y_column,
        neighbors=None,
        text=None,
        point_size=None,
        stop_words=None,
        labels=None,
        is_gis=True,
    )
    # Wire-packing strategy depends on row count:
    #
    # row_count <= 200 M  → f32 wire (Path C). ~2 m precision at the
    #   equator (sub-pixel at any zoom). 9 B/point. For 200 M rows that
    #   is 1.8 GB on the wire — under V8's hard 2 GB ArrayBuffer cap that
    #   ``mosaic-core``'s ``response.arrayBuffer()`` allocation depends on.
    #
    # row_count > 200 M  → u32 wire (Path A: precomputed __x_u32__ /
    #   __y_u32__ columns from ``fast_load_parquet``). 9 B/point with a
    #   category column, 8 B/point without — same on-the-wire size as
    #   f32, so this path is selected for the *quantisation* it enables
    #   (zero per-row cast in SQL, GPU-side u32→f32 unpack instead of
    #   a JS-side Float32Array allocation), not for wire savings. u32
    #   precision at the eubucco 40°-lon span is ~1.5 cm per step —
    #   sub-pixel at any zoom, including the street-level views where
    #   the prior u16 path produced a visible quantization grid.
    #
    #   The 322 M eubucco file has no category column, so the wire is
    #   2.6 GB. Modern V8/Chrome handle this above the historic 2 GB
    #   ceiling on 64-bit hosts; if a future dataset trips it we'll
    #   either split into per-axis queries or enable Arrow IPC zstd
    #   compression server-side (the sort key is __x_u32__,__y_u32__,
    #   so adjacent rows have tiny deltas — zstd squashes the wire to
    #   ~10 % of raw u32).
    #
    # In BOTH cases we advertise ``viewportHint`` so ``queryApproximateDensity``
    # skips its 5 s ``APPROX_QUANTILE + STDDEV`` round trip on cold load.
    LARGE_ROW_THRESHOLD = 200_000_000
    is_very_large = fast_connection.row_count > LARGE_ROW_THRESHOLD
    if (
        fast_connection.x_bounds is not None
        and fast_connection.y_bounds is not None
        and "data" in props
        and "projection" in props["data"]
    ):
        x_min, x_max = fast_connection.x_bounds
        y_min, y_max = fast_connection.y_bounds
        props["data"]["projection"]["viewportHint"] = {
            "centerX": 0.5 * (x_min + x_max),
            "centerY": 0.5 * (y_min + y_max),
            "rangeX": x_max - x_min,
            "rangeY": y_max - y_min,
            "rowCount": fast_connection.row_count,
        }
        if is_very_large:
            # Trigger Path A in EmbeddingViewMosaic: pre-quantized cols,
            # zero per-row arithmetic in the SQL. When the loader produced
            # a Mercator-packed y column, use that — saves ~5.9 s of JS
            # Mercator on 322 M rows by pushing the projection into DuckDB
            # at view-definition time.
            use_merc_y = (
                fast_connection.quantised_merc_y_column is not None
                and fast_connection.merc_y_bounds is not None
            )
            if use_merc_y:
                merc_y_min, merc_y_max = fast_connection.merc_y_bounds
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
                fast_connection.quantised_x_column is not None
                and fast_connection.quantised_y_column is not None
            ):
                y_packed_col = (
                    fast_connection.quantised_merc_y_column
                    if use_merc_y
                    else fast_connection.quantised_y_column
                )
                # API field names ``x_u16``/``y_u16`` are kept as opaque
                # identifiers — they originated when the wire was u16 and
                # are now u32 column references. The viewer treats them
                # as "the precomputed packed column"; the bit width is
                # implied by the typed-array class that arrives over the
                # wire (Uint32Array post-migration).
                props["data"]["projection"]["precomputed"] = {
                    "x_u16": fast_connection.quantised_x_column,
                    "y_u16": y_packed_col,
                    "y_is_mercator": use_merc_y,
                }
            # Tell the browser to skip the deferred density refinement
            # (which is a 200 s+ APPROX_QUANTILE on 322 M rows). The
            # ``viewportHint`` is good enough — sub-percent error in the
            # density colour ramp is invisible at continental zoom.
            props["data"]["projection"]["viewportHint"]["skipDeferredRefine"] = True
    metadata = {"props": props}
    identifier = sha256_hexdigest(
        [__version__, [fast_connection.table], metadata], scope="DataSource"
    )
    data_source = DataSource(identifier, None, metadata)

    if static is None:
        static = str((pathlib.Path(__file__).parent / "static").resolve())

    cors_config = False
    if cors is not None:
        if cors == "":
            cors_config = True
        else:
            cors_config = [
                domain.strip() for domain in cors.split(",") if domain.strip()
            ]

    # Pre-warm the scatter query — this is the single biggest cold-load
    # round trip on the GIS fast path (~3-4 s for 75 M rows). The viewer
    # builds the SQL via Mosaic from these exact columns and casts; we
    # mirror that string here so the cache hit is byte-identical when the
    # browser fires its first scatter request. The prewarm thread runs
    # in parallel with uvicorn startup + JS bundle download, so the user
    # doesn't pay any extra wall time.
    x_col = fast_connection.x_column
    y_col = fast_connection.y_column
    is_gis = props.get("data", {}).get("projection", {}).get("isGis", False)
    # Mirror the wire-packing path the browser actually uses so the
    # cache key is byte-identical and the prewarm hits.
    if is_very_large and "precomputed" in props.get("data", {}).get("projection", {}):
        # Path A — precomputed u32 columns, no per-row arithmetic.
        precomputed = props["data"]["projection"]["precomputed"]
        prewarm_sql = (
            f'SELECT "{precomputed["x_u16"]}" AS "x", '
            f'"{precomputed["y_u16"]}" AS "y" '
            f'FROM "{fast_connection.table}"'
        )
    elif is_very_large and "bounds" in props.get("data", {}).get("projection", {}):
        # Path B — on-the-fly UINTEGER cast (fallback when precomputed
        # cols weren't generated).
        x_min, x_max = fast_connection.x_bounds
        y_min, y_max = fast_connection.y_bounds
        U32_MAX = 4_294_967_295
        x_scale = U32_MAX / (x_max - x_min)
        y_scale = U32_MAX / (y_max - y_min)
        prewarm_sql = (
            f"SELECT "
            f'((COALESCE("{x_col}", {x_min}) - {x_min}) * {x_scale})::UINTEGER AS "x", '
            f'((COALESCE("{y_col}", {y_min}) - {y_min}) * {y_scale})::UINTEGER AS "y" '
            f'FROM "{fast_connection.table}"'
        )
    elif is_gis:
        # Path C with GIS — server-side Mercator on f32 wire.
        prewarm_sql = (
            f'SELECT "{x_col}"::FLOAT AS "x", '
            f'(LN(TAN(PI()/4 + "{y_col}" * PI() / 360))*180/PI())::FLOAT AS "y" '
            f'FROM "{fast_connection.table}"'
        )
    else:
        # Path C non-GIS — plain f32 cast.
        prewarm_sql = (
            f'SELECT "{x_col}"::FLOAT AS "x", '
            f'"{y_col}"::FLOAT AS "y" '
            f'FROM "{fast_connection.table}"'
        )

    app = make_server(
        data_source,
        static_path=static,
        duckdb_uri=duckdb_uri,
        mcp=enable_mcp,
        cors=cors_config,
        duckdb_connection=con,
        dataset_is_view=fast_connection.is_view,
        materialise_thread=fast_connection.materialise_thread,
        materialise_table=fast_connection.materialise_table,
        materialise_error=fast_connection.materialise_error,
        prewarm_arrow_queries=[prewarm_sql],
    )

    new_port = (
        find_available_port(port, max_attempts=10, host=host)
        if enable_auto_port
        else port
    )
    if new_port != port:
        logger.info(f"Port {port} is not available, using {new_port}")

    print()
    print(click.style("-" * 79, dim=True))
    print()
    print(
        f"  {click.style('🚀 Geospatial Atlas', fg='green', bold=True)}  "
        f"{click.style('v' + __version__, fg='green')}  "
        f"{click.style('(fast-load)', fg='yellow')}"
    )
    print(
        f"  {click.style(f'✓ loaded {fast_connection.row_count:,} rows in {fast_connection.duration_seconds:.2f}s', dim=True)}"
    )
    print()
    print(f"  ➜ URL: {click.style(f'http://{host}:{new_port}', fg='cyan', bold=True)}")
    print(click.style("  ➜ Press CTRL+C to quit", dim=True))
    print()
    print(click.style("-" * 79, dim=True))

    import logging as _logging

    uvicorn.run(
        app, port=new_port, host=host, access_log=False, log_level=_logging.ERROR
    )


@click.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("--text", default=None, help="Column containing text data.")
@click.option("--image", default=None, help="Column containing image data.")
@click.option("--audio", default=None, help="Column containing audio data.")
@click.option(
    "--vector", default=None, help="Column containing pre-computed vector embeddings."
)
@click.option(
    "--features",
    default=None,
    help="Column containing features. If specified, a features list view will be created. "
    "The column should be a list of feature strings, or a list of structs, one per feature, "
    "each with a 'feature' field and optional 'source' and 'index' fields.",
)
@click.option(
    "--split",
    default=[],
    multiple=True,
    help="Dataset split name(s) to load from Hugging Face datasets. Can be specified multiple times for multiple splits.",
)
@click.option(
    "--enable-projection/--disable-projection",
    "enable_projection",
    default=True,
    help="Compute embedding projections from text/image/vector data. If disabled without pre-computed projections, the embedding view will be unavailable.",
)
@click.option(
    "--model",
    default=None,
    help="Model name for generating embeddings (e.g., 'all-MiniLM-L6-v2').",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    help="Allow execution of remote code when loading models from Hugging Face Hub.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for processing embeddings (default: 32 for text, 16 for images). Larger values use more memory but may be faster.",
)
@click.option(
    "--embedder",
    type=str,
    default=None,
    help="Embedding backend: 'sentence-transformers' (default for text), 'transformers' (default for image/audio), or 'litellm' (API-based).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="API key for litellm embedding provider.",
)
@click.option(
    "--api-base",
    type=str,
    default=None,
    help="API endpoint for litellm embedding provider.",
)
@click.option(
    "--dimensions",
    type=int,
    default=None,
    help="Number of dimensions for output embeddings (litellm only, supported by OpenAI text-embedding-3+).",
)
@click.option(
    "--max-concurrency",
    type=int,
    default=None,
    help="Maximum number of concurrent embedding batches. Use 1 for local servers like Ollama to avoid memory issues.",
)
@click.option(
    "--x",
    "x_column",
    help="Column containing pre-computed X coordinates for the embedding view.",
)
@click.option(
    "--y",
    "y_column",
    help="Column containing pre-computed Y coordinates for the embedding view.",
)
@click.option(
    "--neighbors",
    "neighbors_column",
    help='Column containing pre-computed nearest neighbors in format: {"ids": [n1, n2, ...], "distances": [d1, d2, ...]}. IDs should be zero-based row indices.',
)
@click.option(
    "--pagerank",
    "pagerank_column",
    default=None,
    is_flag=False,
    flag_value="__compute__",
    help="Compute PageRank scores from the neighbor graph, or specify a column containing pre-computed scores. Automatically computed when --image is specified.",
)
@click.option(
    "--query",
    default=None,
    type=str,
    help="Use the result of the given SQL query as input data. In the query, you may refer to the original data as 'data'.",
)
@click.option(
    "--sample",
    default=None,
    type=int,
    help="Number of random samples to draw from the dataset. Useful for large datasets. If query is specified, sampling applies after the query.",
)
@click.option(
    "--umap-n-neighbors",
    type=int,
    help="Number of neighbors to consider for UMAP dimensionality reduction (default: 15).",
)
@click.option(
    "--umap-min-dist",
    type=float,
    help="The min_dist parameter for UMAP.",
)
@click.option(
    "--umap-metric",
    default="cosine",
    help="Distance metric for UMAP computation (default: 'cosine').",
)
@click.option(
    "--umap-random-state", type=int, help="Random seed for reproducible UMAP results."
)
@click.option(
    "--table",
    "extra_tables",
    type=(str, str),
    multiple=True,
    help="Load an additional table. Format: --table NAME PATH (e.g., --table topics topics.parquet). Can be specified multiple times.",
)
@click.option(
    "--table-relation",
    "table_relations",
    type=(str, str),
    multiple=True,
    help="Declare how an additional table joins to the main table, for cross-table filtering. "
    "Format: --table-relation NAME 'KV', where KV is semicolon-separated key=value pairs describing "
    "the relation: mainKey (join expression over the main table), key (join column on this table). "
    "Wrap an expression in UNNEST(...) when the key is list-valued (e.g. a list/struct-list column). "
    "NAME must match a --table NAME. "
    "Example: --table-relation features "
    "'mainKey=UNNEST(list_transform(features, s -> s.feature));key=feature'. "
    "Can be specified multiple times.",
)
@click.option(
    "--duckdb",
    type=str,
    default="server",
    help="DuckDB connection mode: 'wasm' (run in browser), 'server' (run on this server), or URI (e.g., 'ws://localhost:3000').",
)
@click.option(
    "--host",
    default="localhost",
    help="Host address for the web server (default: localhost).",
)
@click.option(
    "--port", default=5055, help="Port number for the web server (default: 5055)."
)
@click.option(
    "--auto-port/--no-auto-port",
    "enable_auto_port",
    default=True,
    help="Automatically find an available port if the specified port is in use.",
)
@click.option(
    "--cors",
    default=None,
    is_flag=False,
    flag_value="",
    help="Allow cross-origin requests. Use --cors to allow all origins, or --cors http://example.com for a specific domain (or a comma-separated list of domains).",
)
@click.option(
    "--static", type=str, help="Custom path to frontend static files directory."
)
@click.option(
    "--export-application",
    type=str,
    help="Export the visualization as a standalone web application to the specified path. "
    "Use a .zip extension for a ZIP archive, or any other path to export to a folder.",
)
@click.option(
    "--export-metadata",
    type=JSONParamType(),
    default=None,
    help="Custom metadata to merge into the exported metadata.json. "
    'Pass a JSON string (e.g., \'{"database": {"datasetUrl": "https://..."}}\') '
    "or a path to a JSON file.",
)
@click.option(
    "--with",
    "with_modules",
    default=[],
    multiple=True,
    help="Import the given Python module before loading data. For example, you can use this to import fsspec filesystems. Can be specified multiple times to import multiple modules.",
)
@click.option(
    "--point-size",
    type=float,
    default=None,
    help="Size of points in the embedding view (default: automatically calculated based on density).",
)
@click.option(
    "--stop-words",
    type=str,
    default=None,
    help="Path to a file containing stop words to exclude from the text embedding. The file should be a table with column 'word'",
)
@click.option(
    "--labels",
    type=str,
    default=None,
    help="Path to a file containing labels for the embedding view. The file should be a table with columns 'x', 'y', 'text', and optionally 'level' and 'priority'",
)
@click.option(
    "--mcp/--no-mcp",
    "enable_mcp",
    default=False,
    help="Enable MCP (Model Context Protocol) server endpoints for external tool integration.",
)
@click.version_option(version=__version__, package_name="embedding_atlas")
def main(
    inputs,
    text: str | None,
    image: str | None,
    audio: str | None,
    vector: str | None,
    features: str | None,
    split: list[str] | None,
    enable_projection: bool,
    model: str | None,
    trust_remote_code: bool,
    batch_size: int | None,
    embedder: str | None,
    api_key: str | None,
    api_base: str | None,
    dimensions: int | None,
    max_concurrency: int | None,
    x_column: str | None,
    y_column: str | None,
    neighbors_column: str | None,
    pagerank_column: str | None,
    query: str | None,
    sample: int | None,
    umap_n_neighbors: int | None,
    umap_min_dist: float | None,
    umap_metric: str | None,
    umap_random_state: int | None,
    extra_tables: list[tuple[str, str]] | None,
    table_relations: list[tuple[str, str]] | None,
    static: str | None,
    duckdb: str,
    host: str,
    port: int,
    enable_auto_port: bool,
    cors: str | None,
    export_application: str | None,
    export_metadata: dict | None,
    with_modules: list[str] | None,
    point_size: float | None,
    stop_words: str | None,
    labels: str | None,
    enable_mcp: bool,
):
    apply_logging_config()

    if with_modules is not None:
        import_modules(with_modules)

    # DuckDB-native fast path: on a single-parquet GIS input with no
    # pandas-specific transformations, we can skip pandas entirely and
    # let DuckDB read the parquet + extract coordinates with ST_X/ST_Y.
    # On large files (75M+ rows) this takes ~5 s instead of 10+ minutes.
    fast_connection = _try_fast_load(
        inputs=list(inputs),
        splits=list(split) if split else [],
        query=query,
        sample=sample,
        text=text,
        image=image,
        audio=audio,
        vector=vector,
        x_column=x_column,
        y_column=y_column,
        duckdb_uri=duckdb,
    )
    if fast_connection is not None:
        return _run_fast_path(
            fast_connection=fast_connection,
            static=static,
            host=host,
            port=port,
            enable_auto_port=enable_auto_port,
            enable_mcp=enable_mcp,
            cors=cors,
            duckdb_uri=duckdb,
        )

    df = load_datasets(inputs, splits=split, query=query, sample=sample)

    is_gis = False
    if x_column is None or y_column is None:
        # Priority 1: Named lat/lon columns (most explicit)
        detected_x, detected_y = find_gis_columns(df.columns)
        if detected_x is not None and detected_y is not None:
            if x_column is None:
                x_column = detected_x
            if y_column is None:
                y_column = detected_y
            is_gis = True
            print(f"Detected GIS columns: x={x_column}, y={y_column}")

        # Priority 2: Geometry column (geoparquet / WKB) — only if no named columns found
        if x_column is None or y_column is None:
            geom_col = find_geometry_column(df)
            if geom_col is not None:
                print(f"Detected geometry column: {geom_col}")
                df, extracted_x, extracted_y = extract_coordinates_from_geometry(
                    df, geom_col
                )
                if x_column is None:
                    x_column = extracted_x
                if y_column is None:
                    y_column = extracted_y
                is_gis = True
                print(f"Extracted GIS coordinates: x={x_column}, y={y_column}")

    additional_tables = {}

    if extra_tables is not None:
        for name, path in extra_tables:
            additional_tables[name] = load_datasets([path])

    # Give every additional table a synthetic row id (mirroring the main table)
    # so it can be registered in the UI even without a declared relation. Then
    # attach any --table-relation, whose KV maps directly to the join relation.
    additional_tables_meta = {}
    for name, adf in additional_tables.items():
        aux_id = find_column_name(adf.columns, "__row_index__")
        adf[aux_id] = range(adf.shape[0])
        additional_tables_meta[name] = {"id": aux_id}
    if table_relations is not None:
        for name, kv in table_relations:
            if name not in additional_tables:
                raise click.BadParameter(
                    f"--table-relation {name!r} has no matching --table {name!r}"
                )
            # Validate keys/values in parse_kv so a typo (e.g. `mainkey`) or a
            # blank value fails loudly instead of silently falling back to the
            # id-column join (a wrong but non-erroring cross-table filter).
            try:
                relation = parse_kv(
                    kv, allowed_keys={"mainKey", "key"}, allow_empty_values=False
                )
            except ValueError as e:
                raise click.BadParameter(f"--table-relation {name!r}: {e}")
            additional_tables_meta[name]["relation"] = relation

    if enable_projection and (x_column is None or y_column is None):
        # No x, y column selected, first see if text/image/vectors column is specified, if not, ask for it
        if text is None and image is None and audio is None and vector is None:
            selected_column = prompt_for_column(
                df, "Select a column you want to run the embedding on"
            )
        else:
            selected_column = None
        umap_args = {}
        if umap_min_dist is not None:
            umap_args["min_dist"] = umap_min_dist
        if umap_n_neighbors is not None:
            umap_args["n_neighbors"] = umap_n_neighbors
        if umap_random_state is not None:
            umap_args["random_state"] = umap_random_state
        if umap_metric is not None:
            umap_args["metric"] = umap_metric
        # Run embedding and projection
        if (
            text is not None
            or image is not None
            or audio is not None
            or vector is not None
            or selected_column is not None
        ):
            from .projection import compute_projection

            x_column = find_column_name(df.columns, "projection_x")
            y_column = find_column_name(df.columns, "projection_y")
            if neighbors_column is None:
                neighbors_column = find_column_name(df.columns, "__neighbors")
                new_neighbors_column = neighbors_column
            else:
                # If neighbors_column is already specified, don't overwrite it.
                new_neighbors_column = None

            # Determine modality and input column
            if vector is not None:
                modality = "vector"
                input_column = vector
            elif image is not None:
                modality = "image"
                input_column = image
            elif audio is not None:
                modality = "audio"
                input_column = audio
            elif text is not None:
                modality = "text"
                input_column = text
            elif selected_column is not None:
                modality = "auto"
                input_column = selected_column
            else:
                raise RuntimeError("unreachable")

            # Build embedder_args from CLI options
            embedder_args = {}
            if trust_remote_code:
                embedder_args["trust_remote_code"] = True
            if api_key is not None:
                embedder_args["api_key"] = api_key
            if api_base is not None:
                embedder_args["api_base"] = api_base
            if dimensions is not None:
                embedder_args["dimensions"] = dimensions
            # Pass embedder directly; compute_projection handles defaults and validation
            df = compute_projection(
                df,
                inputs=input_column,
                modality=modality,
                x=x_column,
                y=y_column,
                neighbors=new_neighbors_column,
                embedder=embedder,
                model=model,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                embedder_args=embedder_args or None,
                umap_args=umap_args or None,
            )

    id_column = find_column_name(df.columns, "__row_index__")
    df[id_column] = range(df.shape[0])

    stop_words_resolved = None
    if stop_words is not None:
        stop_words_df = load_pandas_data(stop_words)
        stop_words_resolved = stop_words_df["word"].to_list()

    labels_resolved = None
    if labels is not None:
        labels_df = load_pandas_data(labels)
        labels_resolved = labels_df.to_dict("records")

    # Compute PageRank from neighbor graph when requested or when --image is specified
    should_compute_pagerank = (pagerank_column == "__compute__") or (
        image is not None and pagerank_column is None
    )
    if (
        should_compute_pagerank
        and neighbors_column is not None
        and neighbors_column in df.columns
    ):
        from embedding_atlas.pagerank import compute_pagerank_column

        logger.info("Computing PageRank scores from neighbor graph...")
        pagerank_column = find_column_name(df.columns, "pagerank")
        df[pagerank_column] = compute_pagerank_column(df, neighbors=neighbors_column)
    elif pagerank_column == "__compute__":
        logger.warning("Cannot compute PageRank: no neighbor data available.")
        pagerank_column = None

    props = make_embedding_atlas_props(
        row_id=id_column,
        x=x_column,
        y=y_column,
        neighbors=neighbors_column,
        importance=pagerank_column,
        text=text,
        image=image,
        features=features,
        point_size=point_size,
        stop_words=stop_words_resolved,
        labels=labels_resolved,
        is_gis=is_gis,
        additional_tables=additional_tables_meta or None,
    )

    metadata = {
        "props": props,
    }

    identifier = sha256_hexdigest([__version__, inputs, metadata], scope="DataSource")
    dataset = DataSource(identifier, df, metadata, additional_tables=additional_tables)

    if static is None:
        static = (Path(__file__).parent / "static").resolve().as_posix()

    if export_application is not None:
        if export_application.endswith(".zip"):
            with open(export_application, "wb") as f:
                f.write(dataset.make_archive(static, export_metadata))
        else:
            dataset.export_to_folder(static, export_application, export_metadata)
        return

    # Parse CORS configuration
    cors_config = False
    if cors is not None:
        if cors == "":
            # --cors flag without value means allow all origins
            cors_config = True
        else:
            # --cors=domain1.com,domain2.com means specific domains
            cors_config = [
                domain.strip() for domain in cors.split(",") if domain.strip()
            ]

    app = make_server(
        dataset,
        static_path=static,
        duckdb_uri=duckdb,
        mcp=enable_mcp,
        cors=cors_config,
    )

    if enable_auto_port:
        new_port = find_available_port(port, max_attempts=10, host=host)
        if new_port != port:
            logger.info(f"Port {port} is not available, using {new_port}")
    else:
        new_port = port

    print()
    print(click.style("-" * 79, dim=True))
    print()
    print(
        f"  {click.style('🚀 Geospatial Atlas', fg='green', bold=True)}  {click.style('v' + __version__, fg='green')}"
    )
    print()
    print(f"  ➜ URL: {click.style(f'http://{host}:{new_port}', fg='cyan', bold=True)}")
    print(
        click.style(
            "  ➜ Network: use --host to expose, use --cors to enable cross-origin requests",
            dim=True,
        )
    )
    if enable_mcp:
        print(
            f"  ➜ MCP server: {click.style(f'http://{host}:{new_port}/mcp', fg='blue')}"
        )
    else:
        print(click.style("  ➜ MCP server: use --mcp to enable", dim=True))
    print(click.style("  ➜ Press CTRL+C to quit", dim=True))
    print()
    print(click.style("-" * 79, dim=True))

    uvicorn.run(
        app, port=new_port, host=host, access_log=False, log_level=logging.ERROR
    )


if __name__ == "__main__":
    main()
