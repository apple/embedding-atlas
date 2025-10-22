# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import os
import urllib.request
from itertools import product

remote_repo = "https://extensions.duckdb.org"

versions = ["v1.3.2"]
variants = ["wasm_eh", "wasm_mvp"]
extensions = ["json", "fts", "parquet"]

public_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public"
)


def download_file(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Download: " + url)
    headers = {"User-Agent": ""}
    # Create a Request object
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response, open(path, "wb") as out_file:
        while chunk := response.read(1024):  # Read in 1KB chunks
            out_file.write(chunk)


def download_files():
    for version, variant, extension in product(versions, variants, extensions):
        url = f"{remote_repo}/{version}/{variant}/{extension}.duckdb_extension.wasm"
        path = os.path.join(
            public_path,
            "duckdb-wasm",
            version,
            variant,
            f"{extension}.duckdb_extension.wasm",
        )
        if not os.path.exists(path):
            download_file(url, path)


if __name__ == "__main__":
    download_files()
