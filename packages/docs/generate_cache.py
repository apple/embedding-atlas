import hashlib
import json
import logging
import re
from io import BytesIO
from pathlib import Path

import click
import duckdb
import pandas as pd
import requests
from embedding_atlas.projection import compute_text_projection


def load_and_check_integrity(url: str, *, sha256: str) -> bytes:
    resp = requests.get(url)
    data = resp.content
    assert isinstance(data, bytes)
    assert hashlib.sha256(data).hexdigest() == sha256, "checksum mismatch"
    return data


def generate_dataset_embedding(
    *,
    url: str,
    sha256: str,
    query: str,
    output: str,
    output_folder: str,
    model: str = "all-MiniLM-L6-v2",
    umap_args: dict = {},
):
    click.echo(click.style(f"Processing {url}", fg="cyan"))

    binary_data = load_and_check_integrity(url, sha256=sha256)
    if url.endswith(".parquet"):
        data_frame = pd.read_parquet(BytesIO(binary_data))
    elif url.endswith(".jsonl"):
        data_frame = pd.read_json(BytesIO(binary_data), lines=True, orient="records")
    elif url.endswith(".csv"):
        data_frame = pd.read_csv(BytesIO(binary_data))
    else:
        raise ValueError("invalid data format")
    _ = data_frame
    df = duckdb.query(query).to_df()

    umap_args = {"random_state": 42} | umap_args

    compute_text_projection(
        df,
        text="text",
        x="x",
        y="y",
        neighbors="neighbors",
        model=model,
        umap_args=umap_args,
    )

    df = df.drop(columns="text")

    Path(output_folder).mkdir(exist_ok=True, parents=True)
    df.to_parquet(Path(output_folder) / output)

    click.echo(f"Results written to {output}")


@click.command()
@click.option("--output-folder", default=".cache", help="Output folder")
def main(output_folder: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: (%(name)s) %(message)s",
    )

    # Path to the datasets.js file
    datasets_file = Path(__file__).parent / "examples" / "datasets.js"

    # Read the datasets.js file
    content = datasets_file.read_text(encoding="utf-8")

    # Find all special comments with generate_dataset_embedding calls
    # Pattern matches: /*! ... generate_dataset_embedding({ ... }) ... */
    pattern = r"/\*!\s*.*?generate_dataset_embedding\s*\(\s*(\{.*?\})\s*\).*?\*/"
    matches = re.findall(pattern, content, re.DOTALL)

    # Process each match
    for match in matches:
        # Clean up the JSON string - remove trailing comma if present
        json_str = match.strip()
        # Fix common JSON issues like trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Parse the JSON parameters
        params = json.loads(json_str)

        # Call the generate_dataset_embedding function with **params
        generate_dataset_embedding(**params, output_folder=output_folder)


if __name__ == "__main__":
    main()
