# Command Line Utility

The Python package contains a command-line utility for you to quickly explore large text datasets with metadata.

<img style="border-radius: 4px" class="light-only" src="./assets/embedding-atlas-light.png">
<img style="border-radius: 4px" class="dark-only" src="./assets/embedding-atlas-dark.png">

## Installation

```bash
pip install embedding-atlas
```

and then launch the command line tool:

```bash
embedding-atlas [OPTIONS] INPUTS...
```

::: tip
To avoid package installation issues, we recommend using the [uv package manager](https://docs.astral.sh/uv/) to install Embedding Atlas and its dependencies. uv allows you to launch the command line tool with a single command:

```bash
uvx embedding-atlas
```

On Windows, you may install the package on either the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) or directly on Windows. To use NVIDIA GPUs, you'll need to install a PyTorch version that supports CUDA, see [here](https://pytorch.org/get-started/locally/) for more details.
:::

## Loading Data

You can load your data in two ways: locally or from Hugging Face.

### Loading Local Data

To get started with your own data, run:

```bash
embedding-atlas path_to_dataset.parquet
```

### Loading Hugging Face Data

You can instead load datasets from Hugging Face:

```bash
embedding-atlas huggingface_org/dataset_name
```

## Visualizing Embeddings

The script will use [SentenceTransformers](https://sbert.net/) to compute embedding vectors for the specified column containing the text or image data. You may use the `--model` option to specify an embedding model. If not specified, a default model will be used. The current defaults are `all-MiniLM-L6-v2` for text and `google/vit-base-patch16-384` for images, but these are subject to change in future releases.

After embedding vectors are computed, the script will then project the high-dimensional vectors to 2D with [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html).

::: tip
Optionally, if you know what column your text data is in beforehand, you can specify which column to use with the `--text` flag, for example:

```bash
embedding-atlas path_to_dataset.parquet --text text_column
```

Similarly, you may supply the `--image` flag for image data, or the `--vector` flag for pre-computed embedding vectors.
:::

If you've already pre-computed the embedding projection (e.g., by running your own embedding model and projecting them with UMAP), you may store them as two columns such as `projection_x` and `projection_y`, and pass them into `embedding-atlas` with the `--x` and `--y` flags:

```bash
embedding-atlas path_to_dataset.parquet --x projection_x --y projection_y
```

You may also pass in the `--neighbors` flag to specify the column name for pre-computed nearest neighbors.
The `neighbors` column should have values in the following format: `{"ids": [id1, id2, ...], "distances": [d1, d2, ...]}`.
The IDs should be zero-based row indices.
If this column is specified, you'll be able to see nearest neighbors for a selected point in the tool.

Once this script completes, it will print out a URL like `http://localhost:5055/`. Open the URL in a web browser to view the embedding.

## Reproducibility

For reproducible embedding visualizations, we recommend pre-computing both the embedding vectors and their UMAP projections, and storing them with your dataset. This ensures consistency since the default embedding model may change over time, floating-point precision may vary across different devices, and UMAP introduces randomness through both its default random initialization and its use of parallelism (see [here](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)).

The `embedding_atlas` package provides utility functions to compute the embedding projections:

```python
from embedding_atlas.projection import compute_text_projection

compute_text_projection(df, text="text_column",
    x="projection_x", y="projection_y", neighbors="neighbors"
)
```

## Usage

```
Usage: embedding-atlas [OPTIONS] INPUTS...
```

### Command Line Options

<!-- @doc(python-cli): embedding_atlas.cli:main -->
