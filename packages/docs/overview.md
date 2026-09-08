# Overview

Embedding Atlas is a tool that provides interactive visualizations for large embeddings and their metadata. You can visualize, cross-filter, and search across your data.

<img style="border-radius: 4px; margin-top: 16px" class="light-only" src="/assets/embedding-atlas-light.png">
<img style="border-radius: 4px; margin-top: 16px" class="dark-only" src="/assets/embedding-atlas-dark.png">

While embeddings are the focus, Embedding Atlas also works as a dashboard for tabular data. If your dataset has no embedding column, the embedding view is hidden, but linked charts, full-text search, and the instances view still work. Supported column types include text, image, audio, numeric, categorical, and time.

## What you can do

- **Explore embeddings.** Visualize 2D projections of millions of points, browse automatic clusters and labels, find nearest neighbors, and cross-filter against metadata.
- **Build dashboards.** Standard charts (bar, line, bubble, count plot, eCDF) plus a composable chart spec for building custom charts. Configure cross-filtering between any of them. Works with or without an embedding column.
- **Drive analysis with AI agents.** The [command line tool](./tool.md) includes an [MCP](https://modelcontextprotocol.io) server. Agents can query the schema, run SQL, create and modify charts, and capture screenshots.
- **Work with multimodal data.** Text, image, and audio columns are rendered with appropriate viewers; time columns get time-aware charts.

See [Examples](./examples/) for live demos with embedding and tabular datasets.

::: tip
You can use Embedding Atlas directly from this website by [loading your own data](https://apple.github.io/embedding-atlas/app/). In this online version, Embedding Atlas will compute the embedding and projection in your browser. Your data does not leave your machine.
:::

## Packages

Embedding Atlas is released as two packages:

- A Python package `embedding-atlas` that provides:
  - A [command-line tool](./tool.md) for launching Embedding Atlas from the command line.
  - A [Python Notebook widget](./widget.md) for using Embedding Atlas in interactive Python notebooks.
  - A [Streamlit component](./streamlit.md) for using Embedding Atlas in Streamlit apps.
  - All of these approaches allow you to compute embeddings (with custom models) and projections.

- An npm package `embedding-atlas` that exposes the user interface components as APIs so you can use them in your own applications. Below are the exposed components:
  - [EmbeddingView](./embedding-view.md)
  - [EmbeddingViewMosaic](./embedding-view-mosaic.md)
  - [EmbeddingAtlas](./embedding-atlas.md)
  - [Algorithms](./algorithms.md)
