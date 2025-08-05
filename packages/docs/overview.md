# Overview

Embedding Atlas is a tool that provides interactive visualizations for large embeddings. It allows you to visualize, cross-filter, and search embeddings and metadata.

Embedding Atlas is released as two packages:

- A Python package `embedding-atlas` that provides:
  - A [command line utility](./tool.md) for running Embedding Atlas on a data frame.
  - A [Jupyter widget](./widget.md) for using Embedding Atlas in Jupyter notebooks.
  - A [Streamlit component](./streamlit.md) for using Embedding Atlas in Streamlit apps.

- An npm package `embedding-atlas` that exposes the user interface components as API so you can use them in your own applications. Below are the exposed components:
  - [Table](./table.md)
  - [EmbeddingView](./embedding-view.md)
  - [EmbeddingViewMosaic](./embedding-view-mosaic.md)
  - [EmbeddingAtlas](./embedding-atlas.md)
  - [Algorithms](./algorithms.md)
