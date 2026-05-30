---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Embedding Atlas
  text: Scalable, Interactive Visualization
  tagline: Visualize, cross-filter, and search embeddings and metadata.

  image:
    light: ./assets/embedding-atlas-light.png
    dark: ./assets/embedding-atlas-dark.png
    alt: a screenshot of embedding atlas
  actions:
    - theme: brand
      text: Examples
      link: /examples/
      target: _self
    - theme: brand
      text: Load Data
      link: /app/
      target: _self
    - theme: alt
      text: Documentation
      link: /overview

features:
  - icon: 🏷️
    title: Automatic data clustering & labeling
    details: Interactively visualize and navigate overall data structure.

  - icon: 🔍
    title: Real-time search & nearest neighbors
    details: Find similar data to a given query or existing data point.

  - icon: 🚀
    title: Smooth rendering at scale
    details: Render up to a few million points with density contours, powered by WebGPU.

  - icon: 📊
    title: Linked dashboards & cross-filtering
    details: Arrange charts and configure cross-filtering between them. Compose custom charts via a chart spec.

  - icon: 🧩
    title: Multimodal data support
    details: Built-in viewers for text, image, audio, numeric, categorical, and time columns.

  - icon: 🤖
    title: AI agent access via MCP
    details: AI agents can query, chart, and explore your data via Model Context Protocol.
---
