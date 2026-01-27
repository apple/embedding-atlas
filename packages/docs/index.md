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

  - icon: 🫧
    title: Kernel density estimation & density contours
    details: Easily explore and distinguish between dense regions of data and outliers.

  - icon: 🧊
    title: Order-independent transparency
    details: Ensure clear, accurate rendering of overlapping points.

  - icon: 🔍
    title: Real-time search & nearest neighbors
    details: Find similar data to a given query or existing data point.

  - icon: 🚀
    title: WebGPU implementation (with WebGL 2 fallback)
    details: Fast, smooth performance (up to few million points) with modern rendering stack.

  - icon: 📊
    title: Multi-coordinated views for metadata exploration
    details: Interactively link and filter data across metadata columns.
---
