# Embedding Atlas

[![NPM Version](https://img.shields.io/npm/v/embedding-atlas)](https://www.npmjs.com/package/embedding-atlas)
[![PyPI - Version](https://img.shields.io/pypi/v/embedding-atlas)](https://pypi.org/project/embedding-atlas/)
[![Paper](https://img.shields.io/badge/paper-arXiv:2505.06386-b31b1b.svg)](https://arxiv.org/abs/2505.06386)
![Build](https://github.com/apple/embedding-atlas/actions/workflows/ci.yml/badge.svg)
[![GitHub License](https://img.shields.io/github/license/apple/embedding-atlas)](./LICENSE)

Embedding Atlas is a tool that provides interactive visualizations for large embeddings and their metadata. You can visualize, cross-filter, and search across your data.

**For embeddings**

- 🏷️ **Automatic data clustering & labeling:**
  Interactively visualize and navigate overall data structure.

- 🫧 **Kernel density estimation & density contours:**
  Easily explore and distinguish between dense regions of data and outliers.

- 🧊 **Order-independent transparency:**
  Ensure clear, accurate rendering of overlapping points.

- 🔍 **Real-time search & nearest neighbors:**
  Find similar data to a given query or existing data point.

- 🚀 **Smooth performance at scale:**
  Up to a few million points, powered by WebGPU.

**For any tabular data**

- 📊 **Linked dashboards & cross-filtering:**
  Standard chart types (bar, line, bubble, count plot, eCDF) plus a composable chart spec for building custom charts like heatmaps and average-line overlays. Charts can be configured to cross-filter.

- 🧩 **Multimodal data support:**
  Built-in viewers for text, image, audio, numeric, categorical, and time columns.

- 🤖 **AI agent access via MCP:**
  AI agents can query the schema, run SQL, create charts, and capture screenshots via Model Context Protocol.

Please visit <https://apple.github.io/embedding-atlas> for a demo and documentation.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./packages/docs/public/assets/embedding-atlas-dark.png">
  <img alt="screenshot of Embedding Atlas" src="./packages/docs/public/assets/embedding-atlas-light.png">
</picture>

## Get started

To use Embedding Atlas with Python:

```bash
pip install embedding-atlas

embedding-atlas <your-dataset>
```

In addition to the command line tool, Embedding Atlas is available as a Python Notebook (e.g., Jupyter) widget:

```python
from embedding_atlas.widget import EmbeddingAtlasWidget

# Show the Embedding Atlas widget for your data frame:
EmbeddingAtlasWidget(df)
```

Finally, components from Embedding Atlas are also available in an npm package:

```bash
npm install embedding-atlas
```

```js
import { EmbeddingAtlas, EmbeddingView } from "embedding-atlas";

// or with React:
import { EmbeddingAtlas, EmbeddingView } from "embedding-atlas/react";

// or Svelte:
import { EmbeddingAtlas, EmbeddingView } from "embedding-atlas/svelte";
```

For more information, please visit <https://apple.github.io/embedding-atlas/overview.html>.

## Astronomical 3D extension

This fork adds an end-to-end **3D point cloud extension** to the embedding view, plus real
astronomical data (fetched live from [LSDB](https://docs.lsdb.io)) to demonstrate it.

### What's new

- **A navigable 3D point cloud view** (`EmbeddingView3D` / `EmbeddingView3DMosaic`, in
  `packages/component/src/lib/embedding_view/`), built on three.js, alongside the existing 2D
  WebGPU/WebGL2 view. A chart renders in 3D automatically whenever it has a `z` column/field;
  a **"Switch to 2D" / "Switch to 3D" button below the view** lets you flip between a flat
  top-down projection and the full 3D point cloud at any time.
- **Camera**: orbit/pan/zoom with inertia (damping), an auto-fit on first load (not re-applied
  on every filter change, so it doesn't yank your view around), a reset-camera button, and a
  smooth fly-to animation when focusing on a selected point.
- **Picking**: hover shows a live tooltip with the point's coordinates; click resolves to a full
  row and highlights it in the data table below (via the standard Mosaic cross-filter/selection
  machinery, same as the 2D view).
- **Marquee/lasso selection in 3D**: drag a rectangle or lasso in screen space to select points;
  this becomes a real SQL filter (via row identifier when available, or coordinate matching
  otherwise) applied through Mosaic - visible immediately in the histograms and data table.
  Unlike the 2D view, 3D has no persistent brush outline to redraw as the camera moves, so an
  active filter shows a **"Filter active" badge** in the status bar instead of vanishing
  silently.
- **Downsampling for large datasets**: cone-of-view (camera-frustum) culling with idle-scheduled,
  chunked, cancellable processing (`downsample_3d.ts`), so panning/orbiting a million-point
  dataset stays responsive. Points are rendered with a **density-aware size** (scaled to the
  average spacing between points, not just the data's bounding radius), so a wide, sparse field
  reads as sparse rather than looking artificially solid.
- **Backend**: `--z <column>` (precomputed 3D coordinate) and `--umap-n-components 3` (compute a
  3D UMAP projection instead of 2D) CLI flags, plus matching support in the Python
  `compute_projection` API and the in-browser (Transformers.js/WASM) embedding pipeline.
- **Chart builder**: a "Z Field (optional)" selector in the "Create an embedding view" dialog -
  picking a Z field is what turns a chart into a 3D one.

### Demo: real LSDB astronomical data

Two independent, real catalogs are included as a demonstration dataset, fetched live from
LSDB's public HATS server (`server.data.lsdb.io`):

| Script | Catalog | What it does |
| --- | --- | --- |
| `get_real_stars.py` | Gaia DR3 | Cone-searches a 9°-radius field (~357k raw stars) around RA 49°, Dec 3.7° |
| `prepare_stars_parquet.py` | (derived) | Adds `x`/`y`/`z` columns for the CLI's `--x`/`--y`/`--z`, writes `real_stars_xyz.parquet` |
| `convert_stars_to_json.py` | (derived) | Same, as JSON for the in-browser example app (`packages/examples/public/real_stars.json`) |
| `get_euclid_data.py` | Euclid Q1 | Cone-searches the EDF-North deep field (real ESA Euclid telescope data) |
| `prepare_euclid_xyz.py` | (derived) | Adds `x`/`y`/`z` columns, writes `real_euclid_xyz.parquet` |

Run the fetch scripts once (each hits the live LSDB server), then launch the viewer:

```bash
python get_real_stars.py && python prepare_stars_parquet.py
embedding-atlas real_stars_xyz.parquet --x x --y y --z z

# a second, independent real catalog:
python get_euclid_data.py && python prepare_euclid_xyz.py
embedding-atlas real_euclid_xyz.parquet --x x --y y --z z --port 6001
```

X/Y are RA/Dec offsets from the field center (degrees); Z is parallax-derived distance for
Gaia, or a brightness-based depth **proxy** for Euclid Q1 (its base catalog has no
distance/redshift column - this is a visualization convenience, not a real measurement, and is
documented as such in `prepare_euclid_xyz.py`).

Density is controlled by a single **3D ellipsoidal Gaussian** applied to x, y, and z together
(not a 2D sky-plane falloff with an independently-thinned depth axis - that combination let
real, lumpy sky structure and an unrelated depth distribution collide into an oddly-shaped
cloud). Its peak keep-probability is capped well below 1.0, so the field stays airy throughout
rather than solid in the middle, and small positional jitter is added after thinning to blur
out genuine small-scale density structure in the underlying catalog (a probabilistic thin can
only remove stars, never fill in a real gap) into a smooth, round, cluster-like shape - each
point is still a real catalog object, just nudged slightly off its exact position.

**Known limitations, stated plainly:**

- No catalog literally named "LSST" or "Roman" is available: the Roman Space Telescope hasn't
  released public survey data yet, and Rubin/LSST's `dp1` ("Data Preview 1") release exists on
  the LSDB server but isn't structured as a standard `lsdb.open_catalog`-able HATS catalog (only
  flat crossmatch parquet files). Euclid Q1 was used instead as a second genuinely real,
  independently-verified LSDB catalog.
- LSDB's HATS server occasionally returned inconsistent row counts for the same cone-search
  parameters across separate calls during development (observed, not fully root-caused); counts
  reported here were confirmed reproducible by re-running the same query.
- LSDB's internal `_healpix_29` spatial-index column is dropped from both datasets: its int64
  values exceed JavaScript's safe integer range (2^53) for these catalogs, which crashed the
  viewer's data table before the fix.

## BibTeX

For the Embedding Atlas tool:

```bibtex
@misc{ren2025embedding,
  title={Embedding Atlas: Low-Friction, Interactive Embedding Visualization},
  author={Donghao Ren and Fred Hohman and Halden Lin and Dominik Moritz},
  year={2025},
  eprint={2505.06386},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2505.06386},
}
```

For the algorithm that automatically produces clusters and labels in the embedding view:

```bibtex
@misc{ren2025scalable,
  title={A Scalable Approach to Clustering Embedding Projections},
  author={Donghao Ren and Fred Hohman and Dominik Moritz},
  year={2025},
  eprint={2504.07285},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2504.07285},
}
```

For sensemaking on UMAP's internal kNN graph:

```bibtex
@misc{chau2026dimensionality,
  title={Dimensionality Reduction Meets Network Science: Sensemaking on UMAP's kNN Graph},
  author={Duen Horng Chau and Donghao Ren and Fred Hohman and Dominik Moritz},
  year={2026},
  eprint={2607.08746},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2607.08746},
}
```

## Development

For development instructions, please visit <https://apple.github.io/embedding-atlas/develop.html>, or check out `packages/docs/develop.md`.

## License

This code is released under the [`MIT license`](LICENSE).
