"""Generate standalone HTML benchmark reports using Vega-Lite and Tailwind."""

import base64
import csv
import gzip
import json
import os

import numpy as np


def _read_csv(path):
    """Read CSV file and return list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _classify_impl(row):
    """Return display implementation name (rust, rust-gpu, python)."""
    impl = row["implementation"]
    gpu = row.get("gpu", "false")
    if impl == "rust" and gpu in ("true", "True"):
        return "rust-gpu"
    return impl


def _prepare_data(rows, time_col, suite="umap"):
    """Prepare rows for Vega-Lite: add impl_display, cast numeric fields."""
    out = []
    for row in rows:
        r = dict(row)
        r["impl_display"] = _classify_impl(row)
        r["n_points"] = int(r["n_points"])
        r["dim"] = int(r.get("dim", 0))
        r[time_col] = float(r[time_col])
        if suite == "nndescent":
            r["dim_metric"] = f"dim={r['dim']} / {r['metric']}"
            r["dim_metric_sort"] = r["dim"] * 1000 + hash(r["metric"]) % 100
            try:
                r["recall"] = float(r.get("recall", 0))
            except (ValueError, TypeError):
                r["recall"] = None
        out.append(r)
    return out


def _timing_vegalite_spec(data, time_col, title, suite="umap"):
    """Build a Vega-Lite spec for timing charts faceted by metric x threads (and dim for nndescent)."""
    tooltip = [
        {"field": "impl_display", "title": "Implementation"},
        {"field": "n_points", "title": "Points"},
        {"field": "metric", "title": "Metric"},
        {"field": "threads", "title": "Threads"},
        {"field": time_col, "title": "Time (s)", "format": ".3f"},
    ]
    color = {
        "field": "impl_display",
        "type": "nominal",
        "title": "Implementation",
        "scale": {
            "domain": ["rust", "rust-gpu", "python"],
            "range": ["#3b82f6", "#22c55e", "#f97316"],
        },
    }

    if suite == "nndescent":
        # Line chart faceted by (dim x metric) rows and threads columns
        if "dim" not in [t["field"] for t in tooltip]:
            tooltip.insert(3, {"field": "dim", "title": "Dim"})
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": title,
            "data": {"values": data},
            "facet": {
                "column": {"field": "threads", "type": "nominal", "title": "Threads"},
                "row": {
                    "field": "dim_metric",
                    "type": "nominal",
                    "title": "Dim / Metric",
                    "sort": {"field": "dim_metric_sort"},
                },
            },
            "resolve": {"axis": {"x": "independent", "y": "independent"}},
            "spec": {
                "width": 350,
                "height": 200,
                "layer": [
                    {
                        "mark": {"type": "line", "point": True},
                        "encoding": {
                            "x": {
                                "field": "n_points",
                                "type": "quantitative",
                                "title": "Number of points",
                                "scale": {"type": "log"},
                            },
                            "y": {
                                "field": time_col,
                                "type": "quantitative",
                                "title": "Time (s)",
                                "scale": {"type": "log"},
                            },
                            "color": color,
                            "tooltip": tooltip,
                        },
                    }
                ],
            },
        }
    else:
        # Bar chart for UMAP (unchanged)
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": title,
            "data": {"values": data},
            "facet": {
                "column": {"field": "metric", "type": "nominal", "title": "Metric"},
                "row": {"field": "threads", "type": "nominal", "title": "Threads"},
            },
            "spec": {
                "width": 300,
                "height": 200,
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": "n_points",
                        "type": "ordinal",
                        "title": "Number of points",
                        "sort": None,
                    },
                    "y": {
                        "field": time_col,
                        "type": "quantitative",
                        "title": "Time (s)",
                    },
                    "color": color,
                    "xOffset": {"field": "impl_display", "type": "nominal"},
                    "tooltip": tooltip,
                },
            },
        }


def _results_table_html(rows, columns):
    """Build an HTML table from result rows."""
    header = "".join(
        f'<th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{c}</th>'
        for c in columns
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-700">{row.get(c, "")}</td>'
            for c in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)
    return f"""
<div class="overflow-x-auto">
  <table class="min-w-full divide-y divide-gray-200">
    <thead class="bg-gray-50"><tr>{header}</tr></thead>
    <tbody class="bg-white divide-y divide-gray-200">{body}</tbody>
  </table>
</div>"""


def _gz_b64(data: bytes) -> str:
    """Gzip compress and base64 encode bytes."""
    return base64.b64encode(gzip.compress(data, compresslevel=6)).decode("ascii")


_IMPL_ORDER = [
    ("rust", "Rust CPU"),
    ("rust_gpu", "Rust GPU"),
    ("python", "Python"),
]


def _load_embeddings(ds_dir, rows):
    """Load, align, and normalize UMAP embeddings for all (n_points, metric, threads) groups.

    Returns list of dicts with gzipped+base64-encoded embedding/label data.
    """
    from benchmark.align import align_to_reference

    # Determine unique (n_points, metric, threads) triples from results
    triples = set()
    for row in rows:
        triples.add((int(row["n_points"]), row["metric"], row.get("threads", "multi")))

    groups = []
    for n, metric, threads in sorted(triples):
        data_dir = os.path.join(ds_dir, f"{n}_784")
        if not os.path.isdir(data_dir):
            continue

        # Load labels
        labels_path = os.path.join(data_dir, "labels.bin")
        if os.path.exists(labels_path):
            labels = np.fromfile(labels_path, dtype=np.uint8)[:n]
        else:
            labels = np.zeros(n, dtype=np.uint8)

        # Load available embeddings (thread-aware filenames, fall back to old names)
        embeddings = {}
        for prefix, display_name in _IMPL_ORDER:
            emb_path = os.path.join(
                data_dir, f"{prefix}_{metric}_{threads}_embedding.bin"
            )
            if not os.path.exists(emb_path):
                # Fall back to old naming without thread suffix
                emb_path = os.path.join(data_dir, f"{prefix}_{metric}_embedding.bin")
            if os.path.exists(emb_path):
                emb = np.fromfile(emb_path, dtype="<f4").reshape(n, 2)
                if np.isfinite(emb).all():
                    embeddings[display_name] = emb

        if not embeddings:
            continue

        # Align to reference (Rust CPU preferred)
        ref_name = next((name for _, name in _IMPL_ORDER if name in embeddings), None)
        ref_emb = embeddings[ref_name]
        for name in embeddings:
            if name != ref_name:
                embeddings[name] = align_to_reference(embeddings[name], ref_emb)

        # Shared normalization to [0, 1]
        all_coords = np.concatenate(list(embeddings.values()), axis=0)
        mins = all_coords.min(axis=0)
        maxs = all_coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        for name in embeddings:
            embeddings[name] = ((embeddings[name] - mins) / ranges).astype(np.float32)

        # Serialize
        impls = {}
        for name, emb in embeddings.items():
            impls[name] = _gz_b64(emb.tobytes())
            raw_kb = n * 2 * 4 / 1024
            gz_kb = len(impls[name]) * 3 / 4 / 1024
            print(
                f"  {n:>6} {metric:>10} {threads:>6} {name:<10}: {raw_kb:.0f}KB -> {gz_kb:.0f}KB gzipped"
            )

        groups.append(
            {
                "n": n,
                "metric": metric,
                "threads": threads,
                "labels": _gz_b64(labels.tobytes()),
                "impls": impls,
            }
        )

    return groups


_CANVAS_JS = r"""
async function decompressGzB64(b64str) {
  const bin = atob(b64str);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(bytes);
  writer.close();
  const reader = ds.readable.getReader();
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  let totalLen = 0;
  for (const c of chunks) totalLen += c.length;
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of chunks) { result.set(c, offset); offset += c.length; }
  return result.buffer;
}

const COLORS = [
  '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
];

function renderScatter(canvas, coords, labels, n) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = cssW, h = cssH;
  const pad = 10;
  const pw = w - 2 * pad, ph = h - 2 * pad;
  const r = Math.max(0.8, Math.min(3, 3000 / n));

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  // Group by label for batch drawing
  const groups = Array.from({length: 10}, () => []);
  for (let i = 0; i < n; i++) {
    const l = labels[i] < 10 ? labels[i] : 0;
    groups[l].push(i);
  }

  ctx.globalAlpha = 0.6;
  for (let label = 0; label < 10; label++) {
    const pts = groups[label];
    if (pts.length === 0) continue;
    ctx.fillStyle = COLORS[label];
    for (const i of pts) {
      const x = coords[i * 2] * pw + pad;
      const y = coords[i * 2 + 1] * ph + pad;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1.0;
}

async function renderAllEmbeddings() {
  for (const group of EMBEDDING_DATA) {
    const labelsBuf = await decompressGzB64(group.labels);
    const labels = new Uint8Array(labelsBuf);

    for (const [implName, b64] of Object.entries(group.impls)) {
      const coordsBuf = await decompressGzB64(b64);
      const coords = new Float32Array(coordsBuf);
      const canvasId = `emb-${group.n}-${group.metric}-${group.threads}-${implName.replace(/\s+/g, '-')}`;
      const canvas = document.getElementById(canvasId);
      if (canvas) renderScatter(canvas, coords, labels, group.n);
    }
  }
}

document.addEventListener('DOMContentLoaded', renderAllEmbeddings);
"""


def _embeddings_html(groups):
    """Build the HTML section with canvas elements for embedding visualization."""
    if not groups:
        return "", ""

    # Legend
    color_labels = [
        ("#1f77b4", "0"),
        ("#ff7f0e", "1"),
        ("#2ca02c", "2"),
        ("#d62728", "3"),
        ("#9467bd", "4"),
        ("#8c564b", "5"),
        ("#e377c2", "6"),
        ("#7f7f7f", "7"),
        ("#bcbd22", "8"),
        ("#17becf", "9"),
    ]
    legend_items = " ".join(
        f'<span class="inline-flex items-center mr-3">'
        f'<span class="w-3 h-3 rounded-full mr-1" style="background:{color}"></span>'
        f'<span class="text-xs text-gray-600">{label}</span></span>'
        for color, label in color_labels
    )

    sections = []
    for group in groups:
        canvases = []
        for impl_name in group["impls"]:
            canvas_id = f"emb-{group['n']}-{group['metric']}-{group['threads']}-{impl_name.replace(' ', '-')}"
            canvases.append(
                f'<div class="flex flex-col items-center">'
                f'<span class="text-sm font-semibold text-gray-700 mb-1">{impl_name}</span>'
                f'<canvas id="{canvas_id}" class="rounded border border-gray-200" '
                f'style="width:300px;height:300px;"></canvas>'
                f"</div>"
            )
        canvas_row = "\n      ".join(canvases)
        threads_label = (
            "single-thread" if group["threads"] == "single" else "multi-thread"
        )
        sections.append(
            f'<div class="mb-6">\n'
            f'  <h3 class="text-lg font-medium text-gray-700 mb-2">'
            f"{group['n']:,} points &mdash; {group['metric']} &mdash; {threads_label}</h3>\n"
            f'  <div class="flex flex-wrap gap-4">\n      {canvas_row}\n  </div>\n'
            f"</div>"
        )

    data_json = json.dumps(groups)
    script = f"const EMBEDDING_DATA = {data_json};\n{_CANVAS_JS}"

    html_section = (
        '<div class="bg-white rounded-lg shadow p-6 mb-8">\n'
        '  <h2 class="text-xl font-semibold text-gray-800 mb-2">Embedding Visualizations</h2>\n'
        '  <p class="text-sm text-gray-500 mb-2">'
        "Aligned to Rust CPU using Kabsch rotation. Colored by MNIST digit.</p>\n"
        f'  <div class="mb-4">{legend_items}</div>\n' + "\n".join(sections) + "\n</div>"
    )

    return html_section, script


def generate_report(csv_path, output_path, suite, datasets_dir=None):
    """Generate a standalone HTML report from benchmark results CSV.

    Args:
        csv_path: Path to results.csv
        output_path: Path to write report.html
        suite: 'umap' or 'nndescent'
        datasets_dir: Path to datasets directory (enables embedding plots for umap)
    """
    rows = _read_csv(csv_path)
    if not rows:
        print(f"  No results to report in {csv_path}")
        return

    time_col = "build_time_s" if suite == "nndescent" else "time_s"
    data = _prepare_data(rows, time_col, suite=suite)
    columns = list(rows[0].keys())

    spec = _timing_vegalite_spec(
        data, time_col, f"{suite.upper()} Benchmark Timing", suite=suite
    )
    spec_json = json.dumps(spec, indent=2)
    table_html = _results_table_html(rows, columns)

    # Build recall chart for nndescent
    recall_chart_html = ""
    recall_script = ""
    if suite == "nndescent":
        recall_spec = _timing_vegalite_spec(
            data, "recall", f"{suite.upper()} Recall", suite=suite
        )
        recall_spec["spec"]["layer"][0]["encoding"]["y"] = {
            "field": "recall",
            "type": "quantitative",
            "title": "Recall",
            "scale": {"domain": [0, 1]},
        }
        recall_spec["spec"]["layer"][0]["encoding"]["tooltip"] = [
            {"field": "impl_display", "title": "Implementation"},
            {"field": "n_points", "title": "Points"},
            {"field": "dim", "title": "Dim"},
            {"field": "metric", "title": "Metric"},
            {"field": "threads", "title": "Threads"},
            {"field": "recall", "title": "Recall", "format": ".4f"},
        ]
        recall_json = json.dumps(recall_spec, indent=2)
        recall_chart_html = """
    <div class="bg-white rounded-lg shadow p-6 mb-8">
      <h2 class="text-xl font-semibold text-gray-800 mb-4">Recall</h2>
      <div id="recall-chart" class="w-full"></div>
    </div>"""
        recall_script = f"""
    vegaEmbed('#recall-chart', {recall_json}, {{actions: false}})
      .catch(console.error);"""

    # Load and serialize embeddings if available
    embeddings_html = ""
    embeddings_script = ""
    if suite == "umap" and datasets_dir:
        groups = _load_embeddings(datasets_dir, rows)
        embeddings_html, embeddings_script = _embeddings_html(groups)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{suite.upper()} Benchmark Report</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body class="bg-gray-50 min-h-screen">
  <div class="max-w-6xl mx-auto px-4 py-8">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">{suite.upper()} Benchmark Report</h1>
    <p class="text-gray-500 mb-8">Generated from {os.path.basename(csv_path)}</p>

    <div class="bg-white rounded-lg shadow p-6 mb-8">
      <h2 class="text-xl font-semibold text-gray-800 mb-4">Timing Comparison</h2>
      <div id="timing-chart" class="w-full"></div>
    </div>

    {recall_chart_html}

    {embeddings_html}

    <div class="bg-white rounded-lg shadow p-6">
      <h2 class="text-xl font-semibold text-gray-800 mb-4">Results Table</h2>
      {table_html}
    </div>
  </div>

  <script>
    vegaEmbed('#timing-chart', {spec_json}, {{actions: false}})
      .catch(console.error);
    {recall_script}
  </script>
  <script>
    {embeddings_script}
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Report saved to {output_path}")
