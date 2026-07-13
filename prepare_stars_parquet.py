"""Add precomputed x/y/z columns (sky-plane RA/Dec + parallax-derived depth) to
real_stars.parquet so it can be loaded directly via the embedding-atlas CLI with
--x/--y/--z pointing at pre-computed columns (no UMAP needed for real astrometry)."""

import numpy as np
import pandas as pd

df = pd.read_parquet("real_stars.parquet")
print(f"Loaded {len(df)} stars")

df = df[df["distance_parallax"] > 0.05].copy()
df["distance_pc"] = 1000.0 / df["distance_parallax"]
df = df[df["distance_pc"] < df["distance_pc"].quantile(0.98)]
print(f"{len(df)} stars after distance filtering")

ra_center, dec_center = 49.0, 3.7
df["x"] = (df["ra"] - ra_center).round(4)
df["y"] = (df["dec"] - dec_center).round(4)

d = df["distance_pc"].to_numpy()
d_norm = (d - d.min()) / max(d.max() - d.min(), 1e-6)
df["z"] = ((d_norm - 0.5) * 3.0).round(4)

df.to_parquet("real_stars_xyz.parquet")
print(f"Wrote {len(df)} rows to real_stars_xyz.parquet")
print(df[["x", "y", "z", "distance_pc", "star_class"]].head())
