"""Convert real_stars.parquet (RA/Dec/parallax from Gaia DR3) into a browser-ready
star field: X/Y stay the familiar sky-plane RA/Dec (degrees, centered on the cone
center), and Z is distance from parallax, rescaled to a comparable visual range so
the flat 2D sky plot gains real, navigable depth.

Uses the same 3D ellipsoidal thinning (applied to x, y, AND z together) as
prepare_stars_parquet.py - see that script for the full rationale."""

import json

import numpy as np
import pandas as pd

df = pd.read_parquet("real_stars.parquet")
print(f"Loaded {len(df)} stars")

df = df[df["distance_parallax"] > 0.05].copy()
df["distance_pc"] = 1000.0 / df["distance_parallax"]
print(f"{len(df)} stars after parallax-quality filtering")

ra_center, dec_center = 49.0, 3.7
x_all = (df["ra"] - ra_center).to_numpy()
y_all = (df["dec"] - dec_center).to_numpy()

log_d = np.log10(df["distance_pc"].to_numpy())
z_score = (log_d - log_d.mean()) / max(log_d.std(), 1e-6)
z_all = np.tanh(z_score / 2.2) * 2.3

sigma_x, sigma_y, sigma_z = 5.5, 5.5, 1.6
density_scale = 0.04
rng = np.random.RandomState(42)
r2 = (x_all / sigma_x) ** 2 + (y_all / sigma_y) ** 2 + (z_all / sigma_z) ** 2
keep_prob = density_scale * np.exp(-0.5 * r2)
keep = rng.random(len(df)) < keep_prob
kept_x, kept_y, kept_z = x_all[keep], y_all[keep], z_all[keep]

# Blur out real fine-scale density structure with small positional jitter - see
# prepare_stars_parquet.py for the full rationale.
jitter_x = rng.normal(0, sigma_x * 0.16, size=keep.sum())
jitter_y = rng.normal(0, sigma_y * 0.16, size=keep.sum())
jitter_z = rng.normal(0, sigma_z * 0.16, size=keep.sum())

df = df[keep].copy()
x, y, z = kept_x + jitter_x, kept_y + jitter_y, kept_z + jitter_z
print(f"{len(df)} stars after 3D ellipsoidal thinning + jitter (sigma={sigma_x},{sigma_y},{sigma_z}, density_scale={density_scale})")

categories = ["Bright Star", "Medium Star", "Dim Star"]
category_index = df["star_class"].map({name: i for i, name in enumerate(categories)}).to_numpy()

out = {
    "x": x.round(4).tolist(),
    "y": y.round(4).tolist(),
    "z": z.round(4).tolist(),
    "category": category_index.astype(int).tolist(),
    "categoryNames": categories,
    "magnitude": df["brightness_magnitude"].round(3).tolist(),
    "distancePc": df["distance_pc"].round(3).tolist(),
}

with open("packages/examples/public/real_stars.json", "w") as f:
    json.dump(out, f)

print(f"Wrote {len(x)} stars to packages/examples/public/real_stars.json")
print(f"x (RA offset, deg) range: [{x.min():.3f}, {x.max():.3f}]")
print(f"y (Dec offset, deg) range: [{y.min():.3f}, {y.max():.3f}]")
print(f"z (scaled distance) range: [{z.min():.3f}, {z.max():.3f}]")
