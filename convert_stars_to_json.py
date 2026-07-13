"""Convert real_stars.parquet (RA/Dec/parallax from Gaia DR3) into a browser-ready
star field: X/Y stay the familiar sky-plane RA/Dec (degrees, centered on the cone
center), and Z is distance from parallax, rescaled to a comparable visual range so
the flat 2D sky plot gains real, navigable depth."""

import json

import numpy as np
import pandas as pd

df = pd.read_parquet("real_stars.parquet")
print(f"Loaded {len(df)} stars")

# Distance from parallax (mas -> parsecs). Filter out non-positive/unreliable parallaxes.
df = df[df["distance_parallax"] > 0.05].copy()
df["distance_pc"] = 1000.0 / df["distance_parallax"]
# Drop extreme outlier distances (likely noisy parallax measurements) so the point
# cloud isn't dominated by a handful of far-flung points.
df = df[df["distance_pc"] < df["distance_pc"].quantile(0.98)]
print(f"{len(df)} stars after distance filtering")

ra_center, dec_center = 49.0, 3.7
x = df["ra"].to_numpy() - ra_center
y = df["dec"].to_numpy() - dec_center

# Rescale distance into roughly the same visual span as the RA/Dec cone (~[-1.5, 1.5]
# degrees), so the depth axis is comparably navigable rather than either a razor-thin
# sliver or a wildly stretched spike.
d = df["distance_pc"].to_numpy()
d_norm = (d - d.min()) / max(d.max() - d.min(), 1e-6)
z = (d_norm - 0.5) * 3.0

categories = ["Bright Star", "Medium Star", "Dim Star"]
category_index = df["star_class"].map({name: i for i, name in enumerate(categories)}).to_numpy()

out = {
    "x": x.round(4).tolist(),
    "y": y.round(4).tolist(),
    "z": z.round(4).tolist(),
    "category": category_index.astype(int).tolist(),
    "categoryNames": categories,
    "magnitude": df["brightness_magnitude"].round(3).tolist(),
    "distancePc": d.round(3).tolist(),
}

with open("packages/examples/public/real_stars.json", "w") as f:
    json.dump(out, f)

print(f"Wrote {len(x)} stars to packages/examples/public/real_stars.json")
print(f"x (RA offset, deg) range: [{x.min():.3f}, {x.max():.3f}]")
print(f"y (Dec offset, deg) range: [{y.min():.3f}, {y.max():.3f}]")
print(f"z (scaled distance) range: [{z.min():.3f}, {z.max():.3f}]")
print(f"distance_pc range: [{d.min():.1f}, {d.max():.1f}]")
