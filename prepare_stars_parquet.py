"""Add precomputed x/y/z columns (sky-plane RA/Dec + parallax-derived depth) to
real_stars.parquet so it can be loaded directly via the embedding-atlas CLI with
--x/--y/--z pointing at pre-computed columns (no UMAP needed for real astrometry).

The point cloud is thinned with a single 3D ellipsoidal Gaussian applied to x, y, AND z
together (not a 2D sky-plane falloff with an independently-thinned depth axis). A pure
probabilistic thin only ever *removes* stars, though - it can never fill in a real gap,
so a genuine dip in this field's actual star density (a real, lumpy patch of sky, not a
bug) still shows through as a notch no matter how the keep-probability is tuned. Small
Gaussian positional jitter, added after thinning, blurs that real fine structure into a
smooth, round, cluster-like shape without discarding or fabricating any star's identity -
each point is still a real star, just nudged slightly off its exact catalog position."""

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

# log10(distance) is much less skewed than raw parsecs; z-score it and pass through tanh so
# outliers compress smoothly toward the range edges instead of being hard-clipped at a
# percentile cutoff - there is no distance at which points simply vanish or get truncated.
log_d = np.log10(df["distance_pc"].to_numpy())
z_score = (log_d - log_d.mean()) / max(log_d.std(), 1e-6)
z_all = np.tanh(z_score / 2.2) * 2.3

# A single 3D ellipsoidal Gaussian (evaluated on x, y, AND z together, not thinned per-axis
# independently) gives one clean, symmetric falloff instead of a 2D sky-plane shape combined
# with an unrelated depth distribution. `sigma_*` set how far the field spreads on each axis;
# `density_scale` caps the *peak* keep-probability (even dead center) well below 1.0, so the
# field stays airy and spread out everywhere rather than solid in the middle.
sigma_x, sigma_y, sigma_z = 5.5, 5.5, 1.6
density_scale = 0.04
rng = np.random.RandomState(42)
r2 = (x_all / sigma_x) ** 2 + (y_all / sigma_y) ** 2 + (z_all / sigma_z) ** 2
keep_prob = density_scale * np.exp(-0.5 * r2)
keep = rng.random(len(df)) < keep_prob
kept_x, kept_y, kept_z = x_all[keep], y_all[keep], z_all[keep]

# Blur out real fine-scale density structure (see module docstring) with small positional
# jitter - about a sixth of each axis's spread, small enough to preserve the overall
# ellipsoid shape but large enough to round off a sharp real gap into a smooth falloff.
jitter_x = rng.normal(0, sigma_x * 0.16, size=keep.sum())
jitter_y = rng.normal(0, sigma_y * 0.16, size=keep.sum())
jitter_z = rng.normal(0, sigma_z * 0.16, size=keep.sum())

df = df[keep].copy()
df["x"] = (kept_x + jitter_x).round(4)
df["y"] = (kept_y + jitter_y).round(4)
df["z"] = (kept_z + jitter_z).round(4)
print(f"{len(df)} stars after 3D ellipsoidal thinning + jitter (sigma={sigma_x},{sigma_y},{sigma_z}, density_scale={density_scale})")

df.to_parquet("real_stars_xyz.parquet")
print(f"Wrote {len(df)} rows to real_stars_xyz.parquet")
print(df[["x", "y", "z", "distance_pc", "star_class"]].head())
print(f"x range: [{df['x'].min():.3f}, {df['x'].max():.3f}]")
print(f"y range: [{df['y'].min():.3f}, {df['y'].max():.3f}]")
print(f"z range: [{df['z'].min():.3f}, {df['z'].max():.3f}], std: {df['z'].std():.3f}")
