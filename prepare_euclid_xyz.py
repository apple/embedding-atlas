"""Add precomputed x/y/z columns to real_euclid.parquet so it can be loaded directly via
the embedding-atlas CLI with --x/--y/--z. X/Y are RA/Dec offsets from the field center (same
convention as the Gaia star field). Euclid Q1's base catalog has no distance/redshift
column, so Z is a brightness-based depth proxy (fainter objects placed "deeper") rather
than a true distance - this is called out explicitly, not presented as real distance.

Uses the same 3D ellipsoidal thinning (applied to x, y, AND z together) as
prepare_stars_parquet.py, for a consistent soft-edged, symmetric look between the two
catalogs. Euclid Q1's raw source pool here is much smaller than Gaia's (under 10,000
objects), so a gentler density_scale is used - aggressive thinning on top of an
already-small pool would leave the field too sparse to read."""

import numpy as np
import pandas as pd

df = pd.read_parquet("real_euclid.parquet")
print(f"Loaded {len(df)} Euclid Q1 objects")

ra_center = df["ra"].median()
dec_center = df["dec"].median()
x_all = (df["ra"] - ra_center).to_numpy()
y_all = (df["dec"] - dec_center).to_numpy()

# Brightness-based depth proxy: fainter (lower flux) objects are placed further "back". This
# is NOT a real distance/redshift measurement - Euclid Q1's base catalog doesn't carry one -
# it's a visualization convenience so the field has real, data-driven depth structure.
mag_proxy = -2.5 * np.log10(df["flux_vis"].to_numpy())
z_score = (mag_proxy - mag_proxy.mean()) / max(mag_proxy.std(), 1e-6)
z_all = np.tanh(z_score / 2.2) * 2.0

sigma_x, sigma_y, sigma_z = 0.9, 0.9, 1.3
density_scale = 0.2
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
df["x"] = (kept_x + jitter_x).round(4)
df["y"] = (kept_y + jitter_y).round(4)
df["z"] = (kept_z + jitter_z).round(4)
print(f"{len(df)} objects after 3D ellipsoidal thinning + jitter (sigma={sigma_x},{sigma_y},{sigma_z}, density_scale={density_scale})")

df.to_parquet("real_euclid_xyz.parquet")
print(f"Wrote {len(df)} rows to real_euclid_xyz.parquet")
print(f"x range: [{df['x'].min():.3f}, {df['x'].max():.3f}]")
print(f"y range: [{df['y'].min():.3f}, {df['y'].max():.3f}]")
print(f"z range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")
