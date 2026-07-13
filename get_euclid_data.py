"""Fetch real Euclid Q1 (ESA Euclid telescope, Quick Release 1) source catalog data from
LSDB, covering part of the EDF-North deep field. A second, genuinely different LSDB survey
alongside the Gaia DR3 star field, for comparing two real catalogs in the same viewer."""

import lsdb

print("1. Connecting to LSDB Euclid Q1 remote server...")
euclid_cat = lsdb.open_catalog(  # type: ignore
    "https://server.data.lsdb.io/hats/euclid_q1",
    columns=["RIGHT_ASCENSION", "DECLINATION", "FLUX_VIS_2FWHM_APER", "VIS_DET"],
)

print("2. Performing a Cone Search (radius of 2.0 degrees) in the EDF-North deep field...")
cone = euclid_cat.cone_search(ra=269.7, dec=66.0, radius_arcsec=7200)  # type: ignore

print("3. Fetching the Euclid object data...")
df = cone.compute()  # type: ignore

print("4. Filtering to VIS-detected sources with valid flux...")
df = df[(df["VIS_DET"] == True) & (df["FLUX_VIS_2FWHM_APER"] > 0)]  # noqa: E712

df = df.rename(
    columns={
        "RIGHT_ASCENSION": "ra",
        "DECLINATION": "dec",
        "FLUX_VIS_2FWHM_APER": "flux_vis",
    }
)

# LSDB adds an internal `_healpix_29` spatial-partitioning index column; for this catalog its
# int64 values exceed JavaScript's safe integer range (2^53), which crashes the viewer's
# Arrow/BigInt-to-Number conversion when rendering the data table. It's an internal index, not
# meaningful application data, so drop it rather than carry it through to the frontend.
df = df.drop(columns=["_healpix_29"], errors="ignore")

df.to_parquet("real_euclid.parquet")
print(f"5. Success! Saved {len(df)} real Euclid Q1 objects to 'real_euclid.parquet'!")
