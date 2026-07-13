import lsdb

print("1. Connecting to LSDB Gaia DR3 remote server...")
# We load the raw catalog (no server-side filters to avoid bugs)
gaia_cat = lsdb.open_catalog(  # type: ignore
    "https://server.data.lsdb.io/hats/gaia_dr3/gaia",
    columns=["ra", "dec", "phot_g_mean_mag", "parallax", "pmra", "pmdec"]
)

print("2. Performing a Cone Search (radius of 9.0 degrees) for a large, richly-populated field...")
# A much wider radius than before (9 degrees, ~350,000 raw stars vs. the earlier 4 degrees,
# ~12,000 stars), so that after quality filtering and a smooth radial thinning (applied in
# prepare_stars_parquet.py / convert_stars_to_json.py) we comfortably clear 100,000 displayed
# stars with an organically fading density instead of a small, hard-edged patch.
cone = gaia_cat.cone_search(ra=49.0, dec=3.7, radius_arcsec=32400)  # type: ignore

print("3. Fetching the star data...")
# Fetch every star in the cone (not `.head(n)`, which would only return the first N rows
# in HATS/LSDB's internal HEALPix partition order - the first few sky patches, not a
# representative spatial spread of the whole cone). Taking the full result guarantees
# every part of the cone is represented, regardless of partitioning.
df = cone.compute()  # type: ignore

print("4. Filtering and classifying the stars...")
# Rename the columns so they look clean and readable in the dashboard
df = df.rename(
    columns={
        "phot_g_mean_mag": "brightness_magnitude",
        "parallax": "distance_parallax",
        "pmra": "proper_motion_ra",
        "pmdec": "proper_motion_dec",
    }
)

# To make the dashboard colorful, we create a categorical column based on star brightness!
def get_star_class(mag):
    if mag < 15.0:
        return "Bright Star"
    elif mag < 18.0:
        return "Medium Star"
    else:
        return "Dim Star"

df["star_class"] = df["brightness_magnitude"].apply(get_star_class)

# LSDB adds an internal `_healpix_29` spatial-partitioning index column whose int64 values
# exceed JavaScript's safe integer range (2^53), which crashes the viewer's Arrow/BigInt-to-
# Number conversion when rendering the data table. It's an internal index, not meaningful
# application data, so drop it rather than carry it through to the frontend.
df = df.drop(columns=["_healpix_29"], errors="ignore")

# Save the dataset to a Parquet file
df.to_parquet("real_stars.parquet")
print(f"5. Success! Saved {len(df)} real stars to 'real_stars.parquet'!")
