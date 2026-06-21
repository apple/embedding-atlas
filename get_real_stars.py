import lsdb

print("1. Connecting to LSDB Gaia DR3 remote server...")
# We load the raw catalog (no server-side filters to avoid bugs)
gaia_cat = lsdb.open_catalog(  # type: ignore
    "https://server.data.lsdb.io/hats/gaia_dr3/gaia",
    columns=["ra", "dec", "phot_g_mean_mag", "parallax", "pmra", "pmdec"]
)

print("2. Performing a Cone Search (radius of 1.5 degrees) for a beautiful circular cluster...")
# This selects stars in a perfect, natural circle around RA 49.0, Dec 3.7
# (the center of the region you just downloaded)
cone = gaia_cat.cone_search(ra=49.0, dec=3.7, radius_arcsec=5400)  # type: ignore

print("3. Fetching the star data...")
# We fetch 10,000 stars from this circular region
df = cone.head(n=10000)  # type: ignore

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

# Save the dataset to a Parquet file
df.to_parquet("real_stars.parquet")
print("5. Success! Saved 10,000 real stars to 'real_stars.parquet'!")
