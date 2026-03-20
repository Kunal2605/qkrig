#!/usr/bin/env python3
"""
Extract kriged streamflow for a single HOUR and append to per-catchment CSVs.

Reads an hourly .npz file (interp_YYYY-MM-DD_HH.npz), samples the kriged
grid at each catchment centroid in a .gpkg, and appends one row per
catchment to water-year-organised CSVs.

Usage:
    python qkrig_ts_hourly.py YYYY-MM-DD_HH /output_dir

    # Example:
    python qkrig_ts_hourly.py 2024-09-26_04 /mnt/disk1/qkrig/subdaily/

CSV columns: datetime, qkrig, variance
"""

import sys, os, datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd

# --- USER CONFIG ---
GPKG_PATH    = "/mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/"
EXPORT_DIR   = "/mnt/disk1/usgskrig/exports/gridsize/200/conus/range/100km/all_guages/"
LAYER        = "divides"
ID_FIELD     = "divide_id"
GRID_LON_KEY = "grid_lon"
GRID_LAT_KEY = "grid_lat"
GRID_VAL_KEY = "z_interp"
GRID_VAR_KEY = "kriging_variance"

# --- Helpers ---
def npz_path_for_hour(hr_str: str) -> str:
    """Return the .npz path for an hour string like '2024-09-26_04'."""
    return os.path.join(EXPORT_DIR, f"interp_{hr_str}.npz")


def nearest_grid_value(lons, lats, vals, pt_lon, pt_lat):
    ix = np.argmin(np.abs(lons - pt_lon))
    iy = np.argmin(np.abs(lats - pt_lat))
    return float(vals[iy, ix])


def grid_sample_both(npz_path, centroids):
    with np.load(npz_path, allow_pickle=True) as z:
        L, A = z[GRID_LON_KEY], z[GRID_LAT_KEY]
        V, VV = z[GRID_VAL_KEY], z.get(GRID_VAR_KEY, None)
    if VV is None:
        return centroids.apply(lambda pt: (nearest_grid_value(L, A, V, pt.x, pt.y), np.nan))
    return centroids.apply(lambda pt: (
        nearest_grid_value(L, A, V, pt.x, pt.y),
        nearest_grid_value(L, A, VV, pt.x, pt.y)
    ))


def water_year(d: dt.date) -> int:
    """Return the water year for a given date (Oct 1 - Sep 30)."""
    return d.year + 1 if d.month >= 10 else d.year


def load_gpkg():
    """Load all .gpkg files from GPKG_PATH (directory) or a single file."""
    if os.path.isdir(GPKG_PATH):
        gpkg_files = [
            os.path.join(GPKG_PATH, f)
            for f in sorted(os.listdir(GPKG_PATH))
            if f.endswith(".gpkg")
        ]
        if not gpkg_files:
            print(f"No .gpkg files found in {GPKG_PATH}")
            sys.exit(1)
        gdfs = []
        for gf in gpkg_files:
            try:
                gdf = gpd.read_file(gf, layer=LAYER)
                gdfs.append(gdf)
            except Exception as e:
                print(f"  Warning: could not read {gf}: {e}")
        if not gdfs:
            print("No .gpkg files could be loaded")
            sys.exit(1)
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    else:
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER)

    # Project to compute centroids, then back to WGS84
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs("EPSG:5070")
    else:
        gdf_proj = gdf
    cent_proj = gdf_proj.geometry.centroid
    gdf["centroid"] = gpd.GeoSeries(cent_proj, crs=gdf_proj.crs).to_crs(4326)
    return gdf


# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python qkrig_ts_hourly.py YYYY-MM-DD_HH /output_dir")
        sys.exit(1)

    hr_str = sys.argv[1]             # e.g. "2024-09-26_04"
    out_dir = sys.argv[2]

    # Parse the hour string
    try:
        hour_dt = dt.datetime.strptime(hr_str, "%Y-%m-%d_%H")
    except ValueError:
        print(f"Invalid hour format: {hr_str}. Expected YYYY-MM-DD_HH")
        sys.exit(1)

    d = hour_dt.date()
    h = hour_dt.hour

    npz_file = npz_path_for_hour(hr_str)
    if not os.path.exists(npz_file):
        print(f"No NPZ file for {hr_str}, skipping")
        sys.exit(0)

    # Load .gpkg catchments
    gdf = load_gpkg()
    print(f"Loaded {len(gdf)} catchments")

    # Sample hourly kriged values at centroids
    vals = grid_sample_both(npz_file, gdf["centroid"])
    ser_val = vals.apply(lambda x: x[0])
    ser_var = vals.apply(lambda x: x[1])

    wy = water_year(d)
    wy_dir = os.path.join(out_dir, f"WY{wy}")
    os.makedirs(wy_dir, exist_ok=True)

    # Append to per-catchment CSVs (one row per hour)
    datetime_str = hour_dt.strftime("%Y-%m-%d %H:%M")
    for cat_id, val, var_val in zip(gdf[ID_FIELD].values, ser_val.values, ser_var.values):
        cat_file = os.path.join(wy_dir, f"{cat_id}.csv")
        df_hour = pd.DataFrame({
            "datetime": [datetime_str],
            "qkrig": [val],
            "variance": [var_val],
        })
        if os.path.exists(cat_file):
            df_hour.to_csv(cat_file, mode='a', header=False, index=False)
        else:
            df_hour.to_csv(cat_file, mode='w', header=True, index=False)

    print(f"Appended hourly values for {hr_str} to WY{wy} ({len(gdf)} catchments)")
