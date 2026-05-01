#!/usr/bin/env python3
"""
Extract kriged streamflow for a single HOUR and append to per-catchment CSVs.

Reads an hourly .nc file (interp_YYYY-MM-DD_HH.nc), samples the kriged
grid at each catchment centroid in a .gpkg, and appends one row per
catchment to CSVs in the output directory.

Usage:
    python qkrig_ts_hourly.py YYYY-MM-DD_HH /output_dir

    # Example:
    python qkrig_ts_hourly.py 2024-09-26_04 /mnt/disk1/qkrig/subdaily/

Output files per catchment:
    cat-{id}.csv          columns: timestep,time,qkrig_mm_hr
    nex-{id}_output.csv   same data, space-padded columns
"""

import sys, os, datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

# --- USER CONFIG ---
GPKG_PATH  = "/home/ksarna/Documents/qkrig/hydrofabric/gage-03456100_subset.gpkg"
EXPORT_DIR = "/home/ksarna/Documents/qkrig/exports"
LAYER      = "divides"
ID_FIELD   = "divide_id"    # column in the divides layer (values: 'cat-XXXXXXX')


# --- Helpers ---
def nc_path_for_hour(hr_str: str) -> str:
    return os.path.join(EXPORT_DIR, f"interp_{hr_str}.nc")


def load_nc(nc_path: str):
    """Return (lons, lats, z_mm_hr) from a qkrig NetCDF. Converts mm/day → mm/hr."""
    with xr.open_dataset(nc_path) as ds:
        lons = ds["lon"].values
        lats = ds["lat"].values
        z_mm_day = ds["z_interp"].values.astype(np.float64)
    z_mm_hr = z_mm_day / 24.0
    return lons, lats, z_mm_hr


def sample_centroid(lons, lats, grid, pt_lon, pt_lat) -> float:
    ix = np.argmin(np.abs(lons - pt_lon))
    iy = np.argmin(np.abs(lats - pt_lat))
    val = float(grid[iy, ix])
    return val if np.isfinite(val) else 0.0


def load_gpkg() -> gpd.GeoDataFrame:
    """Load divides layer, compute WGS84 centroids."""
    if os.path.isdir(GPKG_PATH):
        gpkg_files = sorted(f for f in os.listdir(GPKG_PATH) if f.endswith(".gpkg"))
        if not gpkg_files:
            print(f"No .gpkg files found in {GPKG_PATH}")
            sys.exit(1)
        gdfs = []
        for gf in gpkg_files:
            try:
                gdfs.append(gpd.read_file(os.path.join(GPKG_PATH, gf), layer=LAYER))
            except Exception as e:
                print(f"  Warning: could not read {gf}: {e}")
        if not gdfs:
            print("No .gpkg files could be loaded")
            sys.exit(1)
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry")
    else:
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER)

    # Centroids in projected CRS, then convert to WGS84 for sampling
    gdf_proj = gdf if not gdf.crs.is_geographic else gdf.to_crs("EPSG:5070")
    gdf["centroid"] = gpd.GeoSeries(
        gdf_proj.geometry.centroid, crs=gdf_proj.crs
    ).to_crs(4326)
    return gdf


def next_timestep(path: str) -> int:
    """Return the next timestep index (count of existing data rows)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header


def write_row(path: str, timestep: int, time_str: str, val: float, spaced: bool):
    """Append one row; write header if file is new."""
    new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if new_file and not spaced:
            f.write("timestep,time,qkrig_mm_hr\n")
        if spaced:
            f.write(f"{timestep}, {time_str}, {val:.6f}\n")
        else:
            f.write(f"{timestep},{time_str},{val:.6f}\n")


# --- MAIN ---
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract per-catchment CSVs from a qkrig NC file.")
    p.add_argument("hr_str", help="Hour string YYYY-MM-DD_HH (e.g. 2024-09-26_04)")
    p.add_argument("out_dir", help="Output directory for per-catchment CSVs")
    p.add_argument("--nc",   default=None, help="Override: direct path to the .nc file")
    p.add_argument("--gpkg", default=None, help="Override: direct path to the .gpkg file")
    args = p.parse_args()

    hr_str  = args.hr_str
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Allow YYYY-MM-DD (daily) or YYYY-MM-DD_HH (hourly)
    hour_dt = None
    for fmt in ("%Y-%m-%d_%H", "%Y-%m-%d"):
        try:
            hour_dt = dt.datetime.strptime(hr_str, fmt)
            break
        except ValueError:
            continue
    if hour_dt is None:
        print(f"Invalid date format: {hr_str}. Expected YYYY-MM-DD_HH or YYYY-MM-DD")
        sys.exit(1)

    nc_file = args.nc if args.nc else nc_path_for_hour(hr_str)
    if not os.path.exists(nc_file):
        print(f"No NC file for {hr_str}: {nc_file}, skipping")
        sys.exit(0)

    if args.gpkg:
        GPKG_PATH = args.gpkg

    gdf = load_gpkg()
    print(f"Loaded {len(gdf)} catchments from {args.gpkg or GPKG_PATH}")

    lons, lats, z_mm_hr = load_nc(nc_file)
    time_str = hour_dt.strftime("%Y-%m-%d %H:%M:%S")

    for _, row in gdf.iterrows():
        raw_id = str(row[ID_FIELD])           # e.g. "cat-1016279"
        num_id = raw_id.split("-")[-1]        # e.g. "1016279"
        centroid = row["centroid"]

        val = sample_centroid(lons, lats, z_mm_hr, centroid.x, centroid.y)

        cat_file = os.path.join(out_dir, f"cat-{num_id}.csv")
        nex_file = os.path.join(out_dir, f"nex-{num_id}_output.csv")

        timestep = next_timestep(cat_file)
        write_row(cat_file, timestep, time_str, val, spaced=False)
        write_row(nex_file, timestep, time_str, val, spaced=True)

    print(f"Appended {hr_str} → {len(gdf)} catchments in {out_dir}")
