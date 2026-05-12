#!/usr/bin/env python3
"""
Extract kriged streamflow per catchment from qkrig hourly NC files and
append to per-catchment CSVs.

Optimized for whole-CONUS hydrofabric scale (~831k divides):
- Centroid sampling is vectorized over all catchments at once.
- A full day's 24 hours can be processed in a single invocation, with each
  per-catchment CSV opened once for one batched append.
- Timestep counter is read in O(1) bytes per file (last-line tail read)
  rather than scanning the whole file.

Usage:
    # Full day (preferred): all 24 hours in one pass
    python qkrig_ts_hourly.py YYYY-MM-DD /output_dir

    # Single hour (legacy): unchanged interface
    python qkrig_ts_hourly.py YYYY-MM-DD_HH /output_dir

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
    """Return (lons, lats, z_mm_hr) from a qkrig hourly NetCDF."""
    with xr.open_dataset(nc_path) as ds:
        lons = ds["lon"].values
        lats = ds["lat"].values
        z_mm_hr = ds["z_interp"].values.astype(np.float64)
    return lons, lats, z_mm_hr


def sample_centroids_vec(lons, lats, grid, pt_lons, pt_lats):
    """Vectorized nearest-cell sampling for many centroids at once.

    For sorted regular grids this returns the same index that
    `np.argmin(|lons - pt_lon|)` would return per point — just computed for
    all points in one pass via `np.searchsorted`. Returns a 1-D float64 array
    of length `len(pt_lons)`.
    """
    # Nearest column index in lons for each pt_lon
    ix_right = np.clip(np.searchsorted(lons, pt_lons, side="left"), 0, len(lons) - 1)
    ix_left = np.maximum(ix_right - 1, 0)
    pick_right_x = np.abs(lons[ix_right] - pt_lons) <= np.abs(lons[ix_left] - pt_lons)
    ix = np.where(pick_right_x, ix_right, ix_left)

    # Nearest row index in lats for each pt_lat
    iy_right = np.clip(np.searchsorted(lats, pt_lats, side="left"), 0, len(lats) - 1)
    iy_left = np.maximum(iy_right - 1, 0)
    pick_right_y = np.abs(lats[iy_right] - pt_lats) <= np.abs(lats[iy_left] - pt_lats)
    iy = np.where(pick_right_y, iy_right, iy_left)

    return grid[iy, ix].astype(np.float64)


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
    """Return the next timestep index for an existing per-catchment CSV.

    Reads only the last ~256 bytes (constant cost per file regardless of how
    long the time series has grown) and parses the integer in column 0. Falls
    back to 0 when the file is missing, empty, or only contains the header.
    """
    if not os.path.exists(path):
        return 0
    size = os.path.getsize(path)
    if size == 0:
        return 0
    with open(path, "rb") as f:
        f.seek(max(0, size - 256))
        tail = f.read()
    last_line = b""
    for line in reversed(tail.split(b"\n")):
        line = line.strip()
        if line:
            last_line = line
            break
    if not last_line:
        return 0
    try:
        last_ts = int(last_line.split(b",", 1)[0])
        return last_ts + 1
    except (ValueError, IndexError):
        return 0  # header-only or otherwise unparseable


def parse_when(s: str):
    """Accept 'YYYY-MM-DD_HH' (single hour) or 'YYYY-MM-DD' (full 24 hours).
    Returns a list of (hr_str, datetime) hour entries to process.
    """
    for fmt, n_hours in (("%Y-%m-%d_%H", 1), ("%Y-%m-%d", 24)):
        try:
            base = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return [
            (
                (base + dt.timedelta(hours=h)).strftime("%Y-%m-%d_%H"),
                base + dt.timedelta(hours=h),
            )
            for h in range(n_hours)
        ]
    print(f"Invalid date format: {s}. Expected YYYY-MM-DD or YYYY-MM-DD_HH")
    sys.exit(1)


# --- MAIN ---
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract per-catchment CSVs from qkrig NC files.")
    p.add_argument("when", help="YYYY-MM-DD (full day) or YYYY-MM-DD_HH (single hour)")
    p.add_argument("out_dir", help="Output directory for per-catchment CSVs")
    p.add_argument("--gpkg", default=None, help="Override: direct path to the .gpkg file")
    p.add_argument("--exports-dir", default=None,
                   help=f"Override: directory holding interp_*.nc files (default: {EXPORT_DIR})")
    args = p.parse_args()

    hours = parse_when(args.when)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.gpkg:
        GPKG_PATH = args.gpkg
    if args.exports_dir:
        EXPORT_DIR = args.exports_dir

    gdf = load_gpkg()
    print(f"Loaded {len(gdf)} catchments from {GPKG_PATH}")

    # Pull centroid coords + ids out as numpy arrays once — avoids per-row
    # geopandas lookup costs on the 831k-row whole-CONUS path.
    raw_ids = gdf[ID_FIELD].astype(str).values
    num_ids = np.array([s.split("-")[-1] for s in raw_ids])
    centroid_lons = gdf["centroid"].x.values.astype(np.float64)
    centroid_lats = gdf["centroid"].y.values.astype(np.float64)
    N = len(raw_ids)

    # Sample each hour vectorized; store as a (N, H) value matrix + parallel
    # list of time strings (None for hours whose NC was missing/unreadable).
    H = len(hours)
    values = np.full((N, H), np.nan, dtype=np.float64)
    time_strs = [None] * H

    for h_idx, (hr_str, hr_dt) in enumerate(hours):
        nc_file = nc_path_for_hour(hr_str)
        if not os.path.exists(nc_file):
            print(f"  No NC file for {hr_str}: {nc_file}, skipping")
            continue
        try:
            lons, lats, z_mm_hr = load_nc(nc_file)
        except Exception as e:
            print(f"  Could not read {nc_file}: {e}, skipping")
            continue
        values[:, h_idx] = sample_centroids_vec(
            lons, lats, z_mm_hr, centroid_lons, centroid_lats
        )
        time_strs[h_idx] = hr_dt.strftime("%Y-%m-%d %H:%M:%S")

    valid_hours = sum(1 for t in time_strs if t is not None)
    if valid_hours == 0:
        print("No usable NC files found; nothing written.")
        sys.exit(0)

    # One batched append per catchment (instead of per-hour, per-row open).
    print(f"Writing CSVs for {N} catchments × {valid_hours} hour(s)...")
    written = 0
    for i in range(N):
        cat_file = os.path.join(out_dir, f"cat-{num_ids[i]}.csv")
        nex_file = os.path.join(out_dir, f"nex-{num_ids[i]}_output.csv")

        ts = next_timestep(cat_file)

        cat_buf = []
        nex_buf = []
        if ts == 0:  # only the very first write to this cat file gets the header
            cat_buf.append("timestep,time,qkrig_mm_hr\n")

        for h_idx in range(H):
            t = time_strs[h_idx]
            if t is None:
                continue  # NC was missing for this hour; skip the row
            v = values[i, h_idx]
            cat_buf.append(f"{ts},{t},{v:.6f}\n")
            nex_buf.append(f"{ts}, {t}, {v:.6f}\n")
            ts += 1

        with open(cat_file, "a") as f:
            f.writelines(cat_buf)
        with open(nex_file, "a") as f:
            f.writelines(nex_buf)
        written += 1

    print(f"Wrote {written} catchment CSV pair(s) to {out_dir}")
