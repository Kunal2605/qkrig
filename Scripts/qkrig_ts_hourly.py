#!/usr/bin/env python3
"""
Aggregate qkrig hourly NetCDFs into a single per-day parquet file.

Output schema (long format, BROTLI-compressed, modeled on T-Route output):
    feature_id   int64        numeric suffix of `divide_id` (e.g. cat-100001 → 100001)
    time         timestamp    UTC hourly
    type         string       always "cat" (sampled at catchment centroid)
    qkrig_mm_hr  float32      kriged streamflow, mm/hr (NaN where NC was missing
                              or grid was masked out of CONUS)

File naming:
    qkrig_output_YYYYMMDD.parquet           full-day run (24 hours, default)
    qkrig_output_YYYYMMDD_HH.parquet        single-hour legacy run

Usage:
    python qkrig_ts_hourly.py YYYY-MM-DD     /output_dir [--gpkg ...] [--exports-dir ...]
    python qkrig_ts_hourly.py YYYY-MM-DD_HH  /output_dir [--gpkg ...] [--exports-dir ...]

Rows are catchment-major (all 24 hours of feature 0, then all 24 of feature 1, ...)
to match T-Route's row ordering. ~20M rows/day at full CONUS (831,777 × 24);
typical compressed size ~80-100 MB.
"""

import sys, os, datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

# --- DEFAULTS (overridden by CLI flags) ---
GPKG_PATH   = "/home/ksarna/Documents/qkrig/hydrofabric/gage-03456100_subset.gpkg"
EXPORT_DIR  = "/home/ksarna/Documents/qkrig/exports"
LAYER       = "divides"
ID_FIELD    = "divide_id"
COMPRESSION = "brotli"


def nc_path_for_hour(hr_str: str) -> str:
    return os.path.join(EXPORT_DIR, f"interp_{hr_str}.nc")


def load_nc(nc_path: str):
    """Return (lons, lats, z_mm_hr) from a qkrig hourly NetCDF."""
    with xr.open_dataset(nc_path) as ds:
        lons = ds["lon"].values
        lats = ds["lat"].values
        z_mm_hr = ds["z_interp"].values.astype(np.float32)
    return lons, lats, z_mm_hr


def sample_centroids_vec(lons, lats, grid, pt_lons, pt_lats):
    """Vectorized nearest-cell sampling for many centroids at once.

    For sorted regular grids this returns the same index that
    `np.argmin(|lons - pt_lon|)` would per point — just computed for all points
    in one pass via `np.searchsorted`. Returns a float32 array of length len(pt_lons).
    """
    ix_right = np.clip(np.searchsorted(lons, pt_lons, side="left"), 0, len(lons) - 1)
    ix_left = np.maximum(ix_right - 1, 0)
    pick_right_x = np.abs(lons[ix_right] - pt_lons) <= np.abs(lons[ix_left] - pt_lons)
    ix = np.where(pick_right_x, ix_right, ix_left)

    iy_right = np.clip(np.searchsorted(lats, pt_lats, side="left"), 0, len(lats) - 1)
    iy_left = np.maximum(iy_right - 1, 0)
    pick_right_y = np.abs(lats[iy_right] - pt_lats) <= np.abs(lats[iy_left] - pt_lats)
    iy = np.where(pick_right_y, iy_right, iy_left)

    return grid[iy, ix].astype(np.float32)


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

    gdf_proj = gdf if not gdf.crs.is_geographic else gdf.to_crs("EPSG:5070")
    gdf["centroid"] = gpd.GeoSeries(
        gdf_proj.geometry.centroid, crs=gdf_proj.crs
    ).to_crs(4326)
    return gdf


def parse_feature_id(divide_id: str) -> int:
    """Strip the `cat-` prefix and parse the suffix as an integer.

    Routes through float() so that the 4 v2.2 hydrofabric oddballs with
    scientific-notation suffixes (`cat-1e+05`, `cat-5e+05`, `cat-8e+05`,
    `cat-3e+06`) land at their numeric values (100000, 500000, 800000, 3000000)
    instead of staying as strings.
    """
    suffix = divide_id.split("-", 1)[1]
    return int(float(suffix))


def parse_when(s: str):
    """Accept 'YYYY-MM-DD_HH' (single hour) or 'YYYY-MM-DD' (full 24 hours).
    Returns (hour_entries, n_hours) where each entry is (hr_str, datetime).
    """
    for fmt, n_hours in (("%Y-%m-%d_%H", 1), ("%Y-%m-%d", 24)):
        try:
            base = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return [
            ((base + dt.timedelta(hours=h)).strftime("%Y-%m-%d_%H"),
             base + dt.timedelta(hours=h))
            for h in range(n_hours)
        ], n_hours
    print(f"Invalid date format: {s}. Expected YYYY-MM-DD or YYYY-MM-DD_HH")
    sys.exit(1)


# --- MAIN ---
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate qkrig NCs into one per-day parquet.")
    p.add_argument("when", help="YYYY-MM-DD (full day) or YYYY-MM-DD_HH (single hour)")
    p.add_argument("out_dir", help="Output directory for the parquet file")
    p.add_argument("--gpkg", default=None, help="Override: path to hydrofabric .gpkg")
    p.add_argument("--exports-dir", default=None,
                   help=f"Override: directory holding interp_*.nc files (default: {EXPORT_DIR})")
    p.add_argument("--compression", default=COMPRESSION,
                   help="Parquet compression codec (default: brotli, matches T-Route output)")
    args = p.parse_args()

    hours, n_hours = parse_when(args.when)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.gpkg:
        GPKG_PATH = args.gpkg
    if args.exports_dir:
        EXPORT_DIR = args.exports_dir

    # --- Load catchments + centroids ---
    gdf = load_gpkg()
    print(f"Loaded {len(gdf)} catchments from {GPKG_PATH}")

    raw_ids = gdf[ID_FIELD].astype(str).values
    feature_ids = np.fromiter(
        (parse_feature_id(s) for s in raw_ids),
        dtype=np.int64, count=len(raw_ids),
    )
    centroid_lons = gdf["centroid"].x.values.astype(np.float64)
    centroid_lats = gdf["centroid"].y.values.astype(np.float64)
    N = len(feature_ids)
    H = len(hours)

    # --- Sample each hour into the (N, H) value matrix ---
    # Missing/unreadable NCs leave that column at NaN — those land in the
    # parquet as `qkrig_mm_hr = NaN`, matching how T-Route ships nulls.
    values = np.full((N, H), np.nan, dtype=np.float32)
    times_dt = []
    valid_count = 0
    for h_idx, (hr_str, hr_dt) in enumerate(hours):
        times_dt.append(hr_dt)
        nc_file = nc_path_for_hour(hr_str)
        if not os.path.exists(nc_file):
            print(f"  No NC file for {hr_str}: {nc_file} — leaving NaN")
            continue
        try:
            lons, lats, z_mm_hr = load_nc(nc_file)
        except Exception as e:
            print(f"  Could not read {nc_file}: {e} — leaving NaN")
            continue
        values[:, h_idx] = sample_centroids_vec(
            lons, lats, z_mm_hr, centroid_lons, centroid_lats
        )
        valid_count += 1

    if valid_count == 0:
        print("No usable NC files found; nothing written.")
        sys.exit(0)

    # --- Build long-format DataFrame (catchment-major, T-Route order) ---
    # repeat:  [f0,f0,...,f0, f1,f1,...,f1, ...] (24 copies of each fid)
    # tile:    [t0,t1,...,t23, t0,t1,...,t23, ...] (24 times cycled N times)
    # ravel:   row-major flatten of (N,H) → matches the above layout
    times_arr = np.array(times_dt, dtype="datetime64[ns]")
    df = pd.DataFrame({
        "feature_id":  np.repeat(feature_ids, H),
        "time":        np.tile(times_arr, N),
        "type":        pd.Categorical.from_codes(
            np.zeros(N * H, dtype=np.int8), categories=["cat"]
        ),
        "qkrig_mm_hr": values.ravel(),
    })

    # --- Output filename ---
    if n_hours == 1:
        fname = f"qkrig_output_{hours[0][1].strftime('%Y%m%d_%H')}.parquet"
    else:
        fname = f"qkrig_output_{hours[0][1].strftime('%Y%m%d')}.parquet"
    out_path = os.path.join(out_dir, fname)

    df.to_parquet(out_path, compression=args.compression, engine="pyarrow", index=False)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {len(df):,} rows ({N} catchments × {H} hours) to {out_path}")
    print(f"  valid hours: {valid_count}/{H}")
    print(f"  file size: {size_mb:.1f} MB ({args.compression})")
