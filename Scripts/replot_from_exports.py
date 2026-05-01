#!/usr/bin/env python3
"""
Rebuild **combined** (map + short variogram) plots from saved kriging exports.

- Looks in exports directory for BOTH:
    interp_YYYY-MM-DD.npz
    variogram_YYYY-MM-DD.csv
- Draws the stacked figure via visualizations.KrigingMapPlotter.plot_interpolation_with_variogram()
  (which calls plot_interpolation + variogram in a 2-row layout).

Usage examples:
  python Scripts/replot_combo_from_exports.py --config configs/usgsgaugekrig.yaml
  python Scripts/replot_combo_from_exports.py --config configs/usgsgaugekrig.yaml \
      --start 2020-01-01 --end 2020-01-31 --save-plots --no-show
  python Scripts/replot_combo_from_exports.py --config configs/usgsgaugekrig.yaml \
      --only 2020-01-05 2020-01-07
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict

import numpy as np
import xarray as xr
import yaml

# Project imports (works if installed with `pip install -e .`)
from vis.visualizations import VariogramPlotter, KrigingMapPlotter, PlotConfig

DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# ----------------------- CLI -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild combined (map + variogram) plots from saved exports.")
    p.add_argument("--config", required=True, help="Path to YAML (e.g., configs/usgsgaugekrig.yaml)")
    p.add_argument("--exports-dir", default=None,
                   help="Override exports directory (defaults to exports.directory in YAML).")
    p.add_argument("--plot-config", default=None,
                   help="Path to plot_config.yaml (defaults to config['plot_config']).")

    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (filter files).")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (filter files).")
    p.add_argument("--only", nargs="*", default=None,
                   help="Specific YYYY-MM-DD dates to render (overrides start/end if provided).")

    # Optional overrides for plotting behavior
    p.add_argument("--save-plots", action="store_true", help="Force PlotConfig.save_plots=True")
    p.add_argument("--no-show", action="store_true", help="Force PlotConfig.show_plots=False")
    return p.parse_args()


# ----------------------- Helpers -----------------------

def parse_date_from_name(path: str) -> Optional[date]:
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return date(y, mth, d)


def list_export_pairs(exports_dir: str) -> Dict[date, Dict[str, str]]:
    """
    Return {date: {"interp": path_npz, "vario": path_csv}}
    (date appears only if both files exist)
    """
    seen: Dict[date, Dict[str, str]] = {}

    for nc in glob.glob(os.path.join(exports_dir, "interp_*.nc")):
        dt = parse_date_from_name(nc)
        if dt is None:
            continue
        seen.setdefault(dt, {})["interp"] = nc

    for csv in glob.glob(os.path.join(exports_dir, "variogram_*.csv")):
        dt = parse_date_from_name(csv)
        if dt is None:
            continue
        seen.setdefault(dt, {})["vario"] = csv

    # Keep only dates that have both
    return {d: v for d, v in seen.items() if "interp" in v and "vario" in v}


def filter_dates(dates: List[date], start: Optional[str], end: Optional[str], only: Optional[List[str]]) -> List[date]:
    if only:
        set_only = {datetime.strptime(s, "%Y-%m-%d").date() for s in only}
        return sorted([d for d in dates if d in set_only])
    if start:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        dates = [d for d in dates if d >= d0]
    if end:
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
        dates = [d for d in dates if d <= d1]
    return sorted(dates)


class RestoredKrig:
    """
    Minimal object that satisfies visualizations.py expectations:
      - grid_lon, grid_lat, z_interp, kriging_variance
      - year, month, day
      - variogram_model (for title; taken from NPZ meta if present, else 'restored')
      - values (empty -> skips scatter if desired)
      - config (full YAML so masks/path work)
      - plot_config_path (for PlotConfig)
      - semivariogram cache so VariogramPlotter can render
      - variogram_plotter (wired here, so combo call can reuse)
    """
    def __init__(self, cfg: dict, plot_cfg_path: Optional[str], dt: date):
        self.config = cfg
        self.plot_config_path = plot_cfg_path
        self.year, self.month, self.day = dt.year, dt.month, dt.day
        self.variogram_model = "restored"
        self.values = np.array([])  # no observations for replot
        self.land_mask = None

        # semivariogram cache
        self._semivar_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # will be set after construction
        self.variogram_plotter: Optional[VariogramPlotter] = None

    def semivariogram_ready(self) -> bool:
        return self._semivar_cache is not None


def load_interp_nc(nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    with xr.open_dataset(nc_path) as ds:
        grid_lon = ds["lon"].values
        grid_lat = ds["lat"].values
        z_interp = ds["z_interp"].values
        krig_var = ds["kriging_variance"].values
        vmodel = ds.attrs.get("variogram_model", None)
    return grid_lon, grid_lat, z_interp, krig_var, vmodel


def load_variogram_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.asarray(arr["distance_km"]), np.asarray(arr["semi_variance"])


# ----------------------- Main -----------------------

def main() -> int:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    exports_dir = args.exports_dir or (cfg.get("exports", {}) or {}).get("directory")
    if not exports_dir:
        raise SystemExit("No exports directory provided (--exports-dir) and none found in config['exports']['directory'].")

    plot_cfg_path = args.plot_config or cfg.get("plot_config")

    pairs = list_export_pairs(exports_dir)
    if not pairs:
        print(f"No complete export pairs (interp+variogram) found in {exports_dir}")
        return 0

    all_dates = sorted(pairs.keys())
    sel_dates = filter_dates(all_dates, args.start, args.end, args.only)
    if not sel_dates:
        print("No dates match the filters.")
        return 0

    # Prepare a PlotConfig and optionally override save/show
    pc = PlotConfig(plot_cfg_path)
    if args.save_plots:
        pc.cfg["save_plots"] = True
    if args.no_show:
        pc.cfg["show_plots"] = False

    for dt in sel_dates:
        nc_path  = pairs[dt]["interp"]
        csv_path = pairs[dt]["vario"]

        try:
            grid_lon, grid_lat, z_interp, krig_var, vmodel = load_interp_nc(nc_path)
            dist_km, semi_var = load_variogram_csv(csv_path)
        except Exception as e:
            print(f"[{dt}] failed to load exports: {e}")
            continue

        # Build a minimal krig-like object
        rk = RestoredKrig(cfg, plot_cfg_path, dt)
        rk.grid_lon = grid_lon
        rk.grid_lat = grid_lat
        rk.grid_lon_mesh, rk.grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        rk.z_interp = z_interp
        rk.kriging_variance = krig_var
        if vmodel:
            rk.variogram_model = vmodel
        rk._semivar_cache = (dist_km, semi_var)

        # Wire plotters
        vp = VariogramPlotter(rk)
        km = KrigingMapPlotter(rk)
        rk.variogram_plotter = vp  # so combo can call it

        # Inject our single PlotConfig into both plotters
        for pl in (vp, km):
            pl.plot_cfg = pc
        vp.config = pc["variogram"]
        km.config_interp = pc["kriging_interpolation"]
        km.config_error = pc["kriging_error"]

        # Render the combined figure (map on top, short variogram below)
        try:
            km.plot_interpolation_with_variogram()
        except Exception as e:
            print(f"[{dt}] combo plot failed: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
