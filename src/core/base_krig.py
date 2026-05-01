# base_krig.py

import os
import numpy as np
import yaml
import xarray as xr
import pandas as pd
from pyproj import Geod
from pykrige.ok import OrdinaryKriging
from shapely.ops import unary_union
from shapely.geometry import box
import cartopy.io.shapereader as shpreader
from typing import Optional, Tuple, Dict


def _build_conus_mask(grid_lon: np.ndarray, grid_lat: np.ndarray) -> np.ndarray:
    """
    Build a boolean CONUS mask of shape (H, W) in native S→N lat order.
    Grid points inside CONUS (with a small 0.05° buffer) are True.
    """
    H, W = len(grid_lat), len(grid_lon)
    xx, yy = np.meshgrid(grid_lon, grid_lat, indexing='xy')   # (H, W)

    shpfilename = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    geoms = [
        rec.geometry for rec in shpreader.Reader(shpfilename).records()
        if rec.attributes.get("NAME") == "United States of America"
    ]
    conus_bbox = box(float(grid_lon[0]), float(grid_lat[0]),
                     float(grid_lon[-1]), float(grid_lat[-1]))
    conus_geom = unary_union(geoms).intersection(conus_bbox).buffer(0.05)

    # vectorised point-in-polygon
    from shapely.vectorized import contains as _contains
    mask = np.asarray(_contains(conus_geom, xx, yy), dtype=bool)  # (H, W)
    if mask.shape != (H, W):
        mask = mask.T
    return mask


class BaseKrig:
    def __init__(self, data, config_path, year, month, day, hour=None):
        if len(data) == 0:
            raise ValueError("Input data is empty.")

        self.data = np.array(data, dtype=float)
        self.lons = self.data[:, 0]
        self.lats = self.data[:, 1]
        self.values = self.data[:, 2]
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour  # None for daily; int 0-23 for hourly
        self.geod = Geod(ellps="WGS84")

        # Load kriging config
        config_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}

        kcfg = self.config.get("kriging", {}) or {}
        raw_plot_cfg = self.config.get("plot_config", None)
        if raw_plot_cfg and not os.path.isabs(raw_plot_cfg):
            self.plot_config_path = os.path.join(config_dir, raw_plot_cfg)
        else:
            self.plot_config_path = raw_plot_cfg

        land_mask_path = self.config.get("data", {}).get("land_mask")
        if land_mask_path and os.path.exists(land_mask_path):
            self.land_mask = np.load(land_mask_path)
        else:
            self.land_mask = None

        # Validate config entries
        required_keys = ["grid_size", "variogram_model", "variogram_bins"]
        for key in required_keys:
            if key not in kcfg:
                raise KeyError(f"Missing '{key}' in kriging config.")

        self.grid_size = int(kcfg["grid_size"])
        self.variogram_model = kcfg["variogram_model"]
        self.variogram_bins = int(kcfg["variogram_bins"])

        # Create interpolation grid
        lon_min, lon_max = np.min(self.lons), np.max(self.lons)
        lat_min, lat_max = np.min(self.lats), np.max(self.lats)
        self.grid_lon = np.linspace(lon_min, lon_max, self.grid_size)
        self.grid_lat = np.linspace(lat_min, lat_max, self.grid_size)
        self.grid_lon_mesh, self.grid_lat_mesh = np.meshgrid(self.grid_lon, self.grid_lat)

        # Placeholders for kriging results
        self.z_interp: Optional[np.ndarray] = None
        self.kriging_variance: Optional[np.ndarray] = None

        # Semivariogram cache
        self._semivar_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._semivar_bins_used: Optional[int] = None

        # CONUS mask cache (built lazily on first export)
        self._conus_mask: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # CONUS mask
    # ---------------------------------------------------------------------
    def _get_conus_mask(self) -> np.ndarray:
        """
        Lazily load and cache the CONUS mask (H, W) in S→N lat order.
        Uses land_mask from config if available, otherwise builds from Natural Earth.
        """
        if self._conus_mask is None:
            land_mask_path = self.config.get("data", {}).get("land_mask")
            if land_mask_path and os.path.exists(land_mask_path):
                self._conus_mask = np.load(land_mask_path)
            else:
                self._conus_mask = _build_conus_mask(self.grid_lon, self.grid_lat)
        return self._conus_mask

    # ---------------------------------------------------------------------
    # Core computations
    # ---------------------------------------------------------------------
    def compute_kriging(self):
        kcfg = self.config.get("kriging", {}) or {}
        variogram_params = None

        if kcfg.get("range"):
            variogram_params = {
                "sill": kcfg.get("sill", None),
                "range": float(kcfg["range"]) / 111.0,  # km -> degrees (approx)
                "nugget": kcfg.get("nugget", 0.0),
            }

        ok = OrdinaryKriging(
            self.lons, self.lats, self.values,
            variogram_model=self.variogram_model,
            exact_values=kcfg.get("exact_values", True),
            nlags=kcfg.get("nlags", 12),
            weight=kcfg.get("weight", True),
            variogram_parameters=variogram_params,
        )

        self.z_interp, self.kriging_variance = ok.execute("grid", self.grid_lon, self.grid_lat)

    def compute_semivariogram(self, bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and cache the empirical semivariogram on geodesic distances.
        Returns:
            bin_centers_km: (B,) array
            semi_variance:  (B,) array (NaN where no pairs fall in bin)
        """
        if bins is None:
            bins = int(self.variogram_bins)

        n = len(self.lons)
        if n < 2:
            raise ValueError("Need at least two points to compute a semivariogram.")

        dists_km = []
        sqdiff = []
        for i in range(n):
            for j in range(i + 1, n):
                _, _, d_m = self.geod.inv(self.lons[i], self.lats[i], self.lons[j], self.lats[j])
                dists_km.append(d_m / 1000.0)
                sqdiff.append((self.values[i] - self.values[j]) ** 2)

        dists_km = np.asarray(dists_km, dtype=float)
        sqdiff = np.asarray(sqdiff, dtype=float)

        if dists_km.size == 0:
            raise ValueError("Not enough unique pairs to compute a semivariogram.")

        bin_edges = np.linspace(0.0, float(np.nanmax(dists_km)), bins + 1)
        bin_idx = np.digitize(dists_km, bin_edges) - 1

        semi_variance = np.full(bins, np.nan, dtype=float)
        for b in range(bins):
            mask = (bin_idx == b)
            if np.any(mask):
                semi_variance[b] = float(np.nanmean(sqdiff[mask]))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        self._semivar_cache = (bin_centers, semi_variance)
        self._semivar_bins_used = int(bins)
        return bin_centers, semi_variance

    # ---------------------------------------------------------------------
    # Readiness helpers
    # ---------------------------------------------------------------------
    def semivariogram_ready(self, bins: Optional[int] = None) -> bool:
        if self._semivar_cache is None:
            return False
        if bins is None:
            return True
        return (self._semivar_bins_used == int(bins))

    # ---------------------------------------------------------------------
    # Export helpers
    # ---------------------------------------------------------------------
    def _date_str(self) -> str:
        base = f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        if self.hour is not None:
            return f"{base}_{self.hour:02d}"
        return base

    def _resolve_exports(self) -> str:
        exp_cfg: Dict = (self.config.get("exports") or {})
        export_dir = exp_cfg.get("directory", "./exports")
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def export_all(self, bins: Optional[int] = None) -> Tuple[str, str]:
        """
        Export both interpolation (.nc) and semivariogram (.csv) to the configured directory.
        Requires that compute_kriging() and compute_semivariogram() were called beforehand.
        """
        if self.z_interp is None or self.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before export_all().")
        if not self.semivariogram_ready(bins):
            raise RuntimeError("compute_semivariogram() must be run (with matching bins) before export_all().")

        export_dir = self._resolve_exports()
        d = self._date_str()

        interp_path = os.path.join(export_dir, f"interp_{d}.nc")
        vario_path  = os.path.join(export_dir, f"variogram_{d}.csv")

        self.export_interpolation(interp_path)
        self.export_variogram(vario_path, bins=bins)

        return interp_path, vario_path

    # ---------------------------------------------------------------------
    # Exports
    # ---------------------------------------------------------------------
    def export_interpolation(self, out_path: str):
        """
        Export interpolation grids (z and variance) to NetCDF (.nc).
        Applies CONUS mask: points outside CONUS and negative values → NaN.
        Requires compute_kriging() beforehand.
        """
        if self.z_interp is None or self.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before exporting interpolation.")

        # Build masked copies — shape (H, W), lat S→N, lon W→E
        conus_mask = self._get_conus_mask()

        z = np.array(self.z_interp, dtype=np.float32)
        var = np.array(self.kriging_variance, dtype=np.float32)

        z[~conus_mask]   = np.nan
        z[z < 0]         = np.nan
        var[~conus_mask] = np.nan

        # Assemble xarray Dataset
        ds = xr.Dataset(
            {
                "z_interp": (["lat", "lon"], z),
                "kriging_variance": (["lat", "lon"], var),
            },
            coords={
                "lat": self.grid_lat,   # S→N (CF convention)
                "lon": self.grid_lon,   # W→E
            },
        )

        ds["z_interp"].attrs["units"]        = "mm/day"
        ds["z_interp"].attrs["long_name"]    = "Kriged streamflow"
        ds["z_interp"].attrs["lat_order"]    = "S→N (CF convention); use origin='lower' with imshow"
        ds["kriging_variance"].attrs["long_name"] = "Kriging error variance"

        ds.attrs["date"]            = f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        if self.hour is not None:
            ds.attrs["hour"]        = int(self.hour)
        ds.attrs["variogram_model"] = self.variogram_model
        ds.attrs["grid_size"]       = int(self.grid_size)

        encoding = {
            "z_interp":         {"dtype": "float32", "zlib": True, "complevel": 4},
            "kriging_variance":  {"dtype": "float32", "zlib": True, "complevel": 4},
        }

        ds.to_netcdf(out_path, encoding=encoding)

    def export_variogram(self, out_path: str, bins: Optional[int] = None):
        """
        Export the empirical semivariogram to CSV.
        Requires compute_semivariogram() beforehand (with same bins if provided).
        """
        if not self.semivariogram_ready(bins):
            raise RuntimeError(
                "compute_semivariogram() must be called before exporting the variogram "
                "(and bins must match if specified)."
            )

        bin_centers, semi_variance = self._semivar_cache
        rows = np.column_stack([bin_centers, semi_variance])
        header = "distance_km,semi_variance"
        np.savetxt(out_path, rows, delimiter=",", header=header, comments="")

    # ---------------------------------------------------------------------
    # Plot delegation
    # ---------------------------------------------------------------------
    def plot_variogram(self):
        raise NotImplementedError("Use visualization module to plot variogram.")

    def map_krig_interpolation(self):
        raise NotImplementedError("Use visualization module to plot interpolation.")

    def map_krig_error_variance(self):
        raise NotImplementedError("Use visualization module to plot error variance.")

    def plot_interpolation_with_variogram(self):
        raise NotImplementedError("Use visualization module to plot combo.")
