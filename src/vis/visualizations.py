# visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from cartopy.io import shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import box
from shapely import vectorized
from matplotlib.colors import LogNorm, PowerNorm, BoundaryNorm
from matplotlib import gridspec

class PlotConfig:
    def __init__(self, path=None):
        self.cfg = self._load_yaml_or_default(path)

    def _load_yaml_or_default(self, path):
        default = {
            "save_plots": False,
            "show_plots": True,
            "plots_directory": "./plots",
            "variogram": {
                "figure_size": [8, 5],
                "color": "blue",
                "label": "Empirical Variogram",
                "xlabel": "Distance (km)",
                "ylabel": "Semi-variance",
                "title_prefix": "Empirical Variogram",
                "legend": True,
                "min_value": None,   # y-axis lower bound
                "max_value": None,   # y-axis upper bound
                "ylog": False,       # log scale on y-axis
            },
            "kriging_interpolation": {
                "figure_size": [8, 6],
                "cmap": "coolwarm",
                "levels": 15,
                "colorbar_label": "Interpolated Streamflow (mm/day)",
                "max_value": None,
                "min_value": None,
                "log_scale": False,
                "scatter": {
                    "cmap": "coolwarm",
                    "s": 8,
                    "edgecolors": "none",
                    "label": "Observed Data",
                },
                "xlabel": "Longitude",
                "ylabel": "Latitude",
                "title_prefix": "Kriging Interpolation",
                "legend": True,
            },
            "kriging_error": {
                "figure_size": [8, 6],
                "cmap": "viridis",
                "levels": 15,
                "colorbar_label": "Kriging Error Variance",
                "xlabel": "Longitude",
                "ylabel": "Latitude",
                "title": "Kriging Error Variance Map",
            },
        }

        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                for k, v in default.items():
                    if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                        merged = v.copy()
                        merged.update(cfg[k])
                        default[k] = merged
                    elif k in cfg:
                        default[k] = cfg[k]
            except Exception:
                pass  # keep defaults
        return default

    def __getitem__(self, item):
        return self.cfg.get(item, {})



def _get_conus_mask(krig):
    """
    Boolean mask (ny, nx) True==inside CONUS, False==outside.

    Prefers the BaseKrig instance's cached mask (built during NC export), which
    avoids racing Natural Earth shapefile downloads when many hours are plotted
    in parallel.
    """
    cached = getattr(krig, "_get_conus_mask", None)
    if callable(cached):
        try:
            return cached()
        except Exception:
            pass  # fall through to local build below

    ny, nx = krig.grid_lat.size, krig.grid_lon.size
    glon = krig.grid_lon.astype(float).copy()
    glon = ((glon + 180.0) % 360.0) - 180.0
    glat = krig.grid_lat.astype(float)
    xx, yy = np.meshgrid(glon, glat)

    # Load US polygon and clip to CONUS bounds
    shpfilename = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    geoms = [rec.geometry for rec in shpreader.Reader(shpfilename).records()
             if rec.attributes.get("NAME") == "United States of America"]
    if not geoms:
        return None

    usa_union = unary_union(geoms)

    # Hard CONUS bbox: approx [-125, -66.5] lon, [24.5, 49.5] lat
    conus_bbox = box(-125.0, 24.5, -66.5, 49.5)
    conus_geom = usa_union.intersection(conus_bbox)

    mask = vectorized.contains(conus_geom, xx, yy) | vectorized.touches(conus_geom, xx, yy)
    return np.asarray(mask, dtype=bool)

def _get_land_mask(krig):
    """
    Returns boolean mask [lat, lon] where True==land, False==water.
    Caches on krig.land_mask. Only accepts masks that match (ny, nx).
    """
    ny, nx = krig.grid_lat.size, krig.grid_lon.size

    # If a mask is cached, ensure it matches the *current* grid
    if getattr(krig, "land_mask", None) is not None:
        m = krig.land_mask
        if isinstance(m, np.ndarray) and m.shape == (ny, nx):
            return m
        # stale or mismatched — drop it
        krig.land_mask = None

    # 1) Only accept external raster if shape matches exactly
    mask_path = krig.config.get("data", {}).get("land_mask")
    if mask_path and os.path.exists(mask_path):
        try:
            arr = np.load(mask_path)
            if arr.shape == (ny, nx):
                krig.land_mask = arr.astype(bool)
                return krig.land_mask
        except Exception:
            pass  # ignore and fall through

    # 2) Build on-the-fly from Natural Earth on the krig grid
    try:

        # Normalize longitudes to [-180, 180] to match Natural Earth
        glon = krig.grid_lon.astype(float).copy()
        glon = ((glon + 180.0) % 360.0) - 180.0
        glat = krig.grid_lat.astype(float)

        xx, yy = np.meshgrid(glon, glat)  # (ny, nx)

        shpfilename = shpreader.natural_earth(
            resolution="50m", category="physical", name="land"
        )
        geoms = list(shpreader.Reader(shpfilename).geometries())
        if not geoms:
            krig.land_mask = None
            return None

        # Crop geometries to our grid bbox for speed
        lon_min, lon_max = float(glon.min()), float(glon.max())
        lat_min, lat_max = float(glat.min()), float(glat.max())
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        geoms = [g for g in geoms if g.intersects(bbox)]
        if not geoms:
            krig.land_mask = None
            return None

        land_union = unary_union(geoms).buffer(0)

        # Prefer covers (includes boundaries); fall back to contains|touches
        if hasattr(vectorized, "covers"):
            mask = vectorized.covers(land_union, xx, yy)
        else:
            mask = vectorized.contains(land_union, xx, yy) | vectorized.touches(land_union, xx, yy)

        mask = np.asarray(mask, dtype=bool)

        # Guarantee mask shape == (ny, nx)
        if mask.shape != (ny, nx):
            # Transpose if needed (some backends flip axes)
            if mask.T.shape == (ny, nx):
                mask = mask.T
            else:
                # As a last resort, bail out rather than returning a mismatched array
                krig.land_mask = None
                return None

        krig.land_mask = mask
        return krig.land_mask

    except Exception:
        krig.land_mask = None
        return None


class VariogramPlotter:
    def __init__(self, krig_obj):
        self.krig = krig_obj
        self.plot_cfg = PlotConfig(getattr(self.krig, "plot_config_path", None))
        self.config = self.plot_cfg["variogram"]

    def plot(self, ax=None):
        if not self.krig.semivariogram_ready():
            raise RuntimeError(
                "Semivariogram not computed. Call `krig.compute_semivariogram(...)` before plotting."
            )

        bin_centers, semi_variance = self.krig._semivar_cache

        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=self.config.get("figure_size", [8, 5]))
            ax = fig.add_subplot(111)
            created_fig = True
        else:
            fig = ax.figure

        ax.scatter(
            bin_centers, semi_variance,
            c=self.config.get("color", "blue"),
            label=self.config.get("label", "Empirical Variogram"),
        )
        ax.set_xlabel(self.config.get("xlabel", "Distance (km)"))
        ax.set_ylabel(self.config.get("ylabel", "Semi-variance"))

        # Title: prefix + variogram model + date (+ hour when hourly)
        title_prefix = self.config.get("title_prefix", "Empirical Variogram")
        date_str = f"{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}"
        hour_str = f" {self.krig.hour:02d}:00 UTC" if getattr(self.krig, "hour", None) is not None else ""
        model = getattr(self.krig, "variogram_model", None) or ""
        model_part = f" ({model})" if model else ""
        ax.set_title(f"{title_prefix}{model_part} — {date_str}{hour_str}")

        # X-axis ticks: adaptive major spacing (so labels don't overlap for
        # CONUS-scale ranges of ~4500 km) plus minor ticks every 100 km so the
        # 100-km granularity is still visible as small tick marks.
        if len(bin_centers) > 0:
            from matplotlib.ticker import MultipleLocator
            xmax = float(np.nanmax(bin_centers))
            if   xmax <= 1000: major_step = 100
            elif xmax <= 2500: major_step = 250
            elif xmax <= 5000: major_step = 500
            else:              major_step = 1000
            ax.xaxis.set_major_locator(MultipleLocator(major_step))
            ax.xaxis.set_minor_locator(MultipleLocator(100))
            ax.set_xlim(left=0, right=xmax + 50)
            ax.tick_params(axis="x", which="minor", length=3)

        # Axis limits / scale
        ymin_cfg = self.config.get("min_value", 1)
        ymax_cfg = self.config.get("max_value", None)
        if self.config.get("ylog", False):
            ax.set_ylim(bottom=ymin_cfg, top=ymax_cfg)
            ax.set_yscale("log")
        else:
            if ymin_cfg is not None or ymax_cfg is not None:
                ax.set_ylim(bottom=ymin_cfg, top=ymax_cfg)

        if self.config.get("legend", True):
            ax.legend(loc="lower left")

        # Only save/show if we created the figure here
        if created_fig:
            save_plots = self.plot_cfg.cfg.get("save_plots", False)
            show_plots = self.plot_cfg.cfg.get("show_plots", True)
            plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")
            if save_plots:
                os.makedirs(plots_dir, exist_ok=True)
                hour_suffix = f"_{self.krig.hour:02d}" if getattr(self.krig, "hour", None) is not None else ""
                fname = f"variogram_{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}{hour_suffix}.png"
                fig.savefig(os.path.join(plots_dir, fname), dpi=300, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)



class KrigingMapPlotter:
    def __init__(self, krig_obj):
        self.krig = krig_obj
        self.plot_cfg = PlotConfig(getattr(self.krig, "plot_config_path", None))
        self.config_interp = self.plot_cfg["kriging_interpolation"]
        self.config_error = self.plot_cfg["kriging_error"]

    def plot_interpolation(self, ax=None):
        if self.krig.z_interp is None:
            raise RuntimeError("compute_kriging() must be run before plotting interpolation.")

        cfg = self.config_interp

        # --- Determine bounds safely ---
        z_raw = np.asarray(self.krig.z_interp)
        has_obs = hasattr(self.krig, "values") and isinstance(self.krig.values, np.ndarray) and self.krig.values.size > 0

        vmin_cfg = cfg.get("min_value", cfg.get("vmin", None))
        vmax_cfg = cfg.get("max_value", cfg.get("vmax", None))

        if vmin_cfg is None or vmax_cfg is None:
            if has_obs:
                data_min = float(np.nanmin(self.krig.values))
                data_max = float(np.nanmax(self.krig.values))
            else:
                data_min = float(np.nanmin(z_raw))
                data_max = float(np.nanmax(z_raw))
        else:
            data_min = data_max = None

        vmin = float(vmin_cfg) if vmin_cfg is not None else data_min
        vmax = float(vmax_cfg) if vmax_cfg is not None else data_max
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        z = np.clip(z_raw, vmin, vmax)

        # Masks
        land_mask = _get_land_mask(self.krig)
        if land_mask is not None:
            z = np.ma.masked_where(~land_mask, z)

        try:
            conus_mask = _get_conus_mask(self.krig)
            if conus_mask is not None:
                z = np.ma.masked_where(~conus_mask, z)
        except NameError:
            pass

        # Norm
        from matplotlib.colors import LogNorm, PowerNorm, BoundaryNorm
        norm_name = cfg.get("norm", "log" if cfg.get("log_scale", False) else "linear").lower()
        cmap = cfg.get("cmap", "viridis")
        norm = None
        eps = 1e-12
        if norm_name == "log":
            vmin_eff = max(vmin, eps)
            z = np.ma.masked_where(z <= 0, z)
            norm = LogNorm(vmin=vmin_eff, vmax=vmax)
        elif norm_name == "power":
            gamma = float(cfg.get("power_gamma", 0.5))
            vmin_eff = max(vmin, 0.0)
            norm = PowerNorm(gamma=gamma, vmin=vmin_eff, vmax=vmax)
        else:
            norm = None

        # Figure/axes
        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=cfg.get("figure_size", [8, 6]))
            ax = fig.add_subplot(111)
            created_fig = True
        else:
            fig = ax.figure

        # Render
        render_mode = cfg.get("render_mode", "pcolormesh").lower()
        if render_mode == "pcolormesh":
            mappable = ax.pcolormesh(
                self.krig.grid_lon, self.krig.grid_lat, z,
                shading="auto",
                cmap=cmap,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
            )
        else:
            levels_cfg = cfg.get("levels", 15)
            levels = levels_cfg
            if isinstance(levels_cfg, int) and norm_name == "log":
                base = float(cfg.get("log_scale_base", 10.0))
                start = np.log(max(vmin, eps)) / np.log(base)
                stop  = np.log(vmax) / np.log(base)
                if stop <= start:
                    stop = start + 1.0
                levels = np.logspace(start, stop, int(levels_cfg), base=base)
                norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)

            mappable = ax.contourf(
                self.krig.grid_lon, self.krig.grid_lat, z,
                levels=levels,
                cmap=cmap,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                extend="both",
            )

        # Colorbar
        cbar = fig.colorbar(mappable, ax=ax, label=cfg.get("colorbar_label", "Interpolated Streamflow (mm/day)"))

        # Observations (only if present)
        if has_obs:
            sc = ax.scatter(
                self.krig.lons, self.krig.lats, c=self.krig.values,
                s=cfg.get("scatter", {}).get("s", 8),
                cmap=cfg.get("scatter", {}).get("cmap", cmap),
                edgecolors=cfg.get("scatter", {}).get("edgecolors", "none"),
                label=cfg.get("scatter", {}).get("label", "Observed Data"),
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
            )

        # Labels & title (date + hour when hourly)
        ax.set_xlabel(cfg.get("xlabel", "Longitude"))
        ax.set_ylabel(cfg.get("ylabel", "Latitude"))
        date_str = f"{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}"
        hour_str = f" {self.krig.hour:02d}:00 UTC" if getattr(self.krig, "hour", None) is not None else ""
        ax.set_title(f"{cfg.get('title_prefix', 'Kriging Interpolation')} "
                     f"({getattr(self.krig, 'variogram_model', 'restored')} model) — {date_str}{hour_str}")
        if cfg.get("legend", True) and has_obs:
            ax.legend(loc="upper right")

        # Only save/show if we created the figure here
        if created_fig:
            save_plots = self.plot_cfg.cfg.get("save_plots", False)
            show_plots = self.plot_cfg.cfg.get("show_plots", True)
            plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")
            if save_plots:
                os.makedirs(plots_dir, exist_ok=True)
                hour_suffix = f"_{self.krig.hour:02d}" if getattr(self.krig, "hour", None) is not None else ""
                fname = f"kriging_interp_{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}{hour_suffix}.png"
                fig.savefig(os.path.join(plots_dir, fname), dpi=300, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    def plot_interpolation_with_variogram(self, figsize=(13, 6)):
        """
        Side-by-side composite for one hour: kriged flow map (left) and
        empirical variogram (right). Single PNG per call. Style matches the
        viz_sep27_composite reference — helene_flow blue→purple cmap, PowerNorm
        gamma=0.35 to keep low values visible, NaN rendered as parchment, no
        axis ticks/spines on the map, percentile-based colorbar limits, clean
        variogram with grid + p99 axis bounds.
        """
        combo_cfg = self.plot_cfg["combo"] if "combo" in self.plot_cfg.cfg else {}
        figsize = tuple(combo_cfg.get("figure_size", figsize))

        if self.krig.z_interp is None:
            raise RuntimeError("compute_kriging() must be run before plotting interpolation.")
        if not self.krig.semivariogram_ready():
            raise RuntimeError("Semivariogram not computed. Call `krig.compute_semivariogram(...)` first.")

        from matplotlib.colors import LinearSegmentedColormap, PowerNorm

        FLOW_COLORS = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5",
                       "#084594", "#4a1486", "#7a0177"]
        cmap = LinearSegmentedColormap.from_list("helene_flow", FLOW_COLORS, N=512)
        cmap.set_bad(color="#e8e4dc")
        POWER_GAMMA = 0.35

        # Hourly kriging map, masked to CONUS + non-negative
        z = self.krig.z_interp.astype(np.float32).copy()
        try:
            mask = _get_conus_mask(self.krig)
            z[~mask] = np.nan
        except Exception:
            pass  # fall back to whatever's already in z (may already be masked)
        z[z < 0] = np.nan

        # Fixed bounds keep the colorbar constant across hours. Fall back to
        # per-hour percentiles only if both bounds aren't set in plot config.
        krig_cfg = self.plot_cfg.cfg.get("kriging_interpolation", {}) or {}
        vmin_cfg = krig_cfg.get("min_value")
        vmax_cfg = krig_cfg.get("max_value")

        fin = z[np.isfinite(z)]
        if vmin_cfg is not None and vmax_cfg is not None:
            vmin, vmax = float(vmin_cfg), float(vmax_cfg)
        elif fin.size:
            vmin = max(float(np.percentile(fin, 1)), 0.0)
            vmax = max(float(np.percentile(fin, 99)), vmin + 0.1)
        else:
            vmin, vmax = 0.0, 1.0
        if vmin >= vmax:
            vmax = vmin + 0.1
        norm = PowerNorm(gamma=POWER_GAMMA, vmin=vmin, vmax=vmax)
        hour_mean = float(np.nanmean(z)) if fin.size else float("nan")

        H, W = z.shape
        DISP = W / H

        date_str = f"{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}"
        hour_str = f"{self.krig.hour:02d}:00 UTC" if getattr(self.krig, "hour", None) is not None else ""
        title_when = f"{date_str} {hour_str}".strip()

        # wspace 0.32 prevents colorbar tick labels crowding the variogram axis.
        fig, (ax_map, ax_v) = plt.subplots(
            1, 2,
            figsize=figsize,
            facecolor="white",
            gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.32},
        )

        # ----- Left: kriged flow map -----
        im = ax_map.imshow(
            np.clip(z, vmin, vmax),
            origin="lower",
            cmap=cmap,
            norm=norm,
            aspect=DISP,
            interpolation="bilinear",
        )
        ax_map.set_xticks([]); ax_map.set_yticks([])
        for sp in ax_map.spines.values():
            sp.set_visible(False)
        ax_map.set_xlabel(
            f"Hour mean: {hour_mean:.3f} mm/hr",
            fontsize=8, color="#333333",
        )
        ax_map.set_title(
            f"{title_when}  —  Kriged Flow",
            fontsize=11, fontweight="bold", pad=7,
        )

        cb = fig.colorbar(im, ax=ax_map, orientation="vertical",
                          fraction=0.030, pad=0.02, shrink=0.82)
        cb.set_label("Flow  (mm hr⁻¹)", fontsize=8.5, labelpad=6)
        ticks = np.linspace(vmin, vmax, 6)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{v:.2f}" for v in ticks], fontsize=7.5)
        cb.ax.tick_params(width=0.5, length=3)
        cb.outline.set_linewidth(0.5)

        # Right: variogram. variogram.{min_value,max_value,x_max} from config
        # keep axes constant across hours; fall back to per-hour p99 otherwise.
        var_cfg  = self.plot_cfg.cfg.get("variogram", {}) or {}
        v_ymin   = var_cfg.get("min_value")
        v_ymax   = var_cfg.get("max_value")
        v_xmax   = var_cfg.get("x_max")

        bin_centers, semi_variance = self.krig._semivar_cache
        fin_sv = semi_variance[np.isfinite(semi_variance)]
        fin_dist = bin_centers[np.isfinite(bin_centers)]

        if v_xmax is not None:
            dist_xmax = float(v_xmax)
        else:
            dist_xmax = (float(np.percentile(fin_dist, 99)) if fin_dist.size else 300.0) * 1.05

        if v_ymin is not None and v_ymax is not None:
            sv_ylim = (float(v_ymin), float(v_ymax))
        else:
            sv_ymax = float(np.percentile(fin_sv, 99)) if fin_sv.size else 1.0
            sv_ylim = (0.0, sv_ymax * 1.10)

        ax_v.scatter(
            bin_centers, semi_variance,
            s=40, color="#2171b5", alpha=0.65, linewidths=0, zorder=3,
            label="Empirical",
        )
        ax_v.set_xlim(0, dist_xmax)
        ax_v.set_ylim(*sv_ylim)
        ax_v.set_xlabel("Distance  (km)", fontsize=9, color="#333")
        ax_v.set_ylabel("Semi-variance", fontsize=9, color="#333")
        model = getattr(self.krig, "variogram_model", None) or ""
        model_part = f"  {model}" if model else ""
        ax_v.set_title(
            f"Variogram{model_part}  |  {hour_str}" if hour_str else f"Variogram{model_part}",
            fontsize=9, fontweight="bold", pad=5,
        )
        ax_v.tick_params(labelsize=8, direction="out", width=0.5, length=3)
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax_v.spines[sp].set_linewidth(0.6)
        ax_v.grid(True, alpha=0.2, lw=0.4, color="#888888")
        if ax_v.get_legend_handles_labels()[0]:
            leg = ax_v.legend(fontsize=8, framealpha=0.85,
                              edgecolor="#aaaaaa", fancybox=False)
            leg.get_frame().set_linewidth(0.5)

        # Save/show using global flags
        save_plots = self.plot_cfg.cfg.get("save_plots", False)
        show_plots = self.plot_cfg.cfg.get("show_plots", True)
        plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            hour_suffix = f"_{self.krig.hour:02d}" if getattr(self.krig, "hour", None) is not None else ""
            fname = f"kriging_combo_{date_str}{hour_suffix}.png"
            fig.savefig(os.path.join(plots_dir, fname), dpi=200, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_error_variance(self):
        if self.krig.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before plotting error variance.")

        var = self.krig.kriging_variance

        # Apply land mask (True==land, False==water)
        land_mask = _get_land_mask(self.krig)
        if land_mask is not None and land_mask.shape == var.shape:
            var = np.ma.masked_where(~land_mask.astype(bool), var)

        # CONUS mask
        conus_mask = _get_conus_mask(self.krig)
        if conus_mask is not None:
            var = np.ma.masked_where(~conus_mask, var)

        plt.figure(figsize=self.config_error.get("figure_size", [8, 6]))
        plt.contourf(
            self.krig.grid_lon, self.krig.grid_lat, var,
            levels=self.config_error.get("levels", 15),
            cmap=self.config_error.get("cmap", "viridis"),
        )
        plt.colorbar(label=self.config_error.get("colorbar_label", "Kriging Error Variance"))
        plt.xlabel(self.config_error.get("xlabel", "Longitude"))
        plt.ylabel(self.config_error.get("ylabel", "Latitude"))
        plt.title(self.config_error.get("title", "Kriging Error Variance Map"))
        save_plots = self.plot_cfg.cfg.get("save_plots", False)
        show_plots = self.plot_cfg.cfg.get("show_plots", True)
        plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")

        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            hour_suffix = f"_{self.krig.hour:02d}" if getattr(self.krig, "hour", None) is not None else ""
            fname = f"kriging_error_{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}{hour_suffix}.png"
            plt.savefig(os.path.join(plots_dir, fname), dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()