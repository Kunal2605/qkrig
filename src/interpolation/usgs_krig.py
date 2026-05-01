# usgs_krig.py

from __future__ import annotations

from core.base_krig import BaseKrig
from vis.visualizations import VariogramPlotter, KrigingMapPlotter


class USGSKrig(BaseKrig):
    """
    Thin wrapper around BaseKrig for USGS gauge data.
    Delegates all plotting to vis.visualizations.
    """

    def __init__(self, data, config_path, year, month, day, hour=None):
        super().__init__(data, config_path, year, month, day, hour=hour)
        self.variogram_plotter = VariogramPlotter(self)
        self.krig_map_plotter = KrigingMapPlotter(self)

    # --- Plotting API ----------------------------------------------------
    def plot_variogram(self):
        self.variogram_plotter.plot()

    def map_krig_interpolation(self):
        self.krig_map_plotter.plot_interpolation()

    def map_krig_error_variance(self):
        self.krig_map_plotter.plot_error_variance()

    def plot_interpolation_with_variogram(self, *args, **kwargs):
        return self.krig_map_plotter.plot_interpolation_with_variogram(*args, **kwargs)