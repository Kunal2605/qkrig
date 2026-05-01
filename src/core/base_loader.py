# base_loader.py

import os
from abc import ABC, abstractmethod
import yaml

class BaseLoader(ABC):
    def __init__(self, config_path):
        self._config_dir = os.path.dirname(os.path.abspath(config_path))
        self.config = self._load_config(config_path)
        self.date_format = self.config["settings"].get("date_format", "%Y-%m-%d")
        self.gauge_metadata = self._load_gauge_metadata()

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_path(self, p):
        """Resolve a path relative to the config file's directory.
        Absolute paths and empty values are returned unchanged."""
        if not p or os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(self._config_dir, p))

    @abstractmethod
    def _load_gauge_metadata(self):
        """
        Load and return a DataFrame of gauge metadata. Must set index to gauge_id.
        """
        pass

    @abstractmethod
    def get_streamflow(self, year, month, day):
        """
        Retrieve streamflow data for a specific date.
        Should return a list of tuples: (lon, lat, streamflow_mm/day, gauge_id)
        """
        pass