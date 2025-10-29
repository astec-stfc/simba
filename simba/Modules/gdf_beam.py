import re
from munch import Munch
import easygdf
import numpy as np


class gdf_beam(Munch):

    def __init__(self, filename) -> None:
        super().__init__(self)
        self.screens_touts = easygdf.load_screens_touts(filename)
        self.sort_screens()
        self._create_positions_dictionary()
        self.sort_touts()
        self._create_times_dictionary()

    @property
    def screens(self) -> dict:
        return self.screens_touts["screens"]

    @property
    def touts(self) -> dict:
        return self.screens_touts["touts"]

    def _create_positions_dictionary(self) -> None:
        self._positions = {s["position"]: s for s in self.screens}

    @property
    def positions(self) -> dict:
        return self._positions

    def single_position_data(self) -> None:
        [setattr(self, k, self.screens_touts[k]) for k in ["x", "y", "z", "GBx", "GBy", "GBz", "m", "q", "nmacro"]]

    def _create_times_dictionary(self) -> None:
        self._times = {s["time"]: s for s in self.touts}

    @property
    def times(self) -> dict:
        return self._times

    def sorted_nicely(self, unsorted_list: list, dict_key: str | None = None) -> list:
        """Sort the given iterable in the way that humans expect."""

        def convert(text):
            return int(text) if text.isdigit() else text

        def alphanum_key(key):
            return [convert(c) for c in re.split("([0-9]+)", str(key[dict_key]))]

        return sorted(unsorted_list, key=alphanum_key)

    def sort_screens(self, **kwargs) -> None:
        self.screens_touts["screens"] = self.sorted_nicely(
            self.screens_touts["screens"], dict_key="position"
        )

    def sort_touts(self, **kwargs) -> None:
        self.screens_touts["touts"] = self.sorted_nicely(
            self.screens_touts["touts"], dict_key="time"
        )

    def get_position(self, position: float) -> dict | None:
        if position in self._positions.keys():
            return Munch(self._positions[position])
        return None

    def get_time(self, time: float) -> dict | None:
        if time in self._times.keys():
            return Munch(self._times[time])
        return None
