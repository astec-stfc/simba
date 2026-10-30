from types import SimpleNamespace as NS

import numpy as np

from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Framework_objects import frameworkLattice


def test_frequency_map_skips_lost_particles():
    turns = np.arange(64)
    track = np.cos(2 * np.pi * 0.31 * turns)
    tracks = {k: np.array([track, track]) for k in ("x", "px", "y", "py")}
    tracks["state"] = np.array([np.ones(64), np.r_[np.ones(10), np.full(54, 2)]])
    fake = NS(
        _grid_points=lambda: [(1e-3, 1e-3), (2e-3, 2e-3)], _track_grid_turn_by_turn=lambda: tracks,
        normalisation_twiss=lambda: None, _ALIVE=1, objectname="ring",
    )
    fake._footprint = lambda tracks: frameworkLattice._footprint(fake, tracks)
    footprint = bmadLattice.run_frequency_map(fake)
    assert [p[:2] for p in footprint] == [(1e-3, 1e-3)]
