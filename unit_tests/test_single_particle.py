"""Single-particle and reference-particle modes.
**Single particle** tracks 13 probes.
"""

import numpy as np
import pytest

from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice
from simba.Modules.Matrices import (
    map_from_probes,
    probe_grid,
    transform_distribution,
)


class FakeLine:
    """A lattice stub carrying only what the flag code reads."""

    def __init__(self, tracking=None, code="madx", supports=True):
        self.file_block = {"tracking": tracking or {}}
        self.objectname = "RING"
        self.code = code
        self.supports_single_particle = supports

    single_particle = frameworkLattice.single_particle
    check_single_particle_supported = frameworkLattice.check_single_particle_supported
    codes_that_can = frameworkLattice.codes_that_can


# --- the flag -----------------------------------------------------------


def test_the_default_is_the_full_distribution():
    assert FakeLine().single_particle is False


def test_it_is_read_from_the_tracking_block():
    assert FakeLine({"single_particle": True}).single_particle is True


def test_madx_can():
    assert madxLattice.supports_single_particle is True


def test_the_base_class_assumes_it_cannot():
    assert frameworkLattice.supports_single_particle is False


def test_asking_a_code_that_cannot_warns_but_does_not_refuse():
    """Falling back to the full distribution is slower, not wrong."""
    line = FakeLine({"single_particle": True}, code="astra", supports=False)
    with pytest.warns(UserWarning, match="single-particle"):
        line.check_single_particle_supported()


def test_the_warning_says_the_results_still_stand():
    line = FakeLine({"single_particle": True}, code="astra", supports=False)
    with pytest.warns(UserWarning, match="results stand"):
        line.check_single_particle_supported()


def test_the_flag_no_longer_lives_on_madx():
    """R1's actual point: one home, not a per-code field. A field on the
    subclass would shadow the base property and quietly win."""
    assert "single_particle" not in madxLattice.model_fields


# --- the probe grid -----------------------------------------------------


def test_the_grid_is_thirteen_probes():
    grid = probe_grid(np.zeros(6), 1e-6)
    assert grid.shape == (6, 13)


def test_the_centroid_comes_first():
    centre = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(probe_grid(centre, 1e-6)[:, 0], centre)


def test_each_coordinate_is_straddled():
    """A pair either side, so the difference is centred and second-order
    accurate rather than one-sided."""
    grid = probe_grid(np.zeros(6), 1e-3)
    for index in range(6):
        assert grid[index, 1 + 2 * index] == pytest.approx(+1e-3)
        assert grid[index, 2 + 2 * index] == pytest.approx(-1e-3)


# --- recovering the map -------------------------------------------------


def known_map():
    matrix = np.eye(6)
    matrix[0, 1] = 2.0
    matrix[2, 3] = 1.5
    matrix[1, 0] = -0.25
    return matrix


def test_a_known_map_is_recovered_exactly():
    """Linear in, linear out -- finite differences are exact on a linear
    map, so this is a round trip and not an approximation."""
    matrix = known_map()
    centre = np.array([1e-3, 0.0, -2e-3, 0.0, 0.0, 0.0])
    tracked = matrix @ probe_grid(centre, 1e-6)
    centroid, recovered = map_from_probes(tracked, 1e-6)
    assert np.allclose(recovered, matrix, atol=1e-9)
    assert np.allclose(centroid, matrix @ centre, atol=1e-12)


def test_the_step_size_cancels():
    """Any step recovers the same linear map; it only matters against
    tracking noise and real nonlinearity."""
    matrix = known_map()
    for delta in (1e-8, 1e-6, 1e-3):
        _, recovered = map_from_probes(matrix @ probe_grid(np.zeros(6), delta), delta)
        assert np.allclose(recovered, matrix, atol=1e-6)


def test_a_lost_probe_gives_no_map():
    """Not a map built from the survivors: that would be quietly wrong
    rather than absent."""
    assert map_from_probes(np.zeros((6, 12)), 1e-6) == (None, None)


def test_a_diverged_probe_gives_no_map():
    tracked = np.zeros((6, 13))
    tracked[0, 5] = np.nan
    assert map_from_probes(tracked, 1e-6) == (None, None)


# --- carrying the distribution ------------------------------------------


def test_the_distribution_follows_the_map():
    """The saving: these particles were never tracked."""
    matrix = known_map()
    centre = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    bunch = centre[:, None] + np.random.default_rng(0).normal(0, 1e-4, (6, 200))
    tracked = matrix @ probe_grid(centre, 1e-6)
    centroid_out, recovered = map_from_probes(tracked, 1e-6)
    moved = transform_distribution(bunch, centre, centroid_out, recovered)
    assert np.allclose(moved, matrix @ bunch, atol=1e-9)


def test_the_centroid_lands_where_the_probe_did():
    matrix = known_map()
    centre = np.array([2e-3, 1e-4, 0.0, 0.0, 0.0, 0.0])
    centroid_out, recovered = map_from_probes(matrix @ probe_grid(centre, 1e-6), 1e-6)
    moved = transform_distribution(centre[:, None], centre, centroid_out, recovered)
    assert np.allclose(moved[:, 0], centroid_out, atol=1e-12)


# --- the reference particle ---------------------------------------------


@pytest.mark.parametrize(
    "cls", [xsuiteLattice, ocelotLattice, bmadLattice], ids=lambda c: c.__name__
)
def test_the_codes_that_record_a_trajectory(cls):
    assert cls.track_reference_particle is not frameworkLattice.track_reference_particle


def test_the_base_class_records_nothing():
    assert frameworkLattice.track_reference_particle(frameworkLattice) == {}
