"""BDSIM's circular-mode sampler hits, split into turns, and its integrator sets."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import simba.Modules.Beams as rbf
from simba.Codes.Generators import frameworkGenerator
from simba.Modules.Beams import bdsim
from simba.Modules.Beams.bdsim import select_turn


def _arrays(*hits):
    """Sampler arrays from ``(eventID, turnNumber, parentID, T)`` hits."""
    columns = np.array(hits, dtype=float).T
    return {
        "eventID": columns[0].astype(int),
        "turnNumber": columns[1].astype(int),
        "parentID": columns[2].astype(int),
        "T": columns[3],
        "x": np.arange(len(hits), dtype=float),
    }


def test_each_primary_once_per_turn():
    arrays = _arrays((0, 1, 0, 10.0), (1, 1, 0, 10.1), (0, 2, 0, 20.0), (1, 2, 0, 20.1))
    assert select_turn(arrays, 2)["x"].tolist() == [2.0, 3.0]


def test_a_crossing_as_the_run_starts_is_dropped():
    """A particle starting behind the first element of a closed ring meets the
    last sampler at T ~ 0, then again at the end of its first turn."""
    arrays = _arrays((0, 1, 0, 1e-4), (0, 1, 0, 136.8), (1, 1, 0, 136.8))
    assert select_turn(arrays, 1)["x"].tolist() == [1.0, 2.0]


def test_secondaries_are_dropped():
    arrays = _arrays((0, 1, 0, 10.0), (0, 1, 1, 10.5))
    assert select_turn(arrays, 1)["x"].tolist() == [0.0]


def test_the_bunch_goes_to_bdsim_as_times(tmp_path):
    """Everything starts on the first element's face; a particle ahead arrives first."""
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)}, filename="seed.openpmd.hdf5",
        initial_momentum=100e6, sigma_x=1e-4, sigma_px=2e4, sigma_y=1e-4, sigma_py=2e4,
        sigma_z=1e-4, sigma_pz=2e4, charge=1e-15, number_of_particles=32,
    ).write()
    seed = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(seed, str(tmp_path / "seed.openpmd.hdf5"))
    bdsim.write_bdsim_beam_file(seed, str(tmp_path / "seed.bdsim"))
    back = rbf.beam()
    bdsim.read_bdsim_beam_file(back, str(tmp_path / "seed.bdsim"))
    z = seed.z.val - np.mean(seed.z.val)
    np.testing.assert_allclose(back.z.val, z, rtol=0, atol=1e-12)
    assert np.corrcoef(back.t.val, z)[0, 1] < -0.999


@pytest.mark.parametrize(
    "turns, angle, circular",
    [(1, 0.0, False), (1, 0.3, False), (1, 2 * math.pi, True), (1, -2 * math.pi, True), (4, 0.0, True)],
)
def test_circular_mode_for_turns_or_a_line_that_comes_round(turns, angle, circular):
    """A single pass round a ring needs the terminator too, or it never ends."""
    pytest.importorskip("pybdsim")
    from simba.Codes.BDSIM.BDSIM import bdsimLattice

    line = SimpleNamespace(turns=turns, net_bend_angle=angle)
    assert bdsimLattice.circular.fget(line) is circular


@pytest.mark.parametrize(
    "tracking, integrator_set",
    [(None, "bdsimmatrix"), ({"turns": 3}, "bdsimmatrix"), ({"integrator_set": "Geant4"}, "geant4")],
)
def test_the_integrator_set_comes_from_the_tracking_block(tracking, integrator_set):
    pytest.importorskip("pybdsim")
    from simba.Codes.BDSIM.BDSIM import bdsimLattice

    line = SimpleNamespace(file_block={"tracking": tracking})
    assert bdsimLattice.integrator_set.fget(line) == integrator_set


def test_an_unknown_integrator_set_is_refused():
    pytest.importorskip("pybdsim")
    from simba.Codes.BDSIM.BDSIM import bdsimLattice

    line = SimpleNamespace(file_block={"tracking": {"integrator_set": "rk4"}})
    with pytest.raises(ValueError, match="integrator_set"):
        bdsimLattice.integrator_set.fget(line)
