"""The one-turn map is the cheapest useful thing a ring model produces.

Tune, periodic beta, chromaticity, momentum compaction and stability all
follow from the 6x6 map of the closed path, and none of them need a bunch
tracked. Every code that can do a ring has a native call for it -- elegant
writes `%s.mat` on every run, Xsuite has `compute_R_matrix`, Ocelot has
`lattice_transfer_map`, Tao has `matrix` -- so simba reads theirs rather than
rebuilding the map itself.

What simba does *not* do is convert them to a common convention. The codes
disagree (`x'` vs `px`, and five different longitudinal pairs) and a wrong
conversion is silent, so each map is stored as its code wrote it with the
convention recorded beside it. The transverse blocks are comparable as they
stand, which is what a cross-code tune check needs.
"""

import warnings

import numpy as np
import pytest

from simba.Codes.ASTRA.ASTRA import astraLattice
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

RING_CODES = [elegantLattice, xsuiteLattice, ocelotLattice, bmadLattice, madxLattice]


def drift_map(length=2.0):
    """A 2 m drift: symplectic, and the one map every convention agrees on."""
    matrix = np.eye(6)
    matrix[0, 1] = length
    matrix[2, 3] = length
    return matrix


class FakeLine:
    """A lattice stub carrying only what the one-turn-map code reads."""

    def __init__(self, matrix=None, periodic=True, supports=True, code="elegant"):
        self.file_block = {"tracking": {"periodic": periodic}}
        self.code = code
        self.objectname = "RING"
        self.supports_periodic = supports
        self.one_turn_map = None
        self.optics_summary = None
        self.closed_orbit = None
        self._matrix = matrix

    def read_optics_summary(self):
        return {}

    def read_closed_orbit(self):
        return None

    def _machine_geometry(self):
        return None

    def read_one_turn_map(self):
        return self._matrix

    periodic = frameworkLattice.periodic
    check_one_turn_map = frameworkLattice.check_one_turn_map
    postProcess = frameworkLattice.postProcess


# --- when the map is read at all ----------------------------------------


def test_a_periodic_line_reads_its_map():
    line = FakeLine(drift_map())
    line.postProcess()
    assert line.one_turn_map is not None


def test_an_open_line_reads_nothing():
    """A transfer line has no one turn, so there is no map to ask for."""
    line = FakeLine(drift_map(), periodic=False)
    line.postProcess()
    assert line.one_turn_map is None


def test_a_code_that_cannot_match_reads_nothing():
    line = FakeLine(drift_map(), supports=False)
    line.postProcess()
    assert line.one_turn_map is None


def test_a_code_with_no_map_to_give_is_not_an_error():
    """elegant writes no matrix file when LSC is on, and that is survivable."""
    line = FakeLine(None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.postProcess()
    assert line.one_turn_map is None


# --- the determinant check ----------------------------------------------
#
# det(R) == 1 for a map that neither creates nor destroys phase-space volume,
# and unlike a full symplecticity test it holds in every convention the codes
# use -- so it can run before anything is normalised.


def test_a_symplectic_map_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FakeLine(drift_map()).postProcess()


def test_a_damped_map_warns():
    """Half the volume: a read that lost a factor somewhere."""
    matrix = drift_map()
    matrix[1, 1] = 0.5
    with pytest.warns(UserWarning, match="determinant"):
        FakeLine(matrix).postProcess()


def test_an_empty_map_warns():
    with pytest.warns(UserWarning, match="determinant"):
        FakeLine(np.zeros((6, 6))).postProcess()


def test_the_warning_says_what_is_untrustworthy():
    """The failure mode is a plausible wrong tune, so the warning has to name
    what should not be believed."""
    with pytest.warns(UserWarning, match="tune, beta, momentum compaction"):
        FakeLine(np.zeros((6, 6))).postProcess()


def test_a_wrong_shaped_map_warns_about_its_shape():
    with pytest.warns(UserWarning, match="not 6x6"):
        FakeLine(np.eye(4)).postProcess()


def test_a_transposed_symplectic_map_still_passes():
    """Honesty about the check's reach: det is necessary, not sufficient, and
    a transpose preserves it. This is why R8 compares codes to each other."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FakeLine(drift_map().T).postProcess()


# --- the convention is recorded, not converted --------------------------


@pytest.mark.parametrize("cls", RING_CODES, ids=lambda c: c.__name__)
def test_every_ring_code_declares_its_convention(cls):
    assert cls.otm_convention, f"{cls.__name__} must say what its map means"
    assert len(cls.otm_convention.split(",")) == 6


def test_the_codes_do_not_all_agree():
    """If they did, this whole apparatus could be deleted."""
    assert len({cls.otm_convention for cls in RING_CODES}) > 1


def test_a_code_that_cannot_do_rings_declares_nothing():
    assert astraLattice.otm_convention == ""


def test_the_base_class_has_no_map():
    assert frameworkLattice.read_one_turn_map(frameworkLattice) is None


# --- MAD-X builds its map rather than reading one -----------------------


class FakeMadx:
    """Only what `madxLattice.read_one_turn_map` touches."""

    def __init__(self, maps):
        self.sector_maps = maps

    read_one_turn_map = madxLattice.read_one_turn_map


def test_madx_with_no_sector_maps_gives_nothing():
    assert FakeMadx([]).read_one_turn_map() is None


def test_madx_composes_in_sequence_order():
    """Two shears that do not commute, so the order is observable."""
    first, second = np.eye(6), np.eye(6)
    first[0, 1] = 2.0
    second[1, 0] = 3.0
    got = FakeMadx([first, second]).read_one_turn_map()
    assert np.allclose(got, second @ first)
    assert not np.allclose(got, first @ second)


def test_madx_identity_maps_compose_to_identity():
    assert np.allclose(FakeMadx([np.eye(6)] * 4).read_one_turn_map(), np.eye(6))
