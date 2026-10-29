"""The one-turn map is the cheapest useful thing a ring model produces."""

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


def test_a_periodic_line_reads_its_map():
    line = FakeLine(drift_map())
    line.postProcess()
    assert line.one_turn_map is not None


def test_an_open_line_reads_nothing():
    line = FakeLine(drift_map(), periodic=False)
    line.postProcess()
    assert line.one_turn_map is None


def test_a_code_that_cannot_match_reads_nothing():
    line = FakeLine(drift_map(), supports=False)
    line.postProcess()
    assert line.one_turn_map is None


@pytest.mark.filterwarnings("error")
def test_a_code_with_no_map_to_give_is_not_an_error():
    """elegant writes no matrix file when LSC is on."""
    line = FakeLine(None)
    line.postProcess()
    assert line.one_turn_map is None


# det(R) == 1 holds in every convention the codes use, unlike a full
# symplecticity test, so it can run before anything is normalised.


@pytest.mark.filterwarnings("error")
def test_a_symplectic_map_is_silent():
    FakeLine(drift_map()).postProcess()


def test_a_damped_map_warns():
    """Half the volume: a read that lost a factor somewhere."""
    matrix = drift_map()
    matrix[1, 1] = 0.5
    with pytest.warns(UserWarning, match="determinant"):
        FakeLine(matrix).postProcess()


def test_an_empty_map_warns_what_is_untrustworthy():
    """The failure is a plausible wrong tune, so the warning names what not to believe."""
    with pytest.warns(UserWarning, match="determinant.*tune, beta, momentum compaction"):
        FakeLine(np.zeros((6, 6))).postProcess()


def test_a_wrong_shaped_map_warns_about_its_shape():
    with pytest.warns(UserWarning, match="not 6x6"):
        FakeLine(np.eye(4)).postProcess()


@pytest.mark.filterwarnings("error")
def test_a_transposed_symplectic_map_still_passes():
    """det is necessary, not sufficient, which is why codes are compared to each other."""
    FakeLine(drift_map().T).postProcess()


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
