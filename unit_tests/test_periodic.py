"""A ring's Twiss is the lattice's, not the beam's."""

import pytest
from laura.models._generated import LatticeGeometryEnum

from simba.Framework_objects import frameworkLattice


class FakeLine:
    """A lattice stub carrying only what the periodic-flag code reads."""

    def __init__(self, file_block=None, code="astra", supports=False, geometry=None):
        self.file_block = file_block or {}
        self.code = code
        self.objectname = "RING"
        self.supports_periodic = supports
        self._geometry = geometry

    def _machine_geometry(self):
        return self._geometry

    periodic = frameworkLattice.periodic
    closed_geometry = frameworkLattice.closed_geometry
    check_periodic_supported = frameworkLattice.check_periodic_supported
    codes_that_can = frameworkLattice.codes_that_can


def test_no_setting_means_open():
    assert FakeLine().periodic is False


def test_an_empty_tracking_block_means_open():
    assert FakeLine({"tracking": {}}).periodic is False


def test_a_null_tracking_block_means_open():
    """How YAML hands over a bare ``tracking:``."""
    assert FakeLine({"tracking": None}).periodic is False


def test_the_flag_is_read_from_the_files_block():
    assert FakeLine({"tracking": {"periodic": True}}).periodic is True


def test_it_sits_beside_the_turn_count():
    line = FakeLine({"tracking": {"turns": 1000, "periodic": True}})
    assert line.periodic is True


# The default comes from LAURA's section `geometry` (Bmad's parameter[geometry]);
# repeating it in the tracking block would be a second source of truth.


def test_a_closed_section_is_periodic_without_being_asked():
    assert FakeLine(geometry=LatticeGeometryEnum.closed).periodic is True


def test_an_open_section_is_not():
    assert FakeLine(geometry=LatticeGeometryEnum.open).periodic is False


def test_a_section_with_no_geometry_is_not():
    assert FakeLine(geometry=None).periodic is False


def test_a_plain_string_geometry_works_too():
    assert FakeLine(geometry="closed").periodic is True


def test_the_tracking_block_overrides_a_closed_section():
    """A mismatched beam in a real ring is a legitimate study of the open solution."""
    line = FakeLine(
        {"tracking": {"periodic": False}}, geometry=LatticeGeometryEnum.closed
    )
    assert line.periodic is False


def test_the_tracking_block_overrides_an_open_section():
    line = FakeLine({"tracking": {"periodic": True}}, geometry=LatticeGeometryEnum.open)
    assert line.periodic is True


def test_asking_an_open_solution_code_to_match_warns_rather_than_refusing():
    """The failure is a plausible wrong tune, so the warning says the numbers
    are not the ring's."""
    line = FakeLine({"tracking": {"periodic": True}}, code="astra")
    with pytest.warns(UserWarning, match="periodic solution.*not the ring's"):
        line.check_periodic_supported()
    assert line.periodic is True


@pytest.mark.filterwarnings("error")
def test_a_capable_code_is_silent():
    line = FakeLine({"tracking": {"periodic": True}}, code="elegant", supports=True)
    line.check_periodic_supported()


@pytest.mark.filterwarnings("error")
def test_an_open_line_is_silent_everywhere():
    FakeLine(code="astra").check_periodic_supported()
