"""A ring's Twiss is the lattice's, not the beam's.

simba hands every backend the incoming beam's Twiss -- Xsuite's `_twiss()`
passes `betx`/`alfx`/`bety`/`alfy`, elegant's `twiss_output` passes
`beta_x`/`alpha_x`/... -- which is right for a transfer line and wrong for a
ring, where the answer is the periodic solution the lattice itself determines.
Nothing about the incoming beam can tell you what that is.

`periodic` is the flag that asks for it. Unlike `turns` it is not really a
tracking choice: whether the reference orbit closes is a fact about the
lattice, and LAURA already records it as section `geometry`. So the flag
defaults to that, and the `tracking` block overrides it either way.

elegant (`matched = 1`), Xsuite (`twiss()` with no initial conditions), Ocelot
(`optics.twiss(tws0=None)`) and Bmad (`parameter[geometry] = closed`) can
honour it; the rest say so rather than quietly returning open-solution optics
whose tune is not the ring's.
"""

import warnings

import pytest
from laura.models._generated import LatticeGeometryEnum

from simba.Codes.ASTRA.ASTRA import astraLattice
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Cheetah.Cheetah import cheetahLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.GPT.GPT import gptLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.OPAL.OPAL import opalLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

CAN_MATCH = [elegantLattice, xsuiteLattice, ocelotLattice, bmadLattice]
CANNOT = [astraLattice, gptLattice, cheetahLattice, opalLattice]


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


# --- reading the flag ---------------------------------------------------


def test_no_setting_means_open():
    assert FakeLine().periodic is False


def test_an_empty_tracking_block_means_open():
    assert FakeLine({"tracking": {}}).periodic is False


def test_a_null_tracking_block_means_open():
    """A key present but empty is how YAML hands over `tracking:`."""
    assert FakeLine({"tracking": None}).periodic is False


def test_the_flag_is_read_from_the_files_block():
    assert FakeLine({"tracking": {"periodic": True}}).periodic is True


def test_it_sits_beside_the_turn_count():
    """A ring normally asks for both."""
    line = FakeLine({"tracking": {"turns": 1000, "periodic": True}})
    assert line.periodic is True


# --- the default comes from LAURA's geometry ----------------------------
#
# `geometry` is section metadata in LAURA -- open or closed, mirroring Bmad's
# `parameter[geometry]`. A ring has already said it closes; saying it again in
# the tracking block would be a second source of truth for one fact.


def test_a_closed_section_is_periodic_without_being_asked():
    assert FakeLine(geometry=LatticeGeometryEnum.closed).periodic is True


def test_an_open_section_is_not():
    assert FakeLine(geometry=LatticeGeometryEnum.open).periodic is False


def test_a_section_with_no_geometry_is_not():
    """LAURA leaves it unset unless the layout says, and most lines are lines."""
    assert FakeLine(geometry=None).periodic is False


def test_a_plain_string_geometry_works_too():
    """The layout may hand over the raw value rather than the enum."""
    assert FakeLine(geometry="closed").periodic is True


def test_the_tracking_block_overrides_a_closed_section():
    """Injecting a mismatched beam into a real ring is a legitimate study, and
    its whole point is the open solution."""
    line = FakeLine(
        {"tracking": {"periodic": False}}, geometry=LatticeGeometryEnum.closed
    )
    assert line.periodic is False


def test_the_tracking_block_overrides_an_open_section():
    line = FakeLine({"tracking": {"periodic": True}}, geometry=LatticeGeometryEnum.open)
    assert line.periodic is True


# --- which codes can do it ----------------------------------------------


@pytest.mark.parametrize("cls", CAN_MATCH, ids=lambda c: c.__name__)
def test_the_four_that_can(cls):
    assert cls.supports_periodic is True


@pytest.mark.parametrize("cls", CANNOT, ids=lambda c: c.__name__)
def test_the_ones_that_cannot(cls):
    assert cls.supports_periodic is False


def test_the_base_class_assumes_it_cannot():
    assert frameworkLattice.supports_periodic is False


# --- saying so ----------------------------------------------------------


def test_asking_an_open_solution_code_to_match_warns():
    line = FakeLine({"tracking": {"periodic": True}}, code="astra")
    with pytest.warns(UserWarning, match="periodic solution"):
        line.check_periodic_supported()


def test_the_warning_says_the_numbers_are_not_the_rings():
    """The failure mode is a plausible wrong tune, not a crash, so the
    warning has to say what is wrong with the answer."""
    line = FakeLine({"tracking": {"periodic": True}}, code="astra")
    with pytest.warns(UserWarning, match="not the ring's"):
        line.check_periodic_supported()


def test_a_capable_code_is_silent():
    line = FakeLine({"tracking": {"periodic": True}}, code="elegant", supports=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_periodic_supported()


def test_an_open_line_is_silent_everywhere():
    """The default costs nothing: no flag, no warning, whatever the code."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FakeLine(code="astra").check_periodic_supported()


def test_it_warns_rather_than_refusing():
    """An unmatched run is still a run; the flag is ignored, not fatal."""
    line = FakeLine({"tracking": {"periodic": True}}, code="astra")
    with pytest.warns(UserWarning):
        line.check_periodic_supported()
    assert line.periodic is True
