"""A turn count is a tracking setting, and only three codes can honour it.

A ring's turns are strict repeats -- nothing in the lattice differs between
them -- so the count lives in the ``files:`` block rather than in the layout,
which is the only shape that survives a million of them. The unit is the line,
and a ring spanning several sections is written as one section naming them
(section orders nest), so the line is the whole group.

elegant (``n_passes``), Xsuite (``num_turns``) and Ocelot (``track_nturns``)
can track turns. Everything else tracks a line once, and says so rather than
quietly tracking one turn when a thousand were asked for.
"""

import warnings

import pytest

from simba.Codes.ASTRA.ASTRA import astraLattice
from simba.Codes.Cheetah.Cheetah import cheetahLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.GPT.GPT import gptLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.OPAL.OPAL import opalLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

CAN_TURN = [elegantLattice, xsuiteLattice, ocelotLattice]
CANNOT = [astraLattice, gptLattice, cheetahLattice, opalLattice]


class FakeLine:
    """A lattice stub carrying only what the turn-count code reads."""

    def __init__(self, file_block=None, code="astra", supports=False):
        self.file_block = file_block or {}
        self.code = code
        self.objectname = "RING"
        self.supports_turns = supports

    turns = frameworkLattice.turns
    check_turns_supported = frameworkLattice.check_turns_supported


# --- reading the count --------------------------------------------------


def test_no_setting_means_one_turn():
    assert FakeLine().turns == 1


def test_an_empty_tracking_block_means_one_turn():
    assert FakeLine({"tracking": {}}).turns == 1


def test_a_null_tracking_block_means_one_turn():
    """A key present but empty is how YAML hands over ``tracking:``."""
    assert FakeLine({"tracking": None}).turns == 1


def test_the_count_is_read_from_the_files_block():
    assert FakeLine({"tracking": {"turns": 1000}}).turns == 1000


def test_a_string_count_is_coerced():
    """Every other numeric setting arrives coerced rather than type-checked."""
    assert FakeLine({"tracking": {"turns": "512"}}).turns == 512


# --- which codes can honour it ------------------------------------------


@pytest.mark.parametrize("cls", CAN_TURN, ids=lambda c: c.__name__)
def test_the_three_that_can(cls):
    assert cls.supports_turns is True


@pytest.mark.parametrize("cls", CANNOT, ids=lambda c: c.__name__)
def test_the_ones_that_cannot(cls):
    assert cls.supports_turns is False


def test_the_base_class_assumes_it_cannot():
    """So a backend gains turns by declaring it, never by omission."""
    assert frameworkLattice.supports_turns is False


# --- the warning --------------------------------------------------------


def test_asking_a_single_pass_code_for_turns_warns():
    line = FakeLine({"tracking": {"turns": 1000}}, code="astra", supports=False)
    with pytest.warns(UserWarning, match="tracks a line once"):
        line.check_turns_supported()


def test_the_warning_names_the_code_and_the_count():
    line = FakeLine({"tracking": {"turns": 1000}}, code="astra", supports=False)
    with pytest.warns(UserWarning, match=r"1000 turns.*astra"):
        line.check_turns_supported()


def test_a_capable_code_is_silent():
    line = FakeLine({"tracking": {"turns": 1000}}, code="elegant", supports=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_turns_supported()


def test_one_turn_is_silent_everywhere():
    """The default must never warn, on any code."""
    for code, supports in (("astra", False), ("elegant", True)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FakeLine({}, code=code, supports=supports).check_turns_supported()


def test_it_warns_rather_than_refusing():
    """One settings file driving several codes is ordinary."""
    line = FakeLine({"tracking": {"turns": 5}}, code="gpt", supports=False)
    with pytest.warns(UserWarning):
        line.check_turns_supported()
    assert line.turns == 5


# --- turns only mean something on a closed path -------------------------
#
# A turn count wraps the line onto its own start. Asking a transfer line for a
# thousand turns is not a smaller ring, it is incoherent -- and until this
# check nothing said so. A superperiod is the legitimate exception: one sector
# of an N-fold-symmetric ring is open on its own.

import math

from laura.models.element import Dipole, Drift
from laura.models.elementList import MachineModel


def ring(nbend, angle, turns=1000):
    """A line of `nbend` bends of `angle`, each followed by a 1 m drift."""
    elements, order = {}, []
    for i in range(nbend):
        bend, drift = f"B{i}", f"D{i}"
        elements[bend] = Dipole(
            name=bend,
            hardware_class="Magnet",
            machine_area="A",
            magnetic={"magnetic_length": 1.0, "k0l": angle},
            physical={"length": 1.0},
        )
        elements[drift] = Drift(
            name=drift,
            hardware_class="Drift",
            hardware_type="Drift",
            machine_area="A",
            physical={"length": 1.0},
        )
        order += [bend, drift]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MachineModel(
            elements=elements,
            section={"sections": {"RING": order}},
            layout={"layouts": {"M": ["RING"]}, "default_layout": "M"},
        )
    return ClosureLine(model, order, turns)


class ClosureLine:
    """A stub exposing what `check_turns_closed` reads off real geometry."""

    def __init__(self, model, order, turns):
        self.startObject = model[order[0]]
        self.endObject = model[order[-1]]
        self.elements = {name: model[name] for name in order}
        self.file_block = {"tracking": {"turns": turns}}
        self.objectname = "RING"
        self.code = "elegant"

    turns = frameworkLattice.turns
    net_bend_angle = frameworkLattice.net_bend_angle
    check_turns_closed = frameworkLattice.check_turns_closed


def test_a_closed_ring_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ring(4, math.pi / 2).check_turns_closed()


def test_an_open_line_warns():
    with pytest.warns(UserWarning, match="does not close"):
        ring(4, 0.0).check_turns_closed()


def test_the_warning_reports_the_gap():
    """Four 1 m bends and four 1 m drifts, dead straight: 8 m from home."""
    with pytest.warns(UserWarning, match=r"ends 8 m from where it starts"):
        ring(4, 0.0).check_turns_closed()


def test_one_turn_never_checks_closure():
    """A single pass down an open line is the ordinary case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ring(4, 0.0, turns=1).check_turns_closed()


def test_a_superperiod_is_named_as_such():
    """Half a ring closes after two, and the message should say so."""
    with pytest.warns(UserWarning, match=r"1/2 fraction of a turn"):
        ring(2, math.pi / 2).check_turns_closed()


def test_a_quarter_ring_is_named_as_such():
    with pytest.warns(UserWarning, match=r"1/4 fraction of a turn"):
        ring(1, math.pi / 2).check_turns_closed()


def test_a_straight_line_gets_no_superperiod_hint():
    """No bending at all is not a sector of anything."""
    with pytest.warns(UserWarning) as caught:
        ring(4, 0.0).check_turns_closed()
    assert "superperiod" not in str(caught[0].message)


def test_the_net_bend_of_a_closed_ring_is_a_full_turn():
    assert ring(4, math.pi / 2).net_bend_angle == pytest.approx(2 * math.pi)


def test_the_net_bend_of_a_straight_line_is_zero():
    assert ring(4, 0.0).net_bend_angle == pytest.approx(0.0)


# --- elegant, the one wired so far --------------------------------------


def test_elegant_no_longer_hardcodes_one_pass():
    """All three ``run_control`` sites used to pass the literal 1."""
    import inspect

    source = inspect.getsource(elegantLattice)
    assert "n_passes=1" not in source
    assert source.count("n_passes=self.turns") == 3
