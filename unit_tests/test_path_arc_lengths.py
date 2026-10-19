"""Where a line starts along the beam path, and who a change reaches."""

import warnings

import pytest

import simba.Framework as sfw
from laura import LAURA
from laura.models.element import Quadrupole
from laura.models.magnetic import brho
from simba.Framework_objects import frameworkLattice

P1, P2 = 100e6, 200e6

SECTIONS = {"INJ": ["INJ_Q"], "LINAC": ["LIN_Q"], "ARC": ["ARC_Q"], "DUMP": ["DMP_Q"]}

ERL = [
    "INJ",
    {"LINAC": {"multipass": 1, "momentum": P1}},
    "ARC",
    {"LINAC": {"multipass": 2, "momentum": P2}},
    "DUMP",
]


def machine(layout):
    elements = [
        Quadrupole(
            name=name,
            hardware_class="Magnet",
            machine_area="A",
            magnetic={"magnetic_length": 0.4, "k1l": 1.0},
            physical={"length": 0.4},
        )
        for name in ("INJ_Q", "LIN_Q", "ARC_Q", "DMP_Q")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LAURA(
            element_list=elements,
            section={"sections": SECTIONS},
            layout={"layouts": {"ERL": layout}, "default_layout": "ERL"},
        )


def framework(layout, tmp_path):
    fw = sfw.Framework(directory=str(tmp_path))
    fw.machine = machine(layout)
    fw.elementObjects = dict(fw.machine.elements)
    return fw


def test_it_names_elements_the_way_a_line_does(tmp_path):
    """`arc_lengths` keys address a pass (`#N`); a line uses `.N`."""
    offsets = framework(ERL, tmp_path).path_arc_lengths()
    assert "LIN_Q.1" in offsets and "LIN_Q.2" in offsets
    assert not any("#" in name for name in offsets)


def test_the_two_passes_are_at_different_path_positions(tmp_path):
    offsets = framework(ERL, tmp_path).path_arc_lengths()
    assert offsets["LIN_Q.2"] > offsets["LIN_Q.1"]


def test_a_single_pass_path_needs_no_qualifying(tmp_path):
    offsets = framework(["INJ", "LINAC", "ARC", "DUMP"], tmp_path).path_arc_lengths()
    assert set(offsets) == {"INJ_Q", "LIN_Q", "ARC_Q", "DMP_Q"}


def test_no_machine_gives_an_empty_map(tmp_path):
    """The signal to fall back to accumulating line lengths."""
    assert sfw.Framework(directory=str(tmp_path)).path_arc_lengths() == {}


def test_modifying_a_multipass_element_warns_with_its_passes_but_applies(tmp_path):
    """Warns rather than refusing: one power supply is usually the point."""
    fw = framework(ERL, tmp_path)
    with pytest.warns(UserWarning, match="LIN_Q#1, LIN_Q#2.*applies to every pass"):
        fw.modifyElement("LIN_Q", "machine_area", "B")
    assert fw.elementObjects["LIN_Q"].machine_area == "B"


# Building a Framework warns about simcodes, so these filter only the change.


def test_an_element_entered_once_does_not_warn(tmp_path):
    fw = framework(ERL, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fw.modifyElement("ARC_Q", "machine_area", "B")


def test_a_single_pass_machine_never_warns(tmp_path):
    fw = framework(["INJ", "LINAC", "ARC", "DUMP"], tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fw.modifyElement("LIN_Q", "machine_area", "B")


def test_no_machine_never_warns(tmp_path):
    fw = sfw.Framework(directory=str(tmp_path))
    fw.elementObjects = {
        "Q1": Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fw.modifyElement("Q1", "machine_area", "B")


# GPT takes `Brho * k1`, with `k` resolved at the pass's stated momentum, so a
# different tracked rigidity makes every field wrong by that ratio, silently.


class FakeLine:
    """A lattice stub carrying only what `check_pass_rigidity` reads."""

    def __init__(self, machine, start, name="LINAC_2"):
        self.machine = machine
        self.start = start
        self.objectname = name

    check_pass_rigidity = frameworkLattice.check_pass_rigidity


def line(layout, start):
    return FakeLine(machine(layout), start)


def test_a_mismatched_rigidity_warns_with_both_numbers():
    with pytest.warns(UserWarning, match="states a momentum of") as caught:
        line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P1))
    message = str(caught[0].message)
    assert f"{brho(P1):.4f}" in message and f"{brho(P2):.4f}" in message


@pytest.mark.filterwarnings("error")
def test_a_matching_rigidity_is_silent():
    line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P2))


@pytest.mark.filterwarnings("error")
def test_a_small_drift_is_tolerated():
    """A real beam never sits exactly on its design momentum."""
    line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P2) * 1.005)


@pytest.mark.filterwarnings("error")
def test_a_pass_stating_no_momentum_is_silent():
    layout = [
        "INJ",
        {"LINAC": {"multipass": 1}},
        "ARC",
        {"LINAC": {"multipass": 2}},
        "DUMP",
    ]
    line(layout, "LIN_Q#2").check_pass_rigidity(1.0)


@pytest.mark.filterwarnings("error")
def test_an_unqualified_start_is_silent():
    """A single-pass line has no pass momentum to disagree with."""
    line(ERL, "ARC_Q").check_pass_rigidity(1.0)


@pytest.mark.filterwarnings("error")
def test_a_zero_rigidity_is_silent():
    """Before a beam is loaded there is nothing to compare."""
    line(ERL, "LIN_Q#2").check_pass_rigidity(0.0)
