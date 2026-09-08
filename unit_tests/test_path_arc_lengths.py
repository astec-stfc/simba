"""Where a line starts along the beam path, and who a change reaches.

A line's own ``getSValues`` restarts at zero, and two passes of one section
restart at the *same* zero. Accumulating line lengths is therefore right only
while the lines happen to tile the path in dictionary order.
``MachineLayout.arc_lengths`` is the authority instead: it walks the layout in
beam order with a running offset and gives one entry per pass.

Also here: modifying a multipass element reaches every pass, because it is one
device. That is usually what the caller means, so it warns rather than refuses.
"""

import warnings

import pytest

import simba.Framework as sfw
from laura import LAURA
from laura.models.element import Quadrupole
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


# --- path_arc_lengths ---------------------------------------------------


def test_it_names_elements_the_way_a_line_does(tmp_path):
    """`arc_lengths` keys address a pass (`#N`); a line uses `.N`."""
    offsets = framework(ERL, tmp_path).path_arc_lengths()
    assert "LIN_Q.1" in offsets and "LIN_Q.2" in offsets
    assert not any("#" in name for name in offsets)


def test_the_two_passes_are_at_different_path_positions(tmp_path):
    """The whole point: one magnet, two arc lengths along the path."""
    offsets = framework(ERL, tmp_path).path_arc_lengths()
    assert offsets["LIN_Q.2"] > offsets["LIN_Q.1"]


def test_a_single_pass_path_needs_no_qualifying(tmp_path):
    offsets = framework(["INJ", "LINAC", "ARC", "DUMP"], tmp_path).path_arc_lengths()
    assert set(offsets) == {"INJ_Q", "LIN_Q", "ARC_Q", "DMP_Q"}


def test_no_machine_gives_an_empty_map(tmp_path):
    """The signal to fall back to accumulating line lengths.

    A Framework's ``machine`` defaults to ``None`` and several tests build one
    that way, so every layout-aware path has to tolerate its absence.
    """
    assert sfw.Framework(directory=str(tmp_path)).path_arc_lengths() == {}


# --- the shared-device warning ------------------------------------------


def test_modifying_a_multipass_element_warns(tmp_path):
    fw = framework(ERL, tmp_path)
    with pytest.warns(UserWarning, match="applies to every pass"):
        fw.modifyElement("LIN_Q", "machine_area", "B")


def test_the_warning_names_the_passes(tmp_path):
    fw = framework(ERL, tmp_path)
    with pytest.warns(UserWarning, match=r"LIN_Q#1, LIN_Q#2"):
        fw.modifyElement("LIN_Q", "machine_area", "B")


def test_it_warns_but_still_applies_the_change(tmp_path):
    """It warns rather than refusing -- one power supply is usually the point."""
    fw = framework(ERL, tmp_path)
    with pytest.warns(UserWarning):
        fw.modifyElement("LIN_Q", "machine_area", "B")
    assert fw.elementObjects["LIN_Q"].machine_area == "B"


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


# --- the rigidity a field is derived with -------------------------------
#
# GPT takes a field, not a normalised strength, and simba hands it
# `Brho * k1`. The `k` was resolved at the momentum the pass states, so if the
# tracked beam's rigidity differs the field is wrong by exactly the ratio --
# on every magnet in the line, and silently.


class FakeLine:
    """A lattice stub carrying only what `check_pass_rigidity` reads."""

    def __init__(self, machine, start, name="LINAC_2"):
        self.machine = machine
        self.start = start
        self.objectname = name

    check_pass_rigidity = frameworkLattice.check_pass_rigidity


def line(layout, start):
    return FakeLine(machine(layout), start)


def test_a_matching_rigidity_is_silent():
    from laura.models.magnetic import brho

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P2))


def test_a_mismatched_rigidity_warns():
    from laura.models.magnetic import brho

    with pytest.warns(UserWarning, match="states a momentum of"):
        line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P1))


def test_the_warning_reports_both_rigidities():
    """Both numbers, so the reader can see the factor for themselves."""
    from laura.models.magnetic import brho

    with pytest.warns(UserWarning) as caught:
        line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P1))
    message = str(caught[0].message)
    assert f"{brho(P1):.4f}" in message and f"{brho(P2):.4f}" in message


def test_a_small_drift_is_tolerated():
    """A real beam never sits exactly on its design momentum."""
    from laura.models.magnetic import brho

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line(ERL, "LIN_Q#2").check_pass_rigidity(brho(P2) * 1.005)


def test_a_pass_stating_no_momentum_is_silent():
    layout = [
        "INJ",
        {"LINAC": {"multipass": 1}},
        "ARC",
        {"LINAC": {"multipass": 2}},
        "DUMP",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line(layout, "LIN_Q#2").check_pass_rigidity(1.0)


def test_an_unqualified_start_is_silent():
    """A single-pass line has no pass momentum to disagree with."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line(ERL, "ARC_Q").check_pass_rigidity(1.0)


def test_a_zero_rigidity_is_silent():
    """Before a beam is loaded there is nothing to compare."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line(ERL, "LIN_Q#2").check_pass_rigidity(0.0)


def test_no_machine_never_warns(tmp_path):
    fw = sfw.Framework(directory=str(tmp_path))
    fw.elementObjects = {
        "Q1": Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fw.modifyElement("Q1", "machine_area", "B")
