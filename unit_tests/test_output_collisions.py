"""Two lines writing the same screen must not overwrite each other's output."""

import pytest

from simba.Framework_objects import OUTPUT_LINE_SEPARATOR, frameworkLattice


class FakeLattice:
    """Just enough of a lattice for the two units under test."""

    def __init__(self, name, screens, end=None, start=None):
        self.objectname = name
        self.screens_and_markers_and_bpms = [Named(s) for s in screens]
        self.start = start
        self.end = end
        self.colliding_outputs = set()

    output_basename = frameworkLattice.output_basename


class Named:
    def __init__(self, name):
        self.name = name


class FakeFramework:
    """Exercises the real collision pass over fake lattices."""

    def __init__(self, lattices):
        self.latticeObjects = lattices

    _line_output_names = None  # bound below


def framework(**lines):
    import simba.Framework as sfw

    fw = FakeFramework(lines)
    fw._line_output_names = sfw.Framework._line_output_names.__get__(fw)
    sfw.Framework._mark_colliding_outputs(fw, list(lines))
    return fw


# --- output_basename in isolation ---------------------------------------


def test_an_uncollided_name_is_unchanged():
    latt = FakeLattice("INJ", ["SCR1"])
    assert latt.output_basename("SCR1") == "SCR1"


def test_a_collided_name_is_qualified():
    latt = FakeLattice("INJ", ["SCR1"])
    latt.colliding_outputs = {"SCR1"}
    assert latt.output_basename("SCR1") == f"INJ{OUTPUT_LINE_SEPARATOR}SCR1"


def test_only_the_collided_name_is_qualified():
    """Per file, not per line -- a line's other outputs keep their names."""
    latt = FakeLattice("INJ", ["SCR1", "SCR2"])
    latt.colliding_outputs = {"SCR1"}
    assert latt.output_basename("SCR2") == "SCR2"


# --- the collision pass -------------------------------------------------


def test_lines_sharing_no_screens_are_untouched():
    """The backwards-compatible case, and it must stay byte-identical."""
    fw = framework(
        INJ=FakeLattice("INJ", ["SCR1"]),
        LINAC=FakeLattice("LINAC", ["SCR2"]),
    )
    assert fw.latticeObjects["INJ"].colliding_outputs == set()
    assert fw.latticeObjects["INJ"].output_basename("SCR1") == "SCR1"


def test_a_shared_screen_is_marked_in_both_lines():
    fw = framework(
        PASS1=FakeLattice("PASS1", ["BPM", "SCR1"]),
        PASS2=FakeLattice("PASS2", ["BPM", "SCR2"]),
    )
    assert fw.latticeObjects["PASS1"].colliding_outputs == {"BPM"}
    assert fw.latticeObjects["PASS2"].colliding_outputs == {"BPM"}


def test_both_occurrences_are_qualified_not_just_the_later():
    """So the name follows from the settings, not from which line ran first."""
    fw = framework(
        PASS1=FakeLattice("PASS1", ["BPM"]),
        PASS2=FakeLattice("PASS2", ["BPM"]),
    )
    assert fw.latticeObjects["PASS1"].output_basename("BPM") == "PASS1-BPM"
    assert fw.latticeObjects["PASS2"].output_basename("BPM") == "PASS2-BPM"


def test_the_two_passes_no_longer_collide():
    """The bug, stated directly: one filename before, two after."""
    fw = framework(
        PASS1=FakeLattice("PASS1", ["BPM"]),
        PASS2=FakeLattice("PASS2", ["BPM"]),
    )
    written = {
        fw.latticeObjects[line].output_basename("BPM") for line in ("PASS1", "PASS2")
    }
    assert len(written) == 2


def test_a_lines_own_unshared_screens_keep_their_names():
    fw = framework(
        PASS1=FakeLattice("PASS1", ["BPM", "SCR1"]),
        PASS2=FakeLattice("PASS2", ["BPM"]),
    )
    assert fw.latticeObjects["PASS1"].output_basename("SCR1") == "SCR1"


def test_the_end_of_a_line_counts_as_an_output():
    """Several codes write the line's final element by name."""
    fw = framework(
        A=FakeLattice("A", ["SCR1"], end="JOIN"),
        B=FakeLattice("B", ["SCR2"], end="JOIN"),
    )
    assert fw.latticeObjects["A"].colliding_outputs == {"JOIN"}
    assert fw.latticeObjects["A"].output_basename("JOIN") == "A-JOIN"


def test_an_end_shared_with_a_screen_collides():
    fw = framework(
        A=FakeLattice("A", ["SCR1"], end="JOIN"),
        B=FakeLattice("B", ["JOIN"]),
    )
    assert fw.latticeObjects["A"].colliding_outputs == {"JOIN"}
    assert fw.latticeObjects["B"].colliding_outputs == {"JOIN"}


def test_three_lines_sharing_one_screen():
    fw = framework(
        **{f"P{n}": FakeLattice(f"P{n}", ["BPM"]) for n in (1, 2, 3)}
    )
    written = {fw.latticeObjects[f"P{n}"].output_basename("BPM") for n in (1, 2, 3)}
    assert written == {"P1-BPM", "P2-BPM", "P3-BPM"}


# --- the handoff, which is the one shared name that must not be qualified ---


def test_the_element_one_line_hands_over_on_keeps_its_plain_name():
    """A ends where B starts, and B goes looking for the plain filename."""
    fw = framework(
        A=FakeLattice("A", ["OTR2"], end="OTR2"),
        B=FakeLattice("B", ["OTR2"], start="OTR2", end="ENDDL1"),
    )
    assert fw.latticeObjects["A"].colliding_outputs == set()
    assert fw.latticeObjects["A"].output_basename("OTR2") == "OTR2"
    assert fw.latticeObjects["B"].output_basename("OTR2") == "OTR2"


def test_a_handoff_name_a_third_line_also_writes_still_collides():
    """Only the two lines either side of it mean the same beam by that name."""
    fw = framework(
        A=FakeLattice("A", ["OTR2"], end="OTR2"),
        B=FakeLattice("B", ["OTR2"], start="OTR2"),
        C=FakeLattice("C", ["OTR2"]),
    )
    # The pair would have been let through on its own; C writing a different
    # beam to the same name is the collision the pass exists to catch, so the
    # exemption lapses and all three qualify.
    for line in ("A", "B", "C"):
        assert fw.latticeObjects[line].colliding_outputs == {"OTR2"}


def test_a_screen_shared_away_from_the_handoff_still_collides():
    fw = framework(
        A=FakeLattice("A", ["BPM", "OTR2"], end="OTR2"),
        B=FakeLattice("B", ["BPM", "OTR2"], start="OTR2"),
    )
    assert fw.latticeObjects["A"].colliding_outputs == {"BPM"}
    assert fw.latticeObjects["B"].output_basename("OTR2") == "OTR2"


def test_the_generator_is_skipped():
    fw = FakeFramework({"INJ": FakeLattice("INJ", ["SCR1"])})
    import simba.Framework as sfw

    fw._line_output_names = sfw.Framework._line_output_names.__get__(fw)
    sfw.Framework._mark_colliding_outputs(fw, ["generator", "INJ"])
    assert fw.latticeObjects["INJ"].colliding_outputs == set()


# --- the default, which is what keeps this backwards compatible ---------


def test_a_lattice_never_marked_qualifies_nothing():
    """A lattice used on its own never goes through Framework.track."""
    latt = FakeLattice("SOLO", ["SCR1"])
    assert latt.colliding_outputs == set()
    assert latt.output_basename("SCR1") == "SCR1"


@pytest.mark.parametrize("name", ["CLA-S01-SCR", "SCR_WITH_UNDERSCORE", "X"])
def test_unqualified_names_pass_through_verbatim(name):
    assert FakeLattice("L", [name]).output_basename(name) == name


# --- a pass selector never reaches a filename ---------------------------
#
# `start_element: CAV_01#2` picks a traversal, and `self.end` is used as an
# output filename by several codes. `#` is not a legal name character in
# elegant or MAD-X, so the selector is converted to the `.N` a flattened
# export writes before it can reach a file.


def test_a_pass_selector_becomes_the_flattened_name():
    latt = FakeLattice("PASS2", ["CAV_01#2"])
    assert latt.output_basename("CAV_01#2") == "CAV_01.2"


def test_no_output_name_can_contain_a_hash():
    latt = FakeLattice("PASS2", ["CAV_01#2"])
    latt.colliding_outputs = {"CAV_01#2"}
    assert "#" not in latt.output_basename("CAV_01#2")


def test_a_selector_and_a_collision_compose():
    latt = FakeLattice("PASS2", ["CAV_01#2"])
    latt.colliding_outputs = {"CAV_01#2"}
    assert latt.output_basename("CAV_01#2") == "PASS2-CAV_01.2"


def test_a_repeat_index_keeps_the_export_spelling():
    """``DRIFT.2#1`` is drift 2 on pass 1, which export writes ``DRIFT.1.2``."""
    latt = FakeLattice("PASS1", ["DRIFT.2#1"])
    assert latt.output_basename("DRIFT.2#1") == "DRIFT.1.2"
