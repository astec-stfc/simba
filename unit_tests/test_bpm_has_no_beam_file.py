"""A BPM never writes a beam file, and that is not a problem."""

import types

import pytest
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Framework_objects import frameworkLattice


class FakeScreen:
    def __init__(self, name, hardware_type):
        self.name = name
        self.hardware_type = hardware_type


def convert(tmp_path, screen):
    """Run the conversion against an empty directory, so nothing is on disk."""
    lattice = types.SimpleNamespace(
        global_parameters={"master_subdir": str(tmp_path)},
        turns=1,
        elementObjects={},
    )
    elegantLattice.sdds_to_hdf5(lattice, screen)


def test_a_missing_bpm_file_is_not_worth_a_warning(tmp_path, recwarn):
    convert(tmp_path, FakeScreen("BPMQD", "Beam_Position_Monitor"))
    assert [str(w.message) for w in recwarn] == []


@pytest.mark.parametrize(
    "hardware_type", ["Screen", "Marker", "Bunch_Length_Monitor", "Diagnostic"]
)
def test_a_missing_watch_file_still_warns(tmp_path, hardware_type):
    with pytest.warns(UserWarning, match="elegant wrote no beam file"):
        convert(tmp_path, FakeScreen("OTRDMP", hardware_type))


@pytest.mark.parametrize("end_beams_only", [False, True])
def test_astra_and_opal_write_beams_where_the_other_codes_do(end_beams_only):
    """The screens and markers, not the BPMs, nor the start, nor the end (the final beam)."""
    elements = [
        FakeScreen("START", "Marker"), FakeScreen("SCR", "Screen"), FakeScreen("BPM", "Beam_Position_Monitor"),
        FakeScreen("MARK", "Marker"), FakeScreen("END", "Marker"),
    ]
    line = types.SimpleNamespace(
        start="START", end="END", end_name="END", screens_and_markers_and_bpms=elements,
        elements={e.name: e for e in elements}, global_parameters={"end_beams_only": end_beams_only},
    )
    line.writes_output = lambda name: frameworkLattice.writes_output(line, name)
    names = [e.name for e in frameworkLattice.output_elements.fget(line)]
    assert names == ([] if end_beams_only else ["SCR", "MARK"])
