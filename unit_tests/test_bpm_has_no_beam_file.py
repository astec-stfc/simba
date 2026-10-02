"""A BPM never writes an elegant beam file, and that is not a problem.

LAURA writes a `Beam_Position_Monitor` as elegant's `moni`, which puts the
centroid in the .cen file and never dumps particles. The conversion pass walks
every screen, marker and BPM looking for a .SDDS to turn into HDF5, so on a
line like the LCLS dump -- 57 BPMs in one stage -- it used to print 57 warnings
about a file that was never going to exist. A `watch` with no file is still
worth hearing about.
"""

import types

import pytest
from simba.Codes.Elegant.Elegant import elegantLattice


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
