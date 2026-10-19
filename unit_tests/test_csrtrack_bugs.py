from types import SimpleNamespace as NS

from simba.Codes.CSRTrack.CSRTrack import csrtrackLattice


def test_default_forces_written_to_headers():
    fake = NS(file_block={}, csrtrack_headers={}, CSRTrackelementObjects={})
    csrtrackLattice.setCSRMode(fake)
    assert "forces" in fake.csrtrack_headers
