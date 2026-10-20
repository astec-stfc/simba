from types import SimpleNamespace as NS

from simba.Codes.CSRTrack.CSRTrack import csrtrackLattice


def test_csr_is_off_unless_asked_for():
    """CSRTrack rejects a type-less forces block; with none it tracks without self force."""
    fake = NS(file_block={}, csrtrack_headers={"forces": object()})
    csrtrackLattice.setCSRMode(fake)
    assert "forces" not in fake.csrtrack_headers


def test_csr_mode_sets_the_force_type():
    fake = NS(file_block={"csr": {"csr_mode": "1D"}}, csrtrack_headers={})
    csrtrackLattice.setCSRMode(fake)
    assert "type=projected" in fake.csrtrack_headers["forces"].write_csrtrack()
