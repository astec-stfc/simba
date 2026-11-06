"""Bmad's per-element CSR and space-charge methods; 3D space charge takes precedence over LSC."""

from types import SimpleNamespace as NS

import pytest
from simba.Codes.Bmad.Bmad import bmadLattice


@pytest.mark.parametrize(
    "csr, lsc, mode, wanted",
    [
        (False, False, "", []),
        (True, False, "", ["*[csr_method] = 1_dim"]),
        (False, True, "", ["*[space_charge_method] = slice"]),
        (False, True, "3d", ["*[space_charge_method] = fft_3d"]),
        (True, True, "3d", ["*[csr_method] = 1_dim", "*[space_charge_method] = fft_3d"]),
    ],
)
def test_collective_methods(csr, lsc, mode, wanted):
    fake = NS(csr_enable=csr, lsc_enable=lsc, space_charge_mode=mode, collective_step=0.1)
    lines = bmadLattice._collective_methods(fake, "").splitlines()
    assert [line for line in lines if line.startswith("*[")] == wanted
    assert ("bmad_com[csr_and_space_charge_on] = T" in lines) == bool(wanted)
