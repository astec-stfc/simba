from types import SimpleNamespace as NS

import pytest

pytest.importorskip("ocelot")
from simba.Codes.Ocelot.mbi import MBI


def test_csr_impedance_negative_bend():
    pos = MBI.csrimpedance(None, 1e-5, NS(l=0.2, angle=0.1))
    assert pos != 0 and MBI.csrimpedance(None, 1e-5, NS(l=0.2, angle=-0.1)) == pos


def test_csr_stays_off_the_cavities():
    """A physics process slices what it spans, and a sliced cavity changes the chirp."""
    from types import SimpleNamespace

    from ocelot import Cavity, Drift, Quadrupole, SBend
    from simba.Codes.Ocelot.Ocelot import ocelotLattice

    seq = [Cavity(l=1, v=0.01), Drift(l=1), SBend(l=0.5, angle=0.1), Drift(l=1),
           SBend(l=0.5, angle=-0.1), Quadrupole(l=0.1), Drift(l=1), Cavity(l=1, v=0.01),
           SBend(l=0.5, angle=0.0), Drift(l=1), SBend(l=0.5, angle=0.05), Drift(l=1)]
    fake = SimpleNamespace(lat_obj=SimpleNamespace(sequence=seq))
    assert ocelotLattice.csr_regions(fake) == [(seq[2], seq[6]), (seq[10], seq[11])]
