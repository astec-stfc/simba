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
    # from just after the cavity before (or the start), for the straight approach
    assert ocelotLattice.csr_regions(fake) == [(seq[1], seq[6]), (seq[8], seq[11])]
    assert ocelotLattice.csr_regions(SimpleNamespace(lat_obj=SimpleNamespace(sequence=seq[1:])))[0][0] is seq[1]


def test_csr_gets_the_design_energy_where_it_starts_and_its_steps():
    """Ocelot's CSR refuses an undulator in its region without the energy."""
    from types import SimpleNamespace

    from ocelot import Cavity, Drift, SBend
    from simba.Codes.Ocelot.Ocelot import ocelotLattice

    seq = [Cavity(l=1, v=0.01), Drift(l=1), SBend(l=0.5, angle=0.1), Drift(l=1),
           Cavity(l=1, v=0.02, phi=60), SBend(l=0.5, angle=0.1), Drift(l=1)]
    fake = SimpleNamespace(lat_obj=SimpleNamespace(sequence=seq), file_block={}, pin=SimpleNamespace(E=0.13),
                           nbin_csr=400, csr_bins=100, mbin_csr=5, sigmamin_csr=1e-5,
                           csr_traj_step=0.0005, csr_apply_step=0.001)
    fake.csr_regions = lambda: ocelotLattice.csr_regions(fake)
    csr, _, _ = ocelotLattice.physproc_csr(fake)
    assert [c.energy for c in csr] == pytest.approx([0.14, 0.15])
    assert {(c.traj_step, c.apply_step) for c in csr} == {(0.0005, 0.001)}
