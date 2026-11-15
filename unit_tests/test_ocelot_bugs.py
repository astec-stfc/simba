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


def test_csr_stays_off_an_undulator():
    """A laser-heater chicane: CSR over each half, not the undulator between them,
    where Ocelot would need the reference energy for its Runge-Kutta orbit."""
    from types import SimpleNamespace

    from ocelot import Drift, SBend, Undulator
    from simba.Codes.Ocelot.Ocelot import ocelotLattice

    seq = [SBend(l=0.1, angle=0.03), Drift(l=0.4), SBend(l=0.1, angle=-0.03), Drift(l=0.4),
           Undulator(lperiod=0.05, nperiods=10, Kx=2.7), Drift(l=0.4),
           SBend(l=0.1, angle=-0.03), Drift(l=0.4), SBend(l=0.1, angle=0.03), Drift(l=1)]
    fake = SimpleNamespace(lat_obj=SimpleNamespace(sequence=seq))
    assert ocelotLattice.csr_regions(fake) == [(seq[0], seq[3]), (seq[6], seq[9])]


def test_mbi_keeps_the_bunching_it_tracked(tmp_path):
    """Ocelot tracks a deep copy of the navigator's processes, so the MBI kept on the
    line (and saved to <line>_mbi.dat) must be that copy, not the empty original."""
    import os
    import shutil

    import numpy as np
    import simba.Framework as fw
    from helpers import fodo_machine
    from simba.Codes.Generators import frameworkGenerator

    np.random.seed(1)
    frameworkGenerator(global_parameters={"master_subdir": str(tmp_path)}, filename="seed.openpmd.hdf5",
                       initial_momentum=100e6, sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
                       sigma_z=1e-3, sigma_pz=1e4, charge=100e-12, number_of_particles=20000).write()
    machine, _, section = fodo_machine(tmp_path)
    settings = fw.FrameworkSettings()
    settings.files = {"FODO": {"code": "ocelot", "input": {},
                               "output": {"start_element": "M1", "end_element": "M3"}}}
    settings.layout, settings.section, settings.element_list = machine.layout, section, f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    line = framework.latticeObjects["FODO"]
    line.mbi = {"set_mbi": True, "min": 1e-5, "max": 2e-5, "nstep": 2, "slices": 0}
    shutil.copy(tmp_path / "seed.openpmd.hdf5", os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    bf = framework.latticeObjects["FODO"].mbi_navi.bf
    assert len(bf) == 2 and len(bf[0]) > 1
