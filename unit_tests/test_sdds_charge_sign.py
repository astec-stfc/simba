"""elegant stores the bunch charge in an SDDS file as a positive number; an
electron beam read from one must come back with negative charge."""

import numpy as np

from simba.Modules import Beams as rbf
from simba.Modules.Beams.ocelot import particle_array_to_beam
from simba.Modules.SDDSFile import SDDSFile
from ocelot.cpbd.beam import ParticleArray


def electron_beam(n=1000, q=-250e-12):
    parray = ParticleArray(n)
    parray.E = 0.1
    parray.rparticles[:] = np.random.default_rng(1).normal(0, 1e-4, (6, n))
    beam = rbf.beam()
    particle_array_to_beam(beam, parray)
    beam._beam.set_total_charge(q)
    return beam


def test_an_sdds_beam_reads_back_with_the_electrons_negative_charge(tmp_path):
    beam = electron_beam()
    path = str(tmp_path / "beam.sdds")
    rbf.sdds.write_SDDS_file(beam, path)
    sdds = SDDSFile(index=1)
    sdds.read_file(path)
    assert sdds.data["Charge"][0] > 0  # as elegant writes it

    back = rbf.beam()
    rbf.sdds.read_SDDS_beam_file(back, path)
    assert np.isclose(float(back._beam.total_charge), -250e-12, rtol=1e-12, atol=0)
    assert np.all(np.asarray(back._beam.charge, dtype=float) < 0)
    assert np.allclose(np.asarray(back._beam.nmacro, dtype=float), 250e-12 / 1.602176634e-19 / len(back.x))
