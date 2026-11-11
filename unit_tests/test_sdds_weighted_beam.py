"""SDDS beam files carry no per-particle charge: a weighted beam is written
resampled to equal weights, so codes that read it (elegant) see the same beam."""

import numpy as np
from ocelot.cpbd.beam import ParticleArray

from simba.Modules import Beams as rbf
from simba.Modules.Beams.ocelot import particle_array_to_beam
from simba.Modules.units import UnitValue


def weighted_beam(n=4000):
    """Half the particles at +1% momentum carrying three times the charge of the half at -1%."""
    parray = ParticleArray(n)
    parray.E = 0.1
    parray.rparticles[:] = 0.0
    parray.rparticles[5, : n // 2] = 0.01
    parray.rparticles[5, n // 2:] = -0.01
    beam = rbf.beam()
    particle_array_to_beam(beam, parray)
    q = -np.where(np.arange(n) < n // 2, 3.0, 1.0) * 1e-15
    beam._beam.charge = UnitValue(q, units="C")
    beam._beam.total_charge = UnitValue(float(q.sum()), units="C")
    return beam


def test_a_weighted_beam_keeps_its_weighted_momentum_in_sdds(tmp_path):
    beam = weighted_beam()
    q = np.abs(np.asarray(beam._beam.charge, dtype=float))
    p = np.asarray(beam.cp, dtype=float)
    path = str(tmp_path / "beam.sdds")
    rbf.sdds.write_SDDS_file(beam, path)
    back = rbf.beam()
    rbf.sdds.read_SDDS_beam_file(back, path)
    assert len(back.x) == len(beam.x)
    weighted = np.average(p, weights=q)
    assert abs(np.mean(np.asarray(back.cp, dtype=float)) - weighted) < 0.05 * abs(weighted - p.mean())
    assert float(back._beam.total_charge) == float(beam._beam.total_charge)
