"""Beam conversion into and out of the plasma codes (Wake-T, FBPIC)."""

import numpy as np
import pytest

from simba.Modules import Beams as rbf
from simba.Modules.units import UnitValue


def _two_bunch_beam():
    """A 500 pC drive bunch and a 50 pC witness, 500 macroparticles each."""
    n = 500
    rng = np.random.default_rng(1)
    beam = rbf.beam()
    p = beam.Particles
    z = np.concatenate([rng.normal(0, 3e-6, n), rng.normal(-80e-6, 1e-6, n)])
    cpz = np.concatenate([np.full(n, 1.0e9), np.full(n, 1.0e8)])
    p.x = UnitValue(rng.normal(0, 2e-6, 2 * n), "m")
    p.y = UnitValue(rng.normal(0, 2e-6, 2 * n), "m")
    p.z = UnitValue(z, "m")
    p.t = UnitValue(-z / 299792458.0, "s")
    beam.set_mass_and_charge(9.1093837015e-31, -1.602176634e-19, 2 * n)
    beam.set_momenta(np.zeros(2 * n), np.zeros(2 * n), cpz)
    p.charge = UnitValue(np.concatenate([np.full(n, -500e-12 / n), np.full(n, -50e-12 / n)]), "C")
    p.total_charge = UnitValue(-550e-12, "C")
    p.status = UnitValue(np.full(2 * n, 5))
    return beam, n


def test_wake_t_bunch_keeps_electron_charge_and_round_trips():
    pytest.importorskip("wake_t")
    from simba.Modules.Beams.wake_t import beam_to_particle_bunch, particle_bunch_to_beam

    beam, n = _two_bunch_beam()
    bunch = beam_to_particle_bunch(beam, zstart=0.0)
    assert (bunch.w > 0).all()
    assert bunch.q_species < 0
    assert bunch.q[:n].sum() == pytest.approx(-500e-12)
    assert bunch.q[n:].sum() == pytest.approx(-50e-12)
    np.testing.assert_allclose(bunch.xi, beam.z.val)

    out = rbf.beam()
    particle_bunch_to_beam(out, bunch, zpos=1.0)
    assert out.Particles.total_charge.val == pytest.approx(-550e-12)
    np.testing.assert_allclose(out.cp.val, beam.cp.val, rtol=1e-12)
    # Arrival time: the head (larger z) arrives first.
    assert np.corrcoef(out.z.val, out.t.val)[0, 1] < -0.99


def test_fbpic_bunch_keeps_per_macroparticle_charge():
    pytest.importorskip("fbpic")
    from fbpic.main import Simulation
    from simba.Modules.Beams.fbpic import beam_to_particles

    beam, n = _two_bunch_beam()
    sim = Simulation(32, 20e-6, 16, 50e-6, 1, 1e-15, zmin=-120e-6, use_cuda=False, verbose_level=0)
    species = beam_to_particles(beam, simulation=sim, zstart=0.0)
    charge = species.q * species.w
    assert charge.sum() == pytest.approx(-550e-12, rel=1e-6)
    drive = species.z > -40e-6
    assert charge[drive].sum() == pytest.approx(-500e-12, rel=1e-6)


def test_fbpic_output_merges_bunch_with_trapped_injected_electrons():
    pytest.importorskip("fbpic")
    from fbpic.main import Simulation
    from scipy.constants import e, m_e
    from simba.Modules.Beams.fbpic import beam_to_particles, particles_to_beam

    beam, n = _two_bunch_beam()
    sim = Simulation(32, 20e-6, 16, 50e-6, 1, 1e-15, zmin=-120e-6, use_cuda=False, verbose_level=0)
    bunch = beam_to_particles(beam, simulation=sim, zstart=0.0)
    injected = sim.add_new_species(q=-e, m=m_e, n=1e20, p_nz=1, p_nr=1, p_nt=4,
                                   p_zmin=-10e-6, p_zmax=0, p_rmax=5e-6)
    injected.uz[: injected.Ntot // 2] = 10.0     # "trapped"
    out = rbf.beam()
    particles_to_beam(out, [bunch, injected], uz_min=[None, 2.0])
    trapped_charge = -e * injected.w[: injected.Ntot // 2].sum()
    assert len(out.x.val) == 2 * n + injected.Ntot // 2
    assert out.Particles.total_charge.val == pytest.approx(-550e-12 + trapped_charge, rel=1e-6)
