import numpy as np
from .. import constants
from ..units import UnitValue
from ocelot.cpbd.beam import ParticleArray


def particle_array_to_beam(self, parray, zstart=0, s=0, ref_index=None, t_reference=None):
    self.set_mass_and_charge(constants.m_e, parray.q_array, len(parray.x()))
    self._beam.x = UnitValue(parray.x(), units="m")
    self._beam.y = UnitValue(parray.y(), units="m")
    if t_reference is None:
        t_reference = (zstart + parray.s) / constants.speed_of_light
    self._beam.t = UnitValue(
        t_reference + parray.tau() / constants.speed_of_light, units="s"
    )
    cp = np.sqrt((parray.energies * 1e9) ** 2 - self.E0_eV**2)
    p0c = np.sqrt((parray.E * 1e9) ** 2 - self.E0_eV**2)
    cpx = parray.px() * p0c
    cpy = parray.py() * p0c
    cpz = np.sqrt(cp**2 - cpx**2 - cpy**2)
    self.set_momenta(cpx, cpy, cpz)
    self._beam.set_total_charge(-1 * abs(np.sum(parray.q_array)))
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1))
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))

    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")


def read_ocelot_beam_file(self, filename):
    from ocelot.cpbd.io import load_particle_array
    self.filename = filename
    self.code = "OCELOT"
    self._beam.particle_rest_energy_eV = self.E0_eV
    parray = load_particle_array(filename)
    particle_array_to_beam(self, parray)


def write_ocelot_beam_file(self, filename, write=True):
    """Save an npz file for ocelot."""
    from ocelot.cpbd.io import save_particle_array
    parray = particle_group_to_parray(self)
    if write:
        save_particle_array(filename, parray)
    return parray


def particle_group_to_parray(self, s_start=0, energy=None, t0=None) -> ParticleArray:
    """Construct an Ocelot ParticleArray from an openPMD-beamphysics ParticleGroup.
    The particle type is assumed to be electrons.

    :param pgroup: ParticleGroup from which to construct the ParticleArray
    :param s_start: deprecated
    :param energy: reference energy in eV; the beam's mean if not given
    :param t0: time tau is measured from, in s; the beam's mean if not given.
        A lattice passes the incoming beam's, so a sampled beam keeps the full
        beam's reference
    :return: ParticleArray corresponding to the provided ParticleGroup
    :rtype: ParticleArray

    """
    E = (self.energy.mean().val if energy is None else energy) * 1e-9
    if t0 is None:
        t0 = np.mean(self.t.val)
    p0c = np.sqrt(E**2 - (self.E0_eV * 1e-9) ** 2)
    x = self.x.val
    y = self.y.val
    # p_x / p0, not the slope p_x / p_z; see particle_array_to_beam
    xp = self.cpx.val / (p0c * 1e9)
    yp = self.cpy.val / (p0c * 1e9)
    p = ((self.energy.val * 1e-9) - E) / p0c
    tau = (self.t.val - t0) * constants.speed_of_light
    s = t0 * constants.speed_of_light

    rparticles = np.array([x, xp, y, yp, tau, p])
    q_array = np.array([np.abs(float(self.Q.val / len(x))) for _ in x])
    parray = ParticleArray(n=len(x))
    parray.rparticles = rparticles
    parray.q_array = q_array
    parray.E = E
    parray.s = s
    return parray
