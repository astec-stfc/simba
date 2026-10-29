import numpy as np
from .. import constants
from ..units import UnitValue
from torch import tensor, ones, get_default_device, float64, as_tensor


def read_cheetah_beam_file(self, filename, beam_energy, zstart=0, s=0, ref_index=None):
    from cheetah import ParticleBeam
    self.filename = filename
    self.code = "Cheetah"
    self._beam.particle_rest_energy_eV = self.E0_eV

    parray = ParticleBeam.from_openpmd_file(
        filename,
        energy=beam_energy,
        dtype=float64,
    )
    interpret_cheetah_ParticleBeam(self, parray, zstart=zstart, s=s, ref_index=ref_index)


def interpret_cheetah_ParticleBeam(self, parray, zstart=0, s=0, ref_index=None):
    self.set_mass_and_charge(constants.m_e, parray.particle_charges.numpy(), len(parray.x.numpy()))
    self._beam.x = UnitValue(parray.x.numpy(), "m")
    self._beam.y = UnitValue(parray.y.numpy(), "m")
    self._beam.t = UnitValue((parray.s.numpy() + parray.tau.numpy()) / constants.speed_of_light, "s")
    cp = np.sqrt(parray.energies.numpy() ** 2 - self.E0_eV**2)
    p0c = np.sqrt(float(parray.energy) ** 2 - self.E0_eV**2)
    cpx = parray.px.numpy() * p0c
    cpy = parray.py.numpy() * p0c
    cpz = np.sqrt(cp**2 - cpx**2 - cpy**2)
    self.set_momenta(cpx, cpy, cpz)
    self._beam.set_total_charge(UnitValue(-1 * abs(np.sum(parray.particle_charges.numpy())), "C"))
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1))
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))

    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")


def write_cheetah_beam_file(self, filename=None, write=True, energy=None, t0=None):
    """Save an openpmd file for cheetah.

    ``energy`` (eV) and ``t0`` (s) are the reference energy and the time tau
    is measured from; the beam's own means if not given.
    """
    # {x, xp, y, yp, t, p, particleID}
    from cheetah import ParticleBeam
    from cheetah.particles.species import Species
    E = self.energy.mean().val if energy is None else energy
    if t0 is None:
        t0 = np.mean(self.t.val)
    p0c = np.sqrt(E**2 - self.E0_eV**2)
    x = self.x.val
    y = self.y.val
    # p_x / p0c, not the slope p_x / p_z; see interpret_cheetah_ParticleBeam
    xp = self.cpx.val / p0c
    yp = self.cpy.val / p0c
    p = (self.energy.val - E) / p0c
    tau = (self.t.val - t0) * constants.speed_of_light
    s = self.s if self.s is not None else 0.0

    rparticles = np.array([x, xp, y, yp, tau, p])
    num_particles = len(x)
    particles = ones((num_particles, 7), dtype=float64)
    particles[:, :6] = tensor(rparticles.transpose(), dtype=float64)
    q_array = np.array([np.abs(float(self.Q.val / len(x))) for _ in x])
    particle_charges = tensor(q_array, dtype=float64)
    particle_beam = ParticleBeam(
        particles=particles,
        energy=as_tensor(E, dtype=float64),
        particle_charges=particle_charges,
        species=Species(self.species),
        s=as_tensor(s, dtype=float64),
        device=get_default_device(),
        dtype=float64,
    )
    if write:
        if filename is None:
            if "cheetah" not in self.filename:
                filename = self.filename.replace(".hdf5", ".cheetah.hdf5")
            else:
                filename = self.filename
        particle_beam.save_as_openpmd_h5(filename)
    return particle_beam
