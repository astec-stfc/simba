import numpy as np
from .. import constants
from ..units import UnitValue
from torch import tensor, ones, get_default_device, float64, as_tensor

def particle_beam_to_beam(self, parray, s_start=0):
    self._beam.particle_rest_energy_eV = self.E0_eV
    self._beam.particle_mass = UnitValue(np.full(len(parray.x.numpy()), constants.m_e), "kg")
    self._beam.particle_rest_energy = UnitValue(
        (
                self._beam.particle_mass * constants.speed_of_light ** 2
        ),
        units="J",
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        (
                self._beam.particle_rest_energy / constants.elementary_charge
        ),
        units="eV/c",
    )
    self._beam.particle_charge = UnitValue(parray.particle_charges.numpy(), "C")
    # self._beam.gamma = UnitValue(parray.relativistic_gamma.numpy(), "")
    self._beam.x = UnitValue(parray.x.numpy(), "m")
    self._beam.y = UnitValue(parray.y.numpy(), "m")
    self._beam.t = UnitValue((parray.tau.numpy()) / constants.speed_of_light, "s")
    # self._beam["p"] = parray.energies.numpy()
    self._beam.px = UnitValue(parray.px.numpy() * parray.energies.numpy() * self.q_over_c, "kg*m/s")
    self._beam.py = UnitValue(parray.py.numpy() * parray.energies.numpy() * self.q_over_c, "kg*m/s")
    cp = parray.energies.numpy()
    self._beam.pz = UnitValue((
            self.q_over_c * cp / np.sqrt(parray.px.numpy() ** 2 + parray.py.numpy() ** 2 + 1)
    ), "kg*m/s")
    self._beam.total_charge = UnitValue(-1 * abs(np.sum(parray.particle_charges.numpy())), "C")
    self._beam.z = UnitValue(s_start + (1 * self._beam.Bz * constants.speed_of_light) * (
            self._beam.t - np.mean(self._beam.t)
    ), "m")  # np.full(len(self.t), 0)
    self._beam.charge = UnitValue(np.full(
        len(self._beam.x), self._beam.total_charge / len(self._beam.x)
    ), "C")
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1), "")
    self.species = parray.species.name

def read_cheetah_beam_file(self, filename, beam_energy):
    from cheetah import ParticleBeam
    self.filename = filename
    self.code = "Cheetah"

    parray = ParticleBeam.from_openpmd_file(
        filename,
        energy=beam_energy,
        dtype=float64,
    )
    particle_beam_to_beam(self, parray)


def write_cheetah_beam_file(self, filename=None, write=True, s_start=0):
    """Save an openpmd file for cheetah."""
    # {x, xp, y, yp, t, p, particleID}
    from cheetah import ParticleBeam
    from cheetah.particles.species import Species
    E = self.energy.mean().val
    x = self.x.val
    y = self.y.val
    xp = self.cpx.val / self.cpz.val
    yp = self.cpy.val / self.cpz.val
    p = (self.energy.val - E) / E
    tau = -(self.t.val - np.mean(self.t.val)) * constants.speed_of_light
    # s = np.mean(self.t.val) * constants.speed_of_light

    rparticles = np.array([x, xp, y, yp, tau, p])
    num_particles = len(x)
    particles = ones((num_particles, 7))
    particles[:, :6] = tensor(rparticles.transpose(), dtype=float64)
    q_array = np.array([np.abs(float(self.Q.val / len(x))) for _ in x])
    particle_charges = tensor(q_array, dtype=float64)
    particle_beam = ParticleBeam(
        particles=particles,
        energy=as_tensor(E, dtype=float64),
        particle_charges=particle_charges,
        species=Species("electron"),
        s=as_tensor(s_start, dtype=float64),
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
