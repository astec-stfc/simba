import numpy as np
from .. import constants
from ..units import UnitValue
from ..gradients import lab_array
from torch import tensor, get_default_device, float64, as_tensor


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


def reference_time(s, p, mass, start):
    """
    :func:`~simba.FrameworkHelperFunctions.reference_time` in torch, so it differentiates: the time
    the reference reaches ``s`` with momentum ``p`` (eV/c, ``mass`` in eV), having left
    ``start = (t0, s0, p0)``, at the mean inverse speed of a linear energy change.
    """
    from torch import as_tensor, ones_like, sqrt, where
    t0, s0, p0 = (as_tensor(v, dtype=float64) for v in start)
    energy, energy0 = sqrt(p**2 + mass**2), sqrt(p0**2 + mass**2)
    gain = energy - energy0
    changed = gain.abs() > 1e-6 * energy
    inv_beta = where(changed, (p - p0) / where(changed, gain, ones_like(gain)), energy / p)
    return t0 + (s - s0) * inv_beta / constants.speed_of_light


def cheetah_lab_coordinates(parray, E0_eV, start=None):
    """
    A Cheetah beam's particles as an (n, 6) tensor of :data:`~simba.Modules.gradients.lab_coordinates`,
    kept in torch so it differentiates.

    ``t`` is ``(s + tau) / c``, or on an open line, with ``start = (t0, s0, p0)``, the reference's
    time of flight from it (:func:`reference_time`) plus ``tau / c``.
    """
    from torch import sqrt, stack
    p0c = sqrt(parray.energy**2 - E0_eV**2)
    cpx = parray.px * p0c
    cpy = parray.py * p0c
    cpz = sqrt(parray.energies**2 - E0_eV**2 - cpx**2 - cpy**2)
    t_reference = (
        parray.s / constants.speed_of_light if start is None
        else reference_time(parray.s, p0c, E0_eV, start)
    )
    t = t_reference + parray.tau / constants.speed_of_light
    return stack([parray.x, cpx, parray.y, cpy, t, cpz], dim=-1)


def cheetah_particles(lab, E0_eV, p0c, t0):
    """
    Cheetah's (n, 7) particles from an (n, 6) tensor of :data:`~simba.Modules.gradients.lab_coordinates`, about a
    reference ``p0c`` (eV) and ``t0`` (s): the inverse of :func:`cheetah_lab_coordinates`,
    kept in torch so it differentiates.
    """
    from torch import ones_like, sqrt, stack
    x, cpx, y, cpy, t, cpz = lab.unbind(-1)
    energy = sqrt(cpx**2 + cpy**2 + cpz**2 + E0_eV**2)
    p = (energy - sqrt(p0c**2 + E0_eV**2)) / p0c
    tau = (t - t0) * constants.speed_of_light
    return stack([x, cpx / p0c, y, cpy / p0c, tau, p, ones_like(x)], dim=-1)


def interpret_cheetah_ParticleBeam(self, parray, zstart=0, s=0, ref_index=None, start=None):
    # Cheetah's Aperture does not remove a lost particle: it sets its survival probability to 0.
    # 0.5 is the threshold Cheetah's own openPMD writer uses to decide alive or lost.
    alive = parray.survival_probabilities.numpy() > 0.5
    charges = parray.particle_charges.numpy()[alive]
    self.set_mass_and_charge(constants.m_e, charges, len(charges))
    # one rest energy for every particle of the Cheetah beam, lost ones included (simba's Cheetah beams are electrons)
    E0_eV = as_tensor(float(np.mean(self.E0_eV)), dtype=float64)
    x, cpx, y, cpy, t, cpz = cheetah_lab_coordinates(parray, E0_eV, start).detach().numpy()[alive].T
    self._beam.x = UnitValue(x, "m")
    self._beam.y = UnitValue(y, "m")
    self._beam.t = UnitValue(t, "s")
    self.set_momenta(cpx, cpy, cpz)
    self._beam.set_total_charge(UnitValue(-1 * abs(np.sum(charges)), "C"))
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1))
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))

    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")


def write_cheetah_beam_file(self, filename=None, write=True, energy=None, t0=None):
    """Save an openpmd file for cheetah.

    ``energy`` (eV) and ``t0`` (s) are the reference energy and the time tau
    is measured from; the beam's own means if not given.
    """
    from cheetah import ParticleBeam
    from cheetah.particles.species import Species
    E = self.energy.mean().val if energy is None else energy
    if t0 is None:
        t0 = np.mean(self.t.val)
    E0_eV = as_tensor(np.asarray(self.E0_eV), dtype=float64)
    p0c = as_tensor(np.sqrt(E**2 - np.asarray(self.E0_eV) ** 2), dtype=float64)
    s = self.s if self.s is not None else 0.0
    x = self.x.val
    lab = as_tensor(lab_array(self), dtype=float64)
    particles = cheetah_particles(lab, E0_eV, p0c, t0)
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
