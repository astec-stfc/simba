from numpy import sqrt
from ocelot.cpbd.physics_proc import PhysProc, _logger


def rereference(particles, p0c_new: float, rest_energy: float):
    """
    Re-reference `particles` to `p0c_new`, keeping every particle as it was.

    * ``p = dE/(p0 c)`` is recomputed so the absolute energy is kept;
    * ``x'`` and ``y'`` are scaled by ``p0_old / p0_new``, so the
      absolute transverse momentum is kept;
    * ``tau`` is ``c`` times a time difference at this ``s``, so is kept as it is.

    Parameters
    ----------
    particles: ParticleArray
        The beam; changed in place
    p0c_new: float
        The new reference momentum [eV/c]
    rest_energy: float
        The species' rest energy [eV]

    Returns
    -------
    ParticleArray
        `particles`
    """
    energy_old = particles.E * 1e9
    p0c_old = sqrt(energy_old**2 - rest_energy**2)
    energy_new = sqrt(p0c_new**2 + rest_energy**2)
    coordinates = particles.rparticles
    coordinates[5] = (coordinates[5] * p0c_old + energy_old - energy_new) / p0c_new
    coordinates[1] *= p0c_old / p0c_new
    coordinates[3] *= p0c_old / p0c_new
    particles.E = energy_new * 1e-9
    return particles


class FixedReference(PhysProc):
    """
    Put the reference back after a cavity, for a line whose reference is its own.

    Ocelot's cavity map always moves the reference energy on by ``V cos(phi)``, see
    (see :attr:`~simba.Framework_objects.frameworkLattice.fixed_reference`).
    Placed on the element after the cavity, so it applies at the cavity's exit.
    """

    def __init__(self, energy: float, rest_energy: float):
        """
        Parameters
        ----------
        energy: float
            The reference total energy to restore [GeV], Ocelot's unit
        rest_energy: float
            The species' rest energy [eV]
        """
        PhysProc.__init__(self)
        self.reference_energy = energy
        self.rest_energy = rest_energy

    def apply(self, p_array, dz):
        _logger.debug(" FixedReference applied, dz =" + str(dz))
        energy = self.reference_energy * 1e9
        rereference(p_array, sqrt(energy**2 - self.rest_energy**2), self.rest_energy)
