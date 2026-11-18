"""
Synchrotron radiation in an Ocelot dipole, particle by particle.

Ocelot's own ``SpontanRadEffects(type="dipole")`` is a planar undulator's
large-K limit. This process radiates at each particle's own energy and path length
through a uniform field.
"""

import numpy as np
from ocelot.cpbd.physics_proc import PhysProc, _logger
from scipy.constants import physical_constants

R_E = physical_constants["classical electron radius"][0]
COMPTON = physical_constants["reduced Compton wavelength"][0]
REST_ENERGY = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6  # eV
QUANTUM = 55 / (24 * np.sqrt(3))


class DipoleRadiation(PhysProc):
    """
    Energy loss and (optionally) quantum excitation in a dipole of curvature ``h``.

    Over a step ``dz`` a particle at ``x`` travels ``(1 + h x) dz`` and bends on
    ``h_p = h p0 / p``; it loses ``2/3 r_e m c^2 gamma^4 h_p^2`` per metre, with
    relative spread ``55/(24 sqrt 3) r_e lambda_c gamma^5 |h_p|^3`` in its variance per metre, and
    its transverse momenta shrink with its energy. Electrons only, and a uniform field;
    a combined-function bend radiates as a pure dipole.
    """

    def __init__(self, curvature: float, quantum: bool):
        """
        Parameters
        ----------
        curvature: float
            ``h = angle / length`` [1/m], signed as Ocelot's ``angle``
        quantum: bool
            Whether to add the quantum excitation
        """
        PhysProc.__init__(self)
        self.curvature = curvature
        self.quantum = quantum

    def apply(self, p_array, dz):
        _logger.debug(" DipoleRadiation applied, dz =" + str(dz))
        coordinates = p_array.rparticles
        energy0 = p_array.E * 1e9
        p0c = np.sqrt(energy0**2 - REST_ENERGY**2)
        energy = energy0 + coordinates[5] * p0c
        gamma = energy / REST_ENERGY
        h = abs(self.curvature) * p0c / np.sqrt(energy**2 - REST_ENERGY**2)
        path = (1 + self.curvature * coordinates[0]) * dz
        loss = 2 / 3 * R_E * REST_ENERGY * gamma**4 * h**2 * path
        if self.quantum:
            spread = np.sqrt(QUANTUM * R_E * COMPTON * gamma**5 * h**3 * path)
            loss -= energy * spread * np.random.randn(len(energy))
        scale = 1 - loss / energy
        coordinates[1] *= scale
        coordinates[3] *= scale
        coordinates[5] -= loss / p0c
