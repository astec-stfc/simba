"""
SIMBA Particles Module

This module defines the class and utilities for storing a particle distribution.

Each beam consists of particles represented in 6-dimensional phase space (x, cpx, y, cpy, z, cpz),
and appropriate transformations of these coordinates are also accessible as properties.

Functions are also available for rematching the beam based on Twiss parameters.

Classes:
    - :class:`~simba.Modules.Particles.Particles`: Container for a particle distribution.
"""

from copy import deepcopy as copy
import warnings
from math import copysign
import numpy as np
from ... import constants
from .emittance import emittance as emittanceobject
from .twiss import twiss as twissobject
from .slice import slice as sliceobject
from .sigmas import sigmas as sigmasobject
from .centroids import centroids as centroidsobject
from .kde import kde as kdeobject

try:
    from .mve import MVE as MVEobject
except ImportError:
    pass
from ...units import UnitValue, unit_multiply
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from typing import Dict, Any


class Particles(BaseModel):
    """
    Class describing particles in 6D phase space
    [x (m), y (m), z (m) / t (s), px (kg*m/s), py (kg*m/s), pz (kg*m/s)].

    The following objects are created based on this distribution:

    - :attr:`~centroids` -- see :class:`~simba.Modules.Beams.Particles.centroids.centroids`

    - :attr:`~emittance` -- see :class:`~simba.Modules.Beams.Particles.emittance.emittance`

    - :attr:`~kde` -- see :class:`~simba.Modules.Beams.Particles.kde.kde`

    - :attr:`~mve` -- see :class:`~simba.Modules.Beams.Particles.mve.MVE`

    - :attr:`~sigmas` -- see :class:`~simba.Modules.Beams.Particles.sigmas.sigmas`

    - :attr:`~slice` -- see :class:`~simba.Modules.Beams.Particles.slice.slice`

    - :attr:`~twiss` -- see :class:`~simba.Modules.Beams.Particles.twiss.twiss`

    The following properties are derived from these arrays:

    - :attr:`~fullbeam` -- the transpose of the 6D array.

    - [:attr:`~xp`, :attr:`~yp`] -- horizontal and vertical angular distributions.

    - [:attr:`~xc`, :attr:`~xpc`, :attr:`~yc`, :attr:`~ypc`] -- horizontal and vertical positions and
    angular distributions, corrected for dispersion.

    - [:attr:`~cpx`, :attr:`~cpy`, :attr:`~cpz`] -- the beam momenta in eV/c.

    - :attr:`~deltap` -- fractional momentum deviation from the mean.

    - [:attr:`~p`, :attr:`~cp`] -- total beam momentum in kg*m/s and eV/c, respectively.

    - [:attr:`~Ex`, :attr:`~Ey`, :attr:`~Ez`] -- beam energies in eV.

    - [:attr:`~Bx`, :attr:`~By`, :attr:`~Bz`] -- relativistic betas.

    - :attr:`~gamma` -- relativistic Lorentz factor.

    - :attr:`~Brho` -- magnetic rigidity.

    - :attr:`~BetaGamma` -- beam momentum as beta*gamma.

    - [:attr:`~kinetic_energy`, :attr:`~mean_energy`] -- kinetic energy in J and its mean.

    - :attr:`~E0_eV` -- rest energy of the particles in eV.

    - :attr:`~Q` -- total charge of the bunch in C.

    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    # properties = {
    #     "x": "m",
    #     "y": "m",
    #     "z": "m",
    #     "t": "s",
    #     "px": "kg*m/s",
    #     "py": "kg*m/s",
    #     "pz": "kg*m/s",
    #     "p": "kg*m/s",
    #     "particle_mass": "kg",
    #     "particle_rest_energy": "J",
    #     "particle_rest_energy_eV": "eV/c",
    #     "particle_charge": "C",
    # }

    # particle_mass = UnitValue(constants.m_e, "kg")
    # E0 = UnitValue(particle_mass * constants.speed_of_light**2, "J")
    # E0_eV = UnitValue(E0 / constants.elementary_charge, "eV/c")
    q_over_c: UnitValue = UnitValue(
        constants.elementary_charge / constants.speed_of_light, "C/c"
    )
    """Elementary charge divided by speed of light"""

    speed_of_light: UnitValue = UnitValue(constants.speed_of_light, "m/s")
    """Speed of light"""

    mass: UnitValue | list | np.ndarray = None
    """Mass of particles [kg] -- can be all the same, or variable
    #TODO deprecated?"""

    particle_mass: UnitValue | list | np.ndarray = None
    """Mass of particles [kg] -- can be all the same, or variable"""

    particle_rest_energy: UnitValue | list | np.ndarray = None
    """Rest mass energy of the particle in kg"""

    particle_rest_energy_eV: UnitValue | list | np.ndarray = None
    """Rest mass energy of the particle in eV"""

    particle_charge: UnitValue | list | np.ndarray = None
    """Charge of the particle [C] -- can be all the same, or variable
    #TODO deprecated?"""

    charge: UnitValue | list | np.ndarray = None
    """Charge of the particle [C] -- can be all the same, or variable"""

    clock: UnitValue | list | np.ndarray = None
    """Time unit of particles (ASTRA-type)"""

    t: UnitValue | list | np.ndarray = None
    """Time coordinates of particles [s]"""

    total_charge: UnitValue | float = None
    """Total charge of particle bunch [C]"""

    x: UnitValue | list | np.ndarray = None
    """Horizontal coordinates of particles [m]"""

    y: UnitValue | list | np.ndarray = None
    """Vertical coordinates of particles [m]"""

    z: UnitValue | list | np.ndarray = None
    """Longitudinal coordinates of particles [m]"""

    px: UnitValue | list | np.ndarray = None
    """Horizontal momentum of particles [kg*m/s]"""

    py: UnitValue | list | np.ndarray = None
    """Vertical momentum of particles [kg*m/s]"""

    pz: UnitValue | list | np.ndarray = None
    """Longitudinal momentum of particles [kg*m/s]"""

    status: UnitValue | list | np.ndarray = None
    """Status of particles for OpenPMD-type distributions"""

    nmacro: int | np.ndarray | UnitValue = None
    """Number of macroparticles in this object"""

    theta: UnitValue | float = 0.0
    """Horizontal rotation of particle distribution with respect to the nominal axis [rad]"""

    reference_particle: list | np.ndarray = None
    """Reference particle for ASTRA-type distributions"""

    toffset: float | UnitValue = None
    """Temporal offset [s]"""

    offset: UnitValue | list | np.ndarray = [0, 0, 0]
    """Beam positional offset in Cartesian coordinates"""

    species_name: Dict = {1: "electron", 2: "positron", 3: "proton", 4: "hydrogen"}

    mass_index: Dict = {
        1: constants.m_e,  # electron
        2: constants.m_e,  # positron
        3: constants.m_p,  # proton
        4: constants.m_p,  # hydrogen ion
    }
    """Dictionary representing the index and mass of supported particles"""

    charge_sign_index: Dict = {1: -1, 2: 1, 3: 1, 4: 1}
    """Dictionary representing the index and charge of supported particles"""

    def sign(self, x):
        return copysign(1, x)

    def get_particle_index(self, m: float, q: int) -> int:
        """
        Get the index of a particle from mass and charge index.

        Parameters
        ----------
        m: float
            Mass of particle
        q: int
            Charge of particle

        Returns
        -------
        int
            Particle index (see :attr:`~mass_index` and :attr:`~charge_sign_index`.
        """
        if m == constants.m_e or (0.9 * constants.m_e) < m < (1.1 * constants.m_e):
            if self.sign(q) > 0:
                # print('found positron')
                return 2
            # print('found electron')
            return 1
        elif m == constants.m_p or (0.9 * constants.m_p) < m < (1.1 * constants.m_p):
            if self.sign(q) > 0:
                # print('found proton')
                return 3
            # print('found h-')
            return 4
        else:
            raise ValueError(f"Particle with mass {m} and charge {q} not supported")

    """ ********************  Statistical Parameters  ************************* """

    def __init__(self, *args, **kwargs):
        super(Particles, self).__init__(*args, **kwargs)

    #
    # def __getitem__(self, key):
    #     if isinstance(super(Particles, self).__getitem__(key), (list, tuple)):
    #         return np.array(super(Particles, self).__getitem__(key))
    #     else:
    #         try:
    #             return super(Particles, self).__getitem__(key)
    #         except KeyError:
    #             raise AttributeError(key)

    def model_dump(self, *args, **kwargs) -> Dict:
        # Only include computed fields
        computed_keys = {f for f in self.__pydantic_decorators__.computed_fields.keys()}
        full_dump = super().model_dump(*args, **kwargs)
        mod_dump = {k: v for k, v in full_dump.items() if k in computed_keys}
        for col in ["x", "y", "z", "cpx", "cpy", "cpz"]:
            mod_dump.update({col: getattr(self, col)})
        for obj in ["emittance", "twiss", "sigmas", "centroids"]:
            mod_dump.update({obj: getattr(self, obj).model_dump()})
        return mod_dump

    @property
    def slice(self) -> sliceobject:
        """
        Get the slice properties from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.slices.slices`
            The slice properties
        """
        if not hasattr(self, "_slice"):
            self._slice = sliceobject(self)
        return self._slice

    @property
    def emittance(self) -> emittanceobject:
        """
        Get the emittance calculations from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.emittance.emittance`
            Beam emittance
        """
        if not hasattr(self, "_emittance"):
            self._emittance = emittanceobject(self)
        return self._emittance

    @property
    def twiss(self) -> twissobject:
        """
        Get the Twiss parameters from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.twiss.twiss`
            Twiss parameters
        """
        if not hasattr(self, "_twiss"):
            self._twiss = twissobject(self)
        return self._twiss

    @property
    def sigmas(self) -> sigmasobject:
        """
        Get the beam sigmas from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.sigmas.sigmas`
            Beam sigmas
        """
        if not hasattr(self, "_sigmas"):
            self._sigmas = sigmasobject(self)
        return self._sigmas

    @property
    def centroids(self) -> centroidsobject:
        """
        Get the centroids from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.centroids.centroids`
            Beam centroids
        """
        if not hasattr(self, "_mean"):
            self._mean = centroidsobject(self)
        return self._mean

    @property
    def kde(self) -> kdeobject:
        """
        Get the kernel density estimator from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.kde.kde`
            KDE
        """
        if not hasattr(self, "_kde"):
            self._kde = kdeobject(self)
        return self._kde

    @property
    def mve(self) -> Any:
        """
        Get the minimum volume ellipse from the distribution.

        Returns
        -------
        :class:`~simba.Modules.Beams.Particles.mve.MVE`
            The MVE
        """
        if not hasattr(self, "_mve"):
            self._mve = MVEobject(self)
        return self._mve

    def covariance(
        self, u: np.ndarray | UnitValue, up: np.ndarray | UnitValue
    ) -> UnitValue | int:
        """
        Get the covariance from two arrays

        Parameters
        ----------
        u: np.ndarray or :class:`~simba.Modules.units.UnitValue`
            First column
        up: np.ndarray or :class:`~simba.Modules.units.UnitValue`
            Second column

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue` or int
            Covariance (returns zero if arrays are not of same length)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ans = 0
            if len(u) > 1 and len(up) > 1:
                ans = UnitValue(
                    np.cov([u, up])[0, 1],
                    unit_multiply(u.units, up.units, divide=False),
                )
            else:
                warnings.warn("Arrays are not of the same length")
            return ans

    def eta_correlation(self, u) -> UnitValue | int:
        """
        Get the covariance between an array and the beam momentum
        :attr:`~p`

        Parameters
        ----------
        u: np.ndarray or :class:`~simba.Modules.units.UnitValue`
            Column to correlate with `p`

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue` or int
            Covariance
        """
        return self.covariance(u, self.p) / self.covariance(self.p, self.p)

    def eta_corrected(self, u) -> UnitValue:
        """
        Correct a column with respect to the beam momentum, subtracting
        :func:`~eta_correlation` from u multiplied with :attr:`~p`

        Parameters
        ----------
        u: np.ndarray or :class:`~simba.Modules.units.UnitValue`
            Column to correct with respect to `p`

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue` or int
            Corrected column
        """
        return u - self.eta_correlation(u) * self.p

    def apply_mask(self, mask: Any) -> None:
        """
        Cut the beam with respect to a mask, removing some particles

        Parameters
        ----------
        mask: int | np.ndarray | list
            Mask to apply
        """
        prebeam = self.fullbeam
        cutbeam = prebeam[mask, :]
        self.fullbeam = cutbeam.T

    @property
    def fullbeam(self) -> np.ndarray:
        """
        Get the full beam as a transpose of all six columns.

        Returns
        -------
        np.ndarray
            The beam object as [x,y,z,px,py,pz]
        """
        return np.array([self.x, self.y, self.z, self.px, self.py, self.pz]).T

    @fullbeam.setter
    def fullbeam(self, beam):
        self.x, self.y, self.z, self.px, self.py, self.pz, self.t, self.nmacro, self.charge = np.array(beam).T

    @property
    def particle_index(self) -> list:
        """
        Get the particle index from the mass and charge of all particles.

        Returns
        -------
        list
            The particle index for all :attr:`~particle_mass` and :attr:`~charge`.

        """
        return [
            self.get_particle_index(m, q)
            for m, q in zip(self.particle_mass, self.charge)
        ]

    @property
    def chargesign(self) -> list:
        """
        Get the sign of charge all particles

        Returns
        -------
        list
            The charge signs of all particles

        """
        return [self.sign(q) for q in self.charge]

    @property
    def xc(self) -> UnitValue:
        """
        Get the horizontal distribution corrected with respect to dispersion

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The corrected horizontal distribution
        """
        return UnitValue(self.eta_corrected(self.x), "m")

    @property
    def xpc(self) -> UnitValue:
        """
        Get the horizontal angle corrected with respect to dispersion

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The corrected horizontal angle
        """
        return UnitValue(self.eta_corrected(self.xp), "")

    @property
    def yc(self) -> UnitValue:
        """
        Get the vertical distribution corrected with respect to dispersion

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The corrected vertical distribution
        """
        return UnitValue(self.eta_corrected(self.y), "m")

    @property
    def ypc(self) -> UnitValue:
        """
        Get the vertical angle corrected with respect to dispersion

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The corrected vertical angle
        """
        return UnitValue(self.eta_corrected(self.yp), "")

    @property
    def cpx(self) -> UnitValue:
        """
        Get the horizontal momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The horizontal momentum
        """
        return UnitValue(self.px / self.q_over_c, "eV/c")

    @property
    def cpy(self) -> UnitValue:
        """
        Get the vertical momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The vertical momentum
        """
        return UnitValue(self.py / self.q_over_c, "eV/c")

    @property
    def cpz(self) -> UnitValue:
        """
        Get the longitudinal momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The longitudinal momentum
        """
        return UnitValue(self.pz / self.q_over_c, "eV/c")

    @property
    def deltap(self) -> UnitValue:
        """
        Get the fractional beam momentum

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The fractional momentum
        """
        return (self.cp - np.mean(self.cp)) / np.mean(self.cp)

    @property
    def xp(self) -> UnitValue:
        """
        Get the horizontal momentum angle in rad

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The horizontal angle
        """
        return UnitValue(np.arctan(self.px / self.pz), "rad")

    @property
    def yp(self) -> UnitValue:
        """
        Get the vertical momentum angle in rad

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The vertical angle
        """
        return UnitValue(np.arctan(self.py / self.pz), "rad")

    @property
    def p(self) -> UnitValue:
        """
        Get the total beam momentum in kg*m/s

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The beam momentum
        """
        return UnitValue(self.cp * self.q_over_c, "kg*m/s")

    @property
    def cp(self) -> UnitValue:
        """
        Get the total beam momentum in eV/C

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The beam momentum
        """
        return UnitValue(np.sqrt(self.cpx**2 + self.cpy**2 + self.cpz**2), "eV/c")

    @property
    def Brho(self) -> UnitValue:
        """
        Get the magnetic rigidity in the longitudinal direction

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Magnetic rigidity
        """
        return UnitValue(np.mean(self.pz) / constants.elementary_charge, "T*m")

    @property
    def E0_eV(self) -> UnitValue:
        """
        Get the particle rest energy in eV;
        see :attr:`~particle_rest_energy_eV`

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Particle rest energy
        """
        return self.particle_rest_energy_eV

    @property
    def gamma(self) -> UnitValue:
        """
        Get the relativistic Lorentz factor of the beam distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Lorentz factor
        """
        return UnitValue(
            np.sqrt(1 + (self.cp.val / self.particle_rest_energy_eV.val) ** 2), ""
        )

    @property
    def BetaGamma(self) -> UnitValue:
        """
        Get the beam momentum as beta*gamma

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Beam momentum
        """
        return UnitValue(self.cp / self.particle_rest_energy_eV, "")

    @property
    def energy(self) -> UnitValue:
        """
        Get the energy of the particles in eV

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The beam energy
        """
        return UnitValue(self.gamma * self.particle_rest_energy_eV, "eV")

    @property
    def Ex(self) -> UnitValue:
        """
        Get the horizontal beam energy in eV

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The horizontal beam energy
        """
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpx**2), "eV")

    @property
    def Ey(self) -> UnitValue:
        """
        Get the longitudinal beam energy in eV

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The longitudinal beam energy
        """
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpy**2), "eV")

    @property
    def Ez(self) -> UnitValue:
        """
        Get the longitudinal beam energy in eV

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The longitudinal beam energy
        """
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpz**2), "eV")

    @property
    def Bx(self) -> UnitValue:
        """
        Get the horizontal relativistic beta

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The horizontal relativistic beta
        """
        return UnitValue(self.cpx / self.energy, "")

    @property
    def By(self) -> UnitValue:
        """
        Get the vertical relativistic beta

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The vertical relativistic beta
        """
        return UnitValue(self.cpy / self.energy, "")

    @property
    def Bz(self) -> UnitValue:
        """
        Get the longitudinal relativistic beta

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The longitudinal relativistic beta
        """
        return UnitValue(self.cpz / self.energy, "")

    @computed_field
    @property
    def Q(self) -> UnitValue:
        """
        Get the total charge of the bunch in C

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The total charge
        """
        return UnitValue(self.total_charge, "C")

    def set_total_charge(self, q: float) -> None:
        """
        Set the total charge of the bunch in C.

        This will also update the `charge` of the individual particles.

        Parameters
        ----------
        q: float
            The total charge
        """
        self.total_charge = UnitValue(q, units="C")
        particle_q = q / (len(self.x))
        self.charge = UnitValue(np.full(len(self.x), particle_q), units="C")

    # @property
    # def sigma_z(self):
    #     return self.rms(self.Bz*constants.speed_of_light*(self['t'] - np.mean(self['t'])))

    @property
    def kinetic_energy(self) -> UnitValue:
        """
        Get the kinetic energy of the particles in J

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Kinetic energy of particles
        """
        if self.particle_rest_energy is None:
            self.particle_rest_energy = self.particle_rest_energy_eV / constants.elementary_charge
        return UnitValue(
            np.array(
                (
                    np.sqrt(self.particle_rest_energy**2 + self.cp**2)
                    - self.particle_rest_energy**2
                )
            ),
            "J",
        )

    @property
    def mean_energy(self) -> UnitValue:
        """
        Get the mean energy of the particles in J (the mean of :attr:`~kinetic_energy`)

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean energy of particles
        """
        return UnitValue(np.mean(self.kinetic_energy), "J")

    def computeCorrelations(
        self, x: UnitValue | np.ndarray, y: UnitValue | np.ndarray
    ) -> tuple:
        """
        Get the covariances `(cov(x,x), cov(x,y), cov(y,y))`,
        see :func:`~covariance`

        Returns
        -------
        tuple
            Covariances between the arrays provided
        """
        return self.covariance(x, x), self.covariance(x, y), self.covariance(y, y)

    def performTransformation(
        self,
        x: UnitValue | np.ndarray,
        xp: UnitValue | np.ndarray,
        beta: bool | float | UnitValue = False,
        alpha: bool | float | UnitValue = False,
        nEmit: bool | float | UnitValue = False,
    ) -> tuple:
        """
        Transform the arrays provided with respect to the Twiss and emittance functions given.

        Parameters
        ----------
        x: :class:`~simba.Modules.units.UnitValue` or np.ndarray
            The first array
        xp: :class:`~simba.Modules.units.UnitValue` or np.ndarray
            The second array
        beta: :class:`~simba.Modules.units.UnitValue` or float or bool
            The beta function to transform the arrays; if `False`, use :func:`~computeCorrelations`
        alpha: :class:`~simba.Modules.units.UnitValue` or float or bool
            The alpha function to transform the arrays; if `False`, use :func:`~computeCorrelations`
        nEmit: :class:`~simba.Modules.units.UnitValue` or float or bool
            The emittance to transform the arrays; if `False`, use :func:`~computeCorrelations`

        Returns
        -------
        tuple
            The transformed arrays
        """
        p = self.cp
        pAve = np.mean(p)
        gamma = np.mean(self.gamma)
        p = p / pAve - 1
        eta1, etap1, _ = self.twiss.calculate_etax()
        x -= p * eta1
        xp -= p * etap1

        S11, S12, S22 = self.computeCorrelations(x, xp)
        emit = np.sqrt(S11 * S22 - S12**2)
        beta1 = S11 / emit
        alpha1 = -S12 / emit
        beta2 = beta if beta is not False else beta1
        alpha2 = alpha if alpha is not False else alpha1
        R11 = beta2 / np.sqrt(beta1 * beta2)
        R12 = 0
        R21 = (alpha1 - alpha2) / np.sqrt(beta1 * beta2)
        R22 = beta1 / np.sqrt(beta1 * beta2)
        if nEmit:
            factor = np.sqrt(float(nEmit) / (emit * gamma))
            R11 *= factor
            R12 *= factor
            R22 *= factor
            R21 *= factor
        x0 = copy(x)
        xp0 = copy(xp)
        x = R11 * x0 + R12 * xp0
        xp = R21 * x0 + R22 * xp0
        return x, xp

    def rematchXPlane(
        self,
        beta: UnitValue | float = None,
        alpha: UnitValue | float = None,
        nEmit: UnitValue | float = None,
    ) -> None:
        """
        Rematch :attr:`~x` and :attr:`~xp` with respect to the Twiss and emittance functions given.

        Parameters
        ----------
        beta: :class:`~simba.Modules.units.UnitValue` or float or bool
            The beta function to transform the arrays; if `False`, raise a warning
        alpha: :class:`~simba.Modules.units.UnitValue` or float or bool
            The alpha function to transform the arrays; if `False`, raise a warning
        nEmit: :class:`~simba.Modules.units.UnitValue` or float or bool
            The emittance to transform the arrays.
        """
        if all([beta is not None and alpha is not None and beta is not False and alpha is not False]):
            x, xp = self.performTransformation(self.x, self.xp, beta, alpha, nEmit)
            self.x = UnitValue(x, "m")

            cpz = self.cp / np.sqrt(xp**2 + self.yp**2 + 1)
            cpx = xp * cpz
            cpy = self.yp * cpz
            self.px = UnitValue(cpx * self.q_over_c, "kg*m/s")
            self.py = UnitValue(cpy * self.q_over_c, "kg*m/s")
            self.pz = UnitValue(cpz * self.q_over_c, "kg*m/s")
        elif all([beta is None or beta is False, alpha is None or alpha is False]):
            pass
        else:
            warnings.warn("Both beta and alpha must be provided to rematch")

    def rematchYPlane(
        self,
        beta: UnitValue | float | bool = False,
        alpha: UnitValue | float | bool = False,
        nEmit: UnitValue | float | bool = False,
    ) -> None:
        """
        Rematch :attr:`~y` and :attr:`~yp` with respect to the Twiss and emittance functions given.

        Parameters
        ----------
        beta: :class:`~simba.Modules.units.UnitValue` or float or bool
            The beta function to transform the arrays; if `False`, raise a warning
        alpha: :class:`~simba.Modules.units.UnitValue` or float or bool
            The alpha function to transform the arrays; if `False`, raise a warning
        nEmit: :class:`~simba.Modules.units.UnitValue` or float or bool
            The emittance to transform the arrays.
        """
        if all([beta is not None and alpha is not None and beta is not False and alpha is not False]):
            y, yp = self.performTransformation(self.y, self.yp, beta, alpha, nEmit)
            self.y = UnitValue(y, "m")

            cpz = self.cp / np.sqrt(self.xp**2 + yp**2 + 1)
            cpx = self.xp * cpz
            cpy = yp * cpz
            self.px = UnitValue(cpx * self.q_over_c, "kg*m/s")
            self.py = UnitValue(cpy * self.q_over_c, "kg*m/s")
            self.pz = UnitValue(cpz * self.q_over_c, "kg*m/s")
        elif all([beta is None or beta is False, alpha is None or alpha is False]):
            pass
        else:
            warnings.warn("Both beta and alpha must be provided to rematch")

    def performTransformationPeakISlice(
        self,
        xslice: UnitValue | np.ndarray,
        xpslice: UnitValue | np.ndarray,
        x: UnitValue | np.ndarray,
        xp: UnitValue | np.ndarray,
        beta: UnitValue | float = None,
        alpha: UnitValue | float = None,
        nEmit: UnitValue | float = None,
    ) -> tuple:
        """
        Transform the arrays provided with respect to the Twiss and emittance functions given,
        or match the arrays with respect to their values at a given slice.

        Parameters
        ----------
        xslice: :class:`~simba.Modules.units.UnitValue` or np.ndarray
            The first array at a given slice
        xpslice: :class:`~simba.Modules.units.UnitValue` or np.ndarray
            The second array at a given slice
        x: :class:`~simba.Modules.units.UnitValue` or np.ndarray
            The first array
        xp: :class:`~simba.Modules.units.UnitValue` or np.ndarray
            The second array
        beta: :class:`~simba.Modules.units.UnitValue` or float or bool
            The beta function to transform the arrays; if `False`, use :func:`~computeCorrelations`
        alpha: :class:`~simba.Modules.units.UnitValue` or float or bool
            The alpha function to transform the arrays; if `False`, use :func:`~computeCorrelations`
        nEmit: :class:`~simba.Modules.units.UnitValue` or float or bool
            The emittance to transform the arrays; if `False`, use :func:`~computeCorrelations`

        Returns
        -------
        tuple
            The transformed arrays
        """
        p = self.cp
        pAve = np.mean(p)
        gamma = np.mean(self.gamma)
        p = p / pAve - 1
        eta1, etap1, _ = self.twiss.calculate_etax()
        x -= p * eta1
        xp -= p * etap1

        S11, S12, S22 = self.computeCorrelations(xslice, xpslice)
        emit = np.sqrt(S11 * S22 - S12**2)
        beta1 = S11 / emit
        alpha1 = -S12 / emit
        beta2 = beta if beta is not None else beta1
        alpha2 = alpha if alpha is not None else alpha1
        R11 = beta2 / np.sqrt(beta1 * beta2)
        R12 = 0
        R21 = (alpha1 - alpha2) / np.sqrt(beta1 * beta2)
        R22 = beta1 / np.sqrt(beta1 * beta2)
        if nEmit is not False:
            factor = np.sqrt(nEmit / (emit * gamma))
            R11 *= factor
            R12 *= factor
            R22 *= factor
            R21 *= factor
        x0 = x
        xp0 = xp
        x = R11 * x0 + R12 * xp0
        xp = R21 * x0 + R22 * xp0
        return x, xp

    def rematchXPlanePeakISlice(
        self,
        beta=False,
        alpha=False,
        nEmit=False,
    ) -> None:
        """
        Rematch :attr:`~x` and :attr:`~xp` with respect to the Twiss and emittance functions given,
        or their values at the peak current slice; see :func:`~performTransformationPeakISlice`.

        Parameters
        ----------
        beta: :class:`~simba.Modules.units.UnitValue` or float or bool
            The beta function to transform the arrays; if `False`, raise a warning
        alpha: :class:`~simba.Modules.units.UnitValue` or float or bool
            The alpha function to transform the arrays; if `False`, raise a warning
        nEmit: :class:`~simba.Modules.units.UnitValue` or float or bool
            The emittance to transform the arrays.
        """
        peakIPosition = self.slice.slice_max_peak_current_slice
        xslice = self.slice.slice_data(self.x)[peakIPosition]
        xpslice = self.slice.slice_data(self.xp)[peakIPosition]
        x, xp = self.performTransformationPeakISlice(
            xslice, xpslice, self.x, self.xp, beta, alpha, nEmit
        )
        self.x = UnitValue(x, "m")
        # self.xp = xp

        cpz = self.cp / np.sqrt(xp**2 + self.yp**2 + 1)
        cpx = xp * cpz
        cpy = self.yp * cpz
        self.px = UnitValue(cpx * self.q_over_c, "kg*m/s")
        self.py = UnitValue(cpy * self.q_over_c, "kg*m/s")
        self.pz = UnitValue(cpz * self.q_over_c, "kg*m/s")

    def rematchYPlanePeakISlice(
        self,
        beta=False,
        alpha=False,
        nEmit=False,
    ) -> None:
        """
        Rematch :attr:`~y` and :attr:`~yp` with respect to the Twiss and emittance functions given,
        or their values at the peak current slice; see :func:`~performTransformationPeakISlice`.

        Parameters
        ----------
        beta: :class:`~simba.Modules.units.UnitValue` or float or bool
            The beta function to transform the arrays; if `False`, raise a warning
        alpha: :class:`~simba.Modules.units.UnitValue` or float or bool
            The alpha function to transform the arrays; if `False`, raise a warning
        nEmit: :class:`~simba.Modules.units.UnitValue` or float or bool
            The emittance to transform the arrays.
        """
        peakIPosition = self.slice.slice_max_peak_current_slice
        yslice = self.slice.slice_data(self.y)[peakIPosition]
        ypslice = self.slice.slice_data(self.yp)[peakIPosition]
        y, yp = self.performTransformationPeakISlice(
            yslice, ypslice, self.y, self.yp, beta, alpha, nEmit
        )
        self.y = UnitValue(y, "m")
        # self.yp = yp

        cpz = self.cp / np.sqrt(self.xp**2 + yp**2 + 1)
        cpx = self.xp * cpz
        cpy = yp * cpz
        self.px = UnitValue(cpx * self.q_over_c, "kg*m/s")
        self.py = UnitValue(cpy * self.q_over_c, "kg*m/s")
        self.pz = UnitValue(cpz * self.q_over_c, "kg*m/s")
