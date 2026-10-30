"""Particle distribution in 6D phase space, its derived quantities and Twiss rematching."""

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
    Particles in 6D phase space: x, y, z [m] (or t [s]) and px, py, pz [kg*m/s].

    The analysis objects (:attr:`centroids`, :attr:`emittance`, :attr:`kde`, :attr:`mve`,
    :attr:`sigmas`, :attr:`slice`, :attr:`twiss`) are built on first access.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    q_over_c: UnitValue = UnitValue(
        constants.elementary_charge / constants.speed_of_light, "C/c"
    )
    """Elementary charge over speed of light."""

    speed_of_light: UnitValue = UnitValue(constants.speed_of_light, "m/s")

    mass: UnitValue | list | np.ndarray = None
    """Particle mass [kg]; unused, see :attr:`particle_mass`."""

    particle_mass: UnitValue | list | np.ndarray = None
    """Per-particle mass [kg]."""

    particle_rest_energy: UnitValue | list | np.ndarray = None
    """Per-particle rest energy [J]."""

    particle_rest_energy_eV: UnitValue | list | np.ndarray = None
    """Per-particle rest energy [eV]."""

    particle_charge: UnitValue | list | np.ndarray = None
    """Charge of one physical particle [C], e.g. -e for electrons."""

    charge: UnitValue | list | np.ndarray = None
    """Per-macroparticle charge [C]."""

    clock: UnitValue | list | np.ndarray = None
    """ASTRA clock column."""

    t: UnitValue | list | np.ndarray = None
    """Time [s]."""

    total_charge: UnitValue | float = None
    """Bunch charge [C]."""

    x: UnitValue | list | np.ndarray = None
    """Horizontal position [m]."""

    y: UnitValue | list | np.ndarray = None
    """Vertical position [m]."""

    z: UnitValue | list | np.ndarray = None
    """Longitudinal position [m]."""

    s: UnitValue | list | np.ndarray | float = None
    """s position [m]."""

    px: UnitValue | list | np.ndarray = None
    """Horizontal momentum [kg*m/s]."""

    py: UnitValue | list | np.ndarray = None
    """Vertical momentum [kg*m/s]."""

    pz: UnitValue | list | np.ndarray = None
    """Longitudinal momentum [kg*m/s]."""

    status: UnitValue | list | np.ndarray = None
    """openPMD particle status."""

    nmacro: int | np.ndarray | UnitValue = None
    """Physical particles per macroparticle."""

    theta: UnitValue | float = 0.0
    """Horizontal rotation from the nominal axis [rad]."""

    toffset: float | UnitValue = None
    """Time offset [s]."""

    offset: UnitValue | list | np.ndarray = [0, 0, 0]
    """Position offset [x, y, z] [m]."""

    species_name: Dict = {1: "electron", 2: "positron", 3: "proton", 4: "hydrogen"}

    mass_index: Dict = {
        1: constants.m_e,  # electron
        2: constants.m_e,  # positron
        3: constants.m_p,  # proton
        4: constants.m_p,  # hydrogen ion
    }
    """Mass [kg] of each particle index."""

    charge_sign_index: Dict = {1: -1, 2: 1, 3: 1, 4: 1}
    """Charge sign of each particle index."""

    def sign(self, x):
        return copysign(1, x)

    def get_particle_index(self, m: float, q: int) -> int:
        """
        Particle index (see :attr:`mass_index`) from mass and charge; masses match to 10%.

        Parameters
        ----------
        m: float
            Mass [kg]
        q: int
            Charge; only its sign is used

        Returns
        -------
        int
        """
        if m == constants.m_e or (0.9 * constants.m_e) < m < (1.1 * constants.m_e):
            if self.sign(q) > 0:
                return 2
            return 1
        elif m == constants.m_p or (0.9 * constants.m_p) < m < (1.1 * constants.m_p):
            if self.sign(q) > 0:
                return 3
            return 4
        else:
            raise ValueError(f"Particle with mass {m} and charge {q} not supported")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_dump(self, *args, **kwargs) -> Dict:
        # Only include computed fields
        computed_keys = set(self.__pydantic_decorators__.computed_fields)
        full_dump = super().model_dump(*args, **kwargs)
        mod_dump = {k: v for k, v in full_dump.items() if k in computed_keys}
        for col in ["x", "y", "z", "cpx", "cpy", "cpz"]:
            mod_dump.update({col: getattr(self, col)})
        for obj in ["emittance", "twiss", "sigmas", "centroids"]:
            mod_dump.update({obj: getattr(self, obj).model_dump()})
        return mod_dump

    @property
    def slice(self) -> sliceobject:
        """Slice properties."""
        if not hasattr(self, "_slice"):
            self._slice = sliceobject(self)
        return self._slice

    @property
    def emittance(self) -> emittanceobject:
        """Emittance calculations."""
        if not hasattr(self, "_emittance"):
            self._emittance = emittanceobject(self)
        return self._emittance

    @property
    def twiss(self) -> twissobject:
        """Twiss parameters."""
        if not hasattr(self, "_twiss"):
            self._twiss = twissobject(self)
        return self._twiss

    @property
    def sigmas(self) -> sigmasobject:
        """Beam sigmas."""
        if not hasattr(self, "_sigmas"):
            self._sigmas = sigmasobject(self)
        return self._sigmas

    @property
    def centroids(self) -> centroidsobject:
        """Beam centroids."""
        if not hasattr(self, "_mean"):
            self._mean = centroidsobject(self)
        return self._mean

    @property
    def kde(self) -> kdeobject:
        """Kernel density estimator."""
        if not hasattr(self, "_kde"):
            self._kde = kdeobject(self)
        return self._kde

    @property
    def mve(self) -> Any:
        """Minimum volume ellipse (:class:`~simba.Modules.Beams.Particles.mve.MVE`)."""
        if not hasattr(self, "_mve"):
            self._mve = MVEobject(self)
        return self._mve

    def covariance(
        self, u: np.ndarray | UnitValue, up: np.ndarray | UnitValue
    ) -> UnitValue | int:
        """
        Covariance of two arrays.

        Parameters
        ----------
        u, up: np.ndarray or UnitValue

        Returns
        -------
        UnitValue or int
            Covariance; 0, with a warning, if either array has fewer than two entries
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
        Correlation of an array with the momentum, cov(u, p) / cov(p, p).

        Parameters
        ----------
        u: np.ndarray or UnitValue

        Returns
        -------
        UnitValue or int
        """
        return self.covariance(u, self.p) / self.covariance(self.p, self.p)

    def eta_corrected(self, u) -> UnitValue:
        """
        Remove the momentum correlation from an array: u - :meth:`eta_correlation` * p.

        Parameters
        ----------
        u: np.ndarray or UnitValue

        Returns
        -------
        UnitValue
        """
        return u - self.eta_correlation(u) * self.p

    def apply_mask(self, mask: Any) -> None:
        """
        Keep only the particles selected by a mask, in every per-particle array.

        Parameters
        ----------
        mask: int | np.ndarray | list
            Index or boolean mask
        """
        n = len(self.x)
        for key, value in self:
            if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
                setattr(self, key, value[mask])

    @property
    def fullbeam(self) -> np.ndarray:
        """(N, 6) array of [x, y, z, px, py, pz]."""
        return np.array([self.x, self.y, self.z, self.px, self.py, self.pz]).T

    @fullbeam.setter
    def fullbeam(self, beam):
        self.x, self.y, self.z, self.px, self.py, self.pz = np.array(beam).T

    @property
    def particle_index(self) -> list:
        """:meth:`get_particle_index` of every particle."""
        return [
            self.get_particle_index(m, q)
            for m, q in zip(self.particle_mass, self.charge)
        ]

    @property
    def chargesign(self) -> list:
        """Charge sign of every particle."""
        return [self.sign(q) for q in self.charge]

    @property
    def xc(self) -> UnitValue:
        """Horizontal position with dispersion removed [m]."""
        return UnitValue(self.eta_corrected(self.x), "m")

    @property
    def xpc(self) -> UnitValue:
        """Horizontal angle with dispersion removed."""
        return UnitValue(self.eta_corrected(self.xp), "")

    @property
    def yc(self) -> UnitValue:
        """Vertical position with dispersion removed [m]."""
        return UnitValue(self.eta_corrected(self.y), "m")

    @property
    def ypc(self) -> UnitValue:
        """Vertical angle with dispersion removed."""
        return UnitValue(self.eta_corrected(self.yp), "")

    @property
    def cpx(self) -> UnitValue:
        """Horizontal momentum [eV/c]."""
        return UnitValue(self.px / self.q_over_c, "eV/c")

    @property
    def cpy(self) -> UnitValue:
        """Vertical momentum [eV/c]."""
        return UnitValue(self.py / self.q_over_c, "eV/c")

    @property
    def cpz(self) -> UnitValue:
        """Longitudinal momentum [eV/c]."""
        return UnitValue(self.pz / self.q_over_c, "eV/c")

    @property
    def deltap(self) -> UnitValue:
        """Fractional momentum deviation from the mean."""
        return (self.cp - np.mean(self.cp)) / np.mean(self.cp)

    @property
    def xp(self) -> UnitValue:
        """Horizontal angle, arctan(px/pz) [rad]."""
        return UnitValue(np.arctan(self.px / self.pz), "rad")

    @property
    def yp(self) -> UnitValue:
        """Vertical angle, arctan(py/pz) [rad]."""
        return UnitValue(np.arctan(self.py / self.pz), "rad")

    @property
    def p(self) -> UnitValue:
        """Total momentum [kg*m/s]."""
        return UnitValue(self.cp * self.q_over_c, "kg*m/s")

    @property
    def cp(self) -> UnitValue:
        """Total momentum [eV/c]."""
        return UnitValue(np.sqrt(self.cpx**2 + self.cpy**2 + self.cpz**2), "eV/c")

    @property
    def Brho(self) -> UnitValue:
        """Magnetic rigidity from the mean pz [T*m]."""
        return UnitValue(np.mean(self.pz) / constants.elementary_charge, "T*m")

    @property
    def E0_eV(self) -> UnitValue:
        """Alias of :attr:`particle_rest_energy_eV`."""
        return self.particle_rest_energy_eV

    @property
    def gamma(self) -> UnitValue:
        """Lorentz factor."""
        return UnitValue(
            np.sqrt(1 + (self.cp.val / self.particle_rest_energy_eV.val) ** 2), ""
        )

    @property
    def BetaGamma(self) -> UnitValue:
        """Momentum as beta*gamma."""
        return UnitValue(self.cp / self.particle_rest_energy_eV, "")

    @property
    def energy(self) -> UnitValue:
        """Total energy, gamma * rest energy [eV]."""
        return UnitValue(self.gamma * self.particle_rest_energy_eV, "eV")

    @property
    def Ex(self) -> UnitValue:
        """sqrt(E0^2 + cpx^2) [eV]."""
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpx**2), "eV")

    @property
    def Ey(self) -> UnitValue:
        """sqrt(E0^2 + cpy^2) [eV]."""
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpy**2), "eV")

    @property
    def Ez(self) -> UnitValue:
        """sqrt(E0^2 + cpz^2) [eV]."""
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpz**2), "eV")

    @property
    def Bx(self) -> UnitValue:
        """Horizontal relativistic beta."""
        return UnitValue(self.cpx / self.energy, "")

    @property
    def By(self) -> UnitValue:
        """Vertical relativistic beta."""
        return UnitValue(self.cpy / self.energy, "")

    @property
    def Bz(self) -> UnitValue:
        """Longitudinal relativistic beta."""
        return UnitValue(self.cpz / self.energy, "")

    @computed_field
    @property
    def Q(self) -> UnitValue:
        """Bunch charge [C]."""
        return UnitValue(self.total_charge, "C")

    def set_total_charge(self, q: float) -> None:
        """
        Set the bunch charge, sharing it equally between the macroparticles.

        Parameters
        ----------
        q: float
            Bunch charge [C]
        """
        self.total_charge = UnitValue(q, units="C")
        particle_q = q / (len(self.x))
        self.charge = UnitValue(np.full(len(self.x), particle_q), units="C")

    @property
    def kinetic_energy(self) -> UnitValue:
        """Kinetic energy [J]."""
        if self.particle_rest_energy is None:
            self.particle_rest_energy = self.particle_rest_energy_eV * constants.elementary_charge
        E0 = np.array(self.particle_rest_energy)
        cp = np.array(self.cp) * constants.elementary_charge
        return UnitValue(np.sqrt(E0**2 + cp**2) - E0, "J")

    @property
    def mean_energy(self) -> UnitValue:
        """Mean :attr:`kinetic_energy` [J]."""
        return UnitValue(np.mean(self.kinetic_energy), "J")

    def computeCorrelations(
        self, x: UnitValue | np.ndarray, y: UnitValue | np.ndarray
    ) -> tuple:
        """
        Covariances (cov(x, x), cov(x, y), cov(y, y)); see :meth:`covariance`.

        Returns
        -------
        tuple
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
        Transform a phase-space plane to the given Twiss parameters and emittance.

        Dispersion is subtracted from ``x`` and ``xp`` (in place) before matching.

        Parameters
        ----------
        x, xp: UnitValue or np.ndarray
            Position and angle
        beta, alpha: UnitValue or float or bool
            Target Twiss parameters; False keeps the beam's own
        nEmit: UnitValue or float or bool
            Target normalised emittance; False keeps the beam's own

        Returns
        -------
        tuple
            The transformed (x, xp)
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
        Rematch the horizontal plane to the given Twiss parameters and emittance.

        Parameters
        ----------
        beta, alpha: UnitValue or float
            Target Twiss parameters; give both or neither (warns if only one)
        nEmit: UnitValue or float, optional
            Target normalised emittance
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
        Rematch the vertical plane to the given Twiss parameters and emittance.

        Parameters
        ----------
        beta, alpha: UnitValue or float or bool
            Target Twiss parameters; give both or neither (warns if only one)
        nEmit: UnitValue or float or bool
            Target normalised emittance; False keeps the beam's own
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
        As :meth:`performTransformation`, but with the starting Twiss and emittance taken from one slice.

        Parameters
        ----------
        xslice, xpslice: UnitValue or np.ndarray
            Position and angle of the reference slice
        x, xp: UnitValue or np.ndarray
            Position and angle of the whole beam
        beta, alpha: UnitValue or float, optional
            Target Twiss parameters; None keeps the slice's own
        nEmit: UnitValue or float or bool, optional
            Target normalised emittance; False keeps the slice's own

        Returns
        -------
        tuple
            The transformed (x, xp)
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
        Rematch the horizontal plane so the peak-current slice has the given Twiss and emittance.

        Parameters
        ----------
        beta, alpha: UnitValue or float
            Target Twiss parameters
        nEmit: UnitValue or float or bool
            Target normalised emittance; False keeps the slice's own
        """
        peakIPosition = self.slice.slice_max_peak_current_slice
        xslice = self.slice.slice_data(self.x)[peakIPosition]
        xpslice = self.slice.slice_data(self.xp)[peakIPosition]
        x, xp = self.performTransformationPeakISlice(
            xslice, xpslice, self.x, self.xp, beta, alpha, nEmit
        )
        self.x = UnitValue(x, "m")

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
        Rematch the vertical plane so the peak-current slice has the given Twiss and emittance.

        Parameters
        ----------
        beta, alpha: UnitValue or float
            Target Twiss parameters
        nEmit: UnitValue or float or bool
            Target normalised emittance; False keeps the slice's own
        """
        peakIPosition = self.slice.slice_max_peak_current_slice
        yslice = self.slice.slice_data(self.y)[peakIPosition]
        ypslice = self.slice.slice_data(self.yp)[peakIPosition]
        y, yp = self.performTransformationPeakISlice(
            yslice, ypslice, self.y, self.yp, beta, alpha, nEmit
        )
        self.y = UnitValue(y, "m")

        cpz = self.cp / np.sqrt(self.xp**2 + yp**2 + 1)
        cpx = self.xp * cpz
        cpy = yp * cpz
        self.px = UnitValue(cpx * self.q_over_c, "kg*m/s")
        self.py = UnitValue(cpy * self.q_over_c, "kg*m/s")
        self.pz = UnitValue(cpz * self.q_over_c, "kg*m/s")
