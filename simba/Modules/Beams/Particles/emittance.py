"""
SIMBA Emittance Module

This module calculates the beam emittances of a particle distribution.
Multiple emittance definitions are provided.

For slice emittance calculations, see :class:`~simba.Modules.Beams.Particles.slice.slice`

Classes:
    - :class:`~simba.Modules.Particles.emittance.emittance`: Emittance calculations.
"""
import numpy as np
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from ...units import UnitValue


class emittance(BaseModel):
    """
    Class for calculating emittances of a particle distribution.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super(emittance, self).__init__(*args, **kwargs)
        self.beam = beam

    # def __repr__(self):
    #     return repr({p: self.emittance(p) for p in ("x", "y")})

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        computed_keys = {
            f for f in self.__pydantic_decorators__.computed_fields.keys()
        }
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def ex(self) -> UnitValue:
        """
        Horizontal emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance
        """
        return self.emittance("x")

    @computed_field
    @property
    def ey(self) -> UnitValue:
        """
        Vertical emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance
        """
        return self.emittance("y")

    @computed_field
    @property
    def enx(self) -> UnitValue:
        """
        Normalised horizontal emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance normalised
        """
        return self.normalized_emittance("x")

    @computed_field
    @property
    def eny(self) -> UnitValue:
        """
        Normalised vertical emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance normalised
        """
        return self.normalized_emittance("y")

    @computed_field
    @property
    def ecx(self) -> UnitValue:
        """
        Horizontal emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance corrected
        """
        return self.horizontal_emittance_corrected

    @computed_field
    @property
    def ecy(self) -> UnitValue:
        """
        Vertical emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance corrected
        """
        return self.vertical_emittance_corrected

    @computed_field
    @property
    def ecnx(self) -> UnitValue:
        """
        Normalised horizontal emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance corrected and normalised
        """
        return self.normalised_horizontal_emittance_corrected

    @computed_field
    @property
    def ecny(self) -> UnitValue:
        """
        Normalised vertical emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance corrected and normalised
        """
        return self.normalised_vertical_emittance_corrected

    def emittance_calc(
            self,
            x: UnitValue | np.ndarray,
            xp: UnitValue | np.ndarray,
            p: UnitValue | np.ndarray=None,
            units: str="m-rad"
    ) -> UnitValue:
        """
        Calculate the emittance from two arrays using
        :func:`~simba.Modules.Beams.Particles.Particles.covariance`

        Parameters
        ----------
        x: UnitValue | np.ndarray
            Spatial column
        xp: UnitValue | np.ndarray
            Angle column
        p: UnitValue | np.ndarray, optional
            Momentum column; if provided, normalise the emittance with respect to this
        units: str
            Unit value (deprecated?)

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Calculated emittance
        """
        cov_x = self.beam.covariance(x, x)
        cov_xp = self.beam.covariance(xp, xp)
        cov_x_xp = self.beam.covariance(x, xp)
        emit = (
            np.sqrt(cov_x * cov_xp - cov_x_xp**2)
            if (cov_x * cov_xp - cov_x_xp**2) > 0
            else 0
        )
        if p is not None:
            beta = np.mean(self.beam.Bz)
            gamma = np.mean(p) / (np.mean(self.beam.particle_rest_energy_eV) * beta)
            emit = gamma * emit
        return emit

    def normalized_emittance(
            self,
            plane: str="x",
            corrected: bool=False
    ) -> UnitValue:
        """
        Calculate the normalised emittance for the plane provided;
        see :func:`~emittance_calc`.

        Parameters
        ----------
        plane: str
            Name of the plane to calculate; must be one of [x, y, z]
        corrected: bool
            If true, correct with respect to dispersion

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The normalised emittance

        Raises
        ------
        ValueError
            If the plane provided is not one of [x, y, z]
        """
        if plane.lower() not in ['x', 'y', 'z']:
            raise ValueError("plane must be in [x, y, z] for normalized_emittance calculation")
        if corrected:
            return self.emittance_calc(
                getattr(self.beam, plane + "c"),
                getattr(self.beam, plane + "pc"),
                self.beam.cpz,
            )
        else:
            return self.emittance_calc(
                getattr(self.beam, plane),
                getattr(self.beam, plane + "p"),
                self.beam.cpz,
            )

    def emittance(
            self,
            plane: str="x",
            corrected: bool=False
    ) -> UnitValue:
        """
        Calculate the emittance for the plane provided;
        see :func:`~emittance_calc`.

        Parameters
        ----------
        plane: str
            Name of the plane to calculate; must be one of [x, y, z]
        corrected: bool
            If true, correct with respect to dispersion

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            The emittance

        Raises
        ------
        ValueError
            If the plane provided is not one of [x, y, z]
        """
        if plane.lower() not in ['x', 'y', 'z']:
            raise ValueError("plane must be in [x, y, z] for normalized_emittance calculation")
        if corrected:
            return self.emittance_calc(
                getattr(self.beam, plane + "c"), getattr(self.beam, plane + "pc"), None
            )
        else:
            return self.emittance_calc(
                getattr(self.beam, plane), getattr(self.beam, plane + "p"), None
            )

    @computed_field
    @property
    def normalized_horizontal_emittance(self) -> UnitValue:
        """
        Normalised horizontal emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance normalised
        """
        return self.emittance_calc(self.beam.x, self.beam.xp, self.beam.cp)

    @computed_field
    @property
    def normalized_vertical_emittance(self) -> UnitValue:
        """
        Normalised vertical emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance normalised
        """
        return self.emittance_calc(self.beam.y, self.beam.yp, self.beam.cp)

    @computed_field
    @property
    def horizontal_emittance(self) -> UnitValue:
        """
        Horizontal emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance
        """
        return self.emittance_calc(self.beam.x, self.beam.xp)

    @computed_field
    @property
    def vertical_emittance(self) -> UnitValue:
        """
        Vertical emittance of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance
        """
        return self.emittance_calc(self.beam.y, self.beam.yp)

    @computed_field
    @property
    def horizontal_emittance_90(self) -> UnitValue:
        """
        Horizontal emittance of 90% of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            90% x-emittance
        """
        alpha = self.beam.twiss.alpha_x
        beta = self.beam.twiss.beta_x
        gamma = self.beam.twiss.gamma_x
        emiti = (
            gamma * self.beam.x**2
            + 2 * alpha * self.beam.x * self.beam.xp
            + beta * self.beam.xp * self.beam.xp
        )
        return sorted(emiti)[int(0.9 * len(emiti) - 0.5)]

    @computed_field
    @property
    def normalized_horizontal_emittance_90(self) -> UnitValue:
        """
        Normalised horizontal emittance of 90% of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            90% x-emittance normalised
        """
        emit = self.horizontal_emittance_90
        return np.mean(self.beam.cp / self.beam.E0_eV) * emit

    @computed_field
    @property
    def vertical_emittance_90(self) -> UnitValue:
        """
        Vertical emittance of 90% of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            90% y-emittance
        """
        alpha = self.beam.twiss.alpha_y
        beta = self.beam.twiss.beta_y
        gamma = self.beam.twiss.gamma_y
        emiti = (
            gamma * self.beam.y**2
            + 2 * alpha * self.beam.y * self.beam.yp
            + beta * self.beam.yp * self.beam.yp
        )
        return sorted(emiti)[int(0.9 * len(emiti) - 0.5)]

    @computed_field
    @property
    def normalized_vertical_emittance_90(self) -> UnitValue:
        """
        Normalised vertical emittance of 90% of the beam in m-rad; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            90% y-emittance normalised
        """
        emit = self.vertical_emittance_90
        return np.mean(self.beam.cp / self.beam.E0_eV) * emit

    @computed_field
    @property
    def horizontal_emittance_corrected(self) -> UnitValue:
        """
        Horizontal emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance corrected
        """
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.emittance_calc(xc, xpc)

    @computed_field
    @property
    def vertical_emittance_corrected(self) -> UnitValue:
        """
        Vertical emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance corrected
        """
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.emittance_calc(yc, ypc)

    @computed_field
    @property
    def normalised_horizontal_emittance_corrected(self) -> UnitValue:
        """
        Normalised horizontal emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            x-emittance corrected and normalised
        """
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.emittance_calc(xc, xpc, self.beam.cp)

    @computed_field
    @property
    def normalised_vertical_emittance_corrected(self) -> UnitValue:
        """
        Normalised vertical emittance of the beam in m-rad corrected for dispersion; see :func:`~emittance_calc`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            y-emittance corrected and normalised
        """
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.emittance_calc(yc, ypc, self.beam.cp)
