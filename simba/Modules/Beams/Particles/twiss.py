"""
SIMBA Twiss Module

This module calculates the Twiss properties of a particle distribution.

Classes:
    - :class:`~simba.Modules.Particles.twiss.twiss`: Twiss calculations.
"""
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
import numpy as np
from ...units import UnitValue
from typing import Dict


class twiss(BaseModel):
    """
    Class for calculating Twiss properties of a particle distribution.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super(twiss, self).__init__(*args, **kwargs)
        self.beam = beam

    # def __repr__(self):
    #     return repr(self.normal)

    def __setattr__(self, name, value):
        if name == "my_field":
            if not isinstance(value, str):
                raise ValueError("my_field must be a string")
        super().__setattr__(name, value)

    @property
    def normal(self) -> Dict:
        """
        Get the following Twiss parameters as a dictionary keyed by name:
        - :attr:`~normalized_horizontal_emittance`
        - :attr:`~horizontal_emittance`
        - :attr:`~alpha_x`
        - :attr:`~beta_x`
        - :attr:`~normalized_vertical_emittance`
        - :attr:`~vertical_emittance`
        - :attr:`~alpha_y`
        - :attr:`~beta_y`

        Returns
        -------
        Dict
            Dictionary of Twiss parameters
        """
        return {
            p: getattr(self, p)
            for p in (
                "normalized_horizontal_emittance",
                "horizontal_emittance",
                "alpha_x",
                "beta_x",
                "normalized_vertical_emittance",
                "vertical_emittance",
                "alpha_y",
                "beta_y",
            )
        }

    @property
    def corrected(self) -> Dict:
        """
        Get the following Twiss parameters (corrected for dispersion) as a dictionary keyed by name:
        - :attr:`~normalized_horizontal_emittance`
        - :attr:`~horizontal_emittance`
        - :attr:`~alpha_x`
        - :attr:`~beta_x`
        - :attr:`~normalized_vertical_emittance`
        - :attr:`~vertical_emittance`
        - :attr:`~alpha_y`
        - :attr:`~beta_y`

        Returns
        -------
        Dict
            Dictionary of corrected Twiss parameters
        """
        return {
            p + "_corrected": getattr(self, p + "_corrected")
            for p in (
                "normalized_horizontal_emittance",
                "horizontal_emittance",
                "alpha_x",
                "beta_x",
                "normalized_vertical_emittance",
                "vertical_emittance",
                "alpha_y",
                "beta_y",
            )
        }

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        computed_keys = {
            f for f in self.__pydantic_decorators__.computed_fields.keys()
        }
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def normalized_horizontal_emittance(self) -> UnitValue:
        """
        Get the normalized horizontal emittance;
        see :attr:`~simba.Modules.Beams.Particles.emittance.normalized_horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Normalized horizontal emittance
        """
        return self.beam.emittance.normalized_horizontal_emittance

    @computed_field
    @property
    def normalized_vertical_emittance(self) -> UnitValue:
        """
        Get the normalized vertical emittance;
        see :attr:`~simba.Modules.Beams.Particles.emittance.normalized_vertical_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Normalized vertical emittance
        """
        return self.beam.emittance.normalized_vertical_emittance

    @computed_field
    @property
    def horizontal_emittance(self) -> UnitValue:
        """
        Get the horizontal emittance;
        see :attr:`~simba.Modules.Beams.Particles.emittance.horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Horizontal emittance
        """
        return self.beam.emittance.horizontal_emittance

    @computed_field
    @property
    def vertical_emittance(self) -> UnitValue:
        """
        Get the vertical emittance;
        see :attr:`~simba.Modules.Beams.Particles.emittance.vertical_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Vertical emittance
        """
        return self.beam.emittance.vertical_emittance

    @computed_field
    @property
    def horizontal_emittance_corrected(self) -> UnitValue:
        """
        Get the horizontal emittance corrected for dispersion;
        see :attr:`~simba.Modules.Beams.Particles.emittance.horizontal_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected horizontal emittance
        """
        return self.beam.emittance.horizontal_emittance_corrected

    @computed_field
    @property
    def vertical_emittance_corrected(self) -> UnitValue:
        """
        Get the vertical emittance corrected for dispersion;
        see :attr:`~simba.Modules.Beams.Particles.emittance.vertical_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected vertical emittance
        """
        return self.beam.emittance.vertical_emittance_corrected

    @computed_field
    @property
    def beta_x(self) -> UnitValue:
        """
        Get the horizontal Twiss beta function as `covariance(x, x) / horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Horizontal Twiss beta
        """
        return (
            self.beam.covariance(self.beam.x, self.beam.x) / self.horizontal_emittance
        )

    @computed_field
    @property
    def alpha_x(self) -> UnitValue:
        """
        Get the horizontal Twiss alpha function as `-covariance(x, xp) / horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Horizontal Twiss alpha
        """
        return (
            -1
            * self.beam.covariance(self.beam.x, self.beam.xp)
            / self.horizontal_emittance
        )

    @computed_field
    @property
    def gamma_x(self) -> UnitValue:
        """
        Get the horizontal Twiss alpha function as `covariance(xp, xp) / horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Horizontal Twiss gamma
        """
        return (
            self.beam.covariance(self.beam.xp, self.beam.xp) / self.horizontal_emittance
        )

    @computed_field
    @property
    def beta_y(self) -> UnitValue:
        """
        Get the vertical Twiss beta function as `covariance(y, y) / horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Vertical Twiss beta
        """
        return self.beam.covariance(self.beam.y, self.beam.y) / self.vertical_emittance

    @computed_field
    @property
    def alpha_y(self) -> UnitValue:
        """
        Get the vertical Twiss beta function as `-covariance(y, yp) / horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Vertical Twiss alpha
        """
        return (
            -1
            * self.beam.covariance(self.beam.y, self.beam.yp)
            / self.vertical_emittance
        )

    @computed_field
    @property
    def gamma_y(self) -> UnitValue:
        """
        Get the vertical Twiss gamma function as `covariance(yp, yp) / horizontal_emittance`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Vertical Twiss gamma
        """
        return (
            self.beam.covariance(self.beam.yp, self.beam.yp) / self.vertical_emittance
        )

    @property
    def twiss_analysis(self) -> tuple:
        """
        Get the calculated Twiss parameters as a tuple.

        Returns
        -------
        tuple
            Calculated Twiss parameters in the following order:
            - :attr:`~simba.Modules.Beams.Particles.emittance.emittance.horizontal_emittance`
            - :attr:`~alpha_x`
            - :attr:`~beta_x`
            - :attr:`~gamma_x`
            - :attr:`~simba.Modules.Beams.Particles.emittance.emittance.vertical_emittance`
            - :attr:`~alpha_y`
            - :attr:`~beta_y`
            - :attr:`~gamma_y`
        """
        return (
            self.beam.emittance.horizontal_emittance,
            self.alpha_x,
            self.beta_x,
            self.gamma_x,
            self.beam.emittance.vertical_emittance,
            self.alpha_y,
            self.beta_y,
            self.gamma_y,
        )

    @computed_field
    @property
    def beta_x_corrected(self) -> UnitValue:
        """
        Get the horizontal Twiss beta corrected for dispersion as
        `covariance(xc, xc) / horizontal_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected horizontal Twiss beta
        """
        xc = self.beam.eta_corrected(self.beam.x)
        return self.beam.covariance(xc, xc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def alpha_x_corrected(self) -> UnitValue:
        """
        Get the horizontal Twiss beta corrected for dispersion as
        `-covariance(xc, xpc) / horizontal_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected horizontal Twiss alpha
        """
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return -1 * self.beam.covariance(xc, xpc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def gamma_x_corrected(self) -> UnitValue:
        """
        Get the horizontal Twiss gamma corrected for dispersion as
        `covariance(xpc, xpc) / horizontal_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected horizontal Twiss gamma
        """
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.beam.covariance(xpc, xpc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def beta_y_corrected(self) -> UnitValue:
        """
        Get the vertical Twiss beta corrected for dispersion as
        `covariance(yc, yc) / vertical_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected vertical Twiss beta
        """
        yc = self.beam.eta_corrected(self.beam.y)
        return self.beam.covariance(yc, yc) / self.vertical_emittance_corrected

    @computed_field
    @property
    def alpha_y_corrected(self) -> UnitValue:
        """
        Get the vertical Twiss alpha corrected for dispersion as
        `-covariance(yc, ypc) / vertical_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected vertical Twiss alpha
        """
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return -1 * self.beam.covariance(yc, ypc) / self.vertical_emittance_corrected

    @computed_field
    @property
    def gamma_y_corrected(self) -> UnitValue:
        """
        Get the vertical Twiss gamma corrected for dispersion as
        `covariance(ypc, ypc) / vertical_emittance_corrected`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Corrected vertical Twiss gamma
        """
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.beam.covariance(ypc, ypc) / self.vertical_emittance_corrected

    @property
    def twiss_analysis_corrected(self) -> tuple:
        """
        Get the calculated Twiss parameters corrected for dispersion as a tuple.

        Returns
        -------
        tuple
            Calculated Twiss parameters in the following order:
            - :attr:`~simba.Modules.Beams.Particles.emittance.emittance.horizontal_emittance_corrected`
            - :attr:`~alpha_x_corrected`
            - :attr:`~beta_x_corrected`
            - :attr:`~gamma_x_corrected`
            - :attr:`~simba.Modules.Beams.Particles.emittance.emittance.vertical_emittance_corrected`
            - :attr:`~alpha_y_corrected`
            - :attr:`~beta_y_corrected`
            - :attr:`~gamma_y_corrected`
        """
        return (
            self.horizontal_emittance_corrected,
            self.alpha_x_corrected,
            self.beta_x_corrected,
            self.gamma_x_corrected,
            self.vertical_emittance_corrected,
            self.alpha_y_corrected,
            self.beta_y_corrected,
            self.gamma_y_corrected,
        )

    @computed_field
    @property
    def eta_x(self) -> UnitValue:
        """
        Get the horizontal dispersion; see :func:`~calculate_etax`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Horizontal dispersion
        """
        return self.calculate_etax()[0]

    @computed_field
    @property
    def eta_xp(self) -> UnitValue:
        """
        Get the derivative of horizontal dispersion; see :func:`~calculate_etax`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Horizontal dispersion derivative
        """
        return self.calculate_etax()[1]

    def calculate_etax(self) -> tuple:
        """
        Get the horizontal dispersion and its derivative.

        Returns
        -------
        tuple
            - Horizontal dispersion
            - Derviative of horizontal dispersion
            - Mean of temporal distribution
        """
        p = self.beam.cpz
        pAve = np.mean(p)
        p = p / pAve - 1  # [(a / pAve) - 1 for a in p]
        S16, S66 = self.beam.covariance(self.beam.x, p), self.beam.covariance(p, p)
        eta1 = S16 / S66 if S66 else 0
        S26 = self.beam.covariance(self.beam.xp, p)
        etap1 = S26 / S66 if S66 else 0
        return eta1, etap1, np.mean(self.beam.t)

    @computed_field
    @property
    def eta_y(self) -> UnitValue:
        """
        Get the vertical dispersion; see :func:`~calculate_etay`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Vertical dispersion
        """
        return self.calculate_etay()[0]

    @computed_field
    @property
    def eta_yp(self) -> UnitValue:
        """
        Get the derivative of vertical dispersion; see :func:`~calculate_etay`.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Vertical dispersion derivative
        """
        return self.calculate_etay()[1]

    def calculate_etay(self) -> tuple:
        """
        Get the vertical dispersion and its derivative.

        Returns
        -------
        tuple
            - Vertical dispersion
            - Derviative of vertical dispersion
            - Mean of temporal distribution
        """
        p = self.beam.cpz
        pAve = np.mean(p)
        p = p / pAve - 1  # [(a / pAve) - 1 for a in p]
        S36, S66 = self.beam.covariance(self.beam.y, p), self.beam.covariance(p, p)
        eta1 = S36 / S66 if S66 else 0
        S46 = self.beam.covariance(self.beam.yp, p)
        etap1 = S46 / S66 if S66 else 0
        return eta1, etap1, np.mean(self.beam.t)
