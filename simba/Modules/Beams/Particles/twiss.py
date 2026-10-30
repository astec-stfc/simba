"""Twiss parameters of a particle distribution."""
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
import numpy as np
from ...units import UnitValue
from typing import Dict


class twiss(BaseModel):
    """Twiss parameters of a particle distribution, from its second moments."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam = beam

    @property
    def normal(self) -> Dict:
        """Emittances, alpha and beta in both planes, keyed by attribute name."""
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
        """As :attr:`normal`, but the dispersion-corrected ``*_corrected`` attributes."""
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
        computed_keys = set(self.__pydantic_decorators__.computed_fields)
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def normalized_horizontal_emittance(self) -> UnitValue:
        """See :attr:`~simba.Modules.Beams.Particles.emittance.emittance.normalized_horizontal_emittance`."""
        return self.beam.emittance.normalized_horizontal_emittance

    @computed_field
    @property
    def normalized_vertical_emittance(self) -> UnitValue:
        """See :attr:`~simba.Modules.Beams.Particles.emittance.emittance.normalized_vertical_emittance`."""
        return self.beam.emittance.normalized_vertical_emittance

    @computed_field
    @property
    def horizontal_emittance(self) -> UnitValue:
        """See :attr:`~simba.Modules.Beams.Particles.emittance.emittance.horizontal_emittance`."""
        return self.beam.emittance.horizontal_emittance

    @computed_field
    @property
    def vertical_emittance(self) -> UnitValue:
        """See :attr:`~simba.Modules.Beams.Particles.emittance.emittance.vertical_emittance`."""
        return self.beam.emittance.vertical_emittance

    @computed_field
    @property
    def horizontal_emittance_corrected(self) -> UnitValue:
        """See :attr:`~simba.Modules.Beams.Particles.emittance.emittance.horizontal_emittance_corrected`."""
        return self.beam.emittance.horizontal_emittance_corrected

    @computed_field
    @property
    def vertical_emittance_corrected(self) -> UnitValue:
        """See :attr:`~simba.Modules.Beams.Particles.emittance.emittance.vertical_emittance_corrected`."""
        return self.beam.emittance.vertical_emittance_corrected

    @computed_field
    @property
    def beta_x(self) -> UnitValue:
        """cov(x, x) / horizontal_emittance."""
        return (
            self.beam.covariance(self.beam.x, self.beam.x) / self.horizontal_emittance
        )

    @computed_field
    @property
    def alpha_x(self) -> UnitValue:
        """-cov(x, xp) / horizontal_emittance."""
        return (
            -1
            * self.beam.covariance(self.beam.x, self.beam.xp)
            / self.horizontal_emittance
        )

    @computed_field
    @property
    def gamma_x(self) -> UnitValue:
        """cov(xp, xp) / horizontal_emittance."""
        return (
            self.beam.covariance(self.beam.xp, self.beam.xp) / self.horizontal_emittance
        )

    @computed_field
    @property
    def beta_y(self) -> UnitValue:
        """cov(y, y) / vertical_emittance."""
        return self.beam.covariance(self.beam.y, self.beam.y) / self.vertical_emittance

    @computed_field
    @property
    def alpha_y(self) -> UnitValue:
        """-cov(y, yp) / vertical_emittance."""
        return (
            -1
            * self.beam.covariance(self.beam.y, self.beam.yp)
            / self.vertical_emittance
        )

    @computed_field
    @property
    def gamma_y(self) -> UnitValue:
        """cov(yp, yp) / vertical_emittance."""
        return (
            self.beam.covariance(self.beam.yp, self.beam.yp) / self.vertical_emittance
        )

    @property
    def twiss_analysis(self) -> tuple:
        """(ex, alpha_x, beta_x, gamma_x, ey, alpha_y, beta_y, gamma_y), geometric emittances."""
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
        """Dispersion-corrected beta_x: cov(xc, xc) / horizontal_emittance_corrected."""
        xc = self.beam.eta_corrected(self.beam.x)
        return self.beam.covariance(xc, xc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def alpha_x_corrected(self) -> UnitValue:
        """Dispersion-corrected alpha_x: -cov(xc, xpc) / horizontal_emittance_corrected."""
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return -1 * self.beam.covariance(xc, xpc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def gamma_x_corrected(self) -> UnitValue:
        """Dispersion-corrected gamma_x: cov(xpc, xpc) / horizontal_emittance_corrected."""
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.beam.covariance(xpc, xpc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def beta_y_corrected(self) -> UnitValue:
        """Dispersion-corrected beta_y: cov(yc, yc) / vertical_emittance_corrected."""
        yc = self.beam.eta_corrected(self.beam.y)
        return self.beam.covariance(yc, yc) / self.vertical_emittance_corrected

    @computed_field
    @property
    def alpha_y_corrected(self) -> UnitValue:
        """Dispersion-corrected alpha_y: -cov(yc, ypc) / vertical_emittance_corrected."""
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return -1 * self.beam.covariance(yc, ypc) / self.vertical_emittance_corrected

    @computed_field
    @property
    def gamma_y_corrected(self) -> UnitValue:
        """Dispersion-corrected gamma_y: cov(ypc, ypc) / vertical_emittance_corrected."""
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.beam.covariance(ypc, ypc) / self.vertical_emittance_corrected

    @property
    def twiss_analysis_corrected(self) -> tuple:
        """As :attr:`twiss_analysis`, but dispersion-corrected."""
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
        """Horizontal dispersion; see :meth:`calculate_etax`."""
        return self.calculate_etax()[0]

    @computed_field
    @property
    def eta_xp(self) -> UnitValue:
        """Horizontal dispersion derivative; see :meth:`calculate_etax`."""
        return self.calculate_etax()[1]

    def calculate_etax(self) -> tuple:
        """
        Horizontal dispersion from the correlation of x and xp with the fractional pz.

        Returns
        -------
        tuple
            (eta_x, eta_xp, mean t); the etas are 0 if pz has no spread
        """
        p = self.beam.cpz
        pAve = np.mean(p)
        p = p / pAve - 1
        S16, S66 = self.beam.covariance(self.beam.x, p), self.beam.covariance(p, p)
        eta1 = S16 / S66 if S66 else 0
        S26 = self.beam.covariance(self.beam.xp, p)
        etap1 = S26 / S66 if S66 else 0
        return eta1, etap1, np.mean(self.beam.t)

    @computed_field
    @property
    def eta_y(self) -> UnitValue:
        """Vertical dispersion; see :meth:`calculate_etay`."""
        return self.calculate_etay()[0]

    @computed_field
    @property
    def eta_yp(self) -> UnitValue:
        """Vertical dispersion derivative; see :meth:`calculate_etay`."""
        return self.calculate_etay()[1]

    def calculate_etay(self) -> tuple:
        """
        Vertical dispersion from the correlation of y and yp with the fractional pz.

        Returns
        -------
        tuple
            (eta_y, eta_yp, mean t); the etas are 0 if pz has no spread
        """
        p = self.beam.cpz
        pAve = np.mean(p)
        p = p / pAve - 1
        S36, S66 = self.beam.covariance(self.beam.y, p), self.beam.covariance(p, p)
        eta1 = S36 / S66 if S66 else 0
        S46 = self.beam.covariance(self.beam.yp, p)
        etap1 = S46 / S66 if S66 else 0
        return eta1, etap1, np.mean(self.beam.t)
