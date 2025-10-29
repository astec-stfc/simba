"""
SIMBA Sigmas Module

This module calculates the sigmas of a particle distribution.

Classes:
    - :class:`~simba.Modules.Particles.sigmas.sigmas`: Sigma calculations.
"""
import numpy as np
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from ...units import UnitValue


class sigmas(BaseModel):
    """
    Class for calculating sigmas of a particle distribution.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super(sigmas, self).__init__(*args, **kwargs)
        self.beam = beam

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        computed_keys = {
            f for f in self.__pydantic_decorators__.computed_fields.keys()
        }
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def sigma_x(self) -> UnitValue:
        """
        Horizontal beam sigma <x^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-x
        """
        return self.Sx

    @computed_field
    @property
    def sigma_y(self) -> UnitValue:
        """
        Vertical beam sigma <y^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-y
        """
        return self.Sy

    @computed_field
    @property
    def sigma_t(self) -> UnitValue:
        """
        Temporal beam sigma <t^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-t
        """
        return self.St

    @computed_field
    @property
    def sigma_z(self) -> UnitValue:
        """
        Longitudinal beam sigma <x^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-z
        """
        return self.Sz

    @computed_field
    @property
    def sigma_px(self) -> UnitValue:
        """
        Horizontal momentum signa <px^2>

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-px
        """
        return np.sqrt(self.beam.covariance(self.beam.px, self.beam.px))

    @computed_field
    @property
    def sigma_py(self) -> UnitValue:
        """
        Vertical momentum signa <px^2>

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-py
        """
        return np.sqrt(self.beam.covariance(self.beam.py, self.beam.py))

    @computed_field
    @property
    def sigma_pz(self) -> UnitValue:
        """
        Longitudinal momentum signa <py^2>

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-pz
        """
        return np.sqrt(self.beam.covariance(self.beam.pz, self.beam.pz))

    @computed_field
    @property
    def sigma_cp(self) -> UnitValue:
        """
        Beam momentum spread.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Momentum spread
        """
        return self.momentum_spread

    @computed_field
    @property
    def sigma_cp_eV(self) -> UnitValue:
        """
        Beam momentum spread.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Momentum spread
        """
        return self.momentum_spread

    @computed_field
    @property
    def Sx(self) -> UnitValue:
        """
        Horizontal beam sigma <x^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-x
        """
        return np.sqrt(self.beam.covariance(self.beam.x, self.beam.x))

    @computed_field
    @property
    def Sy(self) -> UnitValue:
        """
        Vertical beam sigma <x^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-y
        """
        return np.sqrt(self.beam.covariance(self.beam.y, self.beam.y))

    @computed_field
    @property
    def Sz(self) -> UnitValue:
        """
        Longitudinal beam sigma <x^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-z
        """
        return np.sqrt(self.beam.covariance(self.beam.z, self.beam.z))

    @computed_field
    @property
    def St(self) -> UnitValue:
        """
        Temporal beam sigma <x^2>.

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-t
        """
        return np.sqrt(self.beam.covariance(self.beam.t, self.beam.t))

    @computed_field
    @property
    def momentum_spread(self) -> UnitValue:
        """
        Beam momentum spread

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            sigma-x
        """
        return np.std(self.beam.cp)
        # return self.beam.cp.std()/np.mean(self.beam.cp)

    @computed_field
    @property
    def linear_chirp_t_cpz(self) -> UnitValue:
        """
        Linear chirp of the beam as std(t) / (max(cpz) - min(cpz))

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Linear chirp t/cpz
        """
        return -1 * np.std(self.beam.t) / (max(self.beam.cpz) - min(self.beam.cpz))

    @computed_field
    @property
    def linear_chirp_t_pz(self) -> UnitValue:
        """
        Linear chirp of the beam as std(t) / (max(pz) - min(pz))

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Linear chirp t/pz
        """
        return -1 * np.std(self.beam.t) / (max(self.beam.pz) - min(self.beam.pz))

    @computed_field
    @property
    def linear_chirp_z(self) -> UnitValue:
        """
        Linear chirp of the beam as v_z * t / momentum_spread

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Linear chirp in z
        """
        return (
            -1
            * np.std(self.beam.Bz * self.beam.speed_of_light * self.beam.t)
            / self.momentum_spread
            / 100
        )
