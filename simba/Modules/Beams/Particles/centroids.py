"""
SIMBA Centroids Module

This module calculates the beam centroids of a particle distribution.

Classes:
    - :class:`~simba.Modules.Particles.centroids.centroids`: Centroid calculations.
"""
import numpy as np
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from ...units import UnitValue
from ... import constants
from typing import Dict

class centroids(BaseModel):
    """
    Class for calculating centroids of a particle distribution.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super(centroids, self).__init__(*args, **kwargs)
        self.beam = beam

    def model_dump(self, *args, **kwargs) -> Dict:
        # Only include computed fields
        computed_keys = {
            f for f in self.__pydantic_decorators__.computed_fields.keys()
        }
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def mean_x(self) -> UnitValue:
        """
        Mean of horizontal distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of x
        """
        return self.Cx

    @computed_field
    @property
    def mean_y(self) -> UnitValue:
        """
        Mean of vertical distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of y
        """
        return self.Cy

    @computed_field
    @property
    def mean_t(self) -> UnitValue:
        """
        Mean of temporal distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of t
        """
        return self.Ct

    @computed_field
    @property
    def mean_z(self) -> UnitValue:
        """
        Mean of longitudinal distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of t
        """
        return self.Cz

    @computed_field
    @property
    def mean_cpx(self) -> UnitValue:
        """
        Mean of horizontal momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cpx
        """
        return self.Cpx

    @computed_field
    @property
    def mean_cpy(self) -> UnitValue:
        """
        Mean of vertical momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cpy
        """
        return self.Cpy

    @computed_field
    @property
    def mean_cpz(self) -> UnitValue:
        """
        Mean of longitudinal momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cpz
        """
        return self.Cpz

    @computed_field
    @property
    def mean_px(self) -> UnitValue:
        """
        Mean of horizontal momentum in kg*m/s

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of px
        """
        return np.mean(self.beam.px)

    @computed_field
    @property
    def mean_py(self) -> UnitValue:
        """
        Mean of vertical momentum in kg*m/s

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of py
        """
        return np.mean(self.beam.py)

    @computed_field
    @property
    def mean_pz(self) -> UnitValue:
        """
        Mean of longitudinal momentum in kg*m/s

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of pz
        """
        return np.mean(self.beam.pz)

    @computed_field
    @property
    def mean_energy(self) -> UnitValue:
        """
        Mean of beam energy in eV

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of E
        """
        return self.CEn

    @computed_field
    @property
    def mean_gamma(self) -> UnitValue:
        """
        Mean relativistic Lorentz factor

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of gamma
        """
        return self.Cgamma

    @computed_field
    @property
    def mean_cp(self) -> UnitValue:
        """
        Mean of total momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cp
        """
        return self.Ccp

    @computed_field
    @property
    def Cx(self) -> UnitValue:
        """
        Mean of horizontal distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of x
        """
        return np.mean(self.beam.x)

    @computed_field
    @property
    def Cy(self) -> UnitValue:
        """
        Mean of vertical distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of y
        """
        return np.mean(self.beam.y)

    @computed_field
    @property
    def Cz(self) -> UnitValue:
        """
        Mean of longitudinal distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of z
        """
        return np.mean(self.beam.z)

    @computed_field
    @property
    def Ct(self) -> UnitValue:
        """
        Mean of temporal distribution

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of t
        """
        return np.mean(self.beam.t)

    @computed_field
    @property
    def Cp(self) -> UnitValue:
        """
        Mean of total momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cp
        """
        return np.mean(self.beam.cp)

    @computed_field
    @property
    def Cpx(self) -> UnitValue:
        """
        Mean of horizontal momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cpx
        """
        return np.mean(self.beam.cpx)

    @computed_field
    @property
    def Cpy(self) -> UnitValue:
        """
        Mean of vertical momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cpy
        """
        return np.mean(self.beam.cpy)

    @computed_field
    @property
    def Cpz(self) -> UnitValue:
        """
        Mean of longitudinal momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of cpz
        """
        return np.mean(self.beam.cpz)

    @computed_field
    @property
    def Cxp(self) -> UnitValue:
        """
        Mean of horizontal angle in rad

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of xp
        """
        return np.mean(self.beam.xp)

    @computed_field
    @property
    def Cyp(self) -> UnitValue:
        """
        Mean of vertical angle in rad

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of yp
        """
        return np.mean(self.beam.yp)

    @computed_field
    @property
    def Cgamma(self) -> UnitValue:
        """
        Mean of relativistic Lorentz factor

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of gamma
        """
        return np.mean(self.beam.gamma)

    @computed_field
    @property
    def Ccp(self) -> UnitValue:
        """
        Mean of total momentum in eV/c

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean of x
        """
        return np.mean(self.beam.cp)

    @computed_field
    @property
    def CEn(self) -> UnitValue:
        """
        Mean beam energy in eV

        Returns
        -------
        :class:`~simba.Modules.units.UnitValue`
            Mean energy
        """
        return UnitValue(np.mean(self.beam.cp + self.beam.particle_rest_energy_eV), "eV")
