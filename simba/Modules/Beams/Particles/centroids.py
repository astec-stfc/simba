"""Beam centroids of a particle distribution."""
import numpy as np
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from ...units import UnitValue
from typing import Dict

class centroids(BaseModel):
    """Centroids (means) of a particle distribution."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam = beam

    def model_dump(self, *args, **kwargs) -> Dict:
        # Only include computed fields
        computed_keys = set(self.__pydantic_decorators__.computed_fields)
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def mean_x(self) -> UnitValue:
        """Mean x in m."""
        return self.Cx

    @computed_field
    @property
    def mean_y(self) -> UnitValue:
        """Mean y in m."""
        return self.Cy

    @computed_field
    @property
    def mean_t(self) -> UnitValue:
        """Mean t in s."""
        return self.Ct

    @computed_field
    @property
    def mean_z(self) -> UnitValue:
        """Mean z in m."""
        return self.Cz

    @computed_field
    @property
    def mean_cpx(self) -> UnitValue:
        """Mean horizontal momentum in eV/c."""
        return self.Cpx

    @computed_field
    @property
    def mean_cpy(self) -> UnitValue:
        """Mean vertical momentum in eV/c."""
        return self.Cpy

    @computed_field
    @property
    def mean_cpz(self) -> UnitValue:
        """Mean longitudinal momentum in eV/c."""
        return self.Cpz

    @computed_field
    @property
    def mean_px(self) -> UnitValue:
        """Mean horizontal momentum in kg*m/s."""
        return np.mean(self.beam.px)

    @computed_field
    @property
    def mean_py(self) -> UnitValue:
        """Mean vertical momentum in kg*m/s."""
        return np.mean(self.beam.py)

    @computed_field
    @property
    def mean_pz(self) -> UnitValue:
        """Mean longitudinal momentum in kg*m/s."""
        return np.mean(self.beam.pz)

    @computed_field
    @property
    def mean_energy(self) -> UnitValue:
        """Mean total energy in eV."""
        return self.CEn

    @computed_field
    @property
    def mean_gamma(self) -> UnitValue:
        """Mean Lorentz factor."""
        return self.Cgamma

    @computed_field
    @property
    def mean_cp(self) -> UnitValue:
        """Mean total momentum in eV/c."""
        return self.Ccp

    @computed_field
    @property
    def Cx(self) -> UnitValue:
        """Mean x in m."""
        return np.mean(self.beam.x)

    @computed_field
    @property
    def Cy(self) -> UnitValue:
        """Mean y in m."""
        return np.mean(self.beam.y)

    @computed_field
    @property
    def Cz(self) -> UnitValue:
        """Mean z in m."""
        return np.mean(self.beam.z)

    @computed_field
    @property
    def Ct(self) -> UnitValue:
        """Mean t in s."""
        return np.mean(self.beam.t)

    @computed_field
    @property
    def Cp(self) -> UnitValue:
        """Mean total momentum in eV/c."""
        return np.mean(self.beam.cp)

    @computed_field
    @property
    def Cpx(self) -> UnitValue:
        """Mean horizontal momentum in eV/c."""
        return np.mean(self.beam.cpx)

    @computed_field
    @property
    def Cpy(self) -> UnitValue:
        """Mean vertical momentum in eV/c."""
        return np.mean(self.beam.cpy)

    @computed_field
    @property
    def Cpz(self) -> UnitValue:
        """Mean longitudinal momentum in eV/c."""
        return np.mean(self.beam.cpz)

    @computed_field
    @property
    def Cxp(self) -> UnitValue:
        """Mean horizontal angle in rad."""
        return np.mean(self.beam.xp)

    @computed_field
    @property
    def Cyp(self) -> UnitValue:
        """Mean vertical angle in rad."""
        return np.mean(self.beam.yp)

    @computed_field
    @property
    def Cgamma(self) -> UnitValue:
        """Mean Lorentz factor."""
        return np.mean(self.beam.gamma)

    @computed_field
    @property
    def Ccp(self) -> UnitValue:
        """Mean total momentum in eV/c."""
        return np.mean(self.beam.cp)

    @computed_field
    @property
    def CEn(self) -> UnitValue:
        """Mean total energy in eV."""
        return UnitValue(np.mean(np.sqrt(self.beam.cp**2 + self.beam.particle_rest_energy_eV**2)), "eV")
