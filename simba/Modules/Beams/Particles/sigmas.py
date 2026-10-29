"""RMS sizes and spreads of a particle distribution."""
import numpy as np
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from ...units import UnitValue


class sigmas(BaseModel):
    """RMS sizes and spreads of a particle distribution."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, beam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam = beam

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        computed_keys = set(self.__pydantic_decorators__.computed_fields)
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def sigma_x(self) -> UnitValue:
        """RMS x in m."""
        return self.Sx

    @computed_field
    @property
    def sigma_y(self) -> UnitValue:
        """RMS y in m."""
        return self.Sy

    @computed_field
    @property
    def sigma_t(self) -> UnitValue:
        """RMS t in s."""
        return self.St

    @computed_field
    @property
    def sigma_z(self) -> UnitValue:
        """RMS z in m."""
        return self.Sz

    @computed_field
    @property
    def sigma_px(self) -> UnitValue:
        """RMS px in kg*m/s."""
        return np.sqrt(self.beam.covariance(self.beam.px, self.beam.px))

    @computed_field
    @property
    def sigma_py(self) -> UnitValue:
        """RMS py in kg*m/s."""
        return np.sqrt(self.beam.covariance(self.beam.py, self.beam.py))

    @computed_field
    @property
    def sigma_pz(self) -> UnitValue:
        """RMS pz in kg*m/s."""
        return np.sqrt(self.beam.covariance(self.beam.pz, self.beam.pz))

    @computed_field
    @property
    def sigma_cp(self) -> UnitValue:
        """Momentum spread in eV/c; alias of :attr:`momentum_spread`."""
        return self.momentum_spread

    @computed_field
    @property
    def sigma_cp_eV(self) -> UnitValue:
        """Momentum spread in eV/c; alias of :attr:`momentum_spread`."""
        return self.momentum_spread

    @computed_field
    @property
    def Sx(self) -> UnitValue:
        """RMS x in m."""
        return np.sqrt(self.beam.covariance(self.beam.x, self.beam.x))

    @computed_field
    @property
    def Sy(self) -> UnitValue:
        """RMS y in m."""
        return np.sqrt(self.beam.covariance(self.beam.y, self.beam.y))

    @computed_field
    @property
    def Sz(self) -> UnitValue:
        """RMS z in m."""
        return np.sqrt(self.beam.covariance(self.beam.z, self.beam.z))

    @computed_field
    @property
    def St(self) -> UnitValue:
        """RMS t in s."""
        return np.sqrt(self.beam.covariance(self.beam.t, self.beam.t))

    @computed_field
    @property
    def momentum_spread(self) -> UnitValue:
        """Standard deviation of cp in eV/c."""
        return np.std(self.beam.cp)

    @computed_field
    @property
    def linear_chirp_t_cpz(self) -> UnitValue:
        """Linear chirp, -std(t) / (max(cpz) - min(cpz))."""
        return -1 * np.std(self.beam.t) / (max(self.beam.cpz) - min(self.beam.cpz))

    @computed_field
    @property
    def linear_chirp_t_pz(self) -> UnitValue:
        """Linear chirp, -std(t) / (max(pz) - min(pz))."""
        return -1 * np.std(self.beam.t) / (max(self.beam.pz) - min(self.beam.pz))

    @computed_field
    @property
    def linear_chirp_z(self) -> UnitValue:
        """Linear chirp, -std(v_z * t) / momentum_spread / 100."""
        return (
            -1
            * np.std(self.beam.Bz * self.beam.speed_of_light * self.beam.t)
            / self.momentum_spread
            / 100
        )
