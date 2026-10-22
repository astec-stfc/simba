"""ImpactX beams: ``(x, y, t, px, py, pt)`` about the reference particle, with
``t = c (t - t_ref)``, positive behind it, and ``pt = -(E - E_ref) / p_ref c``."""

import numpy as np

from .. import constants
from ..units import UnitValue


def write_impactx_beam(self, p0c: float, t0: float | None = None) -> dict:
    """
    The beam as ImpactX coordinates, keyed as ``add_n_particles`` takes them.

    Parameters
    ----------
    p0c: float
        Reference momentum [eV/c]
    t0: float, optional
        Reference time [s]; the beam's mean if not given

    Returns
    -------
    dict
        ``x``, ``y``, ``t``, ``px``, ``py``, ``pt`` and ``bunch_charge``, the
        total charge's magnitude [C], as ImpactX takes the sign from ``qm``
    """
    if t0 is None:
        t0 = np.mean(self.t.val)
    mass = float(np.mean(self.E0_eV))
    energy = np.hypot(np.asarray(self.cp.val, dtype=float), mass)
    return {
        "x": np.asarray(self.x.val, dtype=float),
        "y": np.asarray(self.y.val, dtype=float),
        "t": constants.speed_of_light * (np.asarray(self.t.val, dtype=float) - t0),
        "px": np.asarray(self.cpx.val, dtype=float) / p0c,
        "py": np.asarray(self.cpy.val, dtype=float) / p0c,
        "pt": (np.hypot(p0c, mass) - energy) / p0c,
        "bunch_charge": abs(float(self._beam.total_charge)),
    }


def read_impactx_beam(
    self, df, ids, p0c: float, t_reference: float, zstart: float = 0, s: float = 0,
    ref_index: int | None = None,
) -> None:
    """
    Set the beam from ImpactX's ``beam.to_df()``, dropping lost particles.

    Parameters
    ----------
    df: pandas.DataFrame
        ``beam.to_df()``
    ids: np.ndarray
        ``idcpu`` of every particle as the beam went in, in the beam's order
    p0c: float
        Reference momentum here [eV/c]
    t_reference: float
        The reference's time here [s]
    zstart: float
        z of the point [m]
    s: float
        s of the point, in the machine [m]
    ref_index: int, optional
        The reference particle's index in ``ids``
    """
    alive = np.isin(ids, df["idcpu"].to_numpy())
    df = df.set_index("idcpu").loc[ids[alive]]
    mass = float(np.mean(self._beam.particle_mass))
    charge = float(np.mean(self._beam.particle_charge))
    macro_charge = abs(float(self._beam.total_charge)) / len(self._beam.x)
    if ref_index is not None:
        ref_index = int(np.count_nonzero(alive[:ref_index])) if alive[ref_index] else None
    rest_energy = float(np.mean(self.E0_eV))
    self.code = "ImpactX"
    self.set_mass_and_charge(mass, charge, len(df))
    self._beam.x = UnitValue(df["position_x"].to_numpy(), units="m")
    self._beam.y = UnitValue(df["position_y"].to_numpy(), units="m")
    self._beam.t = UnitValue(
        t_reference + df["position_t"].to_numpy() / constants.speed_of_light, units="s"
    )
    energy = np.hypot(p0c, rest_energy) - df["momentum_t"].to_numpy() * p0c
    cp = np.sqrt(energy**2 - rest_energy**2)
    cpx, cpy = df["momentum_x"].to_numpy() * p0c, df["momentum_y"].to_numpy() * p0c
    self.set_momenta(cpx, cpy, np.sqrt(cp**2 - cpx**2 - cpy**2))
    self._beam.set_total_charge(np.sign(charge) * macro_charge * len(df))
    self._beam.nmacro = UnitValue(np.full(len(df), 1), units="")
    self._beam.status = UnitValue(np.full(len(df), 5))
    if ref_index is None:
        self.reference_particle_index = None
    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")
