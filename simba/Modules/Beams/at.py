"""pyAT beams: a 6xN array of ``(x, px, y, py, delta, ct)``, with ``ct``
positive behind the reference."""

import numpy as np
from .. import constants
from ..units import UnitValue


def write_at_beam(self, p0c: float, t0: float | None = None) -> np.ndarray:
    """
    The beam as pyAT coordinates, ``ct = beta0 c (t - t0)``.

    Parameters
    ----------
    p0c: float
        Reference momentum [eV/c]
    t0: float, optional
        Reference time [s]; the beam's mean if not given

    Returns
    -------
    np.ndarray
        Fortran-ordered, as pyAT tracks in place
    """
    if t0 is None:
        t0 = np.mean(self.t.val)
    beta0 = p0c / np.hypot(p0c, np.mean(self.E0_eV))
    return np.asfortranarray([
        self.x.val,
        self.cpx.val / p0c,
        self.y.val,
        self.cpy.val / p0c,
        self.cp.val / p0c - 1,
        beta0 * constants.speed_of_light * (self.t.val - t0),
    ], dtype=float)


def read_at_beam(
    self, r, p0c: float, zstart: float = 0, s: float = 0,
    ref_index: int | None = None, t_reference: float | None = None,
) -> None:
    """
    Set the beam from pyAT coordinates, dropping lost (NaN) particles.

    Parameters
    ----------
    r: np.ndarray
        6xN pyAT coordinates
    p0c: float
        Reference momentum [eV/c]
    zstart: float
        z of the point [m]
    s: float
        s of the point, in the machine [m]
    ref_index: int, optional
        The reference particle's index in ``r``
    t_reference: float, optional
        The reference's time here [s]; ``s / c`` if not given
    """
    r = np.asarray(r, dtype=float)
    alive = ~np.isnan(r).any(axis=0)
    mass = float(np.mean(self._beam.particle_mass))
    charge = float(np.mean(self._beam.particle_charge))
    macro_charge = abs(float(self._beam.total_charge)) / len(self._beam.x)
    if ref_index is not None:
        ref_index = int(np.count_nonzero(alive[:ref_index])) if alive[ref_index] else None
    x, px, y, py, delta, ct = r[:, alive]
    self.code = "AT"
    self.set_mass_and_charge(mass, charge, len(x))
    self._beam.x = UnitValue(x, units="m")
    self._beam.y = UnitValue(y, units="m")
    beta0 = p0c / np.hypot(p0c, np.mean(self.E0_eV))
    if t_reference is None:
        t_reference = s / constants.speed_of_light
    self._beam.t = UnitValue(t_reference + ct / (beta0 * constants.speed_of_light), units="s")
    cpx, cpy, cp = px * p0c, py * p0c, (1 + delta) * p0c
    self.set_momenta(cpx, cpy, np.sqrt(cp**2 - cpx**2 - cpy**2))
    self._beam.set_total_charge(np.sign(charge) * macro_charge * len(x))
    self._beam.nmacro = UnitValue(np.full(len(x), 1), units="")
    self._beam.status = UnitValue(np.full(len(x), 5))
    if ref_index is None:
        self.reference_particle_index = None
    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")
