"""
Conversion between SIMBA beams and RF-Track's bunches.

A ``Bunch6d`` (``Lattice`` tracking, and what a ``Volume`` returns) row is
``[x, xp, y, yp, t, Pc, mass, Q, N, id]``: mm, mrad, mm/c, MeV/c, MeV/c^2, charge state
in e, real particles per macroparticle, and the row's index. A ``Bunch6dT`` (``Volume``
tracking, in time) row is ``[x, Px, y, Py, Z, Pz, mass, Q, N, t0]``: mm and MeV/c, with
``t0`` [mm/c] the particle's creation (emission) time.
"""
import numpy as np

from .. import constants
from ..units import UnitValue

_MM_PER_S = constants.speed_of_light * 1e3
"""RF-Track's time unit is mm/c."""


def _species_columns(self, n: int) -> list:
    """The ``mass, Q, N`` columns: MeV/c^2, charge state in e, particles per macroparticle."""
    charge = np.broadcast_to(self._beam.particle_charge.val, (n,)) / constants.elementary_charge
    population = abs(float(self.Q.val)) / constants.elementary_charge / n
    mass = np.broadcast_to(np.asarray(self.E0_eV, dtype=float), (n,)) * 1e-6
    return [mass, np.sign(charge), np.full(n, population)]


def write_rftrack_bunch(self):
    """
    This beam as an ``RF_Track.Bunch6d``, each particle keeping its charge and its index as ``id``.

    Returns
    -------
    RF_Track.Bunch6d
    """
    import RF_Track as rft

    n = len(self.x.val)
    cpz = self.cpz.val
    rows = np.column_stack([
        self.x.val * 1e3,
        self.cpx.val / cpz * 1e3,
        self.y.val * 1e3,
        self.cpy.val / cpz * 1e3,
        self.t.val * _MM_PER_S,
        self.cp.val * 1e-6,
        *_species_columns(self, n),
        np.arange(n, dtype=float),
    ])
    return rft.Bunch6d(rows)


def write_rftrack_cathode_bunch(self, ref_index=None):
    """
    This cathode beam as an ``RF_Track.Bunch6dT``: every particle on the cathode (``Z = 0``,
    the ``Volume``'s origin), emitted at its own time relative to the reference particle's.

    Parameters
    ----------
    ref_index: int, optional
        Index of the reference particle, emitted at ``t0 = 0``; else the mean time is.

    Returns
    -------
    RF_Track.Bunch6dT
    """
    import RF_Track as rft

    n = len(self.x.val)
    t = self.t.val
    t_ref = t[ref_index] if ref_index is not None else np.mean(t)
    rows = np.column_stack([
        self.x.val * 1e3,
        self.cpx.val * 1e-6,
        self.y.val * 1e3,
        self.cpy.val * 1e-6,
        np.zeros(n),
        self.cpz.val * 1e-6,
        *_species_columns(self, n),
        (t - t_ref) * _MM_PER_S,
    ])
    return rft.Bunch6dT(rows)


def interpret_rftrack_bunch(self, bunch, zstart: float = 0.0, s: float = 0.0, ref_index=None) -> None:
    """
    Fill this beam from the surviving particles of an ``RF_Track.Bunch6d``.

    Parameters
    ----------
    bunch: RF_Track.Bunch6d
        The tracked bunch; lost particles are dropped.
    zstart: float
        z of the bunch's reference time [m].
    s: float
        Path length of the bunch [m].
    ref_index: int, optional
        ``id`` (index in the incoming beam) of the reference particle; it becomes this beam's
        reference particle if it survived.
    """
    X = bunch.get_phase_space("%x %y %t %Px %Py %Pz %m %Q %N %id")
    x, y, t, cpx, cpy, cpz, mass, state, population, ids = X.T
    self.code = "RF-Track"
    self.set_mass_and_charge(
        mass * 1e6 * constants.elementary_charge / constants.speed_of_light**2,
        state * constants.elementary_charge,
        len(x),
    )
    self._beam.x = UnitValue(x * 1e-3, "m")
    self._beam.y = UnitValue(y * 1e-3, "m")
    self._beam.t = UnitValue(t / _MM_PER_S, "s")
    self.set_momenta(cpx * 1e6, cpy * 1e6, cpz * 1e6)
    self._beam.set_total_charge(
        UnitValue(np.sum(state * population) * constants.elementary_charge, "C")
    )
    self._beam.nmacro = UnitValue(np.full(len(x), 1))
    self._beam.status = UnitValue(np.full(len(x), 5))
    if ref_index is not None:
        found = np.flatnonzero(ids == ref_index)
        ref_index = int(found[0]) if found.size else None
    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")
