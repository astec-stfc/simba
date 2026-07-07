"""
Convert between SIMBA's generic ``beam`` object and RF-Track's ``Bunch6d``.

RF-Track is used in-process (build/track Python objects directly, no input
deck file) so this module converts objects in memory rather than
parsing/writing files, unlike ``Modules/Beams/astra.py``. See
``laura/RFTrack/PLAN.md`` ("Architecture decision") for why.

ponytail: unit-conversion factors here (SI momentum <-> MeV/c, seconds <->
mm/c) are derived from RF-Track's documented conventions
(``laura/RFTrack/RFTrack_API_notes.md`` §10) but have not been validated
against a real ``RF_Track`` install (not available in this environment yet —
see PROGRESS.md). Re-verify with a real round-trip once it is installed.
"""
import numpy as np

from .. import constants
from ..units import UnitValue


def _momentum_si_to_MeVc(p_si):
    """Convert SI momentum [kg*m/s] to MeV/c."""
    return p_si * constants.speed_of_light / constants.elementary_charge / 1e6


def _momentum_MeVc_to_si(p_mevc):
    """Convert MeV/c momentum to SI [kg*m/s]."""
    return p_mevc * 1e6 * constants.elementary_charge / constants.speed_of_light


def get_P_Q(self, charge: float = -1.0) -> float:
    """
    Return this beam's mean longitudinal momentum-over-charge [MV/c], for use
    as RF-Track's dipole (``SBend``) ``P_Q`` parameter.

    Required because RF-Track's ``SBend`` — unlike ``Quadrupole``/``Multipole``
    — does not support deferring this to ``autophase()`` time; verified
    against the real package that an unset/NaN value silently produces zero
    transmission (see ``rftrack_conversion.build_sbend``).

    Parameters
    ----------
    charge: float
        Particle charge in units of e (default -1, electrons).
    """
    return float(np.mean(_momentum_si_to_MeVc(self.cpz.val))) / charge


def beam_to_bunch6d(self, charge: float = -1.0):
    """
    Convert this SIMBA ``beam`` into an ``RF_Track.Bunch6d``.

    Parameters
    ----------
    charge: float
        Particle charge in units of e (default -1, electrons).

    Returns
    -------
    RF_Track.Bunch6d
    """
    from laura.translator.conversion_rules.codes.rftrack_conversion import get_rftrack

    rft = get_rftrack()

    mass_MeV = (
        float(np.mean(self.particle_mass))
        * constants.speed_of_light**2
        / constants.elementary_charge
        / 1e6
    )
    population = float(abs(np.sum(self.Q.val)) / constants.elementary_charge)

    x = self.x.in_units_of("mm")
    y = self.y.in_units_of("mm")
    cpx_MeVc = _momentum_si_to_MeVc(self.cpx.val)
    cpy_MeVc = _momentum_si_to_MeVc(self.cpy.val)
    cpz_MeVc = _momentum_si_to_MeVc(self.cpz.val)
    xp = (cpx_MeVc / cpz_MeVc) * 1e3  # rad -> mrad
    yp = (cpy_MeVc / cpz_MeVc) * 1e3
    p = np.sqrt(cpx_MeVc**2 + cpy_MeVc**2 + cpz_MeVc**2)
    t = (self.t.val - np.mean(self.t.val)) * constants.speed_of_light * 1e3  # s -> mm/c

    matrix = np.column_stack([x, xp, y, yp, t, p])
    return rft.Bunch6d(mass_MeV, population, charge, matrix)


def bunch6d_to_beam(self, bunch) -> None:
    """
    Update this SIMBA ``beam`` in place from a tracked ``RF_Track.Bunch6d``.

    Parameters
    ----------
    bunch: RF_Track.Bunch6d
        Bunch after tracking through an RF-Track ``Lattice``.
    """
    ps = bunch.get_phase_space("%x %xp %y %yp %t %P %m %Q %N")
    x, xp, y, yp, t, p, m, q, n = ps.T

    xp_rad = xp * 1e-3
    yp_rad = yp * 1e-3
    cpz_MeVc = p / np.sqrt(1 + xp_rad**2 + yp_rad**2)
    cpx_MeVc = cpz_MeVc * xp_rad
    cpy_MeVc = cpz_MeVc * yp_rad

    # NOTE: `cpx`/`cpy`/`cpz` are read-only computed properties (`px / q_over_c`
    # etc, see Modules/Beams/Particles/__init__.py) -- the real settable SI
    # momentum fields are `px`/`py`/`pz`, matching Modules/Beams/ocelot.py's
    # own `particle_array_to_beam` precedent (verified against a real
    # end-to-end simba.Framework().track() run; see PROGRESS.md).
    self._beam.x = UnitValue(x * 1e-3, units="m")
    self._beam.y = UnitValue(y * 1e-3, units="m")
    self._beam.px = UnitValue(_momentum_MeVc_to_si(cpx_MeVc), units="kg*m/s")
    self._beam.py = UnitValue(_momentum_MeVc_to_si(cpy_MeVc), units="kg*m/s")
    self._beam.pz = UnitValue(_momentum_MeVc_to_si(cpz_MeVc), units="kg*m/s")
    self._beam.t = UnitValue(t * 1e-3 / constants.speed_of_light, units="s")
    self._beam.particle_mass = UnitValue(
        m * 1e6 * constants.elementary_charge / constants.speed_of_light**2,
        units="kg",
    )
    self._beam.particle_charge = UnitValue(
        np.sign(q) * n * constants.elementary_charge, units="C"
    )
