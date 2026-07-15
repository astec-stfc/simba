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
    # NOTE: same double-conversion pitfall as `beam_to_bunch6d` -- `pz` (real
    # SI kg*m/s), not `cpz` (already-computed eV/c), is `_momentum_si_to_MeVc`'s
    # expected input. Also divide by `charge` [e] (the parameter this function
    # takes for exactly that purpose), not `constants.elementary_charge` [C] --
    # dividing by the latter turns a ~35 MV/c answer into ~1e28. Verified
    # against a real end-to-end simba.Framework().track() run.
    return float(np.mean(_momentum_si_to_MeVc(self.pz.val))) / charge


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

    # NOTE: `in_units_of` only recognises full-word prefixes ("milli", "kilo",
    # ...) -- short forms like "mm" silently no-op (its dash-appending logic
    # can't match SHORT_PREFIX_FACTOR's dash-less keys), so this used to feed
    # RF-Track x/y still in metres while labelling them mm (1000x too small),
    # corrupting the x-xp/y-yp correlation RF-Track's alpha/beta come from.
    # Verified against a real end-to-end simba.Framework().track() run.
    x = self.x.in_units_of("milli")
    y = self.y.in_units_of("milli")
    # NOTE: use `px`/`py`/`pz` (the real settable SI momentum fields, kg*m/s)
    # here, not `cpx`/`cpy`/`cpz` -- those are already-computed eV/c properties
    # (`px / q_over_c`, see Modules/Beams/Particles/__init__.py), so feeding
    # them into `_momentum_si_to_MeVc` (which expects true SI input) silently
    # double-converted every momentum by a further `1/q_over_c` (~1.87e27):
    # verified against a real end-to-end simba.Framework().track() run, where
    # RF-Track's output momenta came out ~27 orders of magnitude too large.
    cpx_MeVc = _momentum_si_to_MeVc(self.px.val)
    cpy_MeVc = _momentum_si_to_MeVc(self.py.val)
    cpz_MeVc = _momentum_si_to_MeVc(self.pz.val)
    xp = (cpx_MeVc / cpz_MeVc) * 1e3  # rad -> mrad
    yp = (cpy_MeVc / cpz_MeVc) * 1e3
    p = np.sqrt(cpx_MeVc**2 + cpy_MeVc**2 + cpz_MeVc**2)
    t = (self.t.val - np.mean(self.t.val)) * constants.speed_of_light * 1e3  # s -> mm/c

    matrix = np.column_stack([x, xp, y, yp, t, p])
    return rft.Bunch6d(mass_MeV, population, charge, matrix)


def beam_to_bunch6dt(self, charge: float = -1.0):
    """
    Convert this SIMBA ``beam`` into a time-based ``RF_Track.Bunch6dT``, for
    tracking through a ``Volume`` (time integration) -- the environment RF-Track
    recommends for space-charge-dominated, cathode/emission regimes (manual
    §5.1.1). The inverse of :func:`bunch6dt_to_beam`.

    Uses the full ``[X Px Y Py Z Pz MASS Q N T0]`` constructor (manual §2.2.2)
    rather than the 6-column form, so each particle's emission/creation time is
    carried in ``T0`` -- essential for a photocathode bunch, where particles are
    released over a finite emission time and space charge (with mirror charges)
    acts throughout emission.

    Parameters
    ----------
    charge: float
        Single-particle charge state [e] (default -1, electrons); matches
        :func:`beam_to_bunch6d`.

    Returns
    -------
    RF_Track.Bunch6dT
    """
    from laura.translator.conversion_rules.codes.rftrack_conversion import get_rftrack

    rft = get_rftrack()

    n = len(self.x.val)
    mass_MeV = (
        float(np.mean(self.particle_mass))
        * constants.speed_of_light**2
        / constants.elementary_charge
        / 1e6
    )
    population = float(abs(np.sum(self.Q.val)) / constants.elementary_charge)

    # Positions [mm] and Cartesian momenta [MeV/c] -- same unit conventions
    # (and the `in_units_of("milli")` / `px` (not `cpx`) pitfalls) as
    # `beam_to_bunch6d`; see the NOTEs there.
    X = self.x.in_units_of("milli")
    Y = self.y.in_units_of("milli")
    Z = self.z.in_units_of("milli")
    Px = _momentum_si_to_MeVc(self.px.val)
    Py = _momentum_si_to_MeVc(self.py.val)
    Pz = _momentum_si_to_MeVc(self.pz.val)
    # T0 (creation time) [mm/c] -- inverse of bunch6dt_to_beam's `t0*1e-3/c`.
    T0 = self.t.val * constants.speed_of_light * 1e3

    matrix = np.column_stack([
        X, Px, Y, Py, Z, Pz,
        np.full(n, mass_MeV),      # MASS [MeV/c^2]
        np.full(n, charge),        # Q [e]
        np.full(n, population / n),  # N (real particles per macro-particle)
        T0,
    ])
    return rft.Bunch6dT(matrix)


def bunch6dt_to_beam(self, bunch, s: float = 0.0) -> None:
    """
    Update this SIMBA ``beam`` in place from an ``RF_Track.Bunch6dT``.

    Used by :class:`~simba.Codes.Generators.rftrack.RFTrackGenerator` for the
    photocathode/emission bunch produced by ``RF_Track.Bunch6dT_Generator``.
    Unlike :func:`bunch6d_to_beam`, this reads a *time-based* bunch directly
    (Bunch6dT -> Bunch6d conversion is not supported by RF-Track, see
    ``RFTrack/RFTrack_API_notes.md`` §"Conversion"): Cartesian momenta ``Px/Py/Pz``
    [MeV/c] and the per-particle creation time ``t0`` [mm/c] are taken straight
    from the bunch rather than reconstructed from angles.

    Parameters
    ----------
    bunch: RF_Track.Bunch6dT
        Bunch emitted by the generator.
    s: float
        Arc-length position [m] stored on ``beam.s`` (0 at the cathode).
    """
    ps = bunch.get_phase_space("%X %Y %Z %Px %Py %Pz %t0 %m %Q %N")
    if ps is None or len(ps) == 0:
        raise ValueError(
            "Cannot convert empty Bunch6dT to beam. Check RF-Track generator "
            "settings and space-charge configuration for particle losses."
        )
    x, y, z, px, py, pz, t0, m, q, n = ps.T

    self._beam.x = UnitValue(x * 1e-3, units="m")
    self._beam.y = UnitValue(y * 1e-3, units="m")
    self._beam.z = UnitValue(z * 1e-3, units="m")
    self._beam.px = UnitValue(_momentum_MeVc_to_si(px), units="kg*m/s")
    self._beam.py = UnitValue(_momentum_MeVc_to_si(py), units="kg*m/s")
    self._beam.pz = UnitValue(_momentum_MeVc_to_si(pz), units="kg*m/s")
    # %t0 (creation time) is in mm/c; -> seconds. This is the emission-time
    # spread of a cathode bunch, matching what ASTRA/GPT generators store as t.
    self._beam.t = UnitValue(t0 * 1e-3 / constants.speed_of_light, units="s")
    self._beam.particle_mass = UnitValue(
        m * 1e6 * constants.elementary_charge / constants.speed_of_light**2,
        units="kg",
    )
    self._beam.particle_rest_energy = UnitValue(
        self._beam.particle_mass * constants.speed_of_light**2, units="J"
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        self._beam.particle_rest_energy / constants.elementary_charge, units="eV/c"
    )
    self._beam.particle_charge = UnitValue(q * constants.elementary_charge, units="C")
    self._beam.nmacro = UnitValue(n)
    self._beam.charge = UnitValue(q * n * constants.elementary_charge, units="C")
    self._beam.total_charge = UnitValue(np.sum(self._beam.charge), units="C")
    self._beam.status = UnitValue(np.full(len(x), 5))
    self._beam.s = UnitValue(s, units="m")


def bunch6d_to_beam(self, bunch, zstart: float = 0.0, s: float = None) -> None:
    """
    Update this SIMBA ``beam`` in place from a tracked ``RF_Track.Bunch6d``.

    Parameters
    ----------
    bunch: RF_Track.Bunch6d
        Bunch after tracking through an RF-Track ``Lattice``.
    zstart: float
        Absolute Cartesian z position [m] of this bunch snapshot (e.g. the
        screen or lattice-end position it was taken at) -- matches the
        ``zstart`` convention of ``Modules/Beams/ocelot.py``'s
        ``particle_array_to_beam``. Used to offset the per-particle z.
    s: float, optional
        Arc-length position [m] of this bunch snapshot (e.g. the LAURA-
        resolved ``physical.s`` of the screen/lattice-end element), stored on
        ``beam.s``. Defaults to ``zstart`` when not given (matches ``s=0``
        default fallback used before this parameter existed).
    """
    ps = bunch.get_phase_space("%x %xp %y %yp %t %P %m %Q %N")
    if ps is None or len(ps) == 0:
        raise ValueError(
            f"Cannot convert empty Bunch6d to beam at zstart={zstart}. "
            "All particles were lost during tracking; check space-charge "
            "settings, apertures, or lattice configuration."
        )
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
    self._beam.particle_rest_energy = UnitValue(
        self._beam.particle_mass * constants.speed_of_light**2, units="J"
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        self._beam.particle_rest_energy / constants.elementary_charge, units="eV/c"
    )
    # `particle_charge` is the charge of a single real particle [C] (e.g. -e
    # for an electron); `nmacro`/`charge` carry the macro-particle population
    # and its resulting weighted charge -- matching the distinction
    # Modules/Beams/gdf.py's `particle_charge`/`set_total_charge(nmacro *
    # particle_charge)` draws (RF-Track's `%Q` is charge state in units of e,
    # `%N` the real-particle population represented by each macro-particle).
    self._beam.particle_charge = UnitValue(q * constants.elementary_charge, units="C")
    self._beam.nmacro = UnitValue(n)
    self._beam.charge = UnitValue(q * n * constants.elementary_charge, units="C")
    self._beam.total_charge = UnitValue(np.sum(self._beam.charge), units="C")
    self._beam.status = UnitValue(np.full(len(x), 5))
    self._beam.z = UnitValue(
        zstart
        + (-1 * self._beam.Bz * constants.speed_of_light)
        * (self._beam.t - np.mean(self._beam.t)),
        units="m",
    )
    self._beam.s = UnitValue(zstart if s is None else s, units="m")
