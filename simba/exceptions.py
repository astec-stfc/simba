"""
SIMBA warnings and errors, raised when a run will not be what its settings asked for.

Each warning builds its own message, so callers only decide *when* to warn.
Every warning is a :class:`SimbaWarning` (a :class:`UserWarning`) in one of four
groups that can be filtered as a whole::

    import warnings
    from simba.exceptions import UnsupportedWarning

    warnings.simplefilter("ignore", UnsupportedWarning)

The groups are :class:`UnsupportedWarning`, :class:`SettingWarning`,
:class:`GeometryWarning` and :class:`PhysicsWarning`.

Errors are :class:`SimbaError` and also the built-in error they always were,
so ``except ValueError`` still catches :class:`WrongSpeciesError`.
"""

import math

import numpy as np


class SimbaWarning(UserWarning):
    """Every simba warning."""


class UnsupportedWarning(SimbaWarning):
    """Something this code cannot do; the run goes ahead without it."""


class SettingWarning(SimbaWarning):
    """A setting simba cannot read, or one that does not fit the run."""


class GeometryWarning(SimbaWarning):
    """A ring whose geometry does not close as it is tracked."""


class PhysicsWarning(SimbaWarning):
    """A run that tracks, but not the physics its settings intend."""


class SimbaError(Exception):
    """Every simba error."""


def _line(line: str) -> str:
    return f"Line '{line}'"


class NoRadiationWarning(PhysicsWarning):
    """A long lepton ring tracked with radiation off."""

    def __init__(self, line: str, turns: int):
        super().__init__(
            f"{_line(line)} tracks a ring for {turns} turns with no synchrotron "
            "radiation. Without it there is no damping and no quantum "
            "excitation, and the beam may never reach equilibrium: the "
            "emittance, energy spread and bunch length at the end are the ones "
            "you started with, not the ring's. Set tracking: {radiation: "
            "quantum} if you want the equilibrium."
        )


class PeriodicUnsupportedWarning(UnsupportedWarning):
    """The periodic solution, in a code that has none."""

    def __init__(self, line: str, code: str, alternatives: str):
        super().__init__(
            f"{_line(line)} asks for the periodic solution, but {code} is given "
            "the incoming beam's Twiss and has no closed-solution mode here. "
            "The open solution will be used, and its tune and beta functions "
            f"are not the ring's. {alternatives}"
        )


class RadiationUnsupportedWarning(UnsupportedWarning):
    """A radiation model, in a code simba has no radiation switch for."""

    def __init__(
        self, line: str, code: str, radiation: str, radiates_by_default: bool,
        alternatives: str,
    ):
        default = "radiates" if radiates_by_default else "does not radiate"
        super().__init__(
            f"{_line(line)} asks for radiation: {radiation}, but simba has no "
            f"radiation switch for {code}, so it is not applied and {code} "
            f"keeps its own default (it {default}). {alternatives}"
        )


class UnreadableSettingWarning(SettingWarning):
    """A tracking entry simba cannot read, and so drops."""

    def __init__(self, line: str, error: Exception):
        super().__init__(f"{_line(line)}: {error}")


class ProgramsUnsupportedWarning(UnsupportedWarning):
    """Device programs, in a code that cannot change an element between turns."""

    def __init__(self, line: str, code: str, elements: list, alternatives: str):
        super().__init__(
            f"{_line(line)} programs {', '.join(elements)} over turns, but "
            f"{code} has no way to change an element between turns here. "
            f"{alternatives}"
        )


class ProgramOverrunWarning(SettingWarning):
    """A program with knots past the last turn tracked."""

    def __init__(self, line: str, element: str, last_turn: int, turns: int):
        super().__init__(
            f"{_line(line)} programs '{element}' out to turn {last_turn}, but "
            f"only {turns} turns are tracked, so the run ends part-way through "
            "the program."
        )


class ProgramHeldWarning(SettingWarning):
    """A program whose last knot is not zero, held for the rest of the run."""

    def __init__(
        self, line: str, element: str, last_turn: int, value: float, remaining: int
    ):
        super().__init__(
            f"{_line(line)} programs '{element}' up to turn {last_turn} and ends "
            f"at {value:.4g}, which it then holds for the remaining {remaining} "
            "turns of the run. Add a final knot if it should come back down."
        )


class ProgramMissingElementWarning(SettingWarning):
    """A program for an element the code's lattice does not have."""

    def __init__(self, line: str, element: str, code: str):
        super().__init__(
            f"{_line(line)} programs '{element}', which is not in the {code} "
            "lattice. Nothing is varied."
        )


class ProgramNoAttributeWarning(SettingWarning):
    """A program with no ``parameter:``, on a type with no default attribute."""

    def __init__(self, line: str, element: str, code: str, element_type: str):
        super().__init__(
            f"{_line(line)} programs '{element}', a {code} {element_type}, and "
            "simba has no default attribute for that type. Name it with "
            "'parameter:' in the program."
        )


class UnknownRFModeWarning(SettingWarning):
    """An ``rf:`` that is none of :data:`~simba.Modules.EnergyRamp.RF_MODES`."""

    def __init__(self, line: str, mode: str, modes):
        super().__init__(
            f"{_line(line)} asks for rf: {mode}, but the RF is one of "
            f"{', '.join(modes)}. Using follow."
        )


class RFPhasesUnsupportedWarning(UnsupportedWarning):
    """Cavity phases that need moving pass by pass, in a code that cannot."""

    def __init__(self, line: str, mode: str, corrections: dict, code: str):
        """``corrections`` is radians per pass, by cavity name."""
        degrees = np.degrees(max(np.max(np.abs(c)) for c in corrections.values()))
        super().__init__(
            f"{_line(line)} runs its RF as rf: {mode}, which moves "
            f"{', '.join(corrections)} by up to {degrees:.3g} deg over the run, but "
            f"simba cannot move a cavity's phase pass by pass in {code}. Its "
            f"cavities run as {code} has them."
        )


class MissingCavitiesWarning(SettingWarning):
    """Cavities whose phase was to be moved, and which the code does not have."""

    def __init__(self, line: str, cavities: list, code: str):
        super().__init__(
            f"{_line(line)} moves the RF phase of {', '.join(sorted(cavities))}, "
            f"which the {code} lattice does not have. They ran as given."
        )


class RampUnsupportedWarning(UnsupportedWarning):
    """A ramp, in a code that cannot change the reference between turns."""

    def __init__(self, line: str, code: str, alternatives: str):
        super().__init__(
            f"{_line(line)} ramps the reference momentum, but {code} has no way "
            "to change it between turns here. The run is tracked at a fixed "
            f"energy. {alternatives}"
        )


class RampOneTurnWarning(SettingWarning):
    """A ramp on a run of one turn."""

    def __init__(self, line: str):
        super().__init__(
            f"{_line(line)} ramps the reference momentum over turns, but tracks "
            "one turn, so nothing is ramped."
        )


class RampOverrunWarning(SettingWarning):
    """A ramp with knots past the last turn tracked."""

    def __init__(self, line: str, last_turn: int, turns: int):
        super().__init__(
            f"{_line(line)} ramps out to turn {last_turn}, but only {turns} "
            "turns are tracked, so the run ends part-way up the ramp."
        )


class OffRampWarning(PhysicsWarning):
    """A beam that does not enter on the ramp."""

    def __init__(self, line: str, start: float, entering: float):
        super().__init__(
            f"{_line(line)} ramps from p0c = {start:.6g} eV, but the beam enters "
            f"at {entering:.6g} eV/c ({100 * (start / entering - 1):+.3g}%). The "
            "beam starts that far off the ramp."
        )


class OffDesignEnergyWarning(PhysicsWarning):
    """A ring's beam far from the section's design energy."""

    def __init__(self, line: str, design: float, entering: float):
        super().__init__(
            f"{_line(line)} is designed for p0c = {design:.6g} eV, but the beam "
            f"enters at {entering:.6g} eV/c ({100 * (entering / design - 1):+.3g}%). "
            "The ring's reference is its design momentum, so the whole beam "
            "tracks at that momentum offset; check the section's "
            "reference_energy and the beam's energy agree."
        )


class RampWithoutRFWarning(PhysicsWarning):
    """A ramp with no cavity to follow it."""

    def __init__(self, line: str):
        super().__init__(
            f"{_line(line)} ramps the reference momentum but has no RF cavity. "
            "Nothing accelerates the beam, so it keeps its energy and the ramp "
            "leaves it behind."
        )


class RampTooSteepWarning(PhysicsWarning):
    """A ramp needing more energy a turn than the cavities have."""

    def __init__(self, line: str, needed: float, voltage: float):
        super().__init__(
            f"{_line(line)} ramps by up to {needed:.4g} eV a turn, but its "
            f"cavities total {voltage:.4g} V. No synchronous phase supplies "
            "that, so the beam falls off the ramp."
        )


class BadSuperperiodsWarning(SettingWarning):
    """An ``nsuperperiods`` that is not a positive whole number."""

    def __init__(self, line: str, value):
        try:
            shown, reason = int(value), "not a count"
        except (TypeError, ValueError):
            shown, reason = repr(value), "not a whole number"
        super().__init__(
            f"{_line(line)} has nsuperperiods={shown}, which is {reason}. "
            "Tracking the line once per turn."
        )


class SuperperiodsUnsupportedWarning(UnsupportedWarning):
    """Superperiods, in a code that cannot repeat the line within a turn."""

    def __init__(self, line: str, code: str, count: int, alternatives: str):
        super().__init__(
            f"{_line(line)} asks for {count} superperiods, but {code} has no way "
            "to repeat the line within a turn here, so it will be tracked once "
            f"per turn -- which is one {count}th of the intended ring, not a "
            f"coarser version of it. {alternatives}"
        )


class SingleParticleUnsupportedWarning(UnsupportedWarning):
    """Single-particle mode, in a code with no implementation of it."""

    def __init__(self, line: str, code: str, alternatives: str):
        super().__init__(
            f"{_line(line)} asks for single-particle mode, but {code} has no "
            "implementation of it. The full distribution will be tracked, which "
            f"is slower but not wrong -- the results stand. {alternatives}"
        )


class FrequencyMapUnsupportedWarning(UnsupportedWarning):
    """A frequency map, from a code that has none."""

    def __init__(self, line: str, code: str, alternatives: str):
        super().__init__(
            f"{code} has no frequency map here, so line '{line}' returns none. "
            f"{alternatives}"
        )


class NoFootprintWarning(PhysicsWarning):
    """A frequency-map scan in which no particle gave a tune."""

    def __init__(self, line: str):
        super().__init__(f"{_line(line)}: no particle gave a tune in the frequency-map scan.")


class DynamicApertureUnsupportedWarning(UnsupportedWarning):
    """A dynamic-aperture scan, from a code that has none."""

    def __init__(self, line: str, code: str, alternatives: str):
        super().__init__(
            f"{code} has no dynamic-aperture scan here, so line '{line}' returns "
            f"none. {alternatives}"
        )


class TurnsUnsupportedWarning(UnsupportedWarning):
    """Turns, in a code that tracks a line once."""

    def __init__(self, line: str, code: str, turns: int, alternatives: str):
        super().__init__(
            f"{_line(line)} asks for {turns} turns, but {code} tracks a line "
            f"once and has no turn count. One turn will be tracked. {alternatives}"
        )


class NotClosedWarning(GeometryWarning):
    """A line treated as a ring whose geometry does not close."""

    def __init__(self, line: str, turns: int, gap: float, length: float, net_bend: float):
        claim = (
            f"is tracked for {turns} turns"
            if turns > 1
            else "asks for the periodic solution"
        )
        message = (
            f"{_line(line)} {claim}, but its geometry does not close: it ends "
            f"{gap:.4g} m from where it starts, over {length:.4g} m."
        )
        turn = 2 * math.pi
        angle = abs(net_bend)
        if angle > 1e-9 and abs(round(turn / angle) - turn / angle) < 1e-3:
            message += (
                f" Its net bend is a 1/{round(turn / angle)} fraction of a "
                "turn, so if this is one superperiod of a symmetric ring, "
                f"set 'nsuperperiods: {round(turn / angle)}' or track the "
                "whole ring instead."
            )
        super().__init__(message)


class SuperperiodsDoNotCloseWarning(GeometryWarning):
    """A superperiod count whose sectors do not bend through whole turns."""

    def __init__(self, line: str, count: int, net_bend: float):
        angle = abs(net_bend)
        turns_of_bend = count * angle / (2 * math.pi)
        message = (
            f"{_line(line)} declares {count} superperiods, but {count} of them "
            f"bend through {turns_of_bend:.4g} turns rather than a whole number, "
            "so they do not make a closed ring."
        )
        implied = (2 * math.pi) / angle
        if abs(round(implied) - implied) < 1e-3:
            message += f" Its net bend suggests {round(implied)} instead."
        super().__init__(message)


class RigidityMismatchWarning(PhysicsWarning):
    """A beam whose rigidity is not the one its pass states."""

    def __init__(
        self, line: str, brho: float, start: str, stated: float, expected: float
    ):
        super().__init__(
            f"{_line(line)} tracks a beam of rigidity {float(brho):.4f} T.m, but "
            f"pass {start} states a momentum of {stated:.4g} eV/c "
            f"({expected:.4f} T.m)."
        )


class SpaceChargeModeWarning(UnsupportedWarning):
    """A space-charge mode the code has no model for."""

    def __init__(self, line: str, code: str, mode: str, modes: str):
        super().__init__(
            f"{_line(line)} asks for space charge '{mode}', but {code} here has "
            f"only {modes}, so it is tracked without space charge."
        )


class SpaceChargeOffGridWarning(PhysicsWarning):
    """Particles that fell outside a space-charge grid, so felt no field."""

    def __init__(self, line: str, fraction: float):
        super().__init__(
            f"{_line(line)}: {100 * fraction:.2g}% of the particles ended outside "
            "the last space-charge grid. The grids are sized to the beam on its "
            "first pass, and a particle outside one feels no space charge there "
            "and adds none to the field."
        )


class MissingElementWarning(SettingWarning):
    """An element asked for by name that the lattice does not have."""

    def __init__(self, element: str):
        super().__init__(f"WARNING: Element {element} does not exist")


class BadOneTurnMapWarning(PhysicsWarning):
    """A one-turn map that is not 6x6, or does not preserve phase-space volume."""

    def __init__(self, line: str, code: str, shape=None, determinant=None):
        if shape is not None:
            message = (
                f"{_line(line)}: {code} returned a one-turn map of shape "
                f"{shape}, not 6x6."
            )
        else:
            message = (
                f"{_line(line)}: the one-turn map from {code} has determinant "
                f"{determinant:.6g}, not 1. It is not a volume-preserving linear "
                "map, so whatever is derived from it -- tune, beta, momentum "
                "compaction -- is not trustworthy."
            )
        super().__init__(message)


class WrongSpeciesError(SimbaError, ValueError):
    """A beam that is not electrons, in a code that tracks only electrons."""

    def __init__(self, line: str, code: str, rest_energy: float, charge: int):
        super().__init__(
            f"{line} is tracked by {code}, which tracks only electrons, but its "
            f"beam has rest energy {rest_energy:.6g} eV and charge {charge:+d} e. "
            "Track it with another code."
        )
