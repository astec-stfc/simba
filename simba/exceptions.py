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

import contextlib
import math
import re
import warnings

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


class MaterialWithoutPhysicsWarning(PhysicsWarning):
    """Material in a BDSIM line tracked with no physics list, where it does nothing."""

    def __init__(self, line: str, elements: list):
        super().__init__(
            f"{_line(line)} has material in the beam ({', '.join(elements)}) but runs BDSIM "
            "with no physics list, so particles pass through it untouched: no losses, "
            "scattering or secondaries. Set tracking: {physics_list: em} (or another "
            "BDSIM physics list)."
        )


class CrystalWithoutChannellingWarning(PhysicsWarning):
    """A crystal in a BDSIM line whose physics list does not channel."""

    def __init__(self, line: str, crystals: list, physics_list: str):
        super().__init__(
            f"{_line(line)} has crystals ({', '.join(crystals)}) but BDSIM's physics list "
            f"{physics_list!r} has no channelling, so they scatter as amorphous material. "
            "Set tracking: {physics_list: completechannelling}."
        )


class BunchLengthWarning(PhysicsWarning):
    """BDSIM's matrix integrators, which time every particle as the reference."""

    def __init__(self, line: str, integrator_set: str):
        super().__init__(
            f"{_line(line)} runs BDSIM with its {integrator_set} integrators, which "
            "leave each particle's path length through a bend out of its arrival "
            "time, so z, and the bunch length, do not change through dispersion. "
            "Set tracking: {integrator_set: bdsimtwo} (or geant4) to track them."
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


class GradientsUnsupportedWarning(UnsupportedWarning):
    """Gradients asked for in a code that cannot differentiate its tracking."""

    def __init__(self, line: str, code: str, parameters: list, alternatives: str):
        super().__init__(
            f"{_line(line)} carries gradients for {', '.join(parameters)}, but "
            f"{code} cannot differentiate its tracking, so they stop here. "
            f"{alternatives} Or set 'gradients: {{finite_difference: true}}' on the line "
            "to carry them by re-running it."
        )


class GradientNotSmoothWarning(SettingWarning):
    """Finite-difference gradients through output that jumps or kinks at the step taken."""

    def __init__(self, line: str, parameter: str, outputs: list):
        more = f" and {len(outputs) - 5} more" if len(outputs) > 5 else ""
        super().__init__(
            f"{_line(line)}: the finite-difference gradients for {parameter} are not smooth at "
            f"{', '.join(outputs[:5])}{more}: the runs either side of the real one disagree on the slope. "
            "The code's output jumps or kinks at this step (nearest-grid-point space charge "
            "does), or the step is too large for its curvature, so treat these tangents as "
            "unreliable; a smoother solver, or a different gradients: finite_difference: step, "
            "may help."
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


class EdgeOrderWarning(UnsupportedWarning):
    """Bends asking for first-order edges, in a code that always models second."""

    def __init__(self, line: str, code: str, bends: list):
        super().__init__(
            f"{_line(line)} has {len(bends)} bends with edge_order = 1 (e.g. "
            f"{bends[0]}), but {code} always models the second-order edge terms, "
            "so they will not match the other codes."
        )


@contextlib.contextmanager
def one_edge_order_warning(line: str):
    """Gather LAURA's per-bend edge_order warnings into one :class:`EdgeOrderWarning`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    bends, code = [], None
    for w in caught:
        match = _EDGE_ORDER.match(str(w.message))
        if match:
            bends.append(match["bend"])
            code = match["code"]
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if bends:
        warnings.warn(EdgeOrderWarning(line, code, bends), stacklevel=3)


_EDGE_ORDER = re.compile(
    r"(?P<bend>.+): edge_order = 1, but (?P<code>.+) always models the second-order"
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


class SpeciesNotCarriedWarning(PhysicsWarning):
    """Species a line did not track, with no end-of-line beam file to carry them in."""

    def __init__(self, line: str, filename: str, species: list):
        super().__init__(
            f"{_line(line)} wrote no end-of-line beam file {filename}, so the "
            f"species it did not track, {species}, do not reach the next line."
        )


class SpeciesDroppedWarning(SettingWarning):
    """Charged species a line was told not to track, so are not carried on either."""

    def __init__(self, line: str, tracked: str, dropped: list):
        super().__init__(
            f"{_line(line)} tracks only {tracked} (``input: species``), so the other "
            f"charged species in its input, {dropped}, are dropped here."
        )


class MultipleSpeciesError(SimbaError, ValueError):
    """A beam with more than one charged species, in a line that tracks one."""

    def __init__(self, line: str, code: str, tracked: str, others: list):
        super().__init__(
            f"{line} is tracked by {code}, which in simba tracks one species, but its "
            f"input holds {tracked} and also {others}. Carrying charged particles "
            "through untracked would be wrong, so set ``input: species`` to the one "
            "to track (the others are then dropped), or split the run."
        )


class MixedMassError(SimbaError, ValueError):
    """Species of another mass than the tracked one, in a code that cannot track them together."""

    def __init__(self, line: str, code: str, tracked: str, others: list, reason: str):
        super().__init__(
            f"{line} is tracked by {code}, which can track several species only if they "
            f"have the mass of the tracked one ({tracked}), but its input also holds "
            f"{others}: {reason} Set ``input: species`` to the one to track, or track "
            "the line with ASTRA, GPT or Xsuite."
        )


class NearAxisSpeciesWarning(PhysicsWarning):
    """Particles of another species close enough to the reference orbit to be tracked as the reference."""

    def __init__(self, line: str, count: int, tracked: str, where: str):
        super().__init__(
            f"{_line(line)}: {count} particles that are not {tracked} start within 1e-9 "
            f"of the reference orbit, where {where}, so they are tracked as {tracked}."
        )


class CollectiveSpeciesWarning(PhysicsWarning):
    """Collective effects computed as if a beam of several species were all one."""

    def __init__(self, line: str, code: str, effects: str, how: str):
        super().__init__(
            f"{_line(line)} tracks several species with {effects} on, but {code} {how}, "
            "so the collective fields are wrong wherever the species overlap."
        )


class NeutralSpeciesError(SimbaError, ValueError):
    """A neutral species (photons) chosen as the species a tracking code tracks."""

    def __init__(self, line: str, code: str, species: str):
        super().__init__(
            f"{line} is tracked by {code}, but the species it is given to track is "
            f"{species}, which has no charge: no tracking code steers it. Set "
            "``input: species`` to a charged species; photons are carried alongside."
        )


class GradientChainError(SimbaError, ValueError):
    """Gradients reaching a line that changes the beam in a way they cannot follow."""

    def __init__(self, line: str, reason: str):
        super().__init__(
            f"{_line(line)} carries gradients, but {reason}, which "
            "breaks the chain. Remove it, or start the gradients after this line."
        )


class WrongSpeciesError(SimbaError, ValueError):
    """A beam that is not electrons, in a code that tracks only electrons."""

    def __init__(self, line: str, code: str, rest_energy: float, charge: int):
        super().__init__(
            f"{line} is tracked by {code}, which tracks only electrons, but its "
            f"beam has rest energy {rest_energy:.6g} eV and charge {charge:+d} e. "
            "Track it with another code."
        )
