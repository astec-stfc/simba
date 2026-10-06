"""
Device programs: an element's strength as a function of turn number.

An injection kicker or an extraction septum is not a static element: its
strength is a *program over turn number*, zero for the first N turns, up for
one, and down again. The pulse **shape** is hardware and lives in LAURA, as
``ACDipoleSimulationElement.waveform``. *When it fires and at what amplitude*
is the study, and lives here, in simba's ``tracking:`` block alongside
``turns``, ``periodic``, ``radiation`` and ``write_turns``::

    files:
      RING:
        code: elegant
        tracking:
          turns: 10
          programs:
            - element: KICK1
              turns:  [1, 4, 5]
              values: [0.0, 1.0e-3, 0.0]
              interpolation: hold

**Turns are 1-based.** Turn 1 is the first turn tracked, matching
:meth:`~simba.Framework_objects.frameworkLattice.output_turns` and the
``_turn`` suffixes on the beam files.

**``values`` are the element's strength as the lattice states it, not as a
code states it**. A program is read as "put the kicker at this value
on this turn". Naming an attribute with ``parameter:`` sets the
named attribute verbatim.

**The default rule is** ``hold``: the rule those
devices want, no code offers it, so SIMBA implements it and states it here
rather than letting a code's default stand.

Outside the listed turns the value is **clamped** to the first or last knot,
which is what every code's interpolation does too. A pulse that has to come
back down therefore says so, with a final knot at zero; a slow bumper that
stays on simply stops listing knots.

Classes:
    - :class:`DeviceProgram`: one element's strength against turn number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from warnings import warn

import numpy as np

INTERPOLATIONS = ("hold", "linear", "spline")
"""The rules a program may ask for; see the module docstring for the default."""

_HOLD_RISER = (0.75, 0.25)
"""Where a ``hold`` step is written, in turns before the knot it steps to.

A step is a vertical line and the codes only draw straight ones, so ``hold``
is exported as a riser that climbs over the half turn *between* two tracked
turns. Half a turn either side is also the tolerance on the revolution
period.
"""


@dataclass
class DeviceProgram:
    """
    One element's strength against turn number; see the module docstring.

    Attributes
    ----------
    element: str
        Name of the element being programmed, as the lattice names it
    turns: list
        Knot turn numbers, 1-based and ascending
    values: list
        The element attribute's value at each knot, in that attribute's
        own units
    interpolation: str
        ``hold`` (the default), ``linear`` or ``spline``
    parameter: str | None
        The code-native attribute to set, or None to let the backend
        choose one from the element it built
    """

    element: str
    turns: list = field(default_factory=list)
    values: list = field(default_factory=list)
    interpolation: str = "hold"
    parameter: str | None = None

    def __post_init__(self) -> None:
        self.turns = [int(t) for t in self.turns]
        self.values = [float(v) for v in self.values]
        self.interpolation = str(self.interpolation).lower()
        if self.interpolation not in INTERPOLATIONS:
            warn(
                f"Program for '{self.element}' asks for interpolation "
                f"'{self.interpolation}', which is not one of "
                f"{', '.join(INTERPOLATIONS)}. Using 'hold'."
            )
            self.interpolation = "hold"
        if len(self.turns) != len(self.values):
            raise ValueError(
                f"Program for '{self.element}' has {len(self.turns)} turns "
                f"and {len(self.values)} values; they pair up one to one."
            )
        if not self.turns:
            raise ValueError(f"Program for '{self.element}' has no knots.")
        if any(b <= a for a, b in zip(self.turns, self.turns[1:])):
            raise ValueError(
                f"Program for '{self.element}' lists turns {self.turns}, "
                "which are not strictly ascending."
            )
        if self.turns[0] < 1:
            raise ValueError(
                f"Program for '{self.element}' starts at turn "
                f"{self.turns[0]}; turns are numbered from 1."
            )

    @classmethod
    def from_dict(cls, entry: dict) -> DeviceProgram:
        """
        Build one program from a ``tracking: {programs: [...]}`` entry.

        Parameters
        ----------
        entry: dict
            One mapping from the ``programs`` list

        Returns
        -------
        DeviceProgram
            The program described by `entry`

        Raises
        ------
        ValueError
            If the entry names no element, or its knots do not pair up
        """
        if not isinstance(entry, dict):
            raise ValueError(
                f"Each entry in tracking: {{programs: ...}} is a mapping with "
                f"'element', 'turns' and 'values'; got {entry!r}."
            )
        name = entry.get("element")
        if not name:
            raise ValueError(
                f"A program entry names no element: {entry!r}. "
                "Each one needs 'element'."
            )
        unknown = set(entry) - {
            "element",
            "turns",
            "values",
            "interpolation",
            "parameter",
        }
        if unknown:
            warn(
                f"Program for '{name}' carries settings simba does not read: "
                f"{', '.join(sorted(unknown))}."
            )
        return cls(
            element=str(name),
            turns=entry.get("turns") or [],
            values=entry.get("values") or [],
            interpolation=entry.get("interpolation", "hold"),
            parameter=entry.get("parameter"),
        )

    @property
    def first_turn(self) -> int:
        """The first turn the program says anything about; 1-based."""
        return self.turns[0]

    @property
    def last_turn(self) -> int:
        """The last knot's turn; after it the value is held, see the module docstring."""
        return self.turns[-1]

    @property
    def peak(self) -> float:
        """
        The knot value of largest magnitude, keeping its sign.

        What a code wanting ``strength x factor(t)`` -- elegant's ``BUMPER``
        is the one -- is given as the strength, so that every factor lands
        in ``[-1, 1]``.
        """
        return max(self.values, key=abs)

    def value_at(self, turn: int) -> float:
        """
        The programmed value on `turn`.

        Parameters
        ----------
        turn: int
            Turn number, 1-based

        Returns
        -------
        float
            The element attribute's value for that turn, clamped to the
            first or last knot outside the programmed range
        """
        if turn <= self.turns[0]:
            return self.values[0]
        if turn >= self.turns[-1]:
            return self.values[-1]
        if self.interpolation == "hold":
            index = int(np.searchsorted(self.turns, turn, side="right")) - 1
            return self.values[index]
        if self.interpolation == "linear":
            return float(np.interp(turn, self.turns, self.values))
        return float(self._spline()(turn))

    def _spline(self):
        """A cubic through the knots, or linear if there are too few for one."""
        from scipy.interpolate import CubicSpline

        if len(self.turns) < 3:
            warn(
                f"Program for '{self.element}' asks for a spline through "
                f"{len(self.turns)} knots, which do not define one. "
                "Interpolating linearly instead."
            )
            return lambda turn: np.interp(turn, self.turns, self.values)
        return CubicSpline(self.turns, self.values, bc_type="natural")

    def linear_knots(self) -> tuple:
        """
        Knots whose *linear* interpolation reproduces this program exactly.

        Every code that takes a programmed attribute takes it as samples
        joined by straight lines, so ``hold`` and ``spline`` have to be
        rewritten into knots a straight-line reading gets right. The
        guarantee is only at integer turns, which is the only place a
        tracked particle ever asks.

        Returns
        -------
        tuple
            ``(turns, values)``, both lists of float. `turns` are turn
            numbers and need not be integers -- a ``hold`` step is written
            as a riser between two of them.
        """
        if self.interpolation == "linear":
            return ([float(t) for t in self.turns], list(self.values))
        if self.interpolation == "spline":
            sampled = list(range(self.turns[0], self.turns[-1] + 1))
            return ([float(t) for t in sampled], [self.value_at(t) for t in sampled])
        before, after = _HOLD_RISER
        turns = [float(self.turns[0])]
        values = [self.values[0]]
        for knot, value in zip(self.turns[1:], self.values[1:]):
            turns += [knot - before, knot - after]
            values += [values[-1], value]
        turns.append(float(self.turns[-1]))
        values.append(self.values[-1])
        return (turns, values)

    def time_knots(self, revolution_period: float, origin_turn: int = 1) -> tuple:
        """
        :meth:`linear_knots` on a time axis, for the codes that have one.

        Parameters
        ----------
        revolution_period: float
            Seconds per turn, ``C / (beta0 * c)`` -- **beta0 and not c**,
            which matters at the turn counts a ring study uses
        origin_turn: int
            The turn sitting at ``t = 0``. 1 for Xsuite, whose ``t_turn_s``
            is counted from the start of the run; the firing pass for
            elegant, whose ``WAVEFORM`` is counted from there

        Returns
        -------
        tuple
            ``(times, values)``, times in seconds
        """
        turns, values = self.linear_knots()
        return ([(t - origin_turn) * revolution_period for t in turns], values)

    def factor_knots(self, revolution_period: float, origin_turn: int = 1) -> tuple:
        """
        :meth:`time_knots` with the values divided through by :attr:`peak`.

        For a code stating a programmed strength as an amplitude times a
        dimensionless waveform.

        Returns
        -------
        tuple
            ``(times, factors)``, factors in ``[-1, 1]``
        """
        times, values = self.time_knots(revolution_period, origin_turn)
        peak = self.peak
        if not peak:
            return (times, [0.0 for _ in values])
        return (times, [value / peak for value in values])
