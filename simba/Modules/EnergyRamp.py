"""
Energy ramps: the reference momentum as a function of turn number.

Stated once and read the same way by every code that can track one::

    files:
      RING:
        code: xsuite
        tracking:
          turns: 2000
          ramp:
            turns: [1, 1000, 2000]
            kinetic_energy: [160.0e6, 2.0e9, 2.0e9]

``momentum`` (``p0c``, eV) may be given instead of ``kinetic_energy`` (eV).
Turns are 1-based, as for :mod:`simba.Modules.DeviceProgram`; the default
interpolation is ``linear``, with ``hold`` and ``spline`` also available.

The model every backend implements:

* the ramp sets the **reference** momentum at the start of each turn and
  holds it for the turn;
* changing the reference does not change any particle: only coordinates
  measured from the reference move;
* magnet strengths are normalised, so the fields follow the reference
  (R10, ``test_energy_program.py``);
* the RF does the accelerating; with too little voltage the beam falls off
  the ramp, as it would in the machine.

**The clock.** Where a code needs time rather than turn number, turn ``n``
starts at::

    t(1) = 0
    t(n + 1) = t(n) + C / (c * (beta0(n) + beta0(n + 1)) / 2)

with ``C`` the length of one pass. This mid-point rule is Xsuite's
``EnergyProgram.get_t_s_at_turn``.

**The RF.** How a cavity keeps time from pass to pass is its own setting,
``tracking: {rf: follow | fixed}``; see :func:`rf_phase_slip`. A cavity off
a revolution harmonic slips on a flat ring too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from warnings import warn

import numpy as np

from .DeviceProgram import DeviceProgram

QUANTITIES = ("momentum", "kinetic_energy")
"""What a ramp may be stated in: ``p0c`` or kinetic energy, both in eV."""

SPEED_OF_LIGHT = 299792458.0

RF_MODES = ("follow", "fixed")
"""What ``tracking: {rf: ...}`` may be; see :func:`rf_phase_slip`."""


def rf_phase_slip(mode, frequency, s, pass_length, beta0) -> np.ndarray:
    """
    The RF phase the reference sees at a cavity on each pass, relative to the first.

    * ``fixed``: a free-running oscillator. The reference reaches ``s`` on pass
      ``j`` at ``T_j = sum_{k < j} C / (beta0_k c) + s / (beta0_j c)`` (each
      pass at its own speed, not :class:`RampClock`'s mid-point), so the slip
      is ``f (T_j - T_0)`` cycles.
    * ``follow``: frequency ``f beta0_k / beta0_0`` on pass ``k``, so every pass
      is ``h = f C / (beta0_0 c)`` cycles and the slip is ``j h``.
    * ``synchronous``: rephased every pass, so no slip.

    ``fixed`` and ``follow`` agree without a ramp; ``synchronous`` agrees with
    both only when ``h`` is a whole number.

    Parameters
    ----------
    mode: str
        ``follow``, ``fixed`` or ``synchronous``.
    frequency: float
        Frequency on the first pass, in Hz.
    s: float
        Cavity position from the start of the pass, in metres.
    pass_length: float
        Length of one pass, in metres.
    beta0: np.ndarray
        Reference speed over ``c`` on each pass.

    Returns
    -------
    np.ndarray
        Radians in ``(-pi, pi]``, one per pass; the first is 0.
    """
    beta0 = np.asarray(beta0, dtype=float)
    passes = np.arange(len(beta0))
    if mode == "synchronous":
        return np.zeros(len(beta0))
    if mode == "follow":
        cycles = frequency * pass_length / (beta0[0] * SPEED_OF_LIGHT) * passes
    elif mode == "fixed":
        arrival = np.concatenate(
            ([0.0], np.cumsum(pass_length / (beta0[:-1] * SPEED_OF_LIGHT)))
        ) + s / (beta0 * SPEED_OF_LIGHT)
        cycles = frequency * (arrival - arrival[0])
    else:
        raise ValueError(
            f"The RF is one of {', '.join(RF_MODES)}; got '{mode}'."
        )
    return wrap_phase(2 * np.pi * (cycles - np.round(cycles)))


def wrap_phase(phase):
    """
    Wrap ``phase`` into ``(-pi, pi]``.

    Parameters
    ----------
    phase: float | np.ndarray
        Radians.

    Returns
    -------
    float | np.ndarray
    """
    return np.pi - np.mod(np.pi - np.asarray(phase, dtype=float), 2 * np.pi)


def beta_from_p0c(p0c, rest_energy):
    """
    Reference speed over ``c``.

    Parameters
    ----------
    p0c: float | np.ndarray
        Reference momentum times ``c``, in eV.
    rest_energy: float
        In eV.

    Returns
    -------
    float | np.ndarray
        ``beta0``.
    """
    p0c = np.asarray(p0c, dtype=float)
    return p0c / np.sqrt(p0c**2 + rest_energy**2)


@dataclass
class RampClock:
    """
    Seconds at the start of each pass of a ramped run; see the module docstring.

    Attributes
    ----------
    times: np.ndarray
        Seconds at the start of each pass, 0-based, so ``times[0] = 0``.
    passes_per_turn: int
        See :attr:`~simba.Framework_objects.frameworkLattice.passes_per_turn`.
    """

    times: np.ndarray
    passes_per_turn: int = 1

    def __call__(self, turn: float) -> float:
        """
        Seconds at the start of ``turn``, which may be fractional.

        Extrapolated at the last pass's rate, as a device program may outlast the ramp.

        Parameters
        ----------
        turn: float
            Turn number, 1-based.

        Returns
        -------
        float
            Seconds since the start of turn 1.
        """
        passes = (float(turn) - 1.0) * self.passes_per_turn
        last = len(self.times) - 1
        if passes <= last:
            return float(np.interp(passes, np.arange(len(self.times)), self.times))
        step = self.times[-1] - self.times[-2] if last else 0.0
        return float(self.times[-1] + (passes - last) * step)


@dataclass
class EnergyRamp:
    """
    The reference momentum against turn number; see the module docstring.

    Attributes
    ----------
    turns: list
        Knot turn numbers, 1-based and ascending.
    values: list
        The ramp at each knot, in eV, as ``quantity``.
    quantity: str
        ``momentum`` (``p0c``) or ``kinetic_energy``.
    interpolation: str
        ``linear`` (the default), ``hold`` or ``spline``.
    """

    turns: list = field(default_factory=list)
    values: list = field(default_factory=list)
    quantity: str = "momentum"
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        if self.quantity not in QUANTITIES:
            raise ValueError(
                f"A ramp is stated in one of {', '.join(QUANTITIES)}; "
                f"got '{self.quantity}'."
            )
        self._program = DeviceProgram(
            element="ramp",
            turns=self.turns,
            values=self.values,
            interpolation=self.interpolation,
        )
        self.turns = self._program.turns
        self.values = self._program.values
        self.interpolation = self._program.interpolation
        if any(value <= 0 for value in self.values):
            raise ValueError(
                f"A ramp's {self.quantity} must be positive; got {self.values}."
            )

    @classmethod
    def from_dict(cls, entry: dict) -> EnergyRamp:
        """
        Build a ramp from a ``tracking: {ramp: ...}`` mapping.

        Parameters
        ----------
        entry: dict
            The ``ramp`` mapping.

        Returns
        -------
        EnergyRamp

        Raises
        ------
        ValueError
            If the entry states neither or both quantities, or its knots do not pair up.
        """
        if not isinstance(entry, dict):
            raise ValueError(
                "tracking: {ramp: ...} is a mapping with 'turns' and one of "
                f"{', '.join(QUANTITIES)}; got {entry!r}."
            )
        stated = [q for q in QUANTITIES if q in entry]
        if len(stated) != 1:
            raise ValueError(
                f"A ramp states exactly one of {', '.join(QUANTITIES)}; "
                f"this one states {stated or 'neither'}."
            )
        unknown = set(entry) - {"turns", "interpolation", *QUANTITIES}
        if unknown:
            warn(
                "The ramp carries settings simba does not read: "
                f"{', '.join(sorted(unknown))}."
            )
        return cls(
            turns=entry.get("turns") or [],
            values=entry.get(stated[0]) or [],
            quantity=stated[0],
            interpolation=entry.get("interpolation", "linear"),
        )

    @property
    def first_turn(self) -> int:
        """The first knot's turn; before it the first value is held."""
        return self.turns[0]

    @property
    def last_turn(self) -> int:
        """The last knot's turn; after it the last value is held."""
        return self.turns[-1]

    def p0c_at(self, turn: int, rest_energy: float) -> float:
        """
        The reference momentum on ``turn``.

        Parameters
        ----------
        turn: int
            Turn number, 1-based.
        rest_energy: float
            In eV.

        Returns
        -------
        float
            ``p0c`` in eV, clamped to the first or last knot outside them.
        """
        value = self._program.value_at(turn)
        if self.quantity == "momentum":
            return value
        energy = value + rest_energy
        return float(np.sqrt(energy**2 - rest_energy**2))

    def p0c_per_pass(
        self, turns: int, rest_energy: float, passes_per_turn: int = 1
    ) -> np.ndarray:
        """
        The reference momentum at the start of every pass of a run, and one beyond its end.

        Parameters
        ----------
        turns: int
            Turns tracked.
        rest_energy: float
            In eV.
        passes_per_turn: int
            Passes in one turn; the momentum is held across them.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn + 1`` values of ``p0c`` in eV; entry
            ``j`` (0-based) is on turn ``j // passes_per_turn + 1``.
        """
        per_turn = np.array(
            [self.p0c_at(turn, rest_energy) for turn in range(1, turns + 2)]
        )
        passes = turns * passes_per_turn + 1
        return per_turn[np.arange(passes) // passes_per_turn]

    def clock(
        self,
        turns: int,
        pass_length: float,
        rest_energy: float,
        passes_per_turn: int = 1,
    ) -> RampClock:
        """
        Seconds at the start of every pass; the clock in the module docstring.

        Parameters
        ----------
        turns: int
            Turns tracked.
        pass_length: float
            In metres.
        rest_energy: float
            In eV.
        passes_per_turn: int
            Passes in one turn.

        Returns
        -------
        RampClock
            Callable from turn number to seconds.
        """
        beta = beta_from_p0c(
            self.p0c_per_pass(turns, rest_energy, passes_per_turn), rest_energy
        )
        steps = pass_length / (SPEED_OF_LIGHT * 0.5 * (beta[1:] + beta[:-1]))
        return RampClock(
            times=np.concatenate(([0.0], np.cumsum(steps))),
            passes_per_turn=passes_per_turn,
        )

    def energy_gain_per_turn(self, turns: int, rest_energy: float) -> np.ndarray:
        """
        What the RF has to supply on each turn for the beam to stay on the ramp.

        Parameters
        ----------
        turns: int
            Turns tracked.
        rest_energy: float
            In eV.

        Returns
        -------
        np.ndarray
            ``E0(n + 1) - E0(n)`` in eV, one per turn.
        """
        p0c = self.p0c_per_pass(turns, rest_energy)
        energy = np.sqrt(p0c**2 + rest_energy**2)
        return np.diff(energy)
