"""
RF settings for a target longitudinal phase space, after Zagorodnov and Dohlus,
PRST-AB 14, 014403 (2011).

Along the bunch, with ``s = c(<t> - t)`` (the head at positive ``s``), a cavity at
SIMBA phase ``phi`` (crest at 0) and voltage ``V`` adds ``V cos(k s + phi)`` to the
energy. With ``X = V cos(phi)`` and ``Y = V sin(phi)``, each RF system changes the
coefficients of the energy's polynomial ``E(s) = sum e_n s^n`` by

    e_0: X,    e_1: -k Y,    e_2: -k^2 X / 2,    e_3: k^3 Y / 6

so a main system and a harmonic of it give four linear equations, for the energy and
the relative chirp, curvature and skewness; the main system alone gives two, for the
energy and chirp. The beam is taken as ultra-relativistic and rigid through the RF,
with no collective effects.
"""

import numpy as np
from scipy.constants import speed_of_light

_ROWS = (
    lambda k: (1.0, 0.0),
    lambda k: (0.0, -k),
    lambda k: (-(k**2) / 2, 0.0),
    lambda k: (0.0, k**3 / 6),
)


def energy_polynomial(beam, order: int = 3) -> np.ndarray:
    """``[e_0, ..., e_order]``: the beam's energy [eV] as a polynomial in ``s = c(<t> - t)`` [m]."""
    t, energy = np.asarray(beam.t.val, dtype=float), np.asarray(beam.energy.val, dtype=float)
    return np.polyfit(speed_of_light * (t.mean() - t), energy, order)[::-1]


def rf_settings(
    energy_in,
    frequency: float,
    energy_out: float,
    chirp: float,
    curvature: float = 0.0,
    skewness: float = 0.0,
    harmonic: int | None = None,
) -> dict:
    """
    Voltage and phase of each RF system, to take a beam to a target energy and shape.

    Parameters
    ----------
    energy_in: array-like
        The incoming energy polynomial, as :func:`energy_polynomial` gives it.
    frequency: float
        Main RF frequency [Hz].
    energy_out: float
        Energy at the bunch centre afterwards [eV].
    chirp, curvature, skewness: float
        Coefficients of ``s``, ``s^2`` and ``s^3`` in the relative energy
        ``(E - E_0) / E_0`` afterwards [1/m, 1/m^2, 1/m^3]. Curvature and skewness
        need a harmonic system.
    harmonic: int, optional
        The harmonic system's frequency, in units of the main one.

    Returns
    -------
    dict
        ``{"main": (voltage [V], phase [deg])}``, and ``"harmonic"`` likewise if given.
    """
    if not harmonic and (curvature or skewness):
        raise ValueError("Matching curvature or skewness needs harmonic cavities")
    k = 2 * np.pi * frequency / speed_of_light
    ks = [k, harmonic * k] if harmonic else [k]
    n = 2 * len(ks)
    matrix = np.array([[v for q in ks for v in _ROWS[i](q)] for i in range(n)])
    target = np.array([1.0, chirp, curvature, skewness][:n]) * energy_out
    xy = np.linalg.solve(matrix, target - np.asarray(energy_in, dtype=float)[:n])
    return {
        name: (float(np.hypot(x, y)), float(np.degrees(np.arctan2(y, x))))
        for name, (x, y) in zip(("main", "harmonic"), xy.reshape(-1, 2))
    }


def _voltage_factor(cavity) -> float:
    """Voltage per unit ``field_amplitude``: 1, or a travelling-wave cavity's length factor."""
    from laura.translator.converters.converter import translate_elements

    return float(next(iter(translate_elements([cavity]).values())).voltage_factor())


def _system(lattice, names, label: str) -> tuple:
    """``(cavities, frequency)`` of one RF system, which must share a frequency."""
    missing = [n for n in names if n not in lattice.elements]
    if missing:
        raise KeyError(f"longitudinal_match {label} {missing} are not in line {lattice.objectname!r}")
    cavities = [lattice.elements[n] for n in names]
    frequencies = {float(c.cavity.resolved("frequency")) for c in cavities}
    if len(frequencies) > 1:
        raise ValueError(f"longitudinal_match {label} must share one frequency, not {sorted(frequencies)}")
    return cavities, frequencies.pop()


def longitudinal_match(lattice, settings: dict) -> dict:
    """
    Cavity settings for a line's ``longitudinal_match`` block, from its input beam.

    .. code-block:: yaml

        files:
          line:
            longitudinal_match:
              cavities: [L01, L02]   # the main RF system, at one frequency
              harmonics: [L3]        # optional: a harmonic system
              chirp: -5.0            # relative energy chirp afterwards [1/m]
              curvature: 0.0         # [1/m^2], needs harmonics
              skewness: 0.0          # [1/m^3], needs harmonics
              energy_gain: 100.0e6   # [eV]; by default what the cavities give now

    See :func:`rf_settings` for the conventions. Each system's voltage is shared
    between its cavities as they share it now (with equal field amplitudes if they
    give none), and all of a system's cavities take the same phase.

    A phase or field amplitude given by a functional definition keeps it: the
    definition is matched instead, so cavities linked by one definition move
    together.

    Parameters
    ----------
    lattice: :class:`~simba.Framework_objects.frameworkLattice`
        The line, with its input beam loaded.
    settings: dict
        The ``longitudinal_match`` block.

    Returns
    -------
    dict
        ``{cavity: {"simulation.field_amplitude": ..., "cavity.phase": ...}}``, and
        any matched definitions under ``"functional_definitions"``.
    """
    if "cavities" not in settings:
        raise KeyError("longitudinal_match needs `cavities`, the main RF system")
    if "chirp" not in settings:
        raise KeyError("longitudinal_match needs a target `chirp`")
    systems = {"main": _system(lattice, settings["cavities"], "cavities")}
    harmonic = None
    if settings.get("harmonics"):
        systems["harmonic"] = _system(lattice, settings["harmonics"], "harmonics")
        ratio = systems["harmonic"][1] / systems["main"][1]
        harmonic = round(ratio)
        if harmonic < 2 or abs(ratio - harmonic) > 1e-9 * ratio:
            raise ValueError(
                f"longitudinal_match harmonics at {systems['harmonic'][1]} Hz are not a "
                f"harmonic of the cavities at {systems['main'][1]} Hz"
            )

    factors = {c.name: _voltage_factor(c) for cavities, _ in systems.values() for c in cavities}
    voltages = {
        c.name: abs(float(c.simulation.resolved("field_amplitude"))) * factors[c.name]
        for cavities, _ in systems.values()
        for c in cavities
    }
    phases = {
        c.name: np.radians(float(c.cavity.resolved("phase")))
        for cavities, _ in systems.values()
        for c in cavities
    }
    energy_in = energy_polynomial(lattice.global_parameters["beam"])
    gain = settings.get("energy_gain")
    if gain is None:
        gain = sum(v * np.cos(phases[n]) for n, v in voltages.items())

    solved = rf_settings(
        energy_in,
        systems["main"][1],
        energy_in[0] + gain,
        settings["chirp"],
        settings.get("curvature", 0.0),
        settings.get("skewness", 0.0),
        harmonic,
    )
    result, definitions = {}, {}
    for name, (cavities, _) in systems.items():
        voltage, phase = solved[name]
        total = sum(voltages[c.name] for c in cavities)
        for c in cavities:
            amplitude = c.simulation.field_amplitude
            if total:
                new = float(c.simulation.resolved("field_amplitude")) * voltage / total
            else:
                new = voltage / sum(factors[x.name] for x in cavities)
            _set(result, definitions, c, "simulation.field_amplitude", amplitude, new)
            _set(result, definitions, c, "cavity.phase", c.cavity.phase, phase)
    if definitions:
        _check_definitions_stay_inside(lattice, definitions, [c for cs, _ in systems.values() for c in cs])
        result[FUNCTIONAL] = definitions
    return result


FUNCTIONAL = "functional_definitions"
"""Key of the matched functional definitions in a matcher's result."""


def _set(result: dict, definitions: dict, cavity, path: str, current, value: float) -> None:
    """Set ``path`` on ``cavity``, or the functional definition it names."""
    if not isinstance(current, str):
        result.setdefault(cavity.name, {})[path] = value
    elif current in definitions and not np.isclose(definitions[current], value, rtol=1e-12, atol=0):
        raise ValueError(
            f"longitudinal_match needs functional definition {current!r} to be both "
            f"{definitions[current]} and {value} (from {cavity.name}.{path})"
        )
    else:
        definitions[current] = value


def _check_definitions_stay_inside(lattice, definitions: dict, cavities: list) -> None:
    """Refuse to change a definition that elements other than the matched cavities use too."""
    from laura.models.base_models import functional_references

    inside = {c.name for c in cavities}
    machine = getattr(lattice, "machine", None)
    elements = machine.elements if machine is not None else lattice.elements
    outside = {}
    for name in list(elements.keys()):
        if name in inside:
            continue
        for ref in functional_references(elements[name]) & definitions.keys():
            outside.setdefault(ref, []).append(name)
    if outside:
        raise ValueError(
            "longitudinal_match would change functional definitions that other "
            f"elements use too: {outside}; match those cavities as well, or give "
            "the matched ones their own definitions"
        )
