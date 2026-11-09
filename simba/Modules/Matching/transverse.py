"""
Transverse (Twiss) matching of quadrupole strengths, with Ocelot's ``match`` on the
line's own optics, whichever code then tracks it.
"""

import contextlib
import io
import types

import numpy as np

from ._settings import FUNCTIONAL, check_definitions_stay_inside, expand, set_value

_OPERATORS = ("<", ">", "a<", "a>")


def _ocelot_model(lattice):
    """
    ``(MagneticLattice, energy [GeV])``: the line as Ocelot sees it, with SIMBA's
    cavity and bend maps and travelling-wave focusing, from its input beam's energy.
    """
    from ...Codes.Ocelot.Ocelot import ocelotLattice
    from ...Codes.Ocelot.bendmaps import transport_bend_maps
    from ...Codes.Ocelot.cavitymaps import stable_cavity_maps

    stable_cavity_maps()
    transport_bend_maps()
    model = types.SimpleNamespace(
        lat_obj=lattice.section.to_ocelot(save=False, Brho=lattice.rigidity(lattice.reference_p0c)),
        pin=types.SimpleNamespace(
            E=float(np.mean(lattice.global_parameters["beam"].energy.val)) * 1e-9
        ),
        rest_energy=lattice.rest_energy,
        elements=lattice.elements,
    )
    if lattice.tw_focusing:
        ocelotLattice.add_tw1_focusing(model)
    return model.lat_obj, model.pin.E


def _constraints(lattice, sequence, targets: dict) -> dict:
    """Ocelot ``constr`` for ``targets``, checking element names and Twiss keys."""
    from ocelot.cpbd.beam import Twiss

    twiss = Twiss()
    constr = {}
    for name, wanted in targets.items():
        bad = [k for k in wanted if not hasattr(twiss, k)]
        if bad:
            raise KeyError(f"match targets {bad} at {name!r} are not Twiss parameters")
        for key, value in wanted.items():
            if isinstance(value, (list, tuple)) and (len(value) != 2 or value[0] not in _OPERATORS):
                raise ValueError(
                    f"match target {name}.{key} = {value!r}: give a number, or "
                    f"[operator, number] with an operator from {_OPERATORS}"
                )
            if not isinstance(value, (int, float, list, tuple)):
                raise ValueError(f"match target {name}.{key} = {value!r} is not a number or [operator, number]")
        if name == "global":
            constr["global"] = dict(wanted)
            continue
        if name not in lattice.elements:
            raise KeyError(f"match target {name!r} is not in line {lattice.objectname!r}")
        # the element's end: its last Ocelot object
        element = [e for e in sequence if e.id == name][-1]
        constr[element] = {k: list(v) if isinstance(v, tuple) else v for k, v in wanted.items()}
    return constr


def _missed(value: float, target, tolerance: float = 1e-3) -> bool:
    """Whether ``value`` misses ``target`` (a number or ``[operator, number]``), to ``tolerance``."""
    if not isinstance(target, list):
        return abs(value - target) > tolerance * max(1.0, abs(target))
    op, bound = target
    slack = tolerance * max(1.0, abs(bound))
    value = abs(value) if op.startswith("a") else value
    return value > bound + slack if op.endswith("<") else value < bound - slack


def _warn_missed(lat, variables, solution, twiss, constr, line: str) -> None:
    """Warn about targets the solution misses: Ocelot returns its best try either way."""
    from copy import deepcopy
    from warnings import warn

    for variable, x in zip(variables, solution):
        for quad, coefficient in variable.items():
            quad.k1 = coefficient * x
    misses, tw = [], deepcopy(twiss)
    for element in lat.sequence:
        for tm in element.first_order_tms:
            tw = tm * tw
            for key, target in constr.get("global", {}).items():
                if isinstance(target, list) and _missed(getattr(tw, key), target):
                    misses.append(f"global {key} = {getattr(tw, key):.6g} at {element.id} (wanted {target})")
        for key, target in constr.get(element, {}).items():
            if _missed(getattr(tw, key), target):
                misses.append(f"{element.id}.{key} = {getattr(tw, key):.6g} (wanted {target})")
    if misses:
        warn(f"Line '{line}': match missed {'; '.join(dict.fromkeys(misses))}")


def transverse_match(lattice, settings: dict) -> dict:
    """
    Quadrupole strengths for a line's ``match`` block, from its input beam's Twiss.

    .. code-block:: yaml

        files:
          line:
            match:
              variables: [Q1, Q2, Q3]   # quadrupoles, or groups of them
              targets:
                SCR1: {beta_x: 10.0, alpha_x: 0.0}
                SCR2: {beta_y: [">", 8.0]}        # also "<", and "a<"/"a>" on |value|
                global: {beta_x: ["<", 50.0]}     # everywhere along the line
              max_iterations: 10000
              enable: true

    Targets are Ocelot Twiss parameters (``beta_x``, ``alpha_x``, ``Dx``, ``Dxp``,
    ``mux``, ...) at the end of each element. A quadrupole whose ``k1l`` is a
    functional definition keeps it: the definition is matched, and quadrupoles
    sharing it move together (equal ``k1l``); every element using it must be a variable.

    Parameters
    ----------
    lattice: :class:`~simba.Framework_objects.frameworkLattice`
        The line, with its input beam loaded.
    settings: dict
        The ``match`` block.

    Returns
    -------
    dict
        ``{quadrupole: {"magnetic.k1l": ...}}``, and any matched definitions under
        ``"functional_definitions"``.
    """
    from ocelot.cpbd.beam import Twiss
    from ocelot.cpbd.match import match

    if "variables" not in settings:
        raise KeyError("match needs `variables`, the quadrupoles to vary")
    if "targets" not in settings:
        raise KeyError("match needs `targets`")
    names = expand(lattice, settings["variables"])
    missing = [n for n in names if n not in lattice.elements]
    if missing:
        raise KeyError(f"match variables {missing} are not in line {lattice.objectname!r}")
    quads = [lattice.elements[n] for n in names]
    wrong = [q.name for q in quads if q.hardware_type != "Quadrupole"]
    if wrong:
        raise ValueError(
            f"match variables {wrong} are not quadrupoles; only quadrupoles change the "
            "linear optics that match targets"
        )

    lat, energy = _ocelot_model(lattice)
    sequence = lat.sequence
    constr = _constraints(lattice, sequence, settings["targets"])

    # one variable per quadrupole, or per definition its k1l shares: x is k1l
    knobs = {}
    for q in quads:
        k1l = q.magnetic.k1l
        knobs.setdefault(k1l if isinstance(k1l, str) else q.name, []).append(q)
    variables = [
        {o: 1.0 / float(q.magnetic.length) for q in members for o in sequence if o.id == q.name}
        for members in knobs.values()
    ]

    beam = lattice.global_parameters["beam"]
    twiss = Twiss(
        beta_x=float(beam.twiss.beta_x.val), alpha_x=float(beam.twiss.alpha_x.val),
        beta_y=float(beam.twiss.beta_y.val), alpha_y=float(beam.twiss.alpha_y.val),
        E=energy,
    )
    quiet =contextlib.redirect_stdout(io.StringIO()) if not lattice.verbose else contextlib.nullcontext()
    with quiet:
        solution = match(
            lat, constr, variables, twiss, verbose=False,
            max_iter=int(settings.get("max_iterations", 10000)),
        )

    _warn_missed(lat, variables, solution, twiss, constr, lattice.objectname)

    result, definitions = {}, {}
    for members, k1l in zip(knobs.values(), solution):
        for q in members:
            set_value(result, definitions, q, "magnetic.k1l", q.magnetic.k1l, float(k1l), "match")
    if definitions:
        check_definitions_stay_inside(lattice, definitions, set(names), "match")
        result[FUNCTIONAL] = definitions
    return result
