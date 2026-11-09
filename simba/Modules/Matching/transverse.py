"""
Transverse matching of element settings to Twiss and transfer-matrix targets, in any
code with a :mod:`~simba.Modules.Matching.backends` backend.
"""

from dataclasses import dataclass, field
from warnings import warn

from ._settings import FUNCTIONAL, check_definitions_stay_inside, expand, get_path, name_of, set_value
from .backends import MATRIX, OPERATORS, QUANTITIES, backends, residual

ALIASES = {"Dx": "eta_x", "Dxp": "eta_xp", "Dy": "eta_y", "Dyp": "eta_yp"}
"""Ocelot's names for SIMBA's Twiss quantities, still accepted."""

FALLBACK = ("ocelot", "xsuite", "madx", "cheetah")
"""Codes to try, in order, when the line's own cannot match."""


@dataclass
class Knob:
    """One variable: an element's ``field``, or the definition several elements' share."""

    name: str
    elements: list
    field: str
    """Dotted path on each element, e.g. ``magnetic.k1l``."""
    value: float
    definition: str | None = None

    @property
    def is_k1l(self) -> bool:
        """A quadrupole ``k1l``, which every code can vary natively."""
        return self.field == "magnetic.k1l" and all(
            getattr(e, "hardware_type", None) == "Quadrupole" for e in self.elements
        )


@dataclass
class Target:
    """``quantity op value`` at the end of element ``at``, or everywhere if ``at`` is None."""

    at: str | None
    quantity: str
    op: str
    value: float


@dataclass
class Spec:
    """A ``match`` block, checked against the line."""

    knobs: list
    targets: list
    twiss: dict
    """``beta_x``, ``alpha_x``, ``beta_y``, ``alpha_y`` of the input beam."""
    energy: float
    """Mean total energy of the input beam [eV]."""
    p0c: float
    """Reference momentum [eV/c]."""
    max_iterations: int = 10000
    code: str | None = None
    names: list = field(default_factory=list)
    """The variable elements."""

    @property
    def beta0(self) -> float:
        return self.p0c / self.energy


def _targets(lattice, block: dict) -> list:
    targets = []
    for at, wanted in block.items():
        if at != "global" and at not in lattice.elements:
            raise KeyError(f"match target {at!r} is not in line {lattice.objectname!r}")
        for key, value in wanted.items():
            quantity = ALIASES.get(key, key)
            if quantity not in QUANTITIES | MATRIX:
                raise KeyError(
                    f"match target {at}.{key} is not one of {sorted(QUANTITIES)} or R11 to R66"
                )
            if quantity in MATRIX and at == "global":
                raise ValueError(f"match target global.{key}: a transfer matrix is to an element")
            if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] in OPERATORS - {"="}:
                op, value = value
            elif isinstance(value, (int, float)):
                op = "="
            else:
                raise ValueError(
                    f"match target {at}.{key} = {value!r}: give a number, or "
                    f"[operator, number] with an operator from {sorted(OPERATORS - {'='})}"
                )
            targets.append(Target(None if at == "global" else at, quantity, op, float(value)))
    return targets


def _variables(lattice, items) -> list:
    """
    ``[(element, dotted field)]`` for ``variables``: ``NAME`` (a quadrupole's ``k1l``) or
    ``NAME.field``, where ``field`` is a path on the element or its ``magnetic``. A
    group with ``field`` of its own (a chicane's ``angle``) is itself the variable,
    set through the group; any other group's name stands for each member.
    """
    groups = getattr(lattice, "groupObjects", None) or {}
    pairs = []
    for item in [items] if isinstance(items, str) else items:
        name, _, field = item.partition(".")
        group = groups.get(name)
        if field and group is not None and isinstance(getattr(type(group), field, None), property):
            missing = [n for n in expand(lattice, name) if n not in lattice.elements]
            if missing:
                raise KeyError(f"match variable {item!r}: {missing} are not in line {lattice.objectname!r}")
            pairs.append((group, field))
            continue
        members = expand(lattice, name)
        missing = [n for n in members if n not in lattice.elements]
        if missing:
            raise KeyError(f"match variables {missing} are not in line {lattice.objectname!r}")
        for element in (lattice.elements[n] for n in members):
            if not field and element.hardware_type != "Quadrupole":
                raise ValueError(
                    f"match variable {element.name!r}: say which of its settings to vary, "
                    f"e.g. {element.name}.angle; only a quadrupole's (k1l) goes without saying"
                )
            path = None
            # magnetic first: an element also reads its magnetic fields as its own
            for candidate in ([f"magnetic.{field}", field] if field else ["magnetic.k1l"]):
                try:
                    value = get_path(element, candidate)
                except AttributeError:
                    continue
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    path = candidate
                    break
            if path is None:
                raise ValueError(f"match variable {item!r}: {element.name} has no number {field or 'k1l'}")
            pairs.append((element, path))
    return pairs


def parse(lattice, settings: dict) -> Spec:
    """The :class:`Spec` of a ``match`` block, from the line and its input beam."""
    import numpy as np

    if "variables" not in settings:
        raise KeyError("match needs `variables`, the element settings to vary")
    if "targets" not in settings:
        raise KeyError("match needs `targets`")
    from laura.models.base_models import resolve_functional_parameter

    pairs = _variables(lattice, settings["variables"])
    knobs = {}
    for element, path in pairs:
        current = get_path(element, path)
        key = current if isinstance(current, str) else f"{name_of(element)}.{path}"
        if key not in knobs:
            knobs[key] = Knob(key, [], path, float(resolve_functional_parameter(current, force=True)),
                              current if isinstance(current, str) else None)
        elif knobs[key].field != path:
            raise ValueError(f"match variable {key!r} is both {knobs[key].field} and {path}")
        knobs[key].elements.append(element)
    names = list(dict.fromkeys(name_of(e) for e, _ in pairs))
    beam = lattice.global_parameters["beam"]
    return Spec(
        knobs=list(knobs.values()),
        targets=_targets(lattice, settings["targets"]),
        twiss={k: float(getattr(beam.twiss, k).val) for k in ("beta_x", "alpha_x", "beta_y", "alpha_y")},
        energy=float(np.mean(beam.energy.val)),
        p0c=float(lattice.reference_p0c),
        max_iterations=int(settings.get("max_iterations", 10000)),
        code=settings.get("code"),
        names=names,
    )


def choose(lattice, spec: Spec):
    """
    The backend to match with: ``code`` if the block names one, else the line's own
    code, else the first of :data:`FALLBACK` that can.
    """
    every = backends()
    if spec.code is not None:
        if spec.code not in every:
            raise ValueError(f"match code {spec.code!r} has no matcher; codes are {sorted(every)}")
        reasons = every[spec.code].cannot(lattice, spec)
        if reasons:
            raise ValueError(f"match cannot run in {spec.code}: {'; '.join(reasons)}")
        return every[spec.code]
    own = every.get(lattice.code)
    if own is not None and not own.cannot(lattice, spec):
        return own
    for code in FALLBACK:
        if not every[code].cannot(lattice, spec):
            if own is not None:
                warn(
                    f"Line '{lattice.objectname}': {lattice.code} cannot match this "
                    f"({'; '.join(own.cannot(lattice, spec))}); matching in {code}"
                )
            return every[code]
    raise ValueError(
        "no code can match this: "
        + "; ".join(f"{c}: {', '.join(every[c].cannot(lattice, spec))}" for c in FALLBACK)
    )


def _warn_fixed_edges(lattice, spec: Spec) -> None:
    """
    Warn of dipoles whose angle is matched but whose edge angles are numbers, which
    stay as they are: an edge angle written with ``angle`` (``"angle/2"``) follows it.
    """
    dipoles = []
    for knob in spec.knobs:
        if knob.field == "magnetic.angle":
            dipoles += knob.elements
        elif knob.field == "angle" and hasattr(knob.elements[0], "change_Parameter"):
            dipoles += [lattice.elements[n] for n in expand(lattice, knob.elements[0].objectname)]
    fixed = [
        d.name for d in dipoles
        if any(not isinstance(e, str) and e for e in (d.magnetic.entrance_edge_angle, d.magnetic.exit_edge_angle))
    ]
    if fixed:
        warn(
            f"Line '{lattice.objectname}': match varies the angle of {fixed}, whose edge "
            "angles are numbers and stay as they are; write them with `angle` (e.g. "
            "'angle' or 'angle/2') to have them follow"
        )


def _warn_missed(backend, values, line: str) -> None:
    """Warn about targets the solution misses: every matcher returns its best try."""
    points = backend.twiss(values)
    ends = dict(points)
    misses = []
    for t in backend.spec.targets:
        wanted = t.value if t.op == "=" else [t.op, t.value]
        if t.at is None:
            worst = max(points, key=lambda p: abs(residual(p[1][t.quantity], t.op, t.value)))
            if abs(residual(worst[1][t.quantity], t.op, t.value)) > 1e-3:
                misses.append(f"global {t.quantity} = {worst[1][t.quantity]:.6g} at {worst[0]} (wanted {wanted})")
        elif abs(residual(ends[t.at][t.quantity], t.op, t.value)) > 1e-3:
            misses.append(f"{t.at}.{t.quantity} = {ends[t.at][t.quantity]:.6g} (wanted {wanted})")
    if misses:
        warn(f"Line '{line}': match missed {'; '.join(misses)}")


def transverse_match(lattice, settings: dict) -> dict:
    """
    Element settings for a line's ``match`` block, from its input beam's Twiss.

    .. code-block:: yaml

        files:
          line:
            match:
              variables: [Q1, Q2, B1.angle, SOL1.fields.S0L]   # or groups
              targets:
                SCR1: {beta_x: 10.0, alpha_x: 0.0}
                SCR2: {beta_y: [">", 8.0]}     # also "<", and "a<"/"a>" on |value|
                SCR3: {R56: -0.05, R12: 0.0}   # transfer matrix from the line's start
                global: {beta_x: ["<", 50.0]}  # everywhere along the line
              code: xsuite                     # optional; by default the line's own
              max_iterations: 10000
              enable: true

    A variable is ``NAME`` (a quadrupole's ``k1l``) or ``NAME.field``, a number on
    the element or its ``magnetic``. Quadrupole ``k1l`` is varied in each code's own
    model; anything else by rebuilding the model with LAURA's exporters, which is
    slower, and with the shared solver.

    Targets are SIMBA's Twiss quantities (``beta_x``, ``alpha_x``, ``eta_x``,
    ``eta_xp``, ``mux`` in units of 2 pi, ...;) at the end of each element, and
    ``R11`` to ``R66``, the transfer matrix from the line's start, in
    ``(x, px, y, py, zeta, delta)``. The match runs in the named code, else the
    line's own, else the first of :data:`FALLBACK` that can (see each backend's
    capabilities), and warns of any target it misses. A quadrupole whose ``k1l``
    is a functional definition keeps it: the definition is matched, and quadrupoles
    sharing it move together.

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
    spec = parse(lattice, settings)
    _warn_fixed_edges(lattice, spec)
    backend = choose(lattice, spec)(lattice, spec)
    values = backend.solve()
    _warn_missed(backend, values, lattice.objectname)
    if lattice.verbose:
        print(f"Line '{lattice.objectname}': matched in {backend.code}")

    result, definitions = {}, {}
    for knob, value in zip(spec.knobs, values):
        for element in knob.elements:
            set_value(result, definitions, element, knob.field, get_path(element, knob.field), float(value), "match")
    if definitions:
        check_definitions_stay_inside(lattice, definitions, set(spec.names), "match")
        result[FUNCTIONAL] = definitions
    return result
