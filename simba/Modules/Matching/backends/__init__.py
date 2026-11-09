"""
The codes a transverse match can run in, each a :class:`Backend`.

A backend builds its code's model of the line from the line's LAURA section, and
gives the Twiss and transfer matrices along it for given knob values; it may also
solve with its code's own matcher, else :meth:`Backend.solve` minimises the targets'
residuals itself. Quantities are SIMBA's Twiss names, phase advances in units of
2 pi, and ``Rij`` in Xsuite's ``(x, px, y, py, zeta, delta)``.
"""

import importlib

import numpy as np

QUANTITIES = frozenset({
    "beta_x", "beta_y", "alpha_x", "alpha_y",
    "eta_x", "eta_xp", "eta_y", "eta_yp", "mux", "muy",
})
"""The Twiss quantities a match can target."""

MATRIX = frozenset(f"R{i}{j}" for i in range(1, 7) for j in range(1, 7))
"""The transfer-matrix elements a match can target, 1-based."""

OPERATORS = frozenset({"=", "<", ">", "a<", "a>"})
"""``=``, bounds, and bounds on the absolute value."""


def residual(value: float, op: str, target: float) -> float:
    """How far ``value`` misses ``op target``, relative to ``max(1, |target|)``; 0 if met."""
    scale = max(1.0, abs(target))
    if op == "=":
        return (value - target) / scale
    value = abs(value) if op.startswith("a") else value
    miss = value - target if op.endswith("<") else target - value
    return max(0.0, miss) / scale


def matrix_entries(matrix) -> dict:
    """``{"R11": ..., ..., "R66": ...}`` of a 6x6 matrix."""
    matrix = np.asarray(matrix, dtype=float)
    return {f"R{i + 1}{j + 1}": float(matrix[i, j]) for i in range(6) for j in range(6)}


class Backend:
    """
    One code's matcher.

    Subclasses set the class attributes, :meth:`build`, :meth:`set_native` and
    :meth:`points`, and may give :meth:`native_solve` with the code's own matcher.

    Knobs that are quadrupole ``k1l`` are set in the code's model directly
    (:meth:`set_native`); any other knob by setting it on the LAURA element for a
    moment and building the model again (:meth:`build`), which works for any field
    LAURA exports.

    Parameters
    ----------
    lattice: :class:`~simba.Framework_objects.frameworkLattice`
        The line, with its input beam loaded.
    spec: :class:`~simba.Modules.Matching.transverse.Spec`
        What to match.
    """

    code: str = ""
    """The code, as a line's ``code`` names it."""
    package: str = ""
    """The Python package the backend needs."""
    quantities: frozenset = QUANTITIES
    """The Twiss quantities it can target."""
    matrix: bool = True
    """Whether it can target transfer-matrix elements."""
    operators: frozenset = OPERATORS
    """The comparisons it can target."""
    global_targets: bool = True
    """Whether it takes targets along the whole line."""
    acceleration: bool = True
    """Whether its optics follow the energy cavities give."""
    lattice_class: str = ""
    """Its lattice class in :mod:`simba.Framework_lattices`, for its coordinate convention."""

    def __init__(self, lattice, spec):
        from .._settings import temporarily

        self.lattice, self.spec = lattice, spec
        self.native = all(k.is_k1l for k in spec.knobs)
        self.matrix_at = sorted({t.at for t in spec.targets if t.quantity in MATRIX})
        self.scaling = None
        if self.matrix_at:
            from .... import Framework_lattices
            from ...Matrices import canonical_scaling

            cls = getattr(Framework_lattices, self.lattice_class)
            self.scaling = canonical_scaling(cls.otm_longitudinal_sign, cls.otm_longitudinal_scale, spec.beta0)
        with temporarily(lattice, spec.knobs, [k.value for k in spec.knobs]):
            self.build()

    @classmethod
    def available(cls) -> bool:
        """Whether its package is installed."""
        try:
            importlib.import_module(cls.package)
        except ImportError:
            return False
        return True

    @classmethod
    def cannot(cls, lattice, spec) -> list:
        """Why it cannot do ``spec`` on ``lattice``: empty if it can."""
        reasons = []
        if not cls.available():
            reasons.append(f"{cls.package} is not installed")
        quantities = {t.quantity for t in spec.targets} - cls.quantities - (MATRIX if cls.matrix else set())
        if quantities:
            reasons.append(f"it cannot target {sorted(quantities)}")
        operators = {t.op for t in spec.targets} - cls.operators
        if operators:
            reasons.append(f"it cannot compare with {sorted(operators)}")
        if not cls.global_targets and any(t.at is None for t in spec.targets):
            reasons.append("it takes no global targets")
        if not cls.acceleration and lattice.rf_voltage:
            reasons.append("its optics ignore the cavities' energy gain")
        return reasons

    def build(self) -> None:
        """Build the model from the LAURA elements as they stand, wiring any :meth:`set_native` knobs."""
        raise NotImplementedError

    def set_native(self, values) -> None:
        """Set the (quadrupole ``k1l``) knobs in the model."""
        raise NotImplementedError

    def points(self) -> list:
        """
        ``[(element, {quantity: value})]`` at each element's end, as the model stands;
        with this code's ``Rij`` at :attr:`matrix_at`.
        """
        raise NotImplementedError

    def twiss(self, values) -> list:
        """:meth:`points` with the knobs at ``values``, ``Rij`` in Xsuite's coordinates."""
        from .._settings import temporarily

        if self.native:
            self.set_native(values)
        else:
            with temporarily(self.lattice, self.spec.knobs, values):
                self.build()
        points = self.points()
        if self.scaling is not None:
            d = self.scaling
            for _, quantities in points:
                for key in MATRIX & quantities.keys():
                    i, j = int(key[1]) - 1, int(key[2]) - 1
                    quantities[key] *= d[i] / d[j]
        return points

    def code_value(self, quantity: str, value: float) -> float:
        """A target in Xsuite's coordinates, in this code's: the inverse of :meth:`twiss`'s conversion."""
        if quantity not in MATRIX or self.scaling is None:
            return value
        i, j = int(quantity[1]) - 1, int(quantity[2]) - 1
        return value * self.scaling[j] / self.scaling[i]

    def residuals(self, values) -> np.ndarray:
        """Every target's :func:`residual`; a global target's at every point."""
        points = self.twiss(values)
        ends = dict(points)
        out = []
        for t in self.spec.targets:
            if t.at is None:
                out += [residual(p[t.quantity], t.op, t.value) for _, p in points]
            else:
                out.append(residual(ends[t.at][t.quantity], t.op, t.value))
        return np.array(out)

    def native_solves(self) -> bool:
        """Whether :meth:`native_solve` can take this spec; by default, any native-knob spec."""
        return self.native

    def native_solve(self) -> list:
        """Knob values from the code's own matcher."""
        raise NotImplementedError

    def solve(self) -> list:
        """Knob values: the code's own matcher if it can, else least squares on :meth:`residuals`."""
        if type(self).native_solve is not Backend.native_solve and self.native_solves():
            return self.native_solve()
        from scipy.optimize import least_squares

        x0 = np.array([k.value for k in self.spec.knobs], dtype=float)
        fit = least_squares(
            self.residuals, x0, x_scale=np.maximum(np.abs(x0), 0.1),
            max_nfev=self.spec.max_iterations, xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        return list(fit.x)


def backends() -> dict:
    """Every backend, by code."""
    from .cheetah import CheetahBackend
    from .madx import MadxBackend
    from .ocelot import OcelotBackend
    from .xsuite import XsuiteBackend

    return {b.code: b for b in (OcelotBackend, XsuiteBackend, MadxBackend, CheetahBackend)}
