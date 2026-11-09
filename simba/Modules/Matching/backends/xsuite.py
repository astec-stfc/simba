"""Matching with Xsuite's ``line.match``, on SIMBA's Xsuite model of the line."""

import contextlib
import io
import types
import warnings

import numpy as np

from . import MATRIX, Backend, matrix_entries

NAMES = {
    "beta_x": "betx", "beta_y": "bety", "alpha_x": "alfx", "alpha_y": "alfy",
    "eta_x": "dx", "eta_xp": "dpx", "eta_y": "dy", "eta_yp": "dpy", "mux": "mux", "muy": "muy",
}
"""SIMBA's Twiss names, as Xsuite's."""


class XsuiteBackend(Backend):
    code = "xsuite"
    package = "xtrack"
    lattice_class = "xsuiteLattice"

    def build(self) -> None:
        import xtrack as xt

        from ....Codes.Xsuite.Xsuite import xsuiteLattice

        lattice, spec = self.lattice, self.spec
        particle_ref = xt.Particles(p0c=[spec.p0c], mass0=lattice.rest_energy, q0=lattice.reference_charge)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Xsuite is written without structure wakefields")
            line = lattice.section.to_xsuite(1, env=xt.Environment(), particle_ref=particle_ref, save=False)
        if not lattice.fixed_reference and lattice.rf_voltage:
            model = types.SimpleNamespace(line=line, names=None, reference_at_start=None)
            xsuiteLattice.insert_reference_energy_increases(model)
            line = model.line
        if line.tracker is None:
            line.build_tracker()
        self.line = line
        # a k1l knob is a variable; each quadrupole's k1 is it over its length
        self.vars = [f"_match_{i}" for i in range(len(spec.knobs))] if self.native else []
        for var, knob in zip(self.vars, spec.knobs):
            line.vars[var] = knob.value
            for e in knob.elements:
                line.element_refs[e.name].k1 = line.vars[var] / float(e.magnetic.length)
        self.kwargs = {
            "method": "6d", "freeze_energy": False, "compute_R_element_by_element": False,
            "betx": spec.twiss["beta_x"], "alfx": spec.twiss["alpha_x"],
            "bety": spec.twiss["beta_y"], "alfy": spec.twiss["alpha_y"],
        }
        # a row is at an element's entrance: an element's end is the next row
        names = list(line.element_names) + ["_end_point"]
        self.end_row = dict(zip(names, names[1:]))

    def _quiet(self):
        return contextlib.nullcontext() if self.lattice.verbose else contextlib.redirect_stdout(io.StringIO())

    def set_native(self, values) -> None:
        for var, value in zip(self.vars, values):
            self.line.vars[var] = float(value)

    def _matrix(self, name: str) -> np.ndarray:
        """The transfer matrix from the line's start to ``name``'s end."""
        end = self.end_row[name]
        bounds = {} if end == "_end_point" else {"start": self.line.element_names[0], "end": end}
        result = self.line.compute_R_matrix(particle_on_co=self.line.particle_ref.copy(), **bounds)
        return np.asarray(result["R_matrix"], dtype=float)

    def points(self) -> list:
        with self._quiet():
            tw = self.line.twiss(**self.kwargs)
        rows = {n: i for i, n in enumerate(tw.name)}
        points = []
        for name, end in self.end_row.items():
            quantities = {q: float(tw[n][rows[end]]) for q, n in NAMES.items()}
            if name in self.matrix_at:
                quantities.update(matrix_entries(self._matrix(name)))
            points.append((name, quantities))
        return points

    def _target(self, t):
        import xtrack as xt

        value = self.code_value(t.quantity, t.value)
        compare = {"=": lambda v: v, "<": xt.LessThan, ">": xt.GreaterThan}[t.op.lstrip("a")]
        magnitude = abs if t.op.startswith("a") else (lambda v: v)
        if t.quantity in MATRIX:
            i, j, name = int(t.quantity[1]) - 1, int(t.quantity[2]) - 1, t.at
            return xt.Target(lambda tw: magnitude(float(self._matrix(name)[i, j])), compare(value))
        column = NAMES[t.quantity]
        if t.at is None:
            pick = np.max if t.op.endswith("<") else np.min
            return xt.Target(lambda tw: float(pick(np.abs(tw[column]) if t.op.startswith("a") else tw[column])),
                             compare(value))
        at = self.end_row[t.at]
        if t.op.startswith("a"):
            return xt.Target(lambda tw: abs(float(tw[column, at])), compare(value))
        return xt.Target(column, compare(value), at=at)

    def native_solve(self) -> list:
        import xtrack as xt

        with self._quiet():
            self.line.match(
                vary=[xt.Vary(v, step=1e-6) for v in self.vars],
                targets=[self._target(t) for t in self.spec.targets],
                solve=True, verbose=False, n_steps_max=self.spec.max_iterations, **self.kwargs,
            )
        return [float(self.line.vars[v]._value) for v in self.vars]
