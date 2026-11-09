"""Matching with MAD-X's ``MATCH`` (LMDIF), through cpymad, on LAURA's MAD-X sequence of the line."""

from . import MATRIX, Backend

NAMES = {
    "beta_x": "betx", "beta_y": "bety", "alpha_x": "alfx", "alpha_y": "alfy",
    "eta_x": "dx", "eta_xp": "dpx", "eta_y": "dy", "eta_yp": "dpy", "mux": "mux", "muy": "muy",
    **{q: f"re{q[1:]}" for q in MATRIX},
}
"""SIMBA's Twiss names, as MAD-X's."""


class MadxBackend(Backend):
    code = "madx"
    package = "cpymad"
    operators = frozenset({"=", "<", ">"})
    acceleration = False
    lattice_class = "madxLattice"

    def build(self) -> None:
        from cpymad.madx import Madx
        from laura.translator.utils.functions import sanitize_string

        lattice, spec = self.lattice, self.spec
        if getattr(self, "madx", None) is not None:
            self.madx.quit()
        charge = 1 if lattice.reference_charge > 0 else -1
        beam = {"mass": lattice.rest_energy / 1e9, "charge": charge, "pc": spec.p0c / 1e9}
        self.madx = Madx(stdout=False)
        self.madx.input(lattice.section.to_madx(beam=beam, Brho=lattice.rigidity(spec.p0c)))
        self.sequence = sanitize_string(lattice.section.name).lower()
        self.name = {e: sanitize_string(e).lower() for e in lattice.elements}
        # a k1l knob is a variable; each quadrupole's k1 is it over its length
        self.vars = [f"match_knob_{i}" for i in range(len(spec.knobs))] if self.native else []
        for var, knob in zip(self.vars, spec.knobs):
            self.madx.input(f"{var} = {knob.value};")
            for e in knob.elements:
                # not `name->k1 := ...`, which MAD-X accepts and ignores here
                self.madx.input(f"{self.name[e.name]}, k1 := {var} / {float(e.magnetic.length)};")
        t = spec.twiss
        self.initial = f"betx={t['beta_x']}, alfx={t['alpha_x']}, bety={t['beta_y']}, alfy={t['alpha_y']}"
        if self.matrix_at:
            self.initial += ", rmatrix"

    def set_native(self, values) -> None:
        for var, value in zip(self.vars, values):
            self.madx.globals[var] = float(value)

    def points(self) -> list:
        self.madx.input(f"twiss, sequence={self.sequence}, {self.initial};")
        table = self.madx.table.twiss
        laura = {v: k for k, v in self.name.items()}
        wanted = {q: n for q, n in NAMES.items() if q not in MATRIX or self.matrix_at}
        points = []
        for i, row in enumerate(table.name):
            name = laura.get(str(row).split(":")[0])
            if name is not None:
                points.append((name, {q: float(table[n][i]) for q, n in wanted.items()}))
        return points

    def native_solve(self) -> list:
        lines = [f"match, sequence={self.sequence}, {self.initial};"]
        lines += [f"vary, name={v}, step=1e-8;" for v in self.vars]
        for t in self.spec.targets:
            where = "#s/#e" if t.at is None else self.name[t.at]
            value = self.code_value(t.quantity, t.value)
            lines.append(f"constraint, sequence={self.sequence}, range={where}, {NAMES[t.quantity]}{t.op}{value};")
        lines += [f"lmdif, calls={self.spec.max_iterations}, tolerance=1e-20;", "endmatch;"]
        self.madx.input("\n".join(lines))
        return [float(self.madx.globals[v]) for v in self.vars]
