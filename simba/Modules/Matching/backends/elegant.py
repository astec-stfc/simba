"""
Matching with elegant's optimiser, on LAURA's elegant lattice, run in a scratch directory.

Elegant optimises against values at ``fitpoint`` markers, so one goes after each
target element. Its fifth coordinate is path length, not a rescaling of Xsuite's
``zeta``: ``R5j = -R5j(elegant)`` for ``j`` = 1..4 and ``R56 = -R56(elegant) + s / gamma0**2``,
true without RF, so with RF it refuses targets in row or column 5.
"""

import os
import re
import shutil
import subprocess
import tempfile
import weakref

import numpy as np

from . import MATRIX, Backend, matrix_entries

COLUMNS = {
    "beta_x": "betax", "beta_y": "betay", "alpha_x": "alphax", "alpha_y": "alphay",
    "eta_x": "etax", "eta_xp": "etaxp", "eta_y": "etay", "eta_yp": "etayp", "mux": "psix", "muy": "psiy",
}
"""SIMBA's Twiss names, as elegant's columns and optimisation quantities."""

SCALE = {"mux": 2 * np.pi, "muy": 2 * np.pi}
"""Elegant's phase advance is in radians."""

GLOBAL = {"beta_x": "betax", "beta_y": "betay", "eta_x": "etax", "eta_y": "etay"}
"""Quantities elegant gives the ``max.`` and ``min.`` of along the line."""


def _row_column_5(quantity: str) -> bool:
    return quantity in MATRIX and "5" in quantity[1:]


class ElegantBackend(Backend):
    code = "elegant"
    package = "elegant"
    lattice_class = "elegantLattice"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("elegant") is not None

    @classmethod
    def cannot(cls, lattice, spec) -> list:
        reasons = super().cannot(lattice, spec)
        if lattice.rf_voltage and any(_row_column_5(t.quantity) for t in spec.targets):
            reasons.append("with RF, its path length cannot be read as Xsuite's zeta (row and column 5)")
        return reasons

    def build(self) -> None:
        if not hasattr(self, "workdir"):
            self.workdir = tempfile.mkdtemp(prefix="simba_match_elegant_")
            weakref.finalize(self, shutil.rmtree, self.workdir, True)
            self.values = [k.value for k in self.spec.knobs]
        lattice = self.lattice
        charge = abs(lattice.global_parameters["beam"].Q)
        text = re.sub(r"&\s*\n", "", lattice.section.to_elegant(charge=charge))
        lines = list(re.finditer(r"^(\S+)\s*:\s*LINE\s*=\s*\((.*?)\)", text, re.M | re.S))
        self.beamline = lines[-1].group(1)
        body = lines[-1].group(2)
        markers, self.fitpoint = [], {}
        for i, name in enumerate(dict.fromkeys(t.at for t in self.spec.targets if t.at is not None)):
            marker = f"SIMBA_FIT_{i}"
            markers.append(f"{marker}: marker, fitpoint = 1\n")
            body = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", f"{name}, {marker}", body)
            self.fitpoint[name] = marker
        start, end = lines[-1].span(2)
        text = "".join(markers) + text[:start] + body + text[end:]
        with open(os.path.join(self.workdir, "lattice.lte"), "w") as f:
            f.write(text)
        self.gamma0 = self.spec.energy / lattice.rest_energy

    def set_native(self, values) -> None:
        self.values = [float(v) for v in values]

    def _deck(self, middle: str, matrix: bool) -> str:
        t, spec = self.spec.twiss, self.spec
        alters = ""
        if self.native:
            for knob, value in zip(spec.knobs, self.values):
                for e in knob.elements:
                    alters += f"&alter_elements name = {e.name}, item = K1, value = {value / float(e.magnetic.length)} &end\n"
        return (
            f"&run_setup lattice = \"lattice.lte\", use_beamline = \"{self.beamline}\", "
            f"p_central_mev = {spec.p0c / 1e6}, default_order = 2 &end\n"
            f"{alters}"
            f"&twiss_output matched = 0, beta_x = {t['beta_x']}, alpha_x = {t['alpha_x']}, "
            f"beta_y = {t['beta_y']}, alpha_y = {t['alpha_y']}, filename = \"%s.twi\" &end\n"
            + ("&matrix_output SDDS_output = \"%s.mat\", full_matrix_only = 0, SDDS_output_order = 1 &end\n"
               if matrix else "")
            + "&run_control &end\n" + middle
        )

    def _run(self, deck: str, stem: str) -> None:
        with open(os.path.join(self.workdir, f"{stem}.ele"), "w") as f:
            f.write(deck)
        done = subprocess.run(["elegant", f"{stem}.ele"], cwd=self.workdir, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"elegant failed: {(done.stdout + done.stderr)[-2000:]}")

    def _read(self, path: str) -> dict:
        from ...SDDSFile import SDDSFile

        sdds = SDDSFile(index=1)
        sdds.read_file(path)
        return sdds.data

    def points(self) -> list:
        self._run(self._deck("&bunched_beam &end\n&track &end\n", bool(self.matrix_at)), "twiss")
        twi = self._read(os.path.join(self.workdir, "twiss.twi"))
        mat = self._read(os.path.join(self.workdir, "twiss.mat")) if self.matrix_at else None
        laura = {e.upper(): e for e in self.lattice.elements}
        points, self.s_end = [], {}
        for i, raw in enumerate(twi["ElementName"]):
            name = laura.get(str(raw).upper())
            if name is None:
                continue
            self.s_end[name] = float(twi["s"][i])
            quantities = {q: float(twi[c][i]) / SCALE.get(q, 1.0) for q, c in COLUMNS.items()}
            if name in self.matrix_at:
                rows = [j for j, n in enumerate(mat["ElementName"]) if str(n).upper() == name.upper()]
                r = np.array([[mat[f"R{a}{b}"][rows[-1]] for b in range(1, 7)] for a in range(1, 7)], dtype=float)
                quantities.update(matrix_entries(self._canonical(r, float(twi["s"][i]))))
            points.append((name, quantities))
        return points

    def _canonical(self, r: np.ndarray, s: float) -> np.ndarray:
        """Elegant's matrix to ``s`` in Xsuite's coordinates: see the module docstring."""
        r = r.copy()
        r[4, :4] *= -1
        r[4, 5] = -r[4, 5] + s / self.gamma0**2
        return r

    def _elegant_value(self, t) -> float:
        """A target in elegant's coordinates: the inverse of :meth:`_canonical`."""
        if t.quantity in ("R51", "R52", "R53", "R54"):
            return -t.value
        if t.quantity == "R56":
            return -(t.value - self.s_end[t.at] / self.gamma0**2)
        return t.value * SCALE.get(t.quantity, 1.0)

    def native_solves(self) -> bool:
        return self.native and all(
            t.at is not None or (t.quantity in GLOBAL and t.op in ("<", ">")) for t in self.spec.targets
        )

    def native_solve(self) -> list:
        spec = self.spec
        if self.matrix_at:
            self.points()  # for each element's s, which R56's conversion needs
        middle = (
            "&optimization_setup mode = \"minimize\", method = \"simplex\", target = 0, tolerance = 1e-16, "
            f"n_passes = 5, n_evaluations = {spec.max_iterations}, n_restarts = 2, verbose = 0, "
            "log_file = \"/dev/null\" &end\n"
        )
        for knob in spec.knobs:
            first, length = knob.elements[0], float(knob.elements[0].magnetic.length)
            k1 = knob.value / length
            middle += (
                f"&optimization_variable name = {first.name}, item = K1, lower_limit = -1e3, upper_limit = 1e3, "
                f"step_size = {max(1e-3, 0.01 * abs(k1))} &end\n"
            )
            for other in knob.elements[1:]:
                middle += (
                    f"&optimization_covariable name = {other.name}, item = K1, "
                    f"equation = \"{first.name}.K1 {length} * {float(other.magnetic.length)} /\" &end\n"
                )
        for t in spec.targets:
            tolerance = 1e-6 * max(1.0, abs(t.value))
            if t.at is None:
                quantity = f"{'max' if t.op == '<' else 'min'}.{GLOBAL[t.quantity]}"
            else:
                quantity = f"{self.fitpoint[t.at]}#1.{t.quantity if t.quantity in MATRIX else COLUMNS[t.quantity]}"
            if t.op.startswith("a"):
                quantity += " abs"
            value = self._elegant_value(t) if t.at is not None else t.value
            function = {"=": "sene", "<": "selt", ">": "segt"}[t.op.lstrip("a")]
            middle += f"&optimization_term term = \"{quantity} {value} {tolerance} {function}\" &end\n"
        middle += "&bunched_beam &end\n&optimize &end\n&save_lattice filename = \"%s.new\" &end\n"
        self._run(self._deck(middle, False), "match")
        with open(os.path.join(self.workdir, "match.new")) as f:
            saved = re.sub(r"&\s*\n", "", f.read())
        values = []
        for knob in spec.knobs:
            first = knob.elements[0]
            found = re.search(rf"^{re.escape(first.name.upper())}\s*:.*?\bK1=([^,\n]+)", saved, re.M | re.I)
            values.append(float(found.group(1)) * float(first.magnetic.length) if found else 0.0)
        self.values = values
        return values
