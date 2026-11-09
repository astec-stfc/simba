"""
Matching with Tao's optimiser (LMDIF), through pytao, on LAURA's Bmad lattice.

Each quadrupole ``k1l`` knob is a Bmad overlay (``SIMBA_KNOB_i``, var ``k1l``) over
its quadrupoles' ``K1``, so one Tao variable moves them all; targets are Tao data.
Bmad's coordinates are Xsuite's, so matrices need no conversion.
"""

import contextlib
import os
import shutil
import tempfile
import weakref

import numpy as np

from . import MATRIX, Backend, matrix_entries

DATA = {
    "beta_x": "beta.a", "beta_y": "beta.b", "alpha_x": "alpha.a", "alpha_y": "alpha.b",
    "eta_x": "eta.x", "eta_xp": "etap.x", "eta_y": "eta.y", "eta_yp": "etap.y",
    "mux": "phase.a", "muy": "phase.b", **{q: f"r.{q[1:]}" for q in MATRIX},
}
"""SIMBA's names, as Tao data types."""

LAT_LIST = {
    "beta_x": "ele.a.beta", "beta_y": "ele.b.beta", "alpha_x": "ele.a.alpha", "alpha_y": "ele.b.alpha",
    "eta_x": "ele.x.eta", "eta_xp": "ele.x.etap", "eta_y": "ele.y.eta", "eta_yp": "ele.y.etap",
    "mux": "ele.a.phi", "muy": "ele.b.phi",
}
"""SIMBA's Twiss names, as Tao ``lat_list`` columns."""

SCALE = {"mux": 2 * np.pi, "muy": 2 * np.pi}
"""Tao's phase advance is in radians."""

MERIT = {"=": "target", "<": "max", ">": "min", "a<": "abs_max", "a>": "abs_min"}
"""SIMBA's comparisons, as Tao merit types."""


def libtao(lattice) -> str | None:
    """``libtao.so``: the line's own (a Bmad line's), SIMBA's, or ``$ACC_ROOT_DIR``'s."""
    candidates = [getattr(lattice, "libtao", None)]
    try:
        candidates.append(lattice.executables["tao"][0])
    except (KeyError, TypeError, IndexError, AttributeError):
        pass
    if os.environ.get("ACC_ROOT_DIR"):
        candidates.append(os.path.join(os.environ["ACC_ROOT_DIR"], "production", "lib", "libtao.so"))
    return next((c for c in candidates if c and os.path.isfile(c)), None)


class BmadBackend(Backend):
    code = "bmad"
    package = "pytao"
    lattice_class = "bmadLattice"

    @classmethod
    def cannot(cls, lattice, spec) -> list:
        reasons = super().cannot(lattice, spec)
        if cls.available() and libtao(lattice) is None:
            reasons.append("libtao.so is not found (set ACC_ROOT_DIR)")
        bent = [n for k in spec.knobs if k.field == "magnetic.angle" for n in (e.name for e in k.elements)]
        if bent:
            reasons.append(
                f"it keeps elements where they are placed, so the angle of {bent} cannot change "
                "alone (a chicane's angle, which lays them out, can)"
            )
        return reasons

    def build(self) -> None:
        from laura.models.simulation import TwissMatchSimulationElement
        from pytao import Tao

        lattice, spec = self.lattice, self.spec
        if not hasattr(self, "workdir"):
            self.workdir = tempfile.mkdtemp(prefix="simba_match_tao_")
            weakref.finalize(self, shutil.rmtree, self.workdir, True)
        section = lattice.section
        saved = section.reference_energy
        section.reference_energy = spec.energy
        try:
            text = section.to_bmad(
                particle=lattice.global_parameters["beam"].species,
                initial_twiss=TwissMatchSimulationElement(**spec.twiss),
            )
        finally:
            section.reference_energy = saved
        overlays = ""
        if self.native:
            for i, knob in enumerate(spec.knobs):
                slaves = ", ".join(f"{e.name}[K1]: k1l / {float(e.magnetic.length)}" for e in knob.elements)
                overlays += f"SIMBA_KNOB_{i}: overlay = {{{slaves}}}, var = {{k1l}}, k1l = {knob.value}\n"
        use = text.rindex("use,")
        # a new name each build: Bmad otherwise reuses its digested copy of a file rewritten within a second
        self.builds = getattr(self, "builds", -1) + 1
        self.lattice_file = f"lattice_{self.builds}.bmad"
        with open(os.path.join(self.workdir, self.lattice_file), "w") as f:
            f.write(text[:use] + overlays + text[use:])
        with open(os.path.join(self.workdir, "tao.init"), "w") as f:
            f.write(self._init())
        with contextlib.chdir(self.workdir):
            if getattr(self, "tao", None) is None:
                self.tao = Tao("-init tao.init -noplot", so_lib=libtao(lattice) or "")
            else:
                self.tao.init("-init tao.init -noplot")

    def _init(self) -> str:
        """The Tao init file: the lattice, the optimiser, and the targets and knobs as data and variables."""
        spec = self.spec
        text = (
            f'&tao_design_lattice\n  design_lattice(1)%file = "{self.lattice_file}"\n/\n'
            "&tao_params\n  global%optimizer = 'lmdif'\n"
            f"  global%n_opti_cycles = {spec.max_iterations}\n  global%n_opti_loops = 20\n/\n"
        )
        if not spec.targets or not self.native:
            return text
        text += "&tao_d2_data\n  d2_data%name = 'simba'\n  universe = 1\n  n_d1_data = 1\n/\n"
        text += "&tao_d1_data\n  ix_d1_data = 1\n  d1_data%name = 'targets'\n"
        for i, t in enumerate(spec.targets, start=1):
            start, end = ("BEGINNING", "END") if t.at is None else ("", t.at)
            value = t.value * SCALE.get(t.quantity, 1.0)
            weight = 1.0 / max(1.0, abs(t.value)) ** 2
            text += (
                f"  datum({i}) = '{DATA[t.quantity]}' '' '{start}' '{end}' "
                f"'{MERIT[t.op]}' {value} {weight}\n"
            )
        text += "/\n&tao_var\n  v1_var%name = 'simba'\n  default_step = 1e-6\n  default_universe = '1'\n"
        for i in range(len(spec.knobs)):
            text += f"  var({i + 1})%ele_name = 'SIMBA_KNOB_{i}'\n  var({i + 1})%attribute = 'k1l'\n"
        return text + "/\n"

    def set_native(self, values) -> None:
        for i, value in enumerate(values):
            self.tao.cmd(f"set ele SIMBA_KNOB_{i} k1l = {float(value)}")

    def points(self) -> list:
        tao = self.tao
        names = [str(n).upper() for n in tao.lat_list("*", "ele.name")]
        columns = {q: np.asarray(tao.lat_list("*", c), dtype=float) / SCALE.get(q, 1.0) for q, c in LAT_LIST.items()}
        laura = {e.upper(): e for e in self.lattice.elements}
        points = []
        for i, raw in enumerate(names):
            name = laura.get(raw)
            if name is None:
                continue
            quantities = {q: float(column[i]) for q, column in columns.items()}
            if name in self.matrix_at:
                quantities.update(matrix_entries(tao.matrix("beginning", raw)["mat6"]))
            points.append((name, quantities))
        return points

    def native_solves(self) -> bool:
        return self.native and all(t.at is not None or t.op != "=" for t in self.spec.targets)

    def native_solve(self) -> list:
        tao = self.tao
        tao.cmd("use var simba")
        tao.cmd("use dat simba")
        tao.cmd("run")
        values = []
        for knob in self.spec.knobs:
            first = knob.elements[0]
            values.append(float(tao.ele_gen_attribs(first.name)["K1"]) * float(first.magnetic.length))
        return values
