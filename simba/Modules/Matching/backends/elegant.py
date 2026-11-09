"""
Matching with elegant's optimiser, through SIMBA's elegant pieces: the lattice from
:meth:`~simba.Codes.Elegant.Elegant.elegantLattice.writeElements`, the deck from its
command classes, run through the line's executables, in a scratch directory.

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
import types
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


def _command(objecttype: str, name: str | None = None, **keywords):
    """A generic elegant ``&objecttype``; ``name`` is the element it acts on."""
    from ....Codes.Elegant.Elegant import elegantCommandFile

    command = elegantCommandFile(objectname=f"{objecttype}_{name or ''}", objecttype=objecttype, **keywords)
    if name is not None:
        command.add_property("name", name)
    return command


def _quoted(text: str) -> str:
    """A string value with spaces, quoted for elegant's namelists."""
    return f'"{text}"'


class ElegantBackend(Backend):
    code = "elegant"
    package = "elegant"
    lattice_class = "elegantLattice"

    @classmethod
    def available(cls) -> bool:
        return True  # its executable is the line's; see cannot

    @classmethod
    def cannot(cls, lattice, spec) -> list:
        reasons = super().cannot(lattice, spec)
        try:
            command = lattice.executables["elegant"]
        except (KeyError, TypeError, AttributeError):
            command = None
        in_container = lattice.global_parameters.get("container_runtime") in ("docker", "apptainer")
        if not command or not (in_container or shutil.which(command[0])):
            reasons.append("elegant is not installed")
        if lattice.rf_voltage and any(_row_column_5(t.quantity) for t in spec.targets):
            reasons.append("with RF, its path length cannot be read as Xsuite's zeta (row and column 5)")
        return reasons

    def _lattice_text(self) -> str:
        """``elegantLattice.writeElements``: the line's own, else for this line, without a ramp."""
        from ....Codes.Elegant.Elegant import elegantLattice

        lattice = self.lattice
        if isinstance(lattice, elegantLattice):
            return lattice.writeElements()
        model = types.SimpleNamespace(
            bunch_charge=getattr(lattice, "bunch_charge", None),
            global_parameters=lattice.global_parameters,
            section=lattice.section,
            insert_ramp=lambda text: text,
        )
        return elegantLattice.writeElements(model)

    def build(self) -> None:
        from ....Codes.Elegant.Elegant import elegant_element_name

        if not hasattr(self, "workdir"):
            self.workdir = tempfile.mkdtemp(prefix="simba_match_elegant_")
            weakref.finalize(self, shutil.rmtree, self.workdir, True)
            self.values = [k.value for k in self.spec.knobs]
        self.name = {e: elegant_element_name(e) for e in self.lattice.elements}
        text = re.sub(r"&\s*\n", "", self._lattice_text())
        lines = list(re.finditer(r"^(\S+)\s*:\s*LINE\s*=\s*\((.*?)\)", text, re.M | re.S))
        self.beamline = lines[-1].group(1)
        body = lines[-1].group(2)
        markers, self.fitpoint = [], {}
        for i, name in enumerate(dict.fromkeys(t.at for t in self.spec.targets if t.at is not None)):
            marker = f"SIMBA_FIT_{i}"
            markers.append(f"{marker}: marker, fitpoint = 1\n")
            body = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", rf"\g<0>, {marker}", body, flags=re.I)
            self.fitpoint[name] = marker
        start, end = lines[-1].span(2)
        with open(os.path.join(self.workdir, "lattice.lte"), "w") as f:
            f.write("".join(markers) + text[:start] + body + text[end:])
        self.gamma0 = self.spec.energy / self.lattice.rest_energy

    def set_native(self, values) -> None:
        self.values = [float(v) for v in values]

    def _deck(self, after: list, matrix: bool) -> list:
        """The commands before ``after``: run setup, any knob values, Twiss (and matrix) output."""
        from ....Codes.Elegant.Elegant import (
            elegant_matrix_output_command, elegant_run_control_command, elegant_run_setup_command,
            elegant_twiss_output_command, elegantLattice,
        )

        lattice, spec, t = self.lattice, self.spec, self.spec.twiss
        commands = [
            elegant_run_setup_command(
                lattice="lattice.lte", use_beamline=self.beamline, p_central=spec.p0c / lattice.rest_energy,
                **({"always_change_p0": 0} if lattice.fixed_reference else {}),
            ),
            elegantLattice._without_watch_output(),
        ]
        if self.native:
            for knob, value in zip(spec.knobs, self.values):
                for e in knob.elements:
                    commands.append(_command("alter_elements", self.name[e.name], item="K1",
                                             value=value / float(e.magnetic.length)))
        commands.append(elegant_twiss_output_command(
            beam=lattice.global_parameters["beam"], matched=0, filename="%s.twi",
            beta_x=t["beta_x"], alpha_x=t["alpha_x"], beta_y=t["beta_y"], alpha_y=t["alpha_y"],
            eta_x=0.0, eta_xp=0.0,
        ))
        if matrix:
            commands.append(elegant_matrix_output_command(SDDS_output="%s.mat", full_matrix_only=0,
                                                          SDDS_output_order=1))
        return commands + [elegant_run_control_command()] + after

    def _run(self, commands: list, stem: str) -> None:
        """Write ``commands`` to ``stem.ele`` and run elegant on it, as the line's executables do."""
        with open(os.path.join(self.workdir, f"{stem}.ele"), "w") as f:
            f.write("".join(c.write_Elegant() for c in commands))
        executables = self.lattice.executables
        command = executables.build_command(executables["elegant"] + [f"{stem}.ele"], self.workdir)
        done = subprocess.run(command, cwd=self.workdir, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"elegant failed: {(done.stdout + done.stderr)[-2000:]}")

    def _read(self, path: str) -> dict:
        from ...SDDSFile import SDDSFile

        sdds = SDDSFile(index=1)
        sdds.read_file(path)
        return sdds.data

    def _twiss_run(self, matrix: bool, stem: str) -> None:
        from ....Codes.Elegant.Elegant import elegant_bunched_beam_command, elegant_track_command

        self._run(self._deck([elegant_bunched_beam_command(), elegant_track_command()], matrix), stem)

    def points(self) -> list:
        self._twiss_run(bool(self.matrix_at), "twiss")
        twi = self._read(os.path.join(self.workdir, "twiss.twi"))
        mat = self._read(os.path.join(self.workdir, "twiss.mat")) if self.matrix_at else None
        laura = {v: k for k, v in self.name.items()}
        points, self.s_end = [], {}
        for i, raw in enumerate(twi["ElementName"]):
            name = laura.get(str(raw).upper())
            if name is None:
                continue
            self.s_end[name] = float(twi["s"][i])
            quantities = {q: float(twi[c][i]) / SCALE.get(q, 1.0) for q, c in COLUMNS.items()}
            if name in self.matrix_at:
                rows = [j for j, n in enumerate(mat["ElementName"]) if str(n).upper() == self.name[name]]
                r = np.array([[mat[f"R{a}{b}"][rows[-1]] for b in range(1, 7)] for a in range(1, 7)], dtype=float)
                quantities.update(matrix_entries(self._canonical(r, self.s_end[name])))
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
        from ....Codes.Elegant.Elegant import elegant_bunched_beam_command

        spec = self.spec
        if self.matrix_at:
            self.points()  # for each element's s, which R56's conversion needs
        after = [_command(
            "optimization_setup", mode="minimize", method="simplex", target=0, tolerance=1e-16,
            n_passes=5, n_evaluations=spec.max_iterations, n_restarts=2, verbose=0, log_file="/dev/null",
        )]
        for knob in spec.knobs:
            first, length = knob.elements[0], float(knob.elements[0].magnetic.length)
            after.append(_command("optimization_variable", self.name[first.name], item="K1",
                                  lower_limit=-1e3, upper_limit=1e3,
                                  # 5%, scipy's fmin's first simplex: 1% stalled in a local minimum on JFEL S13
                                  step_size=max(1e-2, 0.05 * abs(knob.value / length))))
            for other in knob.elements[1:]:
                equation = f"{self.name[first.name]}.K1 {length} * {float(other.magnetic.length)} /"
                after.append(_command("optimization_covariable", self.name[other.name], item="K1",
                                      equation=_quoted(equation)))
        for t in spec.targets:
            scale = max(1.0, abs(t.value))
            if t.at is None:
                quantity, value = f"{'max' if t.op == '<' else 'min'}.{GLOBAL[t.quantity]}", t.value
            else:
                column = t.quantity if t.quantity in MATRIX else COLUMNS[t.quantity]
                quantity, value = f"{self.fitpoint[t.at]}#1.{column}", self._elegant_value(t)
            if t.op.startswith("a"):
                quantity += " abs"
            if t.op == "=":
                term = f"{quantity} {value} - {scale} / sqr"
            else:
                term = f"{quantity} {value} {1e-4 * scale} {'selt' if t.op.endswith('<') else 'segt'}"
            after.append(_command("optimization_term", term=_quoted(term)))
        after += [elegant_bunched_beam_command(), _command("optimize"), _command("save_lattice", filename="%s.new")]
        self._run(self._deck(after, False), "match")
        with open(os.path.join(self.workdir, "match.new")) as f:
            saved = re.sub(r"&\s*\n", "", f.read())
        values = []
        for knob in spec.knobs:
            first = knob.elements[0]
            line = re.search(rf'^"?{re.escape(self.name[first.name])}"?\s*:(.*)$', saved, re.M | re.I)
            if line is None:
                raise RuntimeError(f"elegant's matched lattice has no {first.name}")
            k1 = re.search(r"\bK1=([^,\n]+)", line.group(1), re.I)  # left out when 0, its default
            values.append(float(k1.group(1)) * float(first.magnetic.length) if k1 else 0.0)
        self.values = values
        return values
