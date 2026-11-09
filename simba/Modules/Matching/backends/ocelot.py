"""Matching with Ocelot's ``match`` (simplex), on SIMBA's Ocelot model of the line."""

import contextlib
import io
import types
from copy import deepcopy

import numpy as np

from . import Backend, matrix_entries

NAMES = {
    "beta_x": "beta_x", "beta_y": "beta_y", "alpha_x": "alpha_x", "alpha_y": "alpha_y",
    "eta_x": "Dx", "eta_xp": "Dxp", "eta_y": "Dy", "eta_yp": "Dyp", "mux": "mux", "muy": "muy",
}
"""SIMBA's Twiss names, as Ocelot's."""

SCALE = {"mux": 2 * np.pi, "muy": 2 * np.pi}
"""Ocelot's phase advance is in radians."""


class OcelotBackend(Backend):
    code = "ocelot"
    package = "ocelot"
    lattice_class = "ocelotLattice"

    def build(self) -> None:
        from ocelot.cpbd.beam import Twiss

        from ....Codes.Ocelot.Ocelot import ocelotLattice
        from ....Codes.Ocelot.bendmaps import transport_bend_maps
        from ....Codes.Ocelot.cavitymaps import stable_cavity_maps

        stable_cavity_maps()
        transport_bend_maps()
        lattice, spec = self.lattice, self.spec
        model = types.SimpleNamespace(
            lat_obj=lattice.section.to_ocelot(save=False, Brho=lattice.rigidity(spec.p0c)),
            pin=types.SimpleNamespace(E=spec.energy * 1e-9),
            rest_energy=lattice.rest_energy,
            elements=lattice.elements,
        )
        if lattice.tw_focusing:
            ocelotLattice.add_tw1_focusing(model)
        self.lat = model.lat_obj
        self.tws0 = Twiss(**spec.twiss, E=spec.energy * 1e-9)
        self.variables = [
            {o: 1.0 / float(e.magnetic.length) for e in knob.elements for o in self.lat.sequence if o.id == e.name}
            for knob in spec.knobs
        ] if self.native else []

    def set_native(self, values) -> None:
        for variable, x in zip(self.variables, values):
            for quad, coefficient in variable.items():
                quad.k1 = coefficient * x

    def points(self) -> list:
        tw, points = deepcopy(self.tws0), []
        energy, matrix = self.spec.energy * 1e-9, np.eye(6)
        for element in self.lat.sequence:
            for tm in element.first_order_tms:
                tw = tm * tw
            quantities = {q: getattr(tw, n) / SCALE.get(q, 1.0) for q, n in NAMES.items()}
            if self.matrix_at:
                for r, tm in zip(element.R(energy), element.tms):
                    matrix = r @ matrix
                    energy += tm.get_delta_e()
                if element.id in self.matrix_at:
                    quantities.update(matrix_entries(matrix))
            points.append((element.id, quantities))
        return points

    def native_solves(self) -> bool:
        # Ocelot's match has no transfer-matrix targets
        return self.native and not self.matrix_at

    def native_solve(self) -> list:
        from ocelot.cpbd.match import match

        constr = {}
        ends = {}
        for element in self.lat.sequence:
            ends[element.id] = element  # the last object of each element is its end
        for t in self.spec.targets:
            value = t.value * SCALE.get(t.quantity, 1.0)
            wanted = value if t.op == "=" else [t.op, value]
            constr.setdefault("global" if t.at is None else ends[t.at], {})[NAMES[t.quantity]] = wanted
        quiet = contextlib.nullcontext() if self.lattice.verbose else contextlib.redirect_stdout(io.StringIO())
        with quiet:
            solution = match(self.lat, constr, self.variables, deepcopy(self.tws0), verbose=False,
                             max_iter=self.spec.max_iterations)
        return [float(x) for x in solution]
