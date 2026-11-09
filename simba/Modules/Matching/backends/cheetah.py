"""Matching in Cheetah, which has no matcher: the shared solver, on Twiss from tracking a ``ParameterBeam``."""

from . import Backend


class CheetahBackend(Backend):
    code = "cheetah"
    package = "cheetah"
    quantities = frozenset({"beta_x", "beta_y", "alpha_x", "alpha_y"})
    matrix = False

    def build(self) -> None:
        import torch
        from cheetah import ParameterBeam

        spec = self.spec
        self.torch = torch
        self.segment = self.lattice.section.to_cheetah()
        f64 = {"dtype": torch.float64}
        t = spec.twiss
        self.beam = ParameterBeam.from_twiss(
            beta_x=torch.tensor(t["beta_x"], **f64), alpha_x=torch.tensor(t["alpha_x"], **f64),
            beta_y=torch.tensor(t["beta_y"], **f64), alpha_y=torch.tensor(t["alpha_y"], **f64),
            emittance_x=torch.tensor(1e-9, **f64), emittance_y=torch.tensor(1e-9, **f64),
            energy=torch.tensor(spec.energy, **f64), **f64,
        )
        by_name = {e.name: e for e in self.segment.elements}
        self.variables = [
            [(by_name[e.name], float(e.magnetic.length)) for e in knob.elements] for knob in spec.knobs
        ] if self.native else []

    def set_native(self, values) -> None:
        for variable, value in zip(self.variables, values):
            for element, length in variable:
                element.k1 = self.torch.tensor(float(value) / length, dtype=self.torch.float64)

    def points(self) -> list:
        beam, points = self.beam, []
        for element in self.segment.elements:
            beam = element.track(beam)
            points.append((element.name, {q: float(getattr(beam, q)) for q in self.quantities}))
        return points
