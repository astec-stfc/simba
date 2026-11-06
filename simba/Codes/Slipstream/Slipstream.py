"""
SIMBA Slipstream module: tracks a lattice once with slipstream's ``LinacTracker``. See `slipstream github`_.

    .. _slipstream github: https://github.com/astec-stfc/slipstream

The elements come from LAURA's ``SectionLatticeTranslator.to_slipstream()``. Slipstream is an optional
dependency, imported when the lattice is written, so importing simba does not import torch.
"""
from ...Framework_objects import frameworkLattice
from ...Modules import constants
from ...Modules.units import UnitValue
from laura.models.element_list import flatten_occurrence

import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, ClassVar, Dict, List

import numpy as np
from yaml import safe_load

with open(os.path.dirname(os.path.abspath(__file__)) + "/slipstream_defaults.yaml") as infile:
    slipstreamglobal = safe_load(infile)


class slipstreamLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` tracked once through slipstream's ``LinacTracker``."""

    code: str = "slipstream"
    """String indicating the lattice object type"""

    electrons_only: ClassVar[bool] = True
    """The beam conversion below assumes electrons."""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    specs: List[Any] = []
    """The lattice as slipstream ``ElementSpec`` objects, in beamline order"""

    marks: Dict[str, int] = {}
    """``{element name: number of specs before it}`` for every element a beam is written at"""

    slipstreamglobal: Dict = {}
    """``slipstream_defaults.yaml``, overridden by ``settings["global"]["SLIPSTREAMsettings"]``"""

    particle_definition: str = None
    """Name of the input beam"""

    pin: Any | None = None
    """Input beam as slipstream's ``(x, px, y, py, tau, delta)`` rows"""

    results: Dict[str, Any] = {}
    """``{element name: (rows, reference energy in eV, reference time in s)}`` at every mark"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.slipstreamglobal = deepcopy(slipstreamglobal)
        self.slipstreamglobal.update(self.settings["global"].get("SLIPSTREAMsettings", {}))
        self.particle_definition = self.input_particle_definition

    def writeElements(self) -> bool:
        """
        Build :attr:`specs` from the section, and :attr:`marks` from its screens, markers, BPMs and end.

        Returns
        -------
        bool
            Always True
        """
        from slipstream.tracking.native.laura_import import _to_element_spec  # ponytail: no public name yet

        outputs = {flatten_occurrence(e.name) for e in self.screens_and_markers_and_bpms}
        self.specs, self.marks = [], {}
        for record in self.section.to_slipstream():
            if record.name in outputs:
                self.marks[record.name] = len(self.specs)
            spec = _to_element_spec(record, load_wakes=self.wakefield_enable)
            if spec is None:
                continue
            if self.csr_enable and spec.kind == "bend":
                spec = replace(spec, csr=True)
            self.specs.append(spec)
        self.marks[self.end] = len(self.specs)  # the exit of the last element, not its entrance
        return True

    def write(self) -> None:
        """Build :attr:`specs`."""
        self.writeElements()

    def preProcess(self) -> None:
        """Load the input beam and convert it to slipstream's coordinates."""
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.load_input_beam(prefix, self.particle_definition)
        beam = self.global_parameters["beam"]
        p0c = self.reference_p0c
        # px, py are p / p0 (not slopes); tau / c is the arrival time after the reference's
        self.pin = np.array([
            beam.x.val, beam.cpx.val / p0c, beam.y.val, beam.cpy.val / p0c,
            constants.speed_of_light * (beam.t.val - self.reference_t0), beam.cp.val / p0c - 1,
        ]).T

    def run(self) -> None:
        """Track :attr:`pin` from one mark to the next and store the state at each in :attr:`results`."""
        import torch
        from slipstream.tracking.native.linac import LinacTracker

        options = dict(self.slipstreamglobal)
        options["dtype"] = getattr(torch, options["dtype"])
        charge = abs(float(self.global_parameters["beam"].Q.val))
        mc2 = self.rest_energy
        energy = float(np.hypot(self.reference_p0c, mc2))
        state = torch.as_tensor(self.pin, dtype=options["dtype"])
        time_s = self.reference_t0
        start = 0
        self.results = {}
        for stop in sorted(set(self.marks.values()) | {len(self.specs)}):
            if stop > start:
                tracker = LinacTracker(
                    self.specs[start:stop], energy_eV=energy, mc2_eV=mc2, bunch_charge_C=charge,
                    lsc=self.lsc_enable, csr=self.csr_enable, **options,
                )
                state, new_energy = tracker.track(state)
                new_energy = float(new_energy)
                length = sum(s.length for s in self.specs[start:stop])
                betas = [np.sqrt(1 - (mc2 / e) ** 2) for e in (energy, new_energy)]
                # ponytail: mean beta is exact where the energy is constant, close in a cavity
                time_s += length / (constants.speed_of_light * np.mean(betas))
                energy, start = new_energy, stop
            for name, mark in self.marks.items():
                if mark == stop:
                    self.results[name] = (state.cpu().numpy(), energy, time_s)

    def postProcess(self) -> None:
        """Write the beam at every mark; the end's also becomes the framework beam."""
        super().postProcess()
        svals = {name: value + self.entrance_s for name, value in self.getSValues(as_dict=True).items()}
        for name, (particles, energy, time_s) in self.results.items():
            end = name == self.end
            element = self.endObject if end else self.elementObjects[name]
            zstart = element.physical.end.z if end else element.physical.start.z
            beam = self.beam_at(particles, energy, time_s, zstart, svals[name])
            self.write_beam_file(beam, name)
            if end:
                self.global_parameters["beam"] = beam

    def beam_at(self, particles: np.ndarray, energy: float, time_s: float, zstart: float, s: float):
        """
        The input beam moved to ``particles``.

        Parameters
        ----------
        particles: np.ndarray
            ``(x, px, y, py, tau, delta)`` rows, with ``px = p_x / p0``
        energy: float
            Reference total energy in eV
        time_s: float
            Time the reference particle is there, in s
        zstart: float
            z of the element
        s: float
            s of the element, in the machine
        """
        beam = deepcopy(self.global_parameters["beam"])
        x, px, y, py, tau, delta = particles.T.astype(float)
        p0c = np.sqrt(energy**2 - self.rest_energy**2)
        cpx, cpy = px * p0c, py * p0c
        cpz = np.sqrt((p0c * (1 + delta)) ** 2 - cpx**2 - cpy**2)
        beam._beam.x = UnitValue(x, "m")
        beam._beam.y = UnitValue(y, "m")
        beam._beam.t = UnitValue(time_s + tau / constants.speed_of_light, "s")
        beam.set_momenta(cpx, cpy, cpz)
        beam.set_z_from_t(zstart, self.ref_idx)
        beam._beam.s = UnitValue(s, units="m")
        return beam
