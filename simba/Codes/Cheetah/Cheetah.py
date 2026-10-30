"""
SIMBA Cheetah module: builds and tracks a Cheetah segment. See `Cheetah github`_.

    .. _Cheetah github: https://github.com/desy-ml/cheetah
"""
from ...Framework_objects import frameworkLattice
from ...FrameworkHelperFunctions import set_deep_attr
from ...Modules import Beams as rbf
from ...Modules.gradients import lab_array, lab_coordinates, write_gradients
from ... import exceptions

import os
import numpy as np
from yaml import safe_load
from copy import deepcopy
from functools import reduce
from warnings import warn
from typing import Dict, Any, ClassVar
import h5py
import lox
from lox.worker.thread import ScatterGatherDescriptor
from laura.models.diagnostic import DiagnosticElement
try:
    from laura.translator.converters.codes.cheetah import choose_cheetah_tracking_method
except ImportError:  # a LAURA before the tracking-method option: Cheetah tracks linearly, as it did
    choose_cheetah_tracking_method = None


with open(
    os.path.dirname(os.path.abspath(__file__)) + "/cheetah_defaults.yaml",
) as infile:
    cheetahglobal = safe_load(infile)

twiss_keys = (
    "beta_x",
    "beta_y",
    "alpha_x",
    "alpha_y",
    "s",
    "energy",
    "emittance_x",
    "emittance_y",
    "projected_emittance_x",
    "projected_emittance_y",
    "sigma_x",
    "sigma_y",
    "sigma_px",
    "sigma_py",
    "mu_x",
    "mu_y",
    "sigma_tau",
    "sigma_p",
)

class cheetahLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` built and tracked as a Cheetah segment."""

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Threaded conversion of Cheetah screen beams to openPMD"""

    code: str = "cheetah"
    """String indicating the lattice object type"""

    electrons_only: ClassVar[bool] = True
    """simba's Cheetah beam conversion assumes electrons."""

    supports_gradients: ClassVar[bool] = True
    """Cheetah tracks in torch, so :meth:`run` can differentiate it."""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    segment: Any | None = None
    """The lattice as a Cheetah `Segment`_

    .. _Segment: https://github.com/desy-ml/cheetah/blob/master/cheetah/accelerator/segment.py
    """

    pin: Any | None = None
    """Initial particle distribution as a Cheetah `ParticleBeam`_

    .. _ParticleBeam: https://github.com/desy-ml/cheetah/blob/master/cheetah/particles/particle_beam.py"""

    pout: Any | None = None
    """Final particle distribution as a Cheetah `ParticleBeam`_"""

    tws: Any | None = None
    """Twiss tensors along the segment (``Any`` so importing simba does not import torch)"""

    tangents: Dict | None = None
    """Derivatives from :meth:`track_tangents`, if :attr:`gradient_parameters`"""

    _lab_in: Any = None
    """The (n, 6) input particles as tracked, for :meth:`track_tangents` and :meth:`backward`."""

    cheetahglobal: Dict = {}
    """``cheetah_defaults.yaml``, overridden by ``settings["global"]["CHEETAHsettings"]``"""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.cheetahglobal = deepcopy(cheetahglobal)
        if "CHEETAHsettings" in list(self.settings["global"].keys()):
            for k, v in self.settings["global"]["CHEETAHsettings"].items():
                if isinstance(v, Dict):
                    for k1, v1 in v.items():
                        self.cheetahglobal[k].update({k1: v1})
                else:
                    self.cheetahglobal.update({k: v})
        self.particle_definition = self.input_particle_definition


    def writeElements(self) -> bool:
        """
        Build :attr:`segment` from the section.

        Returns
        -------
        bool
            Always True
        """
        if choose_cheetah_tracking_method:
            self.section.cheetah_tracking_method = self.cheetahglobal["cheetah_tracking_method"]
        self.segment = self.section.to_cheetah(Brho=self.rigidity(self.reference_p0c))
        return True

    def write(self) -> None:
        """Build :attr:`segment` and save it as JSON to `master_subdir`."""
        success = self.writeElements()
        if success:
            self.segment.to_lattice_json(
                filepath=f'{self.global_parameters["master_subdir"]}/{self.objectname}.json'
            )

    def preProcess(self) -> None:
        """Load the input beam and convert it via :meth:`hdf5_to_openpmd`."""
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.load_input_beam(prefix, self.particle_definition)
        self.hdf5_to_openpmd()
        self.reference_start = (self.reference_t0, float(self.pin.s), self.reference_p0c)

    def hdf5_to_openpmd(self, prefix="", write=True) -> None:
        """
        Rematch the input beam to the initial twiss, convert it to a Cheetah beam and set :attr:`pin`.

        Parameters
        ----------
        prefix: str
            Unused
        write: bool
            Also save it as ``<particle_definition>.cheetah.hdf5``
        """
        cheetahbeamfilename = os.path.join(
            self.global_parameters["master_subdir"],
            f"{os.path.basename(self.particle_definition)}.cheetah.hdf5",
        )
        self.global_parameters["beam"].beam.rematchXPlane(**self.initial_twiss["horizontal"])
        self.global_parameters["beam"].beam.rematchYPlane(**self.initial_twiss["vertical"])

        self.pin = rbf.beam.write_cheetah_beam_file(
            self.global_parameters["beam"],
            cheetahbeamfilename,
            write=write,
            energy=self.reference_energy,
            t0=self.reference_t0,
        )

    def run(self) -> None:
        """Track :attr:`pin` and set :attr:`pout` (and :attr:`tws` if ``save_twiss``, :attr:`tangents` if there are any gradients)."""
        pin = deepcopy(self.pin)
        if choose_cheetah_tracking_method and self.cheetahglobal["cheetah_tracking_method"] == "auto":
            choose_cheetah_tracking_method(self.segment, pin)
        self.pout = self.segment.track(pin)
        if self.cheetahglobal["save_twiss"]:
            self.tws = self.segment.get_beam_attrs_along_segment(twiss_keys, pin)
        self._lab_in = lab_array(self.global_parameters["beam"])
        self.tangents = (
            self.track_tangents()
            if self.forward_tangents and (self.gradient_parameters or self.incoming_tangents) else None
        )

    def cheetah_derivatives(self, parameter: str) -> Dict[str, Any]:
        """
        How :attr:`segment`'s buffers move with a parameter, found by rebuilding the
        segment either side of its value; this covers the name, the units and a keyword
        that sets several buffers.

        Parameters
        ----------
        parameter: str
            ``ELEMENT:keyword``; the keyword may be dotted, e.g. ``magnetic.k1l``

        Returns
        -------
        Dict[str, torch.Tensor]
            d buffer / d parameter, by the buffer's path in :attr:`segment`
        """
        from torch import equal
        name, key = parameter.split(":")
        element = self.elements[name]
        value = reduce(getattr, key.split("."), element)
        step = 1e-6 * max(abs(value), 1.0)
        index = [i for i, e in enumerate(self.segment.elements) if e.name == name]
        faces = self._faces(element)
        buffers = []
        try:
            for v in (value + step, value - step):
                set_deep_attr(element, key, v)
                buffers.append(self._rebuilt_buffers(name, index, faces))
        finally:
            set_deep_attr(element, key, value)
        up, down = buffers
        if up.keys() != down.keys():  # one side rebuilt the whole segment
            unchanged = dict(self.segment.named_buffers())
            up, down = {**unchanged, **up}, {**unchanged, **down}
        derivatives = {k: (up[k] - down[k]) / (2 * step) for k in up if not equal(up[k], down[k])}
        if not derivatives:
            warn(exceptions.SettingWarning(
                f"{exceptions._line(self.objectname)}: gradient parameter {parameter} does not change the "
                "Cheetah lattice, so its derivatives are zero."
            ))
        return derivatives

    @staticmethod
    def _faces(element) -> tuple:
        """Where ``element`` starts and ends: all of it the drifts either side depend on."""
        return element.physical.start.array.tolist(), element.physical.end.array.tolist()

    def _rebuilt_buffers(self, name: str, index: list, faces: tuple) -> Dict[str, Any]:
        """
        :attr:`segment`'s buffers that a change to element ``name`` moves, as
        ``section.to_cheetah`` would build them. If that is only ``name``'s own Cheetah
        element (it is the one element of that name, at ``index``, and its ends have not
        moved from ``faces``), just that element's buffers are rebuilt and
        returned. Otherwise the whole segment is rebuilt and all of its buffers returned.
        """
        from laura.translator.converters.magnet import SolenoidTranslator
        Brho = self.rigidity(self.reference_p0c)
        element = self.section._beamline().get(name)
        if len(index) == 1 and element is not None and self._faces(self.elements[name]) == faces:
            translated = self.section._translate([element])
            self.section._apply_tw_focusing(translated)
            built = []
            for t in translated.values():
                elem = t.to_cheetah(Brho=Brho) if isinstance(t, SolenoidTranslator) else t.to_cheetah()
                if elem is not None:
                    self.section._cheetah_misalignment(t, elem)
                    built.append(elem)
            if len(built) == 1 and built[0].name == name:
                return {f"elements.{index[0]}.{k}": b for k, b in built[0].named_buffers()}
        return dict(self.section.to_cheetah(Brho=Brho).named_buffers())

    def reference_from(self, lab) -> tuple:
        """
        :attr:`reference_p0c` and :attr:`reference_t0` as torch functions of the incoming
        particles, so tangents see an upstream change move this line's reference as the
        forward run does.

        Parameters
        ----------
        lab: torch.Tensor
            (n, 6) incoming :data:`~simba.Modules.gradients.lab_coordinates`

        Returns
        -------
        tuple
            ``(p0c, t0)`` in eV and s
        """
        from torch import as_tensor, float64
        p0c = (
            lab[:, [1, 3, 5]].norm(dim=-1).mean() if self.design_p0c is None
            else as_tensor(self.design_p0c, dtype=float64)
        )
        t0 = lab[:, 4].mean() if not self.fixed_reference else as_tensor(self.reference_t0, dtype=float64)
        return p0c, t0

    def tracking_function(self):
        """
        This line's tracking as a pure torch function, for :meth:`track_tangents` and :meth:`backward`.

        Returns
        -------
        callable
            ``track(t, lab) -> (twiss, beams)``: ``t`` (P,) offsets to :attr:`gradient_parameters`,
            ``lab`` (n, 6) incoming :data:`~simba.Modules.gradients.lab_coordinates`; ``twiss``:
            key -> (n_s,) along :attr:`segment`; ``beams``: element name -> (n, 6) at each screen
            and the line end
        """
        from torch import sqrt, stack
        from cheetah.accelerator import Screen
        from ...Modules.Beams.cheetah import cheetah_lab_coordinates, cheetah_particles

        E0 = self.rest_energy
        end_is_screen = isinstance(self.segment.elements[-1], Screen)
        local = [self.cheetah_derivatives(p) for p in self.gradient_parameters]

        def track(t, lab):
            segment = deepcopy(self.segment)
            for tp, derivatives in zip(t, local):
                for path, derivative in derivatives.items():
                    module, _, leaf = path.rpartition(".")
                    sub = segment.get_submodule(module)
                    setattr(sub, leaf, getattr(sub, leaf) + tp * derivative)
            p0c, t0 = self.reference_from(lab)
            pin = deepcopy(self.pin)
            pin.particles = cheetah_particles(lab, E0, p0c, t0)
            pin.energy = sqrt(p0c**2 + E0**2)
            start = None if self.reference_start is None else (t0, self.reference_start[1], p0c)
            twiss = {k: [] for k in twiss_keys}
            for beam in segment.beam_along_segment_generator(pin):
                for k in twiss_keys:
                    twiss[k].append(getattr(beam, k))
            beams = {
                e.name: cheetah_lab_coordinates(e.get_read_beam(), E0, start)
                for e in segment.elements if isinstance(e, Screen)
            }
            if not end_is_screen:
                beams[self.end] = cheetah_lab_coordinates(beam, E0, start)
            return {k: stack(v) for k, v in twiss.items()}, beams

        return track

    def track_tangents(self) -> Dict:
        """
        Forward-mode derivatives (``torch.func.jvp``) of the twiss along :attr:`segment` and of
        the beam at each screen and the line end. One pass per column: first the parameters
        whose :attr:`incoming_tangents` an upstream line left, then this line's
        :attr:`gradient_parameters`.

        Returns
        -------
        Dict
            ``parameters`` and their ``values``; ``twiss``: key -> (n_s, P) array, with the
            ``s`` -> (n_s,) it is sampled at; ``beams``: element name -> (n, 6, P) array of
            :data:`~simba.Modules.gradients.lab_coordinates`
        """
        from torch import as_tensor, cat, eye, float64, zeros
        from torch.func import jvp, vmap

        track = self.tracking_function()
        lab = as_tensor(self._lab_in, dtype=float64)
        P = len(self.gradient_parameters)
        incoming = self.incoming_tangents or {"parameters": [], "values": [], "tangent": np.zeros((len(lab), 6, 0))}
        upstream = as_tensor(incoming["tangent"], dtype=float64).movedim(-1, 0)
        U = len(upstream)
        none = zeros(P, dtype=float64)
        d_t = cat([zeros(U, P, dtype=float64), eye(P, dtype=float64)])
        d_lab = cat([upstream, zeros(P, *lab.shape, dtype=float64)])
        (twiss, _), (d_twiss, d_beams) = vmap(lambda a, b: jvp(track, (none, lab), (a, b)))(d_t, d_lab)
        return {
            "parameters": incoming["parameters"] + self.gradient_parameters,
            "values": incoming["values"] + [
                reduce(getattr, p.split(":")[1].split("."), self.elements[p.split(":")[0]])
                for p in self.gradient_parameters
            ],
            "twiss": {
                "s": twiss["s"][0].numpy(),
                **{k: d_twiss[k].movedim(0, -1).numpy() for k in twiss_keys if k != "s"},
            },
            "beams": {n: d.movedim(0, -1).numpy() for n, d in d_beams.items()},
        }

    def backward(self, cotangents: Dict) -> tuple:
        """
        One reverse-mode step (``torch.func.vjp``) back through this line.

        Parameters
        ----------
        cotangents: Dict
            Beam file path -> (n, 6) d objective / d particle, for this line's outputs.

        Returns
        -------
        tuple
            d objective / d input particle as an (n, 6) array, and
            :attr:`gradient_parameters` -> d objective / d parameter.
        """
        from torch import as_tensor, float64, zeros
        from torch.func import vjp

        def cotangent(name, beam):
            c = cotangents.get(os.path.abspath(self.output_beam_file(name)))
            return zeros(beam.shape, dtype=float64) if c is None else as_tensor(c, dtype=float64)

        track = self.tracking_function()
        lab = as_tensor(self._lab_in, dtype=float64)
        t = zeros(len(self.gradient_parameters), dtype=float64)
        beams, pullback = vjp(lambda t, lab: track(t, lab)[1], t, lab)
        d_t, d_lab = pullback({name: cotangent(name, beam) for name, beam in beams.items()})
        return d_lab.numpy(), dict(zip(self.gradient_parameters, d_t.tolist()))

    def s_and_z(self, s: np.ndarray) -> tuple:
        """Cheetah's ``s`` along :attr:`segment` as the lattice's ``(s, z)``."""
        svals = s - s[0]
        return svals + self.entrance_s, self.getZAtS(svals)

    @lox.thread(40)
    def screen_threaded_function(self, scr: DiagnosticElement, outname: str, name: str) -> None:
        """
        Write a Cheetah beam as openPMD; the end's also becomes the framework beam.

        Parameters
        ----------
        scr: cheetah.ParticleBeam
            Beam read at the screen (despite the type hint)
        outname: str
            openPMD file to write
        name: str
            Element name
        """
        from ...Modules.Beams import cheetah as rbf_cheetah
        beam = rbf.beam()
        if name == self.end:
            zstart = self.endObject.physical.end.z
        else:
            zstart = self.elementObjects[name].physical.start.z
        rbf_cheetah.interpret_cheetah_ParticleBeam(
            beam,
            scr,
            zstart=zstart,
            s=scr.s.numpy(),
            ref_index=self.ref_idx,
            start=self.reference_start,
        )
        rbf.openpmd.write_openpmd_beam_file(beam, outname)
        if name == self.end:
            self.global_parameters["beam"] = beam

    def postProcess(self) -> None:
        """Write the screen and end beams as openPMD, and the twiss HDF5 if ``save_twiss``."""
        from cheetah.accelerator import Screen

        def sanitized(n: str) -> str:
            s = "".join(c if c.isalnum() or c == "_" else "_" for c in n)
            return "_" + s if s[0].isdigit() else s

        original = {sanitized(n): n for n in self.elementObjects}
        screens = {}
        for element in self.segment.elements:
            if isinstance(element, Screen):
                screens.update({original.get(element.name, element.name): element.get_read_beam()})
        if not isinstance(self.segment.elements[-1], Screen):
            screens.update({self.end: self.pout})
        beam_tangents = {}
        for name, scr in screens.items():
            if name == self.start:
                continue
            outname = self.output_beam_file(name)
            basename = os.path.basename(outname).removesuffix(".openpmd.hdf5")
            self.screen_threaded_function.scatter(scr, outname, name)
            if self.tangents is not None:
                # track_tangents keys a screen by Cheetah's (sanitized) element name, the line end by its own
                tangents = self.tangents["beams"]
                beam_tangents[basename] = tangents[name] if name in tangents else tangents[sanitized(name)]
                # for the line that reads this file next; see check_incoming_tangents
                self.global_parameters.get("tangents", {})[os.path.abspath(outname)] = {
                    "parameters": self.tangents["parameters"],
                    "values": self.tangents["values"],
                    "tangent": beam_tangents[basename],
                }
        self.screen_threaded_function.gather()
        if self.cheetahglobal["save_twiss"] and self.tws is not None:
            twsname = f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.cheetah.hdf5'
            with h5py.File(twsname, "w") as f:
                twsgrp = f.create_group("Twiss")
                for key, val in zip(twiss_keys, self.tws):
                    data = val.numpy()
                    if key == "s":
                        data, z = self.s_and_z(data)
                    twsgrp.create_dataset(key, data=data)
                twsgrp.create_dataset("z", data=z)
        if self.tangents is not None:
            s, z = self.s_and_z(self.tangents["twiss"]["s"])
            write_gradients(
                f'{self.global_parameters["master_subdir"]}/{self.objectname}_gradients.cheetah.hdf5',
                self.tangents["parameters"],
                self.tangents["values"],
                {**self.tangents["twiss"], "s": s, "z": z},
                beam_tangents,
                code=self.code,
                lattice=self.objectname,
                coordinates=list(lab_coordinates),
            )
