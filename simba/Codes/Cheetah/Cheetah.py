"""
SIMBA Cheetah module: builds and tracks a Cheetah segment. See `Cheetah github`_.

    .. _Cheetah github: https://github.com/desy-ml/cheetah
"""
from ...Framework_objects import frameworkLattice
from ...FrameworkHelperFunctions import set_deep_attr
from ...Modules import Beams as rbf
from ...Modules.gradients import write_gradients
from ... import exceptions

import os
import numpy as np
from yaml import safe_load
from copy import deepcopy
from functools import partial, reduce
from warnings import warn
from typing import Dict, Any, ClassVar
import h5py
import lox
from lox.worker.thread import ScatterGatherDescriptor
from laura.models.diagnostic import DiagnosticElement


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
        cheetahbeamfilename = f'{self.global_parameters["master_subdir"]}/{self.particle_definition}.cheetah.hdf5'
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
        """Track :attr:`pin` and set :attr:`pout` (and :attr:`tws` if ``save_twiss``, :attr:`tangents` if :attr:`gradient_parameters`)."""
        pin = deepcopy(self.pin)
        self.pout = self.segment.track(pin)
        if self.cheetahglobal["save_twiss"]:
            self.tws = self.segment.get_beam_attrs_along_segment(twiss_keys, pin)
        self.tangents = self.track_tangents() if self.gradient_parameters else None

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
        buffers = []
        try:
            for v in (value + step, value - step):
                set_deep_attr(element, key, v)
                segment = self.section.to_cheetah(Brho=self.rigidity(self.reference_p0c))
                buffers.append(dict(segment.named_buffers()))
        finally:
            set_deep_attr(element, key, value)
        up, down = buffers
        derivatives = {k: (up[k] - down[k]) / (2 * step) for k in up if not equal(up[k], down[k])}
        if not derivatives:
            warn(exceptions.SettingWarning(
                f"{exceptions._line(self.objectname)}: gradient parameter {parameter} does not change the "
                "Cheetah lattice, so its derivatives are zero."
            ))
        return derivatives

    def track_tangents(self) -> Dict:
        """
        Forward-mode derivatives (``torch.func.jvp``) of the twiss along :attr:`segment` and of
        the beam at each screen and the line end, one pass per :attr:`gradient_parameters`.

        Returns
        -------
        Dict
            ``parameters`` and their ``values``; ``twiss``: key -> (n_s, P) array, with the
            ``s`` -> (n_s,) it is sampled at; ``beams``: element name -> (n, 6, P) array of
            :data:`~simba.Modules.Beams.cheetah.lab_coordinates`
        """
        from torch import stack, tensor, float64
        from torch.func import jvp
        from cheetah.accelerator import Screen
        from ...Modules.Beams.cheetah import cheetah_lab_coordinates

        base = dict(self.segment.named_buffers())
        end_is_screen = isinstance(self.segment.elements[-1], Screen)

        def track(segment, derivatives, t):
            for path, derivative in derivatives.items():
                module, _, leaf = path.rpartition(".")
                setattr(segment.get_submodule(module), leaf, base[path] + t * derivative)
            twiss = {k: [] for k in twiss_keys}
            for beam in segment.beam_along_segment_generator(deepcopy(self.pin)):
                for k in twiss_keys:
                    twiss[k].append(getattr(beam, k))
            beams = {
                e.name: cheetah_lab_coordinates(e.get_read_beam(), self.rest_energy)
                for e in segment.elements if isinstance(e, Screen)
            }
            if not end_is_screen:
                beams[self.end] = cheetah_lab_coordinates(beam, self.rest_energy)
            return {k: stack(v) for k, v in twiss.items()}, beams

        zero, one = tensor(0.0, dtype=float64), tensor(1.0, dtype=float64)
        tangents = []
        for parameter in self.gradient_parameters:
            derivatives = self.cheetah_derivatives(parameter)
            (twiss, _), tangent = jvp(partial(track, deepcopy(self.segment), derivatives), (zero,), (one,))
            tangents.append(tangent)
        return {
            "parameters": self.gradient_parameters,
            "values": [
                reduce(getattr, p.split(":")[1].split("."), self.elements[p.split(":")[0]])
                for p in self.gradient_parameters
            ],
            "twiss": {
                "s": twiss["s"].numpy(),
                **{k: stack([t[0][k] for t in tangents], -1).numpy() for k in twiss_keys if k != "s"},
            },
            "beams": {n: stack([t[1][n] for t in tangents], -1).numpy() for n in tangents[0][1]},
        }

    def s_and_z(self, s: np.ndarray) -> tuple:
        """Cheetah's ``s`` along :attr:`segment` as the lattice's ``(s, z)``."""
        svals = s - s[0]
        lat_s = np.array(self.getSValues(at_entrance=False))
        lat_z = [a[-1] for a in self.getZValues()]
        return svals + self.entrance_s, np.interp(svals, lat_s, lat_z)

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
            try:
                zstart = self.elementObjects[name].physical.start.z
            except KeyError:
                zstart = self.elementObjects[name.replace('_', "-")].physical.start.z
        rbf_cheetah.interpret_cheetah_ParticleBeam(
            beam,
            scr,
            zstart=zstart,
            s=scr.s.numpy(),
            ref_index=self.ref_idx,
        )
        rbf.openpmd.write_openpmd_beam_file(beam, outname)
        if name == self.end:
            self.global_parameters["beam"] = beam

    def postProcess(self) -> None:
        """Write the screen and end beams as openPMD, and the twiss HDF5 if ``save_twiss``."""
        from cheetah.accelerator import Screen
        screens = {}
        for element in self.segment.elements:
            if isinstance(element, Screen):
                screens.update({element.name: element.get_read_beam()})
        if not isinstance(self.segment.elements[-1], Screen):
            screens.update({self.end: self.pout})
        beam_tangents = {}
        for name, scr in screens.items():
            if name.replace("_", "-") == self.start:
                continue
            basename = self.output_basename(name).replace("_", "-")
            outname = f'{self.global_parameters["master_subdir"]}/{basename}.openpmd.hdf5'
            self.screen_threaded_function.scatter(scr, outname, name)
            if self.tangents is not None:
                beam_tangents[basename] = self.tangents["beams"][name]
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
            from ...Modules.Beams.cheetah import lab_coordinates
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