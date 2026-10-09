"""
SIMBA Cheetah module: builds and tracks a Cheetah segment. See `Cheetah github`_.

    .. _Cheetah github: https://github.com/desy-ml/cheetah
"""
from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf

import os
import numpy as np
from yaml import safe_load
from copy import deepcopy
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
        self.segment = self.section.to_cheetah()
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
        """Track :attr:`pin` and set :attr:`pout` (and :attr:`tws` if ``save_twiss``)."""
        pin = deepcopy(self.pin)
        self.pout = self.segment.track(pin)
        if self.cheetahglobal["save_twiss"]:
            self.tws = self.segment.get_beam_attrs_along_segment(twiss_keys, pin)

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
        i = 0
        for name, scr in screens.items():
            if name.replace("_", "-") == self.start:
                continue
            outname = (
                f'{self.global_parameters["master_subdir"]}/'
                f'{self.output_basename(name).replace("_", "-")}.openpmd.hdf5'
            )
            self.screen_threaded_function.scatter(scr, outname, name)
            i += 1
        self.screen_threaded_function.gather()
        if self.cheetahglobal["save_twiss"] and self.tws is not None:
            twsname = f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.cheetah.hdf5'
            with h5py.File(twsname, "w") as f:
                twsgrp = f.create_group("Twiss")
                svals = None
                for key, val in zip(twiss_keys, self.tws):
                    data = val.numpy()
                    if key == "s":
                        svals = data - data[0]
                        data = svals + self.entrance_s
                    twsgrp.create_dataset(key, data=data)
                if svals is not None:
                    lat_s = np.array(self.getSValues(at_entrance=False))
                    lat_z = [a[-1] for a in self.getZValues()]
                    twsgrp.create_dataset("z", data=np.interp(svals, lat_s, lat_z))