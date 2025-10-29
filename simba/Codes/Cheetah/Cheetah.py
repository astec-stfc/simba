"""
SIMBA Cheetah Module

Various objects and functions to handle Cheetah lattices and commands. See `Cheetah github`_ for more details.

    .. _Cheetah github: https://github.com/desy-ml/cheetah

Classes:
    - :class:`~simba.Codes.Cheetah.Cheetah.cheetahLattice`: The Cheetah lattice object, used for
    converting the :class:`~simba.Framework_elements.frameworkObject` s defined in the
    :class:`~simba.Framework_elements.frameworkLattice` into a Cheetah lattice object,
    and for tracking through it.

"""
from torch import Tensor, as_tensor, float64

from ...Framework_objects import frameworkLattice
#from ...Framework_elements import screen
from ...Modules import Beams as rbf

from numpy import mean
import os
from yaml import safe_load
from copy import deepcopy
from typing import Dict, Any
import h5py


with open(
    os.path.dirname(os.path.abspath(__file__)) + "/cheetah_defaults.yaml",
    "r",
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
    """
    Class for defining the Cheetah lattice object, used for
    converting the :class:`~simba.Framework_elements.frameworkObject`s defined in the
    :class:`~simba.Framework_elements.frameworkLattice` into a Cheetah lattice object,
    and for tracking through it.
    """

    code: str = "cheetah"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    segment: Any | None = None
    """
    Lattice elements arranged into a Cheetah `Segment`_
    
    .. _Segment: https://github.com/desy-ml/cheetah/blob/master/cheetah/accelerator/segment.py
    """

    pin: Any | None = None
    """Initial particle distribution as a Cheetah `ParticleArray`_

    .. _ParticleBeam: https://github.com/desy-ml/cheetah/blob/master/cheetah/particles/particle_beam.py"""

    pout: Any | None = None
    """Final particle distribution as a Cheetah `ParticleArray`_"""

    tws: tuple[Tensor, ...] | Tensor | None = None
    """Tensor or tuple of Tensors containing Twiss parameters"""

    cheetahglobal: Dict = {}
    """Global settings for Cheetah, read in from `cheetahLattice.settings["global"]["Cheetahsettings"]` and
    `cheetah_defaults.yaml`"""

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
        if (
            "input" in self.file_block
            and "particle_definition" in self.file_block["input"]
        ):
            if (
                self.file_block["input"]["particle_definition"]
                == "initial_distribution"
            ):
                self.particle_definition = "laser"
            else:
                self.particle_definition = self.file_block["input"][
                    "particle_definition"
                ]
        else:
            self.particle_definition = self.start


    def writeElements(self) -> bool:
        """
        Create Cheetah objects for all the elements in the lattice and set the
        :attr:`~simba.Codes.Cheetah.Cheetah.cheetahLattice.segment`.

        Returns
        -------
        bool
            True if successful
        """
        self.segment = self.section.to_cheetah(save=True)
        return True

    def write(self) -> None:
        """
        Create the lattice object via :func:`~simba.Codes.Cheetah.Cheetah.cheetahLattice.writeElements`
        and save it as a JSON file to `master_subdir`.
        """
        success = self.writeElements()
        if success:
            self.segment.to_lattice_json(
                filepath=f'{self.global_parameters["master_subdir"]}/{self.objectname}.json'
            )

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        super().preProcess()
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.hdf5_to_openpmd(prefix)

    def hdf5_to_openpmd(self, prefix="", write=True) -> None:
        """
        Convert the initial HDF5 particle distribution to OpenPMD format and set
        :attr:`~simba.Codes.Cheetah.Cheetah.cheetahLattice.pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        cheetahbeamfilename = prefix + self.particle_definition + ".openpmd.hdf5"
        self.read_input_file(prefix, self.particle_definition)
        self.pin = rbf.beam.write_cheetah_beam_file(
            self.global_parameters["beam"],
            cheetahbeamfilename,
            write=write,
            s_start=self.startObject.physical.start.z,
        )

    def run(self) -> None:
        """
        Run the code, and set :attr:`~tws` and :attr:`~pout`
        """
        # navi = self.navi_setup()
        pin = deepcopy(self.pin)
        # if self.sample_interval > 1:
        #     pin = pin.thin_out(nth=self.sample_interval)
        self.pout = self.segment.track(pin)
        if self.cheetahglobal["save_twiss"]:
            self.tws = self.segment.get_beam_attrs_along_segment(twiss_keys, pin)
            # print("Twiss parameters:", self.tws)

    def postProcess(self) -> None:
        """
        Convert the outputs from Cheetah to HDF5 format and save them to `master_subdir`.
        """
        from cheetah.accelerator import Screen
        from ...Modules.Beams import cheetah as rbf_cheetah
        screens = {}
        for element in self.segment.elements:
            if isinstance(element, Screen):
                screens.update({element.name: element.get_read_beam()})
        screens.update({self.end: self.pout})
        for name, scr in screens.items():
            beam = rbf.beam()
            outname = f'{self.global_parameters["master_subdir"]}/{name.replace("_", "-")}.openpmd.hdf5'
            s = 0
            try:
                s = self.elementObjects[name].physical.middle.z
            except KeyError:
                s = self.elementObjects[name.replace('_', "-")].physical.middle.z
            # scr.tau -= self.startObject.physical.middle.z
            rbf_cheetah.particle_beam_to_beam(beam, scr, s_start=s)
            beam.write_openpmd_beam_file(filename=outname)
            if name == self.end:
                self.global_parameters["beam"] = beam
        if self.cheetahglobal["save_twiss"] and self.tws is not None:
            twsname = f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.cheetah.hdf5'
            with h5py.File(twsname, "w") as f:
                twsgrp = f.create_group("Twiss")
                for key, val in zip(twiss_keys, self.tws):
                    valn = val.numpy()
                    # if key == "s":
                    #     valn += self.startObject.physical.start.z
                    twsgrp.create_dataset(key, data=val.numpy())