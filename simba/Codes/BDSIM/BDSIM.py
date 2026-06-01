"""
SIMBA BDSIM Module

Various objects and functions to handle BDSIM lattices and commands. See `BDSIM github`_ for more details.

    .. _Cheetah github: https://github.com/bdsim-collaboration/bdsim

Classes:
    - :class:`~simba.Codes.BDSIM.BDSIM.bdsimLattice`: The BDSIM lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a BDSIM lattice object,
    and for tracking through it.

"""

from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf

import subprocess
import os
from typing import Any, ClassVar
import h5py
import lox
from lox.worker.thread import ScatterGatherDescriptor
from laura.models.diagnostic import DiagnosticElement


class bdsimLattice(frameworkLattice):
    """
    Class for defining the BDSIM lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a BDSIM lattice object,
    and for tracking through it.
    """

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ELEGANT into the SIMBA generic 
    :class:`~simba.Modules.Beams.beam` object and writing files"""

    code: str = "bdsim"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    lattice: Any | None = None
    """
    Lattice elements arranged into a BDSIM `Machine`_

    .. _Machine: https://github.com/bdsim-collaboration/pybdsim/blob/develop/src/pybdsim/Builder.py
    """

    pin: Any | None = None
    """Initial particle distribution as a Cheetah `ParticleArray`_

    .. _ParticleBeam: https://github.com/desy-ml/cheetah/blob/master/cheetah/particles/particle_beam.py"""

    pout: Any | None = None
    """Final particle distribution as a Cheetah `ParticleArray`_"""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    ref_s: float = None
    """Reference s position"""

    ref_idx: int = None
    """Reference particle index"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
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

    def write(self) -> None:
        """
        Create the lattice object and save it as a JSON file to `master_subdir`.
        """
        self.lattice = self.section.to_bdsim(save=True)

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.read_input_file(prefix, self.particle_definition)
        self.ref_s = self.global_parameters["beam"].s
        self.ref_idx = self.global_parameters["beam"].reference_particle_index
        self.hdf5_to_bdsim()

    def hdf5_to_bdsim(self) -> None:
        """
        Convert the initial HDF5 particle distribution to BDSIM beam input format and set
        :attr:`~simba.Codes.Cheetah.Cheetah.cheetahLattice.pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        bdsimbeamfilename = self.particle_definition + ".bdsim"
        self.global_parameters["beam"].beam.rematchXPlane(
            **self.initial_twiss["horizontal"]
        )
        self.global_parameters["beam"].beam.rematchYPlane(
            **self.initial_twiss["vertical"]
        )

        rbf.bdsim.write_bdsim_beam_file(
            beam=self.global_parameters["beam"],
            filename=bdsimbeamfilename,
        )

    def run(self) -> None:
        """
        Run the code with input 'filename'
        This method constructs the command to run the simulation using the specified executable
        and the name of the lattice. It redirects the output to a log file in the master subdirectory.

        If  :attr:`~remote_setup` is set, then :func:`~run_remote` will be called instead.

        Raises
        ------
        FileNotFoundError
            If the executable for the specified code is not found in the executables dictionary.
        """
        if self.remote_setup:
            self.run_remote()
        else:
            command = (
                self.executables[self.code]
                + [f"--file={self.global_parameters['master_subdir']}/{self.name}.gmad"]
                + ["--batch"]
                + [
                    f"--output={self.global_parameters['master_subdir']}/{self.name}.root"
                ]
            )
            print(command)
            with open(
                os.path.relpath(
                    self.global_parameters["master_subdir"] + "/" + self.name + ".log",
                    ".",
                ),
                "w",
            ) as f:
                subprocess.call(
                    command,
                    executable="/bin/bash",
                    stdout=f,
                    cwd=self.global_parameters["master_subdir"],
                    env={**os.environ},
                    shell=True,
                )

    @lox.thread(40)
    def screen_threaded_function(
        self, scr: DiagnosticElement, outname: str, name: str
    ) -> None:
        """
        Convert output from Cheetah ParticleBeam to HDF5 format

        Parameters
        ----------
        scr: LAURA DiagnosticElement
            Screen object
        outname: str
            Name of Cheetah beam file
        name: str
            Name of element
        """
        from ...Modules.Beams import cheetah as rbf_cheetah

        beam = rbf.beam()
        s = 0
        try:
            s = self.elementObjects[name].physical.middle.z
        except KeyError:
            s = self.elementObjects[name.replace("_", "-")].physical.middle.z
        # scr.tau -= self.startObject.physical.middle.z
        rbf_cheetah.interpret_cheetah_ParticleBeam(
            beam,
            scr,
            zstart=self.startObject.physical.start.z,
            s=scr.s.numpy(),
            ref_index=self.ref_idx,
        )
        rbf.openpmd.write_openpmd_beam_file(beam, outname)
        if name == self.end:
            self.global_parameters["beam"] = beam

    def postProcess(self) -> None:
        """
        Convert the outputs from Cheetah to HDF5 format and save them to `master_subdir`.
        """
        from cheetah.accelerator import Screen

        screens = {}
        for element in self.segment.elements:
            if isinstance(element, Screen):
                screens.update({element.name: element.get_read_beam()})
        if not isinstance(self.segment.elements[-1], Screen):
            screens.update({self.end: self.pout})
        i = 0
        for name, scr in screens.items():
            outname = f'{self.global_parameters["master_subdir"]}/{name.replace("_", "-")}.openpmd.hdf5'
            self.screen_threaded_function.scatter(scr, outname, name)
            i += 1
        self.screen_threaded_function.gather()
        if self.cheetahglobal["save_twiss"] and self.tws is not None:
            twsname = f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.cheetah.hdf5'
            with h5py.File(twsname, "w") as f:
                twsgrp = f.create_group("Twiss")
                for key, val in zip(twiss_keys, self.tws):
                    twsgrp.create_dataset(key, data=val.numpy())
