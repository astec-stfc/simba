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

import os
import numpy as np
from typing import Any
from warnings import warn
from pybdsim.Run import Bdsim, RebdsimOptics
from pybdsim.Beam import Beam

BDSIM_PARTICLES = {1: "e-", 2: "e+", 3: "proton"}
"""BDSIM particle names, keyed by the SIMBA particle index returned by
:meth:`~simba.Modules.Beams.Particles.Particles.get_particle_index`.

Index 4 is H-, which BDSIM has no built-in name for (it would need a PDG ion
id), so it is deliberately absent rather than mapped to ``antiproton``."""


class bdsimLattice(frameworkLattice):
    """
    Class for defining the BDSIM lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a BDSIM lattice object,
    and for tracking through it.
    """

    code: str = "bdsim"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    lattice: Any | None = None
    """
    Lattice elements arranged into a BDSIM `Machine`_

    .. _Machine: https://github.com/bdsim-collaboration/pybdsim/blob/develop/src/pybdsim/Builder.py
    """

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

    @property
    def gmadname(self) -> str:
        """
        Name of the gmad/ROOT files for this lattice (sanitized).
        """
        return self.section.name.replace("-", "_")

    @property
    def particle_type(self) -> str:
        """
        BDSIM name of the beam particle, from the mass and charge of the beam.
        """
        beam = self.global_parameters["beam"]
        index = beam.beam.get_particle_index(
            float(np.asarray(beam.particle_mass)[0]), np.asarray(beam.charge)[0]
        )
        if index not in BDSIM_PARTICLES:
            raise ValueError(
                f"Particle index {index} has no BDSIM particle name; "
                "BDSIM cannot track this species by name"
            )
        return BDSIM_PARTICLES[index]

    @property
    def charge_sign(self) -> int:
        """
        Sign of the charge of the beam particle; needed for setting magnet polarities.
        """
        return int(self.global_parameters["beam"].chargesign[0])

    def write(self) -> None:
        """
        Create the lattice object and save it as a JSON file to `master_subdir`.
        """
        beam = Beam()
        beam.SetParticleType(self.particle_type)
        beam.SetDistributionType("userfile")
        beam.SetDistrFile(os.path.join(self.global_parameters["master_subdir"], self.particle_definition + ".bdsim"))
        beam.SetDistrFileFormat(rbf.bdsim.BDSIM_DISTRIBUTION_FORMAT)
        beam.SetEnergy(self.global_parameters["beam"].centroids.mean_energy.val, unitsstring="eV")
        self.lattice = self.section.to_bdsim(
            save=True, beam=beam, charge_sign=self.charge_sign
        )

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
        bdsimbeamfilename = os.path.join(self.global_parameters["master_subdir"], self.particle_definition + ".bdsim")
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

    def _generate_optics_config(self):
        with open(f"{self.global_parameters['master_subdir']}/optics_config.txt", "w") as f:
            f.write("CalculateOpticalFunctions 1 \nMergeHistograms 0\n")

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
            Bdsim(
                f"{self.global_parameters['master_subdir']}/{self.gmadname}.gmad",
                f"{self.global_parameters['master_subdir']}/{self.gmadname}",
                ngenerate=len(self.global_parameters["beam"].x),
                bdsimExecutable=self.executables[self.code][-1],
            )
            RebdsimOptics(
                f"{self.global_parameters['master_subdir']}/{self.gmadname}.root",
                f"{self.global_parameters['master_subdir']}/{self.gmadname}.optics.root",
            )

    def screen_function(self, name: str, arrays: dict, outname: str) -> None:
        """
        Convert one BDSIM sampler into a SIMBA beam and write it to HDF5.

        Parameters
        ----------
        name: str
            Name of the element the sampler belongs to.
        arrays: dict
            Raw sampler arrays, see
            :func:`~simba.Modules.Beams.bdsim.read_bdsim_sampler_arrays`.
        outname: str
            Name of the openPMD file to write.
        """
        beam = rbf.beam()
        rbf.bdsim.interpret_bdsim_sampler(
            beam,
            arrays,
            charge=self.global_parameters["beam"].total_charge,
            zstart=self.startObject.physical.start.z,
            ref_index=self.ref_idx,
        )
        rbf.openpmd.write_openpmd_beam_file(beam, outname)
        if name == self.end:
            self.global_parameters["beam"] = beam

    def postProcess(self) -> None:
        """
        Convert the BDSIM sampler outputs to HDF5 format and save them to
        `master_subdir`.

        BDSIM writes one sampler per marker into the ``Event`` tree of its raw ROOT
        output.
        """
        super().postProcess()
        if not self.trackBeam:
            return
        rootfile = os.path.join(
            self.global_parameters["master_subdir"], self.gmadname + ".root"
        )
        if not os.path.isfile(rootfile):
            raise FileNotFoundError(
                f"BDSIM output file {rootfile} not found; the tracking run failed"
            )
        available = rbf.bdsim.get_bdsim_sampler_names(rootfile)
        wanted = {
            scr.name: scr.name.replace("-", "_")
            for scr in self.screens_and_markers_and_bpms
        }
        missing = [n for n, s in wanted.items() if s not in available]
        for name in missing:
            warn(f"No BDSIM sampler found for {name}; no beam file will be written")
        wanted = {n: s for n, s in wanted.items() if n not in missing}
        if not wanted:
            warn(f"No BDSIM samplers found in {rootfile}")
            return
        arrays = rbf.bdsim.read_bdsim_sampler_arrays(
            rootfile, samplers=list(wanted.values())
        )
        for name, sampler in wanted.items():
            outname = os.path.join(
                self.global_parameters["master_subdir"], f"{name}.openpmd.hdf5"
            )
            self.screen_function(name, arrays[sampler], outname)
