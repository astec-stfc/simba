"""
Simframe GPT Module

Various objects and functions to handle GPT lattices and commands.

Classes:
    - :class:`~simba.Codes.GPT.GPT.gptLattice`: The GPT lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
    the lattice suitable for GPT input and lattice files.

    - :class:`~simba.Codes.GPT.GPT.gpt_element`: Base class for defining
    commands in a GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_setfile`: Class for defining the
    input files for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_charge`: Class for defining the
    bunch charge for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_setreduce`: Class for reducing the
    number of particles for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_accuracy`: Class for setting the
    accuracy for GPT tracking.

    - :class:`~simba.Codes.GPT.GPT.gpt_spacecharge`: Class for defining the
    space charge setup for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_tout`: Class for defining the
    number of steps for particle distribution output for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_csr1d`: Class for defining the
    CSR calculations for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_writefloorplan`: Class for setting up the
    writing of the lattice floor plan for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_Zminmax`: Class for defining the
    minimum and maximum z-positions for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_forwardscatter`: Class for defining
    scattering parameters for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_scatterplate`: Class for defining a
    scattering object for the GPT input file.

    - :class:`~simba.Codes.GPT.GPT.gpt_dtmaxt`: Class for defining the
    step size(s) for the GPT input file.
"""

import os
import subprocess
import numpy as np
from PAdantic.models.diagnostic import DiagnosticElement

from ...Framework_objects import frameworkLattice
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from ...Modules.constants import speed_of_light
from ...Modules.units import UnitValue
from ...Modules.gdf_beam import gdf_beam
from typing import Dict, Literal, Any
from nala.translator.converters.codes.gpt import (
    gpt_setfile,
    gpt_charge,
    gpt_setreduce,
    gpt_accuracy,
    gpt_spacecharge,
    gpt_tout,
    gpt_csr1d,
    gpt_writefloorplan,
    gpt_Zminmax,
    gpt_forwardscatter,
    gpt_scatterplate,
    gpt_dtmaxt,
)

gpt_defaults = {}


class gptLattice(frameworkLattice):
    """
    Class for defining the GPT lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
    the lattice suitable for a GPT input file.
    """

    code: str = "gpt"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = True
    """Flag to indicate whether negative drifts are allowed"""

    bunch_charge: float | None = None
    """Bunch charge"""

    headers: Dict = {}
    """Headers to be included in the GPT lattice file"""

    ignore_start_screen: Any = None
    """Flag to indicate whether to ignore the first screen in the lattice"""

    screen_step_size: float = 0.1
    """Step size for screen output"""

    time_step_size: float = 1e-11
    """Step size for tracking"""

    override_meanBz: float | int | None = None
    """Set the average particle longitudinal velocity manually"""

    override_tout: float | int | None = None
    """Set the time step output manually"""

    accuracy: int = 6
    """Tracking accuracy"""

    endScreenObject: Any = None
    """Final screen object for dumping particle distributions"""

    Brho: UnitValue | None = None

    particle_definition: str = None

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
        self.headers["setfile"] = gpt_setfile(
            set='"beam"', filename='"' + self.name + '.gdf"'
        )
        self.headers["floorplan"] = gpt_writefloorplan(
            filename='"' + self.objectname + '_floor.gdf"'
        )

    @property
    def space_charge_mode(self) -> str | None:
        """
        Get the space charge mode based on
        :attr:`~simba.Framework_objects.frameworkLattice.globalSettings` or
        :attr:`~simba.Framework_objects.frameworkLattice.file_block`.

        Returns
        -------
        str
            Space charge mode as string, or None if not provided.

        """
        if (
            "charge" in self.file_block
            and "space_charge_mode" in self.file_block["charge"]
        ):
            return self.file_block["charge"]["space_charge_mode"]
        elif (
            "charge" in self.globalSettings
            and "space_charge_mode" in self.globalSettings["charge"]
        ):
            return self.globalSettings["charge"]["space_charge_mode"]
        else:
            return None

    @space_charge_mode.setter
    def space_charge_mode(self, mode: Literal["2d", "3d", "2D", "3D"]) -> None:
        """
        Set the space charge mode manually ["2D", "3D"].

        Parameters
        ----------
        mode: Literal["2d", "3d", "2D", "3D"]
            The space charge calculation mode
        """
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        self.file_block["charge"]["space_charge_mode"] = mode

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into a GPT-compatible format; see
        :attr:`~simba.Framework_objects.frameworkLattice.elementObjects`.

        The appropriate headers required for GPT are written at the top of the file, see the `write_GPT`
        function in :class:`~simba.Codes.GPT.gpt_element`.

        Returns
        -------
        str
            The lattice represented as a string compatible with GPT
        """
        self.headers["accuracy"] = gpt_accuracy(accuracy=self.accuracy)
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        if "charge" not in self.globalSettings:
            self.globalSettings["charge"] = {}
        space_charge_dict = self.file_block["charge"] | self.globalSettings["charge"]
        space_charge = self.global_parameters | space_charge_dict
        self.headers["spacecharge"] = gpt_spacecharge(**space_charge)
        if self.particle_definition == "laser" and self.space_charge_mode is not None:
            self.headers["spacecharge"].npart = len(self.global_parameters["beam"].x)
            self.headers["spacecharge"].sample_interval = self.sample_interval
            # self.headers["spacecharge"].space_charge_mode = "cathode"
        if (
            self.csr_enable
            and len(self.dipoles) > 0
            and max([abs(d.angle) for d in self.dipoles]) > 0
        ):  # and not os.name == 'nt':
            self.headers["csr1d"] = gpt_csr1d()
            # print('CSR Enabled!', self.objectname, len(self.dipoles))
        # self.headers['forwardscatter'] = gpt_forwardscatter(ECS='"wcs", "I"', name='cathode', probability=0)
        # self.headers['scatterplate'] = gpt_scatterplate(ECS='"wcs", "z", -1e-6', model='cathode', a=1, b=1)
        self.headers["setfile"].particle_definition = self.particle_definition
        self.section.gpt_headers = self.headers
        fulltext = self.section.to_gpt(
            startz=self.startObject.physical.start.z,
            endz=self.endObject.physical.end.z,
            Brho=self.global_parameters["beam"].Brho,
            # screen_step_size=self.screen_step_size,
        )
        return fulltext

    def write(self) -> None:
        """
        Writes the GPT input file from :func:`~simba.Codes.GPT.gptLattice.writeElements`
        to <master_subdir>/<self.objectname>.in.
        """
        code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(code_file, self.writeElements())
        self.files.append(code_file)

    def preProcess(self) -> None:
        """
        Convert the beam file from the previous lattice section into GPT format and set the number of
        particles based on the input distribution, see
        :func:`~simba.Codes.GPT.GPT.gptLattice.hdf5_to_astra`.
        """
        super().preProcess()
        self.headers["setfile"].particle_definition = self.objectname + ".gdf"
        prefix = self.get_prefix()
        self.hdf5_to_gdf(prefix)

    def run(self) -> None:
        """
        Run the code with input 'filename'

        `GPTLICENSE` must be provided in
        :attr:`~simba.Framework_objects.frameworkLattice.global_parameters`.

        Average properties of the distribution are also calculated and written
        to an `<>emit.gdf` file in `master_subdir`.
        """
        main_command = (
            self.executables[self.code]
            + ["-o", self.objectname + "_out.gdf"]
            + ["GPTLICENSE=" + self.global_parameters["GPTLICENSE"]]
            + [self.objectname + ".in"]
        )
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = (
            my_env["LD_LIBRARY_PATH"] + ":/opt/GPT3.3.6/lib/"
            if "LD_LIBRARY_PATH" in my_env
            else "/opt/GPT3.3.6/lib/"
        )
        my_env["OMP_WAIT_POLICY"] = "PASSIVE"
        post_command = (
            [self.executables[self.code][0].replace("gpt", "gdfa")]
            + ["-o", self.objectname + "_emit.gdf"]
            + [self.objectname + "_out.gdf"]
            + [
                "position",
                "Q",
                "avgx",
                "avgy",
                "avgz",
                "stdx",
                "stdBx",
                "stdy",
                "stdBy",
                "stdz",
                "stdt",
                "nemixrms",
                "nemiyrms",
                "nemizrms",
                "numpar",
                "nemirrms",
                "avgG",
                "avgp",
                "stdG",
                "avgt",
                "avgBx",
                "avgBy",
                "avgBz",
                "CSalphax",
                "CSalphay",
                "CSbetax",
                "CSbetay",
            ]
        )
        post_command_t = (
            [self.executables[self.code][0].replace("gpt", "gdfa")]
            + ["-o", self.objectname + "_emitt.gdf"]
            + [self.objectname + "_out.gdf"]
            + [
                "time",
                "Q",
                "avgx",
                "avgy",
                "avgz",
                "stdx",
                "stdBx",
                "stdy",
                "stdBy",
                "stdz",
                "nemixrms",
                "nemiyrms",
                "nemizrms",
                "numpar",
                "nemirrms",
                "avgG",
                "avgp",
                "stdG",
                "avgBx",
                "avgBy",
                "avgBz",
                "CSalphax",
                "CSalphay",
                "CSbetax",
                "CSbetay",
                "avgfBx",
                "avgfEx",
                "avgfBy",
                "avgfEy",
                "avgfBz",
                "avgfEz",
            ]
        )
        post_command_traj = (
            [self.executables[self.code][0].replace("gpt", "gdfa")]
            + ["-o", self.objectname + "traj.gdf"]
            + [self.objectname + "_out.gdf"]
            + ["time", "Q", "avgx", "avgy", "avgz"]
        )
        with open(
            os.path.abspath(
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".bat"
            ),
            "w",
        ) as batfile:
            for command in [
                main_command,
                post_command,
                post_command_t,
                post_command_traj,
            ]:
                output = '"' + command[0] + '" '
                for c in command[1:]:
                    output += c + " "
                output += "\n"
                batfile.write(output)
        with open(
            os.path.abspath(
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".log"
            ),
            "w",
        ) as f:
            # print('gpt command = ', command)
            subprocess.call(
                main_command,
                stdout=f,
                cwd=self.global_parameters["master_subdir"],
                env=my_env,
            )
            subprocess.call(
                post_command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )
            subprocess.call(
                post_command_t, stdout=f, cwd=self.global_parameters["master_subdir"]
            )
            subprocess.call(
                post_command_traj, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    def postProcess(self) -> None:
        """
        Convert the beam file(s) from the GPT output into HDF5 format, see
        :func:`~simba.Elements.screen.screen.gdf_to_hdf5`.
        """
        super().postProcess()
        cathode = self.particle_definition == "laser"
        gdfbeam = rbf.gdf.read_gdf_beam_file_object(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_out.gdf'
        )
        for e in self.screens_and_markers_and_bpms:
            if not e.name == self.start:
                self.gdf_to_hdf5(
                    gptbeamfilename=self.objectname + "_out.gdf",
                    screen=e,
                    cathode=cathode,
                    gdf=gdfbeam,
                    # t0=self.headers["setfile"].time,
                )
            # else:
            # print('Ignoring', self.ignore_start_screen.objectname)
        self.gdf_to_hdf5(
            gptbeamfilename=self.objectname + "_out.gdf",
            screen=self.endObject,
            cathode=cathode,
            gdf=gdfbeam,
        )

    def hdf5_to_gdf(self, prefix: str="") -> None:
        """
        Convert the HDF5 beam distribution to GDF format.

        Certain properties of this class, including
        :attr:`~simba.Codes.GPT.GPT.gptLattice.sample_interval`,
        :attr:`~simba.Codes.GPT.GPT.gptLattice.override_meanBz`,
        :attr:`~simba.Codes.GPT.GPT.gptLattice.override_tout` are also
        used to update
        :attr:`~simba.Codes.GPT.GPT.gptLattice.headers`.

        Parameters
        ----------
        prefix: str
            HDF5 file prefix
        """
        self.read_input_file(prefix, self.particle_definition)
        if self.particle_definition == "laser":
            self.global_parameters["beam"].z = UnitValue(0 * self.global_parameters["beam"].t, units="m")
        self.headers["setfile"].time = np.mean(self.global_parameters["beam"].t)
        if self.sample_interval > 1:
            self.headers["setreduce"] = gpt_setreduce(
                set='"beam"',
                setreduce=int(
                    len(self.global_parameters["beam"].x) / self.sample_interval
                ),
            )
        if self.override_meanBz is not None and isinstance(
            self.override_meanBz, (int, float)
        ):
            meanBz = self.override_meanBz
        else:
            meanBz = np.mean(self.global_parameters["beam"].Bz)
            if meanBz < 0.5:
                meanBz = 0.75

        if self.override_tout is not None and isinstance(
            self.override_tout, (int, float)
        ):
            self.headers["tout"] = gpt_tout(
                starttime=0, endpos=self.override_tout, step=str(self.time_step_size)
            )
        else:
            endpos = self.findS(self.end)[0][1]
            startpos = self.findS(self.start)[0][1]
            endpos += self.global_parameters["beam"].centroids.mean_t.val * speed_of_light
            startpos += self.global_parameters["beam"].centroids.mean_t.val * speed_of_light
            self.headers["tout"] = gpt_tout(
                starttime=startpos / meanBz / speed_of_light,
                endpos=endpos / meanBz / speed_of_light,
                step=str(self.time_step_size),
            )
        self.global_parameters["beam"].beam.rematchXPlane(
            **self.initial_twiss["horizontal"]
        )
        self.global_parameters["beam"].beam.rematchYPlane(
            **self.initial_twiss["vertical"]
        )
        gdfbeamfilename = self.objectname + ".gdf"
        cathode = self.particle_definition == "laser"
        rbf.gdf.write_gdf_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + gdfbeamfilename,
            normaliseX=self.startObject.physical.middle.x,
            cathode=cathode,
        )
        self.Brho = self.global_parameters["beam"].Brho
        self.files.append(self.global_parameters["master_subdir"] + "/" + gdfbeamfilename)

    def gdf_to_hdf5(
            self,
            screen: DiagnosticElement,
            gptbeamfilename: str,
            cathode: bool = False,
            gdf: gdf_beam | None = None,
    ) -> None:
        """
        Convert the GDF beam file to HDF5 format and write the beam file.

        Parameters
        ----------
        screen: nala.models.diagnostic.DiagnosticElement
            Diagnostic element
        gptbeamfilename: str
            Name of GPT beam file
        cathode: bool
            True if beam was emitted from a cathode
        gdf: gdfbeam or None
            GDF beam object
        """
        # gptbeamfilename = self.objectname + '.' + str(int(round((self.allElementObjects[self.end].position_end[2])*100))).zfill(4) + '.' + str(master_run_no).zfill(3)
        # try:
        # print('Converting screen', self.objectname,'at', self.gpt_screen_position)
        beam = rbf.beam()
        rbf.gdf.read_gdf_beam_file(
            beam,
            os.path.join(self.global_parameters["master_subdir"], gptbeamfilename),
            position=screen.physical.middle.z,
            gdfbeam=gdf,
        )
        HDF5filename = screen.name + ".openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            zoffset=screen.physical.middle.z,
        )
        # except:
        #     print('Error with screen', self.objectname,'at', self.gpt_screen_position)
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                (
                    os.path.join(
                        self.global_parameters["master_subdir"], gptbeamfilename
                    )
                ).strip('"')
            )
