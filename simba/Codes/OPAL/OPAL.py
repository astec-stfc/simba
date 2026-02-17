import os
import subprocess
import numpy as np
import yaml
from typing import Dict, Literal

from ...Framework_objects import (
    frameworkLattice,
    getGrids,
)
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from ...Modules.Beams.opal import find_opal_s_positions
from ...Modules.SDDSFile import SDDSFile
# import mpi4py
# mpi4py.rc.initialize = False

from laura.translator.converters.codes.opal import (
    opal_option,
    opal_distribution,
    opal_fieldsolver,
    opal_beam,
    opal_track,
    opal_run,
)

from ...Modules.constants import speed_of_light
from ...Modules.units import UnitValue


def update_globals(global_settings, beamlen=None, sample_interval=1):
    grids = getGrids()
    with open(
            os.path.join(os.path.dirname(__file__), "globals_Opal.yaml"), "r"
    ) as file:
        opalglobal = yaml.load(file, Loader=yaml.Loader)
    for sc in ['x', 'y', 'z']:
        if f"SC_3D_N{sc}f" in list(global_settings.keys()):
            scconv = sc.upper().replace('Z', 'T')
            global_settings.update({f"M{scconv}": global_settings[f"SC_3D_N{sc}f"]})
    for typ, vals in opalglobal.items():
        for k, v in vals.items():
            if k in global_settings.keys():
                opalglobal[typ].update({k: v})
    if beamlen:
        gridsize = grids.getGridSizes(
            (beamlen / sample_interval)
        )
        opalglobal["fieldsolver"].update({"MX": gridsize, "MY": gridsize, "MT": gridsize})
    return opalglobal

class opalLattice(frameworkLattice):
    """
    Class for defining the GPT lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
    the lattice suitable for an OPAL input file.
    """

    code: str = "opal"
    """String indicating the lattice object type"""

    headers: Dict = {}
    """Headers to be included in the OPAL lattice file"""

    particle_definition: str = None
    """Name of initial particle distribution"""

    time_step_size: float = 2e-12
    """Step size for tracking"""

    breakstr: str = "//----------------------------------------------------------------------------"
    """String used for separating headers in the input file"""

    version: str = "202210"
    """Version of OPAL"""

    maxsteps: int = 1000000
    """Maximum number of steps for tracking; will be set dynamically once the lattice is parsed"""

    headers: Dict = {}
    """Section headers for OPAL input file"""

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

    def write(self):
        self.section.opal_headers = self.headers
        output = self.section.to_opal(
            energy=self.global_parameters["beam"].centroids.mean_cpz.val / 1e6,
            breakstr=self.breakstr,
        )
        command_file = (
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(command_file, output, "w")
        self.files.append(command_file)

    def preProcess(self):
        super().preProcess()
        prefix = self.get_prefix()
        fpath = self.read_input_file(prefix, self.particle_definition)
        self.ref_s = self.global_parameters["beam"].s
        self.ref_idx = self.global_parameters["beam"].reference_particle_index
        self.hdf5_to_opal()
        beamlen = len(self.global_parameters["beam"].x)
        pc = np.mean(self.global_parameters["beam"].cpz.val) / 1e9
        bcurrent = abs(self.global_parameters["beam"].total_charge * 1e6)
        chargesign = int(self.global_parameters["beam"].chargesign[0])
        if "particle_definition" in list(self.file_block["input"].keys()):
            initobj = "laser" if self.file_block["input"]["particle_definition"] == "initial_distribution" else self.start
        else:
            initobj = self.start
        self.headers["option"] = opal_option()
        self.headers["distribution"] = opal_distribution(input_particle_definition=f"\"{initobj}.opal\"")
        self.headers["fieldsolver"] = opal_fieldsolver(
            npart=beamlen,
            sample_interval=self.sample_interval,
            space_charge_mode=str(self.space_charge_mode),
        )
        self.headers["beam"] = opal_beam(
            PC=pc,
            NPART=beamlen,
            CHARGE=chargesign,
            PARTICLE=self.global_parameters["beam"].species.upper(),
            BCURRENT=bcurrent,
        )
        self.headers["track"] = opal_track(
            DT=self.time_step_size,
            MAXSTEPS=self.maxsteps,
            LINE=self.objectname,
            ZSTOP=self.endObject.physical.end.z - self.startObject.physical.start.z,
        )
        self.headers["run"] = opal_run()
        self.files.append(f"{self.global_parameters['master_subdir']}/{initobj}.opal")
        self.write()

    def postProcess(self):
        elems = self.getSValues(as_dict=True)
        svals = {}
        for s in self.screens_and_bpms:
            svals.update({s.name: elems[s.name] - self.startObject.physical.start.z})
        spositions = find_opal_s_positions(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}.h5',
            svals,
            tolerance=1.0,
        )
        for elem in self.screens_and_bpms:
            if elem.name in spositions:
                opalbeamname = f'{self.global_parameters["master_subdir"]}/{self.objectname}.h5'
                beam = rbf.beam()
                beam.read_opal_beam_file(filename=opalbeamname, step=spositions[elem.name])
                beam._beam.z = UnitValue(beam._beam.z.val + self.startObject.physical.middle.z, "m")
                beam._beam.t = UnitValue(beam._beam.t.val + (self.startObject.physical.middle.z / speed_of_light), "s")
                rbf.openpmd.write_openpmd_beam_file(
                    beam,
                    f'{self.global_parameters["master_subdir"]}/{elem.name}.openpmd.hdf5',
                )
        opalbeamname = f'{self.global_parameters["master_subdir"]}/{self.objectname}.h5'
        beam = rbf.beam()
        beam.read_opal_beam_file(filename=opalbeamname, step=-1)
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            f'{self.global_parameters["master_subdir"]}/{self.endObject.name}.openpmd.hdf5',
        )
        self.commandFiles = {}
        opalObject = SDDSFile()
        opalObject.read_file(f"{self.global_parameters['master_subdir']}/{self.objectname}.stat")
        opalData = opalObject.data
        for k in opalData:
            # handling for multiple elegant runs per file (e.g. error simulations)
            # by default extract only the first run (in ELEGANT this is the fiducial)
            if isinstance(opalData[k], np.ndarray) and (opalData[k].ndim > 1):
                opalData[k] = opalData[k][0]
            else:
                opalData[k] = np.array(opalData[k])
        if self.ref_s is not None:
            opalData["s"] += self.ref_s
        import h5py
        with h5py.File(f"{self.global_parameters['master_subdir']}/{self.objectname}.opal_twiss.h5", "w") as f:
            for k, v in opalData.items():
                try:
                    f.create_dataset(k, data=np.array(v))
                except TypeError as e:
                    pass

    def hdf5_to_opal(self):
        emitted = True if self.particle_definition == "laser" else False
        rbf.opal.write_opal_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + self.particle_definition + '.opal',
            subz=self.startObject.physical.start.z,
            emitted=emitted,
        )

    def run(self):
        """Run the code with input 'filename'"""
        if self.remote_setup:
            self.run_remote()
        else:
            if not os.name == "nt":
                command = "bash -c '" + " ".join(self.executables[self.code] + [self.objectname + ".in"]) + "'"
                with open(
                    os.path.abspath(
                        self.global_parameters["master_subdir"]
                        + "/"
                        + self.objectname
                        + ".log"
                    ),
                    "w",
                ) as f:
                    subprocess.call(
                        command,
                        stdout=f,
                        cwd=self.global_parameters["master_subdir"],
                        env={**os.environ},
                        shell=True
                    )
