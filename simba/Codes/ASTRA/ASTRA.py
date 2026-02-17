"""
SIMBA ASTRA Module

Various objects and functions to handle ASTRA lattices and commands. See `ASTRA manual`_ for more details.

    .. _ASTRA manual: https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf

Classes:
    - :class:`~simba.Codes.ASTRA.ASTRA.astraLattice`: The ASTRA lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
    the lattice suitable for an ASTRA input file.

    - :class:`~simba.Codes.ASTRA.ASTRA.astra_header`: Class for defining the &HEADER portion
    of the ASTRA input file.

    - :class:`~simba.Codes.ASTRA.ASTRA.astra_newrun`: Class for defining the &NEWRUN portion
    of the ASTRA input file.

    - :class:`~simba.Codes.ASTRA.ASTRA.astra_charge`: Class for defining the &CHARGE portion
    of the ASTRA input file.

    - :class:`~simba.Codes.ASTRA.ASTRA.astra_output`: Class for defining the &OUTPUT portion
    of the ASTRA input file.

    - :class:`~simba.Codes.ASTRA.ASTRA.astra_errors`: Class for defining the &ERRORS portion
    of the ASTRA input file.
"""

import os
from warnings import warn
import numpy as np
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import ClassVar, Dict, List, Any, Tuple
from pydantic import Field, field_validator, ConfigDict

from ...Framework_objects import frameworkLattice, global_error
from ...FrameworkHelperFunctions import expand_substitution, saveFile
from ...Modules import Beams as rbf
from laura.models.diagnostic import DiagnosticElement
from laura.models.element import PhysicalBaseElement
from laura.models.physical import PhysicalElement
from laura.translator.converters.codes.astra import (
    astra_newrun,
    astra_charge,
    astra_output,
    astra_errors,
)

from ...Modules.units import UnitValue

section_header_text_ASTRA = {
    "cavities": {"header": "CAVITY", "bool": "LEField"},
    "wakefields": {"header": "WAKE", "bool": "LWAKE"},
    "solenoids": {"header": "SOLENOID", "bool": "LBField"},
    "quadrupoles": {"header": "QUADRUPOLE", "bool": "LQuad"},
    "dipoles": {"header": "DIPOLE", "bool": "LDipole"},
    "astra_newrun": {"header": "NEWRUN"},
    "astra_output": {"header": "OUTPUT"},
    "astra_charge": {"header": "CHARGE"},
    "global_error": {"header": "ERROR"},
    "apertures": {"header": "APERTURE", "bool": "LApert"},
}


class astraLattice(frameworkLattice):
    """
    Class for defining the ASTRA lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
    the lattice suitable for an ASTRA input file.
    """

    model_config = ConfigDict(validate_assignment=True)

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ASTRA into the SIMBA generic 
    :class:`~simba.Modules.Beams.beam` object and writing files"""

    code: str = "astra"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = True
    """Flag to indicate whether negative drifts are allowed"""

    _bunch_charge: float | None = None
    """Bunch charge"""

    _toffset: float | None = None
    """Time offset of reference particle"""

    _space_charge_mode: str | None = None

    headers: Dict = {}
    """Headers to be included in the ASTRA lattice file"""

    starting_offset: list[float] = [0.0, 0.0, 0.0]
    """Initial offset of first element"""

    starting_rotation: list[float] = [0.0, 0.0, 0.0]
    """Initial rotation of first element"""

    zstop: float = None
    """End z position of lattice"""

    astra_headers: Dict[str, Any] = Field(default_factory=dict)
    """Headers for ASTRA input file"""

    ref_s: float = None
    """Reference s position"""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.starting_offset = (
            eval(expand_substitution(self, self.file_block["starting_offset"]))
            if "starting_offset" in self.file_block
            else [0, 0, 0]
        )

        # This calculated the starting rotation based on the input file and the number of dipoles
        self.starting_rotation = (
            [0.0, 0.0, float(-1 * self.startObject.physical.global_rotation.theta)]
        )
        self.starting_rotation = (
            eval(expand_substitution(self, str(self.file_block["starting_rotation"])))
            if "starting_rotation" in self.file_block
            else self.starting_rotation
        )

        # Create a "newrun" block
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "ASTRAsettings" not in self.globalSettings:
            self.globalSettings["ASTRAsettings"] = {}
        newrun_settings = self.file_block["input"] | self.globalSettings["ASTRAsettings"]
        starting_offset = [a + b for a, b in zip(self.startObject.physical.start, self.starting_offset)]
        self.section.astra_headers["newrun"] = astra_newrun(
            starting_offset=starting_offset,
            starting_rotation=self.starting_rotation,
            global_parameters=self.global_parameters,
            input_particle_definition = self.startObject.name,
            **newrun_settings,
        )
        # If the initial distribution is derived from a generator file, we should use that
        if (
            "input" in self.file_block
            and "particle_definition" in self.file_block["input"]
        ):
            if (
                self.file_block["input"]["particle_definition"]
                == "initial_distribution"
            ):
                self.section.astra_headers["newrun"].input_particle_definition = "laser.astra"
                self.section.astra_headers["newrun"].output_particle_definition = "laser.astra"
            else:
                self.section.astra_headers["newrun"].input_particle_definition = self.file_block[
                    "input"
                ]["particle_definition"]
                self.section.astra_headers["newrun"].output_particle_definition = (
                    self.objectname + ".astra"
                )
        else:
            self.section.astra_headers["newrun"].input_particle_definition = (
                self.start + ".astra"
            )
            self.section.astra_headers["newrun"].output_particle_definition = (
                self.objectname + ".astra"
            )
        # Create an "output" block
        if "output" not in self.file_block:
            self.file_block["output"] = {}
        output_settings = self.file_block["output"] | self.globalSettings["ASTRAsettings"]
        zstart = self.startObject.physical.start.z
        self.zstop = self.endObject.physical.end.z
        screens = [e for e in self.section.elements.elements.values() if e.hardware_class == "Diagnostic"]
        if "zstart" in output_settings:
            output_settings.pop("zstart")
        self.section.astra_headers["output"] = astra_output(
            starting_offset=self.starting_offset,
            starting_rotation=self.starting_rotation,
            global_parameters=self.global_parameters,
            zstart=zstart,
            zstop=self.zstop,
            zemit=int((self.zstop - zstart) / 0.01),
            screens=screens,
            **output_settings,
        )
        #
        # Create a "charge" block
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        if "charge" not in self.globalSettings:
            self.globalSettings["charge"] = {}
        space_charge_dict = self.file_block["charge"] | self.globalSettings["charge"]
        charge_settings = space_charge_dict | self.globalSettings["ASTRAsettings"]
        self.section.astra_headers["charge"] = astra_charge(
            global_parameters=self.global_parameters,
            **charge_settings,
        )
        #
        # Create an "error" block
        if "global_errors" not in self.file_block:
            self.file_block["global_errors"] = {}
        if "global_errors" not in self.globalSettings:
            self.globalSettings["global_errors"] = {}
        if "global_errors" in self.file_block or "global_errors" in self.globalSettings:
            globalerror = global_error(
                objectname=self.objectname + "_global_error",
                objecttype="global_error",
                global_parameters=self.global_parameters,
            )
            error_settings = self.file_block["global_errors"] | self.globalSettings["global_errors"]
            self.section.astra_headers["global_errors"] = astra_errors(
                element=globalerror,
                global_parameters=self.global_parameters,
                **error_settings,
            )
        self.astra_headers = self.section.astra_headers
        # print 'errors = ', self.file_block, self.headers['global_errors']

    @property
    def space_charge_mode(self) -> str:
        """
        The space charge type for ASTRA, i.e. "2D", "3D".

        Returns
        -------
        str
            The space charge type for ASTRA
        """
        return str(self.astra_headers["charge"].space_charge_mode)

    @space_charge_mode.setter
    def space_charge_mode(self, mode: str) -> None:
        """
        Sets the space charge mode for the &HEADER object

        Parameters
        ----------
        mode: str
            Space charge mode
        """
        self.astra_headers["charge"].space_charge_mode = str(mode)

    @property
    def sample_interval(self) -> int:
        """
        Factor by which to reduce the number of particles in the simulation, i.e. every 10th particle.

        Returns
        -------
        int
            The sampling interval `n_red` in ASTRA
        """
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, interval: int) -> None:
        """
        Sets the factor by which to reduce the number of particles in the simulation in the &NEWRUN header,
        and scales the number of space charge bins in the &CHARGE header accordingly;
        see :func:`~simba.Codes.ASTRA.ASTRA.astra_newrun.framework_dict`,
        :func:`~simba.Codes.ASTRA.ASTRA.astra_charge.grid_size`.

        Parameters
        ----------
        interval:
            Sampling interval
        """
        # print('Setting new ASTRA sample_interval = ', interval)
        self._sample_interval = interval
        self.astra_headers["newrun"].sample_interval = interval
        self.astra_headers["charge"].sample_interval = interval

    @property
    def bunch_charge(self) -> float:
        """
        Bunch charge in coulombs

        Returns
        -------
        float:
            Bunch charge
        """
        return self._bunch_charge

    @bunch_charge.setter
    def bunch_charge(self, charge: float) -> None:
        """
        Sets the bunch charge for this object and also in :class:`~simba.Codes.ASTRA.ASTRA.astra_newrun`.

        Parameters
        ----------
        charge: float
            Bunch charge in coulombs
        """
        # print('Setting new ASTRA sample_interval = ', interval)
        self._bunch_charge = charge
        self.astra_headers["newrun"].bunch_charge = charge

    @property
    def toffset(self) -> float:
        """
        Get the time offset for the reference particle.

        Returns
        -------
        float
            The time offset in seconds
        """
        return self._toffset

    @toffset.setter
    def toffset(self, toffset: float) -> None:
        """
        Set the time offset for this object and the :class:`~simba.Codes.ASTRA.ASTRA.astra_newrun` object.

        Parameters
        ----------
        toffset: float
            The time offset in seconds
        """
        # print('Setting new ASTRA sample_interval = ', interval)
        self._toffset = toffset
        self.astra_headers["newrun"].toffset = 1e9 * toffset

    def write(self) -> None:
        """
        Writes the ASTRA input file from :func:`~simba.Codes.ASTRA.ASTRA.astraLattice.writeElements`
        to <master_subdir>/<self.objectname>.in.
        """
        code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        self.section.astra_headers = self.astra_headers
        saveFile(code_file, self.section.to_astra())
        self.files.append(code_file)

    def preProcess(self) -> None:
        """
        Convert the beam file from the previous lattice section into ASTRA format and set the number of
        particles based on the input distribution, see
        :func:`~simba.Codes.ASTRA.ASTRA.astra_newrun.hdf5_to_astra`.
        """
        super().preProcess()
        prefix = self.get_prefix()
        astrabeamfilename = self.read_input_file(
            prefix,
            self.astra_headers["newrun"].input_particle_definition.replace(".astra", "")
        )
        self.ref_s = self.global_parameters["beam"].s if self.global_parameters["beam"].s is not None else 0
        self.astra_headers["newrun"].input_particle_definition = self.hdf5_to_astra()
        self.astra_headers["charge"].npart = len(self.global_parameters["beam"].x)

    @lox.thread
    def screen_threaded_function(
        self,
        objectname: str,
        scr: DiagnosticElement,
        cathode: bool,
        mult: int,
        sval: float = 0.0,
    ) -> None:
        """
        Convert output from ASTRA screen to HDF5 format

        Parameters
        ----------
        objectname: str
            Name of screen object
        scr: :class:`~laura.models.diagnostic.DiagnosticElement`
            Screen object
        cathode: bool
            True if beam was emitted from a cathode
        mult: int
            Multiplication factor for ASTRA-type filenames
        sval: float
            S-position of beam
        """
        return self.astra_to_hdf5(objectname, scr, cathode, mult, sval)

    def get_screen_scaling(self) -> int:
        """
        Determine the screen scaling factor for screens and BPMs

        Returns
        -------
        int
            The scaling factor depending on the `master_run_no` parameter
        """
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        for mult in [100, 1000, 10]:
            foundscreens = [
                self.find_ASTRA_filename(self.objectname, e, master_run_no, mult)
                for e in self.screens_and_bpms
            ]
            if all(foundscreens):
                return mult
        return 100

    def postProcess(self) -> None:
        """
        Convert the beam file(s) from the ASTRA output into HDF5 format, see
        :func:`~simba.Codes.ASTRA.ASTRA.astra_to_hdf5`.
        """
        super().postProcess()
        cathode = (
            self.astra_headers["newrun"].input_particle_definition == "initial_distribution"
        )
        mult = self.get_screen_scaling()
        svals = np.array(self.getSValues(at_entrance=False)) + self.ref_s
        zvals = [a[-1] for a in self.getZValues()]
        for e in self.screens_and_bpms:
            sval = np.interp(e.middle.z, zvals, svals)
            self.screen_threaded_function.scatter(
                scr=e,
                objectname=self.objectname,
                cathode=cathode,
                mult=mult,
                sval=sval,
            )
        self.screen_threaded_function.gather()
        endelem = PhysicalBaseElement(
            name=self.end,
            hardware_class="",
            hardware_type="",
            machine_area="",
            physical=PhysicalElement(middle=[0, 0, self.zstop])
        )
        self.astra_to_hdf5(lattice=self.objectname, scr=endelem, cathode=cathode, mult=mult, final=True)

    def astra_to_hdf5(
            self,
            lattice: str,
            scr: DiagnosticElement | PhysicalBaseElement,
            cathode: bool = False,
            mult: int = 100,
            final: bool = False,
            sval: float = 0.0,
    ) -> None:
        """
        Convert the ASTRA beam file name to HDF5 format and write the beam file.

        Parameters
        ----------
        lattice: str
            Lattice name
        scr: laura.models.diagnostic.DiagnosticElement
            LAURA DiagnosticElement
        cathode: bool
            True if beam was emitted from a cathode
        mult: int
            Multiplication factor for ASTRA-type filenames
        sval: float
            S-position of beam
        """
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        astrabeamfilename = self.find_ASTRA_filename(lattice, scr, master_run_no, mult)
        if astrabeamfilename is None:
            warn(f"Screen Error: {lattice}, {scr.physical.middle.z}, {astrabeamfilename}")
        else:
            beam = rbf.beam()
            rbf.astra.read_astra_beam_file(
                beam,
                (
                    os.path.join(
                        self.global_parameters["master_subdir"], astrabeamfilename
                    )
                ).strip('"'),
                normaliseZ=False,
            )
            rbf.hdf5.rotate_beamXZ(
                beam,
                -1 * self.starting_rotation[2],
                preOffset=[0, 0, 0],
                postOffset=-1 * np.array(self.starting_offset),
            )
            beam.s = UnitValue(sval, units="m")
            HDF5filename = scr.name + ".openpmd.hdf5"
            rbf.openpmd.write_openpmd_beam_file(
                beam,
                self.global_parameters["master_subdir"] + "/" + HDF5filename,
                )
            if self.global_parameters["delete_tracking_files"]:
                os.remove(
                    (
                        os.path.join(
                            self.global_parameters["master_subdir"], astrabeamfilename
                        )
                    ).strip('"')
                )
            if final:
                self.global_parameters["beam"] = beam

    def find_ASTRA_filename(
            self,
            lattice: str,
            scr: DiagnosticElement | PhysicalBaseElement,
            master_run_no: int,
            mult: int
    ) -> str | None:
        """
        Determine the ASTRA filename for the screen object.

        Parameters
        ----------
        lattice: str
            The name of the lattice
        scr: laura.models.diagnostic.DiagnosticElement
            LAURA DiagnosticElement
        master_run_no: int
            The run number
        mult: int
            Multiplication factor for ASTRA-type output
        zstart: float
            Start position of lattice

        Returns
        -------
        str or None
            The ASTRA filename for the screen object, or None if the file does not exist.
        """
        for i in [0, -0.001, 0.001]:
            tempfilename = (
                    lattice
                    + "."
                    + str(int(round((scr.physical.middle.z + i - self.startObject.physical.start.z) * mult))).zfill(4)
                    + "."
                    + str(master_run_no).zfill(3)
            )
            tempfilenamenozstart = (
                    lattice
                    + "."
                    + str(int(round((scr.physical.middle.z + i) * mult))).zfill(4)
                    + "."
                    + str(master_run_no).zfill(3)
            )
            tempfilenameend = (
                    lattice
                    + "."
                    + str(int(round((self.zstop + i - self.startObject.physical.start.z) * mult))).zfill(4)
                    + "."
                    + str(master_run_no).zfill(3)
            )
            tempfilenameendnozstart = (
                    lattice
                    + "."
                    + str(int(round((self.zstop + i) * mult))).zfill(4)
                    + "."
                    + str(master_run_no).zfill(3)
            )
            for f in [
                tempfilename,
                tempfilenameendnozstart,
                tempfilenameend,
                tempfilenamenozstart
            ]:
                if os.path.isfile(
                    os.path.join(self.global_parameters["master_subdir"], f)
                ):
                    return f
        return None

    def hdf5_to_astra(self) -> str:
        """
        Convert beam input file to ASTRA format and write to `master_subdir`.

        Returns
        -------
        str:
            Name of ASTRA beam file
        """
        astrabeamfilename = self.astra_headers["newrun"].output_particle_definition
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        return astrabeamfilename