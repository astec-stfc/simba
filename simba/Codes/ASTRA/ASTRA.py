"""
SIMBA ASTRA module: writes ASTRA input files, runs ASTRA and converts its output.
See `ASTRA manual`_.

The namelist headers (``AstraNewRun``, ``AstraCharge``, ...) live in
:mod:`laura.translator.converters.codes.astra`.

    .. _ASTRA manual: https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf
"""

import os
from copy import deepcopy
from warnings import warn
import numpy as np
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import ClassVar, Dict, Any
from pydantic import Field, ConfigDict

from ...Framework_objects import frameworkLattice, global_error
from ...FrameworkHelperFunctions import expand_substitution, saveFile
from ...Modules import Beams as rbf
from laura.models.diagnostic import DiagnosticElement
from laura.models.element import Screen
from laura.models.physical import PhysicalElement, Position
from laura.translator.converters.codes.astra import (
    AstraNewRun,
    AstraCharge,
    AstraOutput,
    AstraErrors,
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
    """A :class:`~simba.Framework_objects.frameworkLattice` written as an ASTRA input file."""

    model_config = ConfigDict(validate_assignment=True)

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Threaded conversion of ASTRA screen outputs to openPMD"""

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

    zstep: float = 0.01
    """Tracking step size [m]"""

    astra_headers: Dict[str, Any] = Field(default_factory=dict)
    """Headers for ASTRA input file"""

    local_frame: bool | None = None
    """Write the deck in the lattice's own frame (see :meth:`to_local`) rather than world coordinates."""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.starting_offset = (
            eval(expand_substitution(self, self.file_block["starting_offset"]))
            if "starting_offset" in self.file_block
            else [0, 0, 0]
        )

        self.starting_rotation = (
            [0.0, 0.0, float(-1 * self.startObject.physical.global_rotation.theta)]
        )
        self.starting_rotation = (
            eval(expand_substitution(self, str(self.file_block["starting_rotation"])))
            if "starting_rotation" in self.file_block
            else self.starting_rotation
        )

        if "local_frame" in self.file_block:
            self.local_frame = bool(self.file_block["local_frame"])
        elif self.local_frame is None:
            self.local_frame = abs(self.starting_rotation[2]) > 1e-9

        # Create a "newrun" block
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "ASTRAsettings" not in self.globalSettings:
            self.globalSettings["ASTRAsettings"] = {}
        newrun_settings = self.file_block["input"] | self.globalSettings["ASTRAsettings"]
        settings = deepcopy(newrun_settings)
        if "twiss" in settings:
            settings.pop("twiss")
        self.section.astra_headers["newrun"] = AstraNewRun(
            global_parameters=self.global_parameters,
            input_particle_definition = self.startObject.name,
            **settings,
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
        zstart = self.to_local(self.startObject.physical.start)[2]
        self.zstop = self.to_local(self.endObject.physical.end)[2]
        screens = [e for e in self.deck_section.elements.elements.values() if e.hardware_class == "Diagnostic"]
        if "zstart" in output_settings:
            output_settings.pop("zstart")
        self.section.astra_headers["output"] = AstraOutput(
            global_parameters=self.global_parameters,
            zstart=zstart,
            zstop=self.zstop,
            zemit=int((self.zstop - zstart) / self.zstep),
            screens=screens,
            **output_settings,
        )
        # Create a "charge" block
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        if "charge" not in self.globalSettings:
            self.globalSettings["charge"] = {}
        space_charge_dict = self.file_block["charge"] | self.globalSettings["charge"]
        charge_settings = space_charge_dict | self.globalSettings["ASTRAsettings"]
        self.section.astra_headers["charge"] = AstraCharge(
            global_parameters=self.global_parameters,
            **charge_settings,
        )
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
            self.section.astra_headers["global_errors"] = AstraErrors(
                element=globalerror,
                global_parameters=self.global_parameters,
                **error_settings,
            )
        self.astra_headers = self.section.astra_headers

    @property
    def space_charge_mode(self) -> str:
        """
        Space charge mode of the &CHARGE header, e.g. "2D", "3D".

        Returns
        -------
        str
        """
        return str(self.astra_headers["charge"].space_charge_mode)

    @space_charge_mode.setter
    def space_charge_mode(self, mode: str) -> None:
        """
        Set the space charge mode of the &CHARGE header.

        Parameters
        ----------
        mode: str
        """
        self.astra_headers["charge"].space_charge_mode = str(mode)

    @property
    def bunch_charge(self) -> float:
        """
        Bunch charge [C].

        Returns
        -------
        float
        """
        return self._bunch_charge

    @bunch_charge.setter
    def bunch_charge(self, charge: float) -> None:
        """
        Set the bunch charge here and in the &NEWRUN header.

        Parameters
        ----------
        charge: float
            Bunch charge [C]
        """
        self._bunch_charge = charge
        self.astra_headers["newrun"].bunch_charge = charge

    @property
    def toffset(self) -> float:
        """
        Time offset of the reference particle [s].

        Returns
        -------
        float
        """
        return self._toffset

    @toffset.setter
    def toffset(self, toffset: float) -> None:
        """
        Set the time offset here and in the &NEWRUN header (which takes ns).

        Parameters
        ----------
        toffset: float
            Time offset [s]
        """
        self._toffset = toffset
        self.astra_headers["newrun"].toffset = 1e9 * toffset

    def to_local(self, point) -> np.ndarray:
        """
        Map a world position into the deck's frame: origin at the first element's entrance, along +z.

        Parameters
        ----------
        point: Position | Sequence[float]
            LAURA ``Position`` or ``(x, y, z)``

        Returns
        -------
        np.ndarray
            ``(x, y, z)`` in the deck's frame; unchanged if not :attr:`local_frame`
        """
        p = np.asarray(getattr(point, "array", point), dtype=float)
        if not self.local_frame:
            return p
        physical = self.startObject.physical
        origin = np.asarray(physical.start.array, dtype=float)
        local = physical.rotation_matrix.T @ (p - origin)
        return np.round(local, 9)

    @property
    def deck_section(self):
        """
        The section as written out; with :attr:`local_frame`, a deep copy moved into that frame.

        Returns
        -------
        SectionLatticeTranslator
        """
        if not self.local_frame:
            return self.section
        section = self.section.model_copy(deep=True)
        for element in section.elements.elements.values():
            physical = getattr(element, "physical", None)
            if physical is None or physical.middle is None:
                continue
            x, y, z = self.to_local(physical.middle)
            physical.middle = Position(x=x, y=y, z=z)
            physical.global_rotation.theta += self.starting_rotation[2]
        return section

    def write(self) -> None:
        """Write :attr:`deck_section` as ASTRA input to ``<master_subdir>/<objectname>.in``."""
        code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        section = self.deck_section
        section.directory = self.global_parameters["master_subdir"]
        section.astra_headers = self.astra_headers
        saveFile(code_file, section.to_astra())
        self.files.append(code_file)

    def preProcess(self) -> None:
        """Convert the input beam via :meth:`hdf5_to_astra` and set the particle count."""
        super().preProcess()
        prefix = self.get_prefix()
        self.load_input_beam(
            prefix,
            self.astra_headers["newrun"].input_particle_definition.replace(".astra", "")
        )
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
        Threaded :meth:`astra_to_hdf5` for one screen.

        Parameters
        ----------
        objectname: str
            Lattice name
        scr: :class:`~laura.models.diagnostic.DiagnosticElement`
            Screen
        cathode: bool
            Unused by :meth:`astra_to_hdf5`
        mult: int
            Position multiplier in ASTRA output filenames
        sval: float
            s of the screen [m]
        """
        return self.astra_to_hdf5(
            lattice=objectname, scr=scr, cathode=cathode, mult=mult, sval=sval
        )

    def get_screen_scaling(self) -> int:
        """
        Find the position multiplier (100, 1000 or 10) ASTRA used in the screen filenames.

        Returns
        -------
        int
            100 if no multiplier matches every screen
        """
        master_run_no = self.global_parameters.get("run_no", 1)
        for mult in [100, 1000, 10]:
            foundscreens = [
                self.find_ASTRA_filename(self.objectname, e, master_run_no, mult)
                for e in self.screens_and_bpms
            ]
            if all(foundscreens):
                return mult
        return 100

    def postProcess(self) -> None:
        """Convert the ASTRA screen and final beams to openPMD via :meth:`astra_to_hdf5`."""
        super().postProcess()
        cathode = self.input_particle_definition == "laser"
        mult = self.get_screen_scaling()
        offset = self.s_offset
        self.write_s_offset()
        for e in self.screens_and_bpms:
            self.screen_threaded_function.scatter(
                scr=e,
                objectname=self.objectname,
                cathode=cathode,
                mult=mult,
                sval=offset + self.to_local(e.middle)[2],
            )
        self.screen_threaded_function.gather()
        endelem = Screen(
            name=self.end,
            hardware_class="Diagnostic",
            hardware_type="",
            machine_area="",
            physical=PhysicalElement(middle=self.endObject.physical.end.array),
        )
        self.astra_to_hdf5(
            lattice=self.objectname,
            scr=endelem,
            cathode=cathode,
            mult=mult,
            final=True,
            sval=offset + self.zstop,
        )

    @property
    def s_offset(self) -> float:
        """
        s minus z at the start of this lattice: path length gained upstream by bending.

        Returns
        -------
        float
        """
        return float(
            self.entrance_s - self.to_local(self.startObject.physical.start)[2]
        )

    def write_s_offset(self) -> str:
        """
        Write :attr:`s_offset` next to the ASTRA output files.

        ASTRA's emit files record z only, so :func:`~simba.Modules.Twiss.astra.read_s_offset`
        needs this to place them in s.

        Returns
        -------
        str
            Path of the offset file
        """
        path = os.path.join(
            self.global_parameters["master_subdir"], self.objectname + ".s_offset"
        )
        with open(path, "w") as f:
            f.write(repr(self.s_offset))
        return path

    def astra_to_hdf5(
            self,
            lattice: str,
            scr: DiagnosticElement | Screen,
            cathode: bool = False,
            mult: int = 100,
            final: bool = False,
            sval: float = 0.0,
    ) -> None:
        """
        Read the ASTRA beam at a screen and write it as openPMD.

        Parameters
        ----------
        lattice: str
            Lattice name
        scr: laura.models.diagnostic.DiagnosticElement | Screen
            Screen
        cathode: bool
            Unused
        mult: int
            Position multiplier in ASTRA output filenames
        final: bool
            Also make this the beam in ``global_parameters``
        sval: float
            s of the screen [m]
        """
        master_run_no = self.global_parameters.get("run_no", 1)
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
            if not self.local_frame:
                rbf.hdf5.rotate_beamXZ(
                    beam,
                    -1 * self.starting_rotation[2],
                    preOffset=[0, 0, 0],
                    postOffset=-1 * np.array(self.starting_offset),
                )

            beam.Particles.s = UnitValue(sval, units="m")
            HDF5filename = self.output_basename(scr.name) + ".openpmd.hdf5"
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
            scr: DiagnosticElement | Screen,
            master_run_no: int,
            mult: int
    ) -> str | None:
        """
        Find the ASTRA output file for a screen.

        Parameters
        ----------
        lattice: str
            Lattice name
        scr: laura.models.diagnostic.DiagnosticElement | Screen
            Screen
        master_run_no: int
            Run number
        mult: int
            Position multiplier to try first

        Returns
        -------
        str or None
            None if no matching file exists
        """
        # ASTRA names outputs in cm, but in mm for sections shorter than 1 m, and a
        # lattice without screens gives no `mult`, so try the other conventions too.
        scr_z = self.to_local(scr.physical.middle)[2]
        start_z = self.to_local(self.startObject.physical.start)[2]
        for m in dict.fromkeys([mult, 1000, 100, 10]):
            for i in [0, -0.001, 0.001]:
                tempfilename = (
                        lattice
                        + "."
                        + str(int(round((scr_z + i - start_z) * m))).zfill(4)
                        + "."
                        + str(master_run_no).zfill(3)
                )
                tempfilenamenozstart = (
                        lattice
                        + "."
                        + str(int(round((scr_z + i) * m))).zfill(4)
                        + "."
                        + str(master_run_no).zfill(3)
                )
                tempfilenameend = (
                        lattice
                        + "."
                        + str(int(round((self.zstop + i - start_z) * m))).zfill(4)
                        + "."
                        + str(master_run_no).zfill(3)
                )
                tempfilenameendnozstart = (
                        lattice
                        + "."
                        + str(int(round((self.zstop + i) * m))).zfill(4)
                        + "."
                        + str(master_run_no).zfill(3)
                )
                # the screen's own position first (relative, then absolute); the
                # end-of-lattice names are a last resort, otherwise every screen that
                # misses on the relative name silently picks up the final distribution
                for f in [
                    tempfilename,
                    tempfilenamenozstart,
                    tempfilenameendnozstart,
                    tempfilenameend,
                ]:
                    if os.path.isfile(
                        os.path.join(self.global_parameters["master_subdir"], f)
                    ):
                        return f
        return None

    def hdf5_to_astra(self) -> str:
        """
        Write the input beam in ASTRA format to `master_subdir`.

        Returns
        -------
        str
            ASTRA beam filename
        """
        astrabeamfilename = self.astra_headers["newrun"].output_particle_definition
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        return astrabeamfilename