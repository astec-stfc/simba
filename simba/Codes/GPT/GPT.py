"""
SIMBA GPT module: writes GPT input files, runs GPT and converts its output.

The GPT header commands (``GptSetFile``, ``GptTout``, ...) live in
:mod:`laura.translator.converters.codes.gpt`.
"""

import os
import re
import subprocess
import numpy as np
from laura.models.diagnostic import DiagnosticElement

from ...Framework_objects import frameworkLattice
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from ...Modules.constants import speed_of_light
from ...Modules.units import UnitValue
from ...Modules.gdf_beam import gdf_beam
from typing import Dict, Literal, Any
from laura.translator.converters.codes.gpt import (
    GptSetFile,
    GptAccuracy,
    GptSpaceCharge,
    GptTout,
    GptCsr1D,
    GptWriteFloorPlan,
)

gpt_defaults = {}


class gptLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` written as a GPT input file."""

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

    time_step_size: float = 5e-10
    """Interval between ``tout`` beam dumps [s]; GPT's own step is adaptive (:attr:`accuracy`)."""

    override_meanBz: float | int | None = None
    """Set the average particle longitudinal velocity manually"""

    override_tout: float | int | None = None
    """Set the time step output manually"""

    accuracy: int = 6
    """Tracking accuracy"""

    endScreenObject: Any = None
    """Final screen object for dumping particle distributions"""

    Brho: UnitValue | None = None
    """Magnetic rigidity"""

    particle_definition: str = None
    """Initial particle definition"""

    dtmin: float | None = None
    """Integration time step size"""

    crest_scan_particles: int = 10
    """Particles kept for :meth:`find_crest`; the crest is single-particle, so a handful will do."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition
        self.headers["setfile"] = GptSetFile(
            set='"beam"', filename='"' + self.name + '.gdf"'
        )
        self.headers["floorplan"] = GptWriteFloorPlan(
            filename='"' + self.objectname + '_floor.gdf"'
        )

    @property
    def space_charge_mode(self) -> str | None:
        """
        Space charge mode from :attr:`file_block`, else :attr:`globalSettings`.

        Returns
        -------
        str | None
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
        Set the space charge mode in :attr:`file_block`.

        Parameters
        ----------
        mode: Literal["2d", "3d", "2D", "3D"]
        """
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        self.file_block["charge"]["space_charge_mode"] = mode

    def writeElements(self) -> str:
        """
        Build the :attr:`headers` and render the section as GPT input.

        Returns
        -------
        str
            GPT input file text
        """
        self.headers["accuracy"] = GptAccuracy(accuracy=self.accuracy)
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        if "charge" not in self.globalSettings:
            self.globalSettings["charge"] = {}
        space_charge_dict = self.file_block["charge"] | self.globalSettings["charge"]
        space_charge = self.global_parameters | space_charge_dict
        self.headers["spacecharge"] = GptSpaceCharge(**space_charge)
        if self.particle_definition == "laser" and self.space_charge_mode is not None:
            self.headers["spacecharge"].npart = len(self.global_parameters["beam"].x)
        if (
            self.csr_enable
            and len(self.dipoles) > 0
            and max([abs(d.magnetic.KnL(0)) for d in self.dipoles]) > 0
        ):
            self.headers["csr1d"] = GptCsr1D()
        self.headers["setfile"].particle_definition = self.particle_definition
        self.section.gpt_headers = self.headers
        self.check_pass_rigidity(self.global_parameters["beam"].Brho)
        return self.section.to_gpt(
            startz=self.startObject.physical.start.z,
            endz=self.endObject.physical.end.z,
            Brho=self.global_parameters["beam"].Brho,
            dtmin=self.dtmin
        )

    def write(self) -> None:
        """Write :meth:`writeElements` to ``<master_subdir>/<objectname>.in``."""
        code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(code_file, self.writeElements())
        self.files.append(code_file)

    def _cavity_phase_variable(self, name: str, text: str) -> str:
        """
        Find the GPT phase variable of a cavity, by matching ``map1D_TM`` lines to its start z.

        Parameters
        ----------
        name: str
            Cavity name
        text: str
            Generated GPT input file

        Returns
        -------
        str
            e.g. ``phi119357``
        """
        element = self.elementObjects[name]
        zpos = float(element.physical.start.z)
        pattern = re.compile(
            r'map1D_TM\(\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*([-\d.eE+]+)\s*,'
            r'[^)]*?,\s*(phi\w+)\s*,'
        )
        matches = [(float(z), var) for z, var in pattern.findall(text)]
        if not matches:
            raise ValueError(f"no map1D_TM cavity found in the GPT input for {name}")
        z, var = min(matches, key=lambda m: abs(m[0] - zpos))
        if abs(z - zpos) > 1e-3:
            raise ValueError(
                f"no GPT cavity within 1 mm of {name} at z = {zpos}; "
                f"closest is {var} at z = {z}"
            )
        return var

    def find_crest(
        self,
        name: str,
        phase_range: tuple = (0.0, 350.0),
        step: float = 10.0,
        refine: bool = True,
    ) -> float:
        """
        Find a cavity's crest phase as the maximum ``avgG`` of a GPT ``mr`` phase scan.

        The converter writes ``phi = crest + 90 - phase``, so a peak at ``phi*`` is ``crest = phi* - 90``.

        Parameters
        ----------
        name: str
            Cavity name
        phase_range: tuple
            ``(from, to)`` of the coarse scan [deg]
        step: float
            Coarse scan step [deg]
        refine: bool
            Rescan one coarse step either side of the peak at a tenth of the step

        Returns
        -------
        float
            Crest phase [deg], also written back onto the element
        """
        self.write()
        subdir = self.global_parameters["master_subdir"]
        base = os.path.join(subdir, self.objectname)
        with open(base + ".in") as f:
            text = f.read()
        var = self._cavity_phase_variable(name, text)

        # The scanned symbol has to be *undefined* in the input file: mr supplies
        # it on the GPT command line, and a local assignment would shadow it.
        scan_text = re.sub(
            rf"^\s*{var}\s*=.*$", f"{var} = crestscan/deg;", text, flags=re.MULTILINE
        )
        if "crestscan" not in scan_text:
            raise ValueError(f"could not substitute the phase assignment for {var}")

        for collective in (r"spacecharge\w*", "wakefield", r"csr\w*", "Wakefield"):
            scan_text = re.sub(
                rf"^\s*{collective}\(.*$", "", scan_text, flags=re.MULTILINE
            )
        scan_text = re.sub(r"^\s*tout\(.*$", "", scan_text, flags=re.MULTILINE)
        scan_text = re.sub(
            r'^(\s*setfile\(.*)$',
            rf'\1\nsetreduce("beam",{self.crest_scan_particles});',
            scan_text, count=1, flags=re.MULTILINE,
        )

        def scan(lo, hi, dx):
            saveFile(base + "_crest.in", scan_text)
            saveFile(base + "_crest.mr", f"crestscan {lo} {hi} {dx}\n")
            gpt = self.executables[self.code]
            mr = [gpt[0].replace("gpt", "mr")]
            gdfa = [gpt[0].replace("gpt", "gdfa")]
            env = os.environ.copy()
            env["OMP_WAIT_POLICY"] = "PASSIVE"
            subprocess.call(
                mr + ["-o", self.objectname + "_crest_out.gdf",
                      self.objectname + "_crest.mr"]
                + gpt + [self.objectname + "_crest.in",
                         "GPTLICENSE=" + str(self.global_parameters["GPTLICENSE"])],
                cwd=subdir, env=env,
            )
            subprocess.call(
                gdfa + ["-o", self.objectname + "_crest_avg.gdf",
                        self.objectname + "_crest_out.gdf",
                        "crestscan", "avgG", "numpar"],
                cwd=subdir, env=env,
            )
            return self._read_crest_scan(base + "_crest_avg.gdf", z_eval)

        z_eval = float(self.elementObjects[name].physical.end.z)
        phases, energies = scan(phase_range[0], phase_range[1], step)
        peak = phases[int(np.argmax(energies))]
        if refine:
            fine_p, fine_e = scan(peak - step, peak + step, step / 10.0)
            if len(fine_e):
                peak = fine_p[int(np.argmax(fine_e))]

        element = self.elementObjects[name]
        crest = (float(peak) - 90.0) % 360.0
        element.crest = crest
        return crest

    def autophase(
        self,
        names: list | None = None,
        phase_range: tuple = (0.0, 350.0),
        step: float = 10.0,
        refine: bool = True,
    ) -> dict:
        """
        Phase every RF cavity in the section with :meth:`find_crest`, upstream to downstream.

        A slow beam's arrival time depends on upstream energy gain, so each cavity
        is scanned with those before it already on crest.

        Parameters
        ----------
        names: list or None
            Cavities to phase, in any order; None means every accelerating cavity in the section
        phase_range: tuple
            ``(from, to)`` of the coarse scan [deg]
        step: float
            Coarse scan step [deg]
        refine: bool
            Follow each coarse scan with a finer one around the peak

        Returns
        -------
        dict
            ``{name: crest}`` in phasing order
        """
        if names is None:
            # elementObjects spans the whole machine; keep accelerating cavities in
            # this section (an avgG scan says nothing about a deflector).
            z0 = float(self.startObject.physical.start.z)
            z1 = float(self.endObject.physical.end.z)
            names = []
            for n, e in self.elementObjects.items():
                if getattr(e, "crest", None) is None:
                    continue
                if not float(getattr(e, "field_amplitude", 0.0) or 0.0):
                    continue
                z = float(e.physical.start.z)
                if z0 - 1e-6 <= z <= z1 + 1e-6:
                    names.append(n)
        ordered = sorted(names, key=lambda n: float(self.elementObjects[n].physical.start.z))
        crests = {}
        for name in ordered:
            crests[name] = self.find_crest(
                name, phase_range=phase_range, step=step, refine=refine
            )
        return crests

    @staticmethod
    def _read_crest_scan(filename: str, z_eval: float | None = None) -> tuple:
        """
        Read ``(crestscan, avgG)`` arrays from a ``gdfa`` aggregate file.

        Parameters
        ----------
        filename: str
            ``gdfa`` aggregate file
        z_eval: float or None
            Use the screen at or just beyond this z; None means the furthest downstream
        """
        import easygdf

        data = easygdf.load(filename)
        found = []
        for block in data["blocks"]:
            children = {
                str(np.atleast_1d(c["name"])[0]): np.atleast_1d(c["value"])
                for c in block.get("children", [])
            }
            if "crestscan" in children and "avgG" in children:
                pos = float(np.atleast_1d(block.get("value", np.nan)).ravel()[0])
                npar = children.get("numpar")
                found.append((pos, children["crestscan"], children["avgG"], npar))
        if not found:
            return np.array([]), np.array([])
        if z_eval is None:
            pos, ph, en, npar = max(found, key=lambda f: f[0])
        else:
            downstream = [f for f in found if f[0] >= z_eval - 1e-6]
            pos, ph, en, npar = min(
                downstream or found, key=lambda f: abs(f[0] - z_eval)
            )
        if npar is not None and len(npar) == len(en):
            keep = npar >= np.median(npar)
            if keep.any():
                ph, en = ph[keep], en[keep]
        return ph, en

    def preProcess(self) -> None:
        """Convert the input beam to GDF via :meth:`hdf5_to_gdf`."""
        super().preProcess()
        self.headers["setfile"].particle_definition = self.objectname + ".gdf"
        prefix = self.get_prefix()
        self.hdf5_to_gdf(prefix)

    def run(self) -> None:
        """
        Run GPT, then ``gdfa`` for the averaged beam properties (``_emit.gdf``, ``_emitt.gdf``, ``traj.gdf``).

        Needs ``GPTLICENSE`` in :attr:`global_parameters`.
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
        """Write the beam at each screen and the end from the GPT output, via :meth:`gdf_to_hdf5`."""
        super().postProcess()
        cathode = self.particle_definition == "laser"
        svals = np.array(self.getSValues(at_entrance=False)) + self.entrance_s
        zvals = [a[-1] for a in self.getZValues()]
        gdfbeam = rbf.gdf.read_gdf_beam_file_object(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_out.gdf'
        )
        for e in self.screens_and_markers_and_bpms:
            if e.name != self.start:
                sval = np.interp(e.physical.middle.z, zvals, svals)
                self.gdf_to_hdf5(
                    gptbeamfilename=self.objectname + "_out.gdf",
                    screen=e,
                    cathode=cathode,
                    gdf=gdfbeam,
                    t0=self.headers["setfile"].time,
                    sval=sval,
                )
        sval = np.interp(self.endObject.physical.middle.z, zvals, svals)
        self.gdf_to_hdf5(
            gptbeamfilename=self.objectname + "_out.gdf",
            screen=self.endObject,
            cathode=cathode,
            gdf=gdfbeam,
            t0=self.headers["setfile"].time,
            sval=sval,
        )
        # every screen reads the one shared _out.gdf, so delete it once at the end
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                os.path.join(self.global_parameters["master_subdir"], self.objectname + "_out.gdf")
            )

    def hdf5_to_gdf(self, prefix: str="") -> None:
        """
        Load the input beam, write it as GDF and set the ``setfile``/``tout`` :attr:`headers`.

        Parameters
        ----------
        prefix: str
            HDF5 file prefix
        """
        self.load_input_beam(prefix, self.particle_definition)
        if self.particle_definition == "laser":
            self.global_parameters["beam"].z = UnitValue(0 * self.global_parameters["beam"].t, units="m")
        self.headers["setfile"].time = np.mean(self.global_parameters["beam"].t)
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
            self.headers["tout"] = GptTout(
                starttime=0, endpos=self.override_tout, step=str(self.time_step_size)
            )
        else:
            endpos = (
                    self.findS(self.endObject.name)[0][1]
                    - self.findS(self.startObject.name)[0][1]
            )
            self.headers["tout"] = GptTout(
                starttime=0,
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
            t0: float = 0.0,
            sval: float = 0.0,
    ) -> None:
        """
        Read the beam at a screen from GPT output and write it as openPMD.

        Parameters
        ----------
        screen: laura.models.diagnostic.DiagnosticElement
        gptbeamfilename: str
            GPT output file, relative to `master_subdir`
        cathode: bool
            Unused
        gdf: gdf_beam or None
            Already-loaded GDF file, to avoid re-reading it
        t0: float
            Time offset added to the beam [s]
        sval: float
            s of the screen [m]
        """
        beam = rbf.beam()
        rbf.gdf.read_gdf_beam_file(
            beam,
            os.path.join(self.global_parameters["master_subdir"], gptbeamfilename),
            position=screen.physical.middle.z,
            gdfbeam=gdf,
        )
        beam._beam.t = UnitValue(beam._beam.t.val + t0, units="s")
        beam._beam.s = UnitValue(sval, units="m")
        HDF5filename = self.output_basename(screen.name) + ".openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )
