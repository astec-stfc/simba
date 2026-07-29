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
    """Step size for output data during tracking"""

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
    """Particles retained during a crest scan (:func:`~find_crest`). The crest is
    a single-particle property of the RF, so a handful is enough."""

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
            dtmin=self.dtmin
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

    def _cavity_phase_variable(self, name: str, text: str) -> str:
        """
        Find the GPT variable holding the RF phase of a named cavity.

        The converter names a cavity's variables after its position with the
        decimal point stripped. The ``map1D_TM`` lines are parsed and matched
        against the element's own start position.

        Parameters
        ----------
        name: str
            Name of the cavity element.
        text: str
            Contents of the generated GPT input file.

        Returns
        -------
        str
            The name of the phase variable, e.g. ``phi119357``.
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
        Find the crest phase of an RF cavity by scanning it with GPT's ``mr``.
        ``mr`` runs GPT once per phase and concatenates the output, and ``gdfa``
        reduces each run to an average Lorentz factor, so the crest is simply the
        phase of maximum ``avgG``.

        The converter writes ``phi = (crest + 90 - phase)``, so running on crest
        means ``phase = 0`` and therefore ``phi = crest + 90``. A scan peaking at
        ``phi*`` gives ``crest = phi* - 90``.

        Parameters
        ----------
        name: str
            Name of the cavity element to phase.
        phase_range: tuple
            ``(from, to)`` of the coarse scan in degrees.
        step: float
            Coarse scan step in degrees.
        refine: bool
            Follow the coarse scan with a finer one, one coarse step either side
            of the peak, to a tenth of the step.

        Returns
        -------
        float
            The measured crest phase in degrees, also written back onto the
            element.
        """
        self.write()
        subdir = self.global_parameters["master_subdir"]
        base = os.path.join(subdir, self.objectname)
        text = open(base + ".in").read()
        var = self._cavity_phase_variable(name, text)

        # The scanned symbol has to be *undefined* in the input file: mr supplies
        # it on the GPT command line, and a local assignment would shadow it.
        scan_text = re.sub(
            rf"^\s*{var}\s*=.*$", f"{var} = crestscan/deg;", text, flags=re.MULTILINE
        )
        if "crestscan" not in scan_text:
            raise ValueError(f"could not substitute the phase assignment for {var}")

        # The crest is a single-particle property of the RF, which is how ASTRA
        # and OPAL phase internally.
        # Collective effects have no place in a single-particle crest scan, and
        # GPT refuses to run a wakefield on a reduced bunch ("Bunch length cannot
        # be zero"), so strip them along with the space charge.
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

        # Read the energy just past this cavity, not at the end of the line: with
        # downstream cavities still present the final energy depends on their
        # phases too, which would confound the scan.
        z_eval = float(self.elementObjects[name].physical.end.z)
        phases, energies = scan(phase_range[0], phase_range[1], step)
        peak = phases[int(np.argmax(energies))]
        if refine:
            fine_p, fine_e = scan(peak - step, peak + step, step / 10.0)
            if len(fine_e):
                peak = fine_p[int(np.argmax(fine_e))]

        element = self.elementObjects[name]
        # The converter writes phi = (crest + 90 - phase), so running on crest
        # means phase = 0, i.e. phi = crest + 90. A scan peaking at phi* therefore
        # gives crest = phi* - 90 directly; it does not depend on the crest the
        # element happened to be carrying beforehand.
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
        Phase every RF cavity in the section, upstream to downstream.

        The order matters and the cavities cannot be done independently. While
        the beam is still slow, the time it arrives at a cavity depends on how
        much energy it gained in the ones before it, so the crest of the second
        cavity is only meaningful once the first is already on crest. Scanning
        them in isolation, or in the wrong order, measures the crest of a beam
        that will never exist.

        Each cavity is therefore scanned with every upstream cavity already set
        to its measured crest -- which happens naturally, because
        :func:`~find_crest` writes the result back onto the element and the input
        file is regenerated for the next scan.

        Parameters
        ----------
        names: list or None
            Cavities to phase, in any order; they are sorted by position. When
            None, every cavity in the section is phased.
        phase_range: tuple
            ``(from, to)`` of the coarse scan in degrees.
        step: float
            Coarse scan step in degrees.
        refine: bool
            Follow each coarse scan with a finer one around the peak.

        Returns
        -------
        dict
            ``{name: crest}`` in the order the cavities were phased.
        """
        if names is None:
            # elementObjects spans the whole machine, so restrict to accelerating
            # cavities that actually sit inside this section. Deflecting cavities
            # are excluded: they are not run on crest and their energy gain is
            # nominally zero, so an avgG scan says nothing about them.
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
        Pull the scanned phase and average Lorentz factor out of a ``gdfa``
        aggregate, which nests them one level below the screen position.

        Parameters
        ----------
        filename: str
            The ``gdfa`` aggregate file.
        z_eval: float or None
            Preferred evaluation position; the screen at or just beyond it is
            used. When None the furthest downstream screen is taken.
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
        # Drop phases that lose particles. avgG averages over survivors, so
        # losing the low-energy tail *raises* it -- off-crest phases where the
        # beam is being scraped produce spurious maxima higher than the real
        # crest, and which particles survive varies from run to run.
        if npar is not None and len(npar) == len(en):
            # Compare against the median rather than the maximum: the count is
            # occasionally one *above* the nominal at a single phase, and keying
            # off the maximum would then discard every other point and let one
            # sample decide the crest.
            keep = npar >= np.median(npar)
            if keep.any():
                ph, en = ph[keep], en[keep]
        return ph, en

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
        svals = np.array(self.getSValues(at_entrance=False)) + self.startObject.physical.start.z
        zvals = [a[-1] for a in self.getZValues()]
        gdfbeam = rbf.gdf.read_gdf_beam_file_object(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_out.gdf'
        )
        for e in self.screens_and_markers_and_bpms:
            if not e.name == self.start:
                sval = np.interp(e.physical.middle.z, zvals, svals)
                self.gdf_to_hdf5(
                    gptbeamfilename=self.objectname + "_out.gdf",
                    screen=e,
                    cathode=cathode,
                    gdf=gdfbeam,
                    t0=self.headers["setfile"].time,
                    sval=sval,
                )
            # else:
            # print('Ignoring', self.ignore_start_screen.objectname)
        sval = np.interp(self.endObject.physical.middle.z, zvals, svals)
        self.gdf_to_hdf5(
            gptbeamfilename=self.objectname + "_out.gdf",
            screen=self.endObject,
            cathode=cathode,
            gdf=gdfbeam,
            t0=self.headers["setfile"].time,
            sval=sval,
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
            endpos = (
                    self.findS(self.endObject.name)[0][1]
                    - self.findS(self.startObject.name)[0][1]
            )
            self.headers["tout"] = gpt_tout(
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
        Convert the GDF beam file to HDF5 format and write the beam file.

        Parameters
        ----------
        screen: laura.models.diagnostic.DiagnosticElement
            Diagnostic element
        gptbeamfilename: str
            Name of GPT beam file
        cathode: bool
            True if beam was emitted from a cathode
        gdf: gdfbeam or None
            GDF beam object
        t0: float
            Initial time co-ordinate
        sval: float
            S-position of screen
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
        # The beam just read is the local one; gptLattice has no `beam` attribute,
        # and it is the local object that gets written out below.
        beam._beam.t = UnitValue(beam._beam.t.val + t0, units="s")
        beam._beam.s = UnitValue(sval, units="m")
        HDF5filename = screen.name + ".openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
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
