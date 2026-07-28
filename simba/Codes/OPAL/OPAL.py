import os
from warnings import warn
import subprocess
import numpy as np
import yaml
from typing import Any, Dict, Literal

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

    generator: Any = None
    """The framework's beam generator, if any. Set by
    :func:`~simba.Framework.Framework.add_Generator` so that a section starting
    at the cathode can generate its own distribution inside OPAL rather than
    importing one produced by a different code (see :func:`~all_in_one`)."""

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
        self.section.opal_version = self.version
        output = self.section.to_opal(
            energy=self.global_parameters["beam"].centroids.mean_cpz.val / 1e6,
            breakstr=self.breakstr,
        )
        command_file = (
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(command_file, output, "w")
        self.files.append(command_file)

    @property
    def emitted(self) -> bool:
        """
        Whether the bunch is emitted from a cathode rather than started as a
        free-space distribution. Matches the condition used by
        :func:`~hdf5_to_opal`, which writes emission *times* into the
        longitudinal column of the OPAL distribution file when this is True.
        """
        return self.particle_definition == "laser"

    @property
    def all_in_one(self) -> bool:
        """
        Whether this section should generate its own bunch inside OPAL rather
        than importing a distribution produced by another code.

        True when the section starts at a cathode -- ``charge.cathode`` is set
        and the input is the generator's ``initial_distribution``. OPAL models
        generation and acceleration in a single run, and splitting the two loses
        information that no particle file carries: the emission time structure
        has to be reconstructed from the file, and OPAL's photocathode emission
        model never sees the generator's own pulse shape.
        """
        charge = self.file_block.get("charge", {}) or {}
        return bool(self.emitted and charge.get("cathode") and self.generator is not None)

    def native_distribution_block(self) -> str | None:
        """
        Render an OPAL ``DISTRIBUTION`` block from the framework's generator
        settings, so that OPAL generates the bunch at the cathode itself.

        The framework's generator is re-expressed as an
        :class:`~simba.Codes.Generators.opal.OPALGenerator` via its
        ``model_dump()``, so every generator setting carries over rather than only those a
        particle file happens to preserve.

        Returns
        -------
        str or None
            The rendered block, or None if it could not be built
        """
        if not self.all_in_one:
            return None
        from ..Generators.opal import OPALGenerator

        kwargs = self.generator.model_dump()
        kwargs["code"] = "opal"
        try:
            generator = OPALGenerator(**kwargs)
            return generator._write_distribution()
        except Exception as e:
            warn(
                f"Could not build a native OPAL distribution from the generator "
                f"settings ({type(e).__name__}: {e}); falling back to importing "
                f"the generated particle file."
            )
            return None

    def emission_settings(self) -> dict:
        """
        Cathode-emission settings for the OPAL ``DISTRIBUTION`` namelist.

        Returns an empty dict when the bunch is not emitted from a cathode. When
        it is, `EMITTED` must be declared: SIMBA writes emission times (of order
        1e-12 s) into the longitudinal column of the distribution file, and
        without this OPAL reads them as metres, collapsing the bunch to a point
        and producing a huge spurious space-charge kick.

        The model and step/bin counts come from `globals_Opal.yaml`, overridden
        by `settings["global"]`. Note that `TEMISSION` is deliberately not set:
        for a `FROMFILE` distribution the emission window is defined by the
        times in the file, and OPAL rejects the keyword ("Object DIST has no
        attribute TEMISSION").

        Returns
        -------
        dict
            Keyword arguments for :class:`~laura.translator.converters.codes.opal.opal_distribution`
        """
        charge = self.file_block.get("charge", {}) or {}
        mirror = charge.get("mirror_charge", False)
        if not self.emitted:
            if mirror:
                warn(
                    "mirror_charge is set but the bunch is not emitted from a "
                    "cathode, so OPAL will not apply image charges: its FFT "
                    "Poisson solver only adds the image charges at -z while a "
                    "bunch is being emitted. Set the section to start from the "
                    "cathode (particle_definition: initial_distribution) if the "
                    "image charge is wanted."
                )
            return {}
        opalglobal = update_globals(self.globalSettings)
        dist = opalglobal.get("distribution", {})
        settings = {"emitted": True}
        if "EMISSIONMODEL" in dist:
            settings["emission_model"] = dist["EMISSIONMODEL"]
        if "EMISSIONSTEPS" in dist:
            settings["emission_steps"] = int(dist["EMISSIONSTEPS"])
        if "NBIN" in dist:
            settings["n_bins"] = int(dist["NBIN"])
        return settings

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
        native = self.native_distribution_block()
        self.headers["distribution"] = opal_distribution(
            input_particle_definition=f"\"{initobj}.opal\"",
            raw_block=native,
            **({} if native else self.emission_settings()),
        )
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
        start_z = self.startObject.physical.start.z
        svals = {
            s.name: s.physical.middle.z - start_z for s in self.screens_and_bpms
        }
        opalbeamname = f'{self.global_parameters["master_subdir"]}/{self.objectname}.h5'
        spositions = find_opal_s_positions(opalbeamname, svals, tolerance=0.05)
        for elem in self.screens_and_bpms:
            if elem.name in spositions:
                beam = rbf.beam()
                beam.read_opal_beam_file(filename=opalbeamname, step=spositions[elem.name])
                zpos = elem.physical.middle.z
                beam._beam.z = UnitValue(beam._beam.z.val + zpos, "m")
                beam._beam.t = UnitValue(
                    beam._beam.t.val + (zpos / speed_of_light), "s"
                )
                rbf.openpmd.write_openpmd_beam_file(
                    beam,
                    f'{self.global_parameters["master_subdir"]}/{elem.name}.openpmd.hdf5',
                )
        beam = rbf.beam()
        beam.read_opal_beam_file(filename=opalbeamname, step=-1)
        zpos = self.endObject.physical.end.z
        beam._beam.z = UnitValue(beam._beam.z.val + zpos, "m")
        beam._beam.t = UnitValue(beam._beam.t.val + (zpos / speed_of_light), "s")
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
        emitted = self.emitted
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
