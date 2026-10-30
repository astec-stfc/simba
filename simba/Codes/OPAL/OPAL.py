import contextlib
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

from laura.translator.converters.codes.opal import (
    OpalOption,
    OpalDistribution,
    OpalFieldSolver,
    OpalBeam,
    OpalTrack,
    OpalRun,
)

from ...Modules.constants import speed_of_light, elementary_charge, m_e
from ...Modules.units import UnitValue

#: Magnetic rigidity of a particle with beta*gamma = 1 [T m], i.e. m_e*c/e.
BRHO_BETAGAMMA_1 = m_e * speed_of_light / elementary_charge


def _rms_emittance(u: np.ndarray, pu: np.ndarray) -> float:
    """
    RMS emittance of a single plane, with ``pu`` normalised to ``beta*gamma``.
    """
    u = u - u.mean()
    pu = pu - pu.mean()
    return np.sqrt(
        max((u * u).mean() * (pu * pu).mean() - (u * pu).mean() ** 2, 0.0)
    )


def canonical_emittances(filename: str, b_threshold: float = 1e-6) -> Dict[float, tuple]:
    """
    Transverse emittances of an OPAL particle dump from canonical momenta, keyed by s.

    OPAL uses mechanical momenta, which inside a solenoid inflate the emittance
    (often tenfold); every other code reports the canonical one.

    Parameters
    ----------
    filename: str
        OPAL particle dump (``<name>.h5``)
    b_threshold: float
        Field below which no correction is needed [T]

    Returns
    -------
    Dict[float, tuple]
        ``{s: (emit_x, emit_y)}`` for the steps that needed correcting
    """
    import h5py

    corrected = {}
    with h5py.File(filename, "r") as f:
        for key in f:
            if not key.startswith("Step#"):
                continue
            step = f[key]
            bz = float(np.atleast_1d(step.attrs["B-ref"])[-1])
            if abs(bz) < b_threshold:
                continue
            k = bz / (2 * BRHO_BETAGAMMA_1)
            x, y = step["x"][()], step["y"][()]
            spos = float(np.atleast_1d(step.attrs["SPOS"])[0])
            corrected[spos] = (
                _rms_emittance(x, step["px"][()] + k * y),
                _rms_emittance(y, step["py"][()] - k * x),
            )
    return corrected


def _dispersion(u: np.ndarray, delta: np.ndarray, var_delta: float) -> float:
    """
    Linear regression of ``u`` against ``delta``: ``<u*delta>/<delta**2>``.
    """
    return float(((u - u.mean()) * delta).mean() / var_delta)


def dispersions(filename: str, min_spread: float = 1e-6) -> Dict[float, tuple]:
    """
    Dispersion and its derivative at each step of an OPAL particle dump, keyed by s.

    Parameters
    ----------
    filename: str
        OPAL particle dump (``<name>.h5``)
    min_spread: float
        Relative momentum spread below which no dispersion is reported

    Returns
    -------
    Dict[float, tuple]
        ``{s: (Dx, Dxp, Dy, Dyp)}`` for the steps with enough spread to fit
    """
    import h5py

    disp = {}
    with h5py.File(filename, "r") as f:
        for key in f:
            if not key.startswith("Step#"):
                continue
            step = f[key]
            px, py, pz = step["px"][()], step["py"][()], step["pz"][()]
            if pz.size == 0 or pz.min() <= 0:
                continue
            p = np.sqrt(px**2 + py**2 + pz**2)
            p0 = p.mean()
            if p0 <= 0:
                continue
            delta = p / p0 - 1.0
            var_delta = (delta * delta).mean()
            if np.sqrt(var_delta) < min_spread:
                continue
            spos = float(np.atleast_1d(step.attrs["SPOS"])[0])
            disp[spos] = tuple(
                _dispersion(u, delta, var_delta)
                for u in (step["x"][()], px / pz, step["y"][()], py / pz)
            )
    return disp


def update_globals(global_settings, beamlen=None, sample_interval=1):
    grids = getGrids()
    with open(
            os.path.join(os.path.dirname(__file__), "globals_Opal.yaml")
    ) as file:
        opalglobal = yaml.load(file, Loader=yaml.Loader)
    for sc in ['x', 'y', 'z']:
        if f"SC_3D_N{sc}f" in global_settings:
            scconv = sc.upper().replace('Z', 'T')
            global_settings.update({f"M{scconv}": global_settings[f"SC_3D_N{sc}f"]})
    for typ, vals in opalglobal.items():
        for k in vals:
            if k in global_settings:
                opalglobal[typ][k] = global_settings[k]
    if beamlen:
        gridsize = grids.getGridSizes(beamlen / sample_interval)
        opalglobal["fieldsolver"].update({"MX": gridsize, "MY": gridsize, "MT": gridsize})
    return opalglobal

class opalLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` written as an OPAL input file."""

    code: str = "opal"
    """String indicating the lattice object type"""

    headers: Dict = {}
    """Headers to be included in the OPAL lattice file"""

    particle_definition: str = None
    """Name of initial particle distribution"""

    time_step_size: float | list | tuple = 2e-12
    """Tracking step [s]; a sequence stages it along the line at :attr:`time_step_boundaries`
    (e.g. a fine step only near the cathode)."""

    time_step_boundaries: list | tuple | None = None
    """z [m] from the section start where :attr:`time_step_size` moves to its next value;
    one fewer entry than ``time_step_size``."""

    breakstr: str = "//----------------------------------------------------------------------------"
    """String used for separating headers in the input file"""

    version: str = "202210"
    """Version of OPAL"""

    maxsteps: int = 1000000
    """Maximum number of tracking steps"""

    space_charge_grid: int | tuple[int, int, int] | list | None = None
    """Space-charge mesh: one size for all three axes, or ``(MX, MY, MT)``; None sizes
    it from the particle count. OPAL requires ``npart >= MX*MY*MT``."""

    bbox_increase: float | None = None
    """OPAL's ``BBOXINCR``: % enlargement of the space-charge box; None means OPAL's 2%."""

    force_all_in_one: bool | None = None
    """Override :attr:`all_in_one`: True generates the bunch in OPAL, False imports a file."""

    generator: Any = None
    """The framework's beam generator, set by the framework; used by :attr:`all_in_one`."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition


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
        """Whether the bunch is emitted from a cathode; :meth:`hdf5_to_opal` then writes emission times."""
        return self.particle_definition == "laser"

    def option_settings(self) -> dict:
        """
        OPAL ``OPTION`` settings: ``globals_Opal.yaml``'s ``global`` block, overridden by global settings.

        Returns
        -------
        dict
            Keyword arguments for :class:`~laura.translator.converters.codes.opal.OpalOption`
        """
        opalglobal = update_globals(self.globalSettings)
        settings = dict(opalglobal.get("global", {}) or {})
        allowed = OpalOption.model_fields
        dropped = [k for k in settings if k not in allowed]
        for k in dropped:
            warn(f"OPAL OPTION has no attribute {k}; ignoring")
            settings.pop(k)
        return settings

    @property
    def all_in_one(self) -> bool:
        """
        Whether OPAL generates the bunch itself rather than importing a particle file.

        By default, true for an emitted beam with ``charge.cathode`` and a generator:
        a particle file loses the emission time structure. :attr:`force_all_in_one` overrides.
        """
        auto = bool(
            self.emitted
            and (self.file_block.get("charge", {}) or {}).get("cathode")
            and self.generator is not None
        )
        if self.force_all_in_one is None:
            return auto
        if self.force_all_in_one and self.generator is None:
            warn(
                "force_all_in_one=True but no generator is attached; falling "
                "back to importing a distribution file."
            )
            return False
        return self.force_all_in_one

    def native_distribution_block(self) -> str | None:
        """
        OPAL ``DISTRIBUTION`` block from the generator, via
        :class:`~simba.Codes.Generators.opal.OPALGenerator`, if :attr:`all_in_one`.

        Returns
        -------
        str or None
            None if not :attr:`all_in_one` or it could not be built
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
        Cathode-emission settings for the OPAL ``DISTRIBUTION`` namelist; empty if not :attr:`emitted`.

        ``EMITTED`` is required, else OPAL reads the file's emission times as metres.
        ``TEMISSION`` is left out: OPAL rejects it for ``FROMFILE``.

        Returns
        -------
        dict
            Keyword arguments for :class:`~laura.translator.converters.codes.opal.OpalDistribution`
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
        self.load_input_beam(prefix, self.particle_definition)
        self.hdf5_to_opal()
        beamlen = len(self.global_parameters["beam"].x)
        pc = np.mean(self.global_parameters["beam"].cpz.val) / 1e9
        bcurrent = abs(self.global_parameters["beam"].total_charge * 1e6)
        chargesign = int(self.global_parameters["beam"].chargesign[0])
        # the file hdf5_to_opal wrote
        initobj = self.particle_definition
        self.headers["option"] = OpalOption(**self.option_settings())
        native = self.native_distribution_block()
        self.headers["distribution"] = OpalDistribution(
            input_particle_definition=f"\"{initobj}.opal\"",
            raw_block=native,
            **({} if native else self.emission_settings()),
        )
        self.headers["fieldsolver"] = OpalFieldSolver(
            npart=beamlen,
            space_charge_mode=str(self.space_charge_mode),
            grid_size_override=self.space_charge_grid,
            BBOXINCR=self.bbox_increase,
        )
        self.headers["beam"] = OpalBeam(
            PC=pc,
            NPART=beamlen,
            CHARGE=chargesign,
            PARTICLE=self.global_parameters["beam"].species.upper(),
            BCURRENT=bcurrent,
        )
        bounds = None
        if isinstance(self.time_step_size, (list, tuple)):
            bounds = list(self.time_step_boundaries or [])
            if len(bounds) != len(self.time_step_size) - 1:
                raise ValueError(
                    f"time_step_boundaries needs {len(self.time_step_size) - 1} "
                    f"entries for {len(self.time_step_size)} time steps, got {len(bounds)}"
                )
        self.headers["track"] = OpalTrack(
            DT=self.time_step_size,
            MAXSTEPS=self.maxsteps,
            LINE=self.objectname,
            ZSTOP=self.endObject.physical.end.z - self.startObject.physical.start.z,
            ZSTOP_STAGES=bounds,
        )
        self.headers["run"] = OpalRun()
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
                    f'{self.global_parameters["master_subdir"]}/'
                    f'{self.output_basename(elem.name)}.openpmd.hdf5',
                )
        beam = rbf.beam()
        beam.read_opal_beam_file(filename=opalbeamname, step=-1)
        zpos = self.endObject.physical.end.z
        beam._beam.z = UnitValue(beam._beam.z.val + zpos, "m")
        beam._beam.t = UnitValue(beam._beam.t.val + (zpos / speed_of_light), "s")
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            f'{self.global_parameters["master_subdir"]}/'
            f'{self.output_basename(self.endObject.name)}.openpmd.hdf5',
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
        # OPAL's own emittances use mechanical momenta, so replace them with
        # canonical ones wherever a solenoid is on -- see canonical_emittances.
        svals_stat = np.asarray(opalData["s"], dtype=float)
        corrected = canonical_emittances(opalbeamname)
        if corrected:
            for spos, (ex, ey) in corrected.items():
                idx = int(np.argmin(np.abs(svals_stat - spos)))
                if abs(svals_stat[idx] - spos) < 1e-6:
                    opalData["emit_x"][idx] = ex
                    opalData["emit_y"][idx] = ey
        DISPERSION_COLUMNS = ("Dx", "Dxp", "Dy", "Dyp")
        for name in DISPERSION_COLUMNS:
            opalData[name] = np.zeros(len(svals_stat))
        for spos, values in dispersions(opalbeamname).items():
            idx = int(np.argmin(np.abs(svals_stat - spos)))
            if abs(svals_stat[idx] - spos) < 1e-6:
                for name, value in zip(DISPERSION_COLUMNS, values):
                    opalData[name][idx] = value
        opalData["s"] += self.entrance_s
        import h5py
        with h5py.File(f"{self.global_parameters['master_subdir']}/{self.objectname}.opal_twiss.h5", "w") as f:
            for k, v in opalData.items():
                with contextlib.suppress(TypeError):
                    f.create_dataset(k, data=np.array(v))

    def hdf5_to_opal(self):
        emitted = self.emitted
        rbf.opal.write_opal_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + self.particle_definition + '.opal',
            subz=self.startObject.physical.start.z,
            emitted=emitted,
        )

    def run(self):
        """Run OPAL on ``<objectname>.in``, locally or remotely."""
        if self.remote_setup:
            self.run_remote()
        else:
            workdir = os.path.abspath(self.global_parameters["master_subdir"])
            command_list = self.executables.build_command(
                self.executables[self.code] + [self.objectname + ".in"], workdir
            )
            command = "bash -c '" + " ".join(command_list) + "'"
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
