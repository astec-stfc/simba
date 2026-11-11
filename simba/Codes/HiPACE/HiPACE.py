"""
SIMBA HiPACE++ module: one HiPACE++ run per line, of its plasma stage or plasma lens.
See the `HiPACE++ docs`_.

The stage's physics comes from LAURA's
:meth:`~laura.translator.converters.plasma.PlasmaTranslator.to_hipace`; the incoming
beam, run control and anything in :attr:`~hipaceLattice.inputs` are added here.

    .. _HiPACE++ docs: https://hipace.readthedocs.io
"""

import glob
import os
from math import ceil
from typing import Dict

import h5py
from scipy.constants import c

from laura.translator.utils.hipace import hipace_inputs

from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules.Beams.fbpic import read_fbpic_beam_file
from ...Modules.Beams.hipace import write_hipace_beam_file


class hipaceLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` run as one HiPACE++ simulation."""

    code: str = "hipace"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    allow_negative_drifts: bool = True
    """Allow drifts to be of negative length (could be necessary for plasma injection)"""

    particle_definition: str | None = None
    """Name of the input particle distribution. ``input: {particle_definition: null}``
    in the settings file means there is no incoming beam (a laser-driven stage)."""

    bunch_z_position: float = 0
    """Position in the HiPACE++ box, in metres, at which
    :attr:`~simba.Framework_objects.frameworkLattice.plasma_box_origin` -- the
    incoming reference particle, or else the centroid -- is placed."""

    include_ions: bool = False
    """Model the background ions as mobile protons"""

    inputs: Dict = {
        "boundary.field": "Dirichlet",
        "boundary.particle": "Absorbing",
        "diagnostic.dimensions": "xz",
    }
    """HiPACE++ inputs applied over the stage's own, e.g. ``{"hipace.verbose": 1}``;
    ``HIPACEsettings: {inputs: {...}}`` adds to them"""

    _zstart: float = 0.0
    _t0: float = 0.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.inputs = dict(self.inputs)
        for k, v in (self.settings["global"].get("HIPACEsettings") or {}).items():
            if isinstance(v, Dict):
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)
        stated = self.file_block.get("input") or {}
        if "particle_definition" in stated and stated["particle_definition"] is None:
            self.particle_definition = None
        else:
            self.particle_definition = self.input_particle_definition

    @property
    def subdir(self) -> str:
        return self.global_parameters["master_subdir"]

    @property
    def zstart(self) -> float:
        """Lattice z of the box's z = 0 at :attr:`t0`."""
        return self._zstart

    @property
    def t0(self) -> float:
        """
        HiPACE++ time at the start: the stage's density is a function of ``c*t``
        from its entrance, so a beam already past it starts later.
        """
        return self._t0

    def preProcess(self) -> None:
        """Read the input beam and fix :attr:`zstart` and :attr:`t0`."""
        super().preProcess()
        if self.particle_definition is not None:
            prefix = self.get_prefix()
            prefix = prefix if self.trackBeam else prefix + self.particle_definition
            self.load_input_beam(prefix, self.particle_definition)
            self._zstart = self.plasma_box_origin - self.bunch_z_position
        else:
            self._zstart = float(self.startObject.physical.start.z)

    def write(self) -> None:
        """
        Write ``<name>.in`` and, with an incoming beam, ``<name>_beam.h5`` to the
        master subdirectory.

        Raises
        ------
        ValueError
            If the line does not hold exactly one plasma stage or plasma lens, or
            the beam starts past its end
        """
        stages = self.section.to_hipace(include_ions=self.include_ions, directory=self.subdir)
        if len(stages) != 1:
            raise ValueError(
                f"{self.objectname}: a HiPACE++ line holds one plasma stage or plasma "
                f"lens; it has {len(stages) or 'none'} ({', '.join(stages)})."
            )
        (stage, params), = stages.items()
        entrance = float(self.elementObjects[stage].physical.start.z)
        self._t0 = (self._zstart + self.bunch_z_position - entrance) / c
        if self._t0 >= params["hipace.max_time"]:
            raise ValueError(f"{self.objectname}: the beam starts past the end of {stage}.")
        if params["hipace.dt"] != "adaptive":
            params["max_step"] = max(
                1, ceil((params["hipace.max_time"] - self._t0) / params["hipace.dt"])
            )
        params |= {"hipace.initial_time": self._t0, "hipace.output_folder": f"{self.objectname}_diags"}
        if self.particle_definition is None:
            params["beams.names"] = "no_beam"
        else:
            beam_file = f"{self.objectname}_beam.h5"
            charge, mass = write_hipace_beam_file(
                self.global_parameters["beam"], os.path.join(self.subdir, beam_file), self._zstart
            )
            params |= {
                "beams.names": "beam",
                "beam.injection_type": "from_file",
                "beam.input_file": beam_file,
                "beam.charge": charge,
                "beam.mass": mass,
            }
        params |= self.inputs
        with open(os.path.join(self.subdir, f"{self.objectname}.in"), "w") as f:
            f.write(hipace_inputs(params))

    def run(self) -> None:
        """Run HiPACE++ on ``<name>.in`` in the master subdirectory."""
        workdir = os.path.abspath(self.subdir)
        command = self.executables.build_command(
            self.executables[self.code] + [f"{self.objectname}.in"], workdir
        )
        self.run_command(command, os.path.join(workdir, f"{self.objectname}.log"), cwd=workdir)

    def final_diagnostic(self) -> str | None:
        """The last openPMD dump HiPACE++ wrote, or None."""
        dumps = sorted(glob.glob(os.path.join(self.subdir, f"{self.objectname}_diags", "hdf5", "*.h5")))
        return dumps[-1] if dumps else None

    def postProcess(self) -> None:
        """Read the beam from the last dump and write it as openPMD for the next line."""
        super().postProcess()
        if self.particle_definition is None:
            return  # no beam out of a laser-only stage; HiPACE++ cannot trap plasma electrons
        dump = self.final_diagnostic()
        if dump is None:
            raise RuntimeError(f"{self.objectname}: HiPACE++ wrote no diagnostics.")
        with h5py.File(dump) as f:
            time = float(next(iter(f["data"].values())).attrs["time"])
        beam = self.global_parameters["beam"]
        read_fbpic_beam_file(beam, dump, z_offset=self._zstart + c * (time - self._t0), species="beam")
        beam.code = "hipace"
        rbf.openpmd.write_openpmd_beam_file(
            beam, os.path.join(self.subdir, f"{self.output_basename(self.end)}.openpmd.hdf5")
        )
