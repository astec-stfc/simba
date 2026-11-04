"""
SIMBA Ptarmigan Module

A laser colliding with the beam: Compton scattering (linear and nonlinear), radiation
reaction and pair creation, by the strong-field QED Monte Carlo
`Ptarmigan <https://github.com/tgblackburn/ptarmigan>`_.

A Ptarmigan line holds one LAURA ``Laser``, the interaction point, at the laser's focus.
It does no transport, and needs none: there are no fields once the laser has gone, so the
particles it leaves with (the beam, and the photons and pairs it makes) are drifted in
straight lines to the end of the line.

Classes:
    - :class:`~simba.Codes.Ptarmigan.Ptarmigan.ptarmiganLattice`
"""

import os
from typing import ClassVar
from warnings import warn

import numpy as np
import yaml
from scipy.constants import epsilon_0

from ... import exceptions
from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules import constants

PTARMIGAN_POLARIZATIONS = {"linear": "linear", "circular": "circular", None: "linear"}
"""LAURA polarizations Ptarmigan has."""

PASSIVE_HARDWARE_CLASSES = frozenset({
    "Drift", "Marker", "Diagnostic", "Monitor", "Simulation", "Vacuum", "Laser", "TwissMatch",
    "Lighting", "Stage", "Feedback",
})
"""LAURA hardware classes that do nothing to the beam, so a straight-line drift through
them is exact. Anything else in a Ptarmigan line is warned about."""

CONTROL_KEYS = frozenset({
    "radiation_reaction", "pair_creation", "pol_resolved", "classical", "lcfa",
    "bandwidth_correction", "dt_multiplier", "increase_pair_rate_by", "rng_seed",
    "track_secondaries",
})
"""Ptarmigan ``control`` keys the ``tracking`` block may set."""

SIMBA_KEYS = frozenset({"a0", "timing_offset"})
"""``tracking`` keys simba itself uses: see :attr:`ptarmiganLattice.a0` and
:attr:`ptarmiganLattice.timing_offset`."""


def gaussian_a0(wavelength: float, pulse_energy: float, waist: float, fwhm: float) -> float:
    """
    a0 (by peak field, as Ptarmigan defines it, for linear polarization) of a Gaussian
    pulse: peak power E / (sigma_t sqrt(2 pi)) with sigma_t = fwhm / (2 sqrt(2 ln 2)),
    focused to a 1/e^2 radius ``waist``.
    """
    sigma_t = fwhm / (2 * np.sqrt(2 * np.log(2)))
    peak_power = pulse_energy / (sigma_t * np.sqrt(2 * np.pi))
    intensity = 2 * peak_power / (np.pi * waist**2)
    field = np.sqrt(2 * intensity / (epsilon_0 * constants.speed_of_light))
    omega = 2 * np.pi * constants.speed_of_light / wavelength
    return float(constants.elementary_charge * field / (constants.m_e * omega * constants.speed_of_light))


class ptarmiganLattice(frameworkLattice):
    """A line holding a laser that collides with the beam, run by Ptarmigan."""

    code: str = "ptarmigan"
    """String indicating the lattice object type"""

    supports_turns: ClassVar[bool] = False

    particle_definition: str | None = None
    """Input beam file stem"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition

    @property
    def laser(self):
        """The line's ``Laser``: the interaction point, at its focus."""
        lasers = [e for e in self.elements.values() if getattr(e, "hardware_type", None) == "Laser"]
        if len(lasers) != 1:
            raise ValueError(
                f"A Ptarmigan line holds one Laser, the interaction point; "
                f"'{self.objectname}' has {len(lasers)}"
            )
        return lasers[0]

    @property
    def interaction_z(self) -> float:
        """z of the interaction point: the laser element's middle [m]."""
        return float(self.laser.physical.middle.z)

    @property
    def control(self) -> dict:
        """
        Ptarmigan's ``control`` section, from the ``tracking`` block::

            files:
              IP:
                code: ptarmigan
                tracking: {radiation_reaction: true, pair_creation: true, classical: false}
        """
        tracking = self.file_block.get("tracking") or {}
        unknown = sorted(set(tracking) - CONTROL_KEYS - SIMBA_KEYS)
        if unknown:
            raise ValueError(f"Unknown Ptarmigan tracking keys {unknown}; one of {sorted(CONTROL_KEYS)}")
        return {key: value for key, value in tracking.items() if key in CONTROL_KEYS}

    @property
    def a0(self) -> float:
        """
        a0 from ``tracking: {a0: ...}``, else the laser's own ``a0``,
        both as Ptarmigan defines it (by peak field); else from its
        energy, waist and duration: :func:`gaussian_a0`, which is for linear
        polarization, over sqrt(2) for circular.
        """
        tracking = self.file_block.get("tracking") or {}
        if "a0" in tracking:
            return float(tracking["a0"])
        laser = self.laser.laser
        if getattr(laser, "a0", None):
            return float(laser.a0)
        missing = [k for k in ("wavelength", "pulse_energy", "waist", "pulse_duration_fwhm") if not getattr(laser, k, None)]
        if missing:
            raise ValueError(f"Laser {self.laser.name} needs {missing} (or tracking: {{a0: ...}}) for Ptarmigan")
        a0 = gaussian_a0(laser.wavelength, laser.pulse_energy, laser.waist, laser.pulse_duration_fwhm)
        if self.polarization == "circular":
            a0 /= np.sqrt(2)
        return float(a0)

    @property
    def polarization(self) -> str:
        """The laser's polarization as Ptarmigan names it (linear if unset)."""
        laser = self.laser.laser
        value = getattr(laser, "polarization", None)
        value = getattr(value, "value", value)
        if value not in PTARMIGAN_POLARIZATIONS:
            raise ValueError(f"Ptarmigan has linear or circular polarization, not {value}")
        return PTARMIGAN_POLARIZATIONS[value]

    @property
    def timing_offset(self) -> float:
        """
        How much later than the bunch centre the laser peak reaches its focus [s]: positive,
        the laser is late; 0, they meet at the focus. From the ``tracking`` block::

            tracking: {timing_offset: 1.0e-12}

        A head-on collision then happens ``c timing_offset / 2`` from the focus.
        """
        tracking = self.file_block.get("tracking") or {}
        return float(tracking.get("timing_offset", 0.0))

    @property
    def laser_offset(self) -> tuple:
        """
        Transverse misalignment of the laser focus off the beam axis, x and y [m]: the
        ``Laser`` element's ``physical.error.position``. Its longitudinal and rotation
        errors are not modelled (their meaning for a crossing laser is not one thing),
        with a warning.
        """
        error = getattr(self.laser.physical, "error", None)
        if error is None:
            return (0.0, 0.0)
        rotation = [getattr(error.rotation, k, 0.0) or 0.0 for k in ("theta", "phi", "psi")]
        if (error.position.z or 0.0) or any(rotation):
            warn(
                f"{self.objectname}: the laser's longitudinal and rotation errors are not "
                "modelled by Ptarmigan here, only its transverse (x, y) ones; set "
                "collision_angle for the crossing angle"
            )
        return (float(error.position.x or 0.0), float(error.position.y or 0.0))

    @property
    def input_file(self) -> str:
        return os.path.join(self.global_parameters["master_subdir"], f"{self.objectname}.ptarmigan.h5")

    def check_elements(self) -> None:
        """
        Warn of elements that act on the beam (magnets, cavities, collimators, plasmas...),
        which Ptarmigan, like every element but the laser, passes straight through.
        """
        active = [
            name for name, e in self.elements.items()
            if getattr(e, "hardware_class", None) not in PASSIVE_HARDWARE_CLASSES
        ]
        if active:
            warn(exceptions.UnmodelledElementsWarning(self.objectname, "Ptarmigan", active))

    def preProcess(self) -> None:
        """Load the input beam and write it for Ptarmigan, relative to the bunch's centre."""
        super().preProcess()
        self.check_elements()
        self.load_input_beam(self.get_prefix(), self.particle_definition)
        beam = self.global_parameters["beam"]
        self._laser_offset = self.laser_offset
        rbf.ptarmigan.write_ptarmigan_beam_file(
            beam, self.input_file, laser_offset=self._laser_offset, laser_delay=self.timing_offset
        )
        # where and when the bunch centre reaches the focus, to put the output back
        index = beam.reference_particle_index
        t = np.asarray(beam.t.val)
        z = np.asarray(beam.z.val)
        use = index if index is not None and 0 <= index < len(t) else None
        t0 = t[use] if use is not None else np.mean(t)
        z0 = z[use] if use is not None else np.mean(z)
        speed = np.mean(beam.Bz.val) * constants.speed_of_light
        # Ptarmigan's time zero: the laser peak at the focus, timing_offset after the bunch
        self._laser_peak_time = float(t0 + (self.interaction_z - z0) / speed) + self.timing_offset

    def write(self) -> None:
        """Write the Ptarmigan input file, ``<objectname>.yml``."""
        laser = self.laser.laser
        species = self.global_parameters["beam"].species
        config = {
            "control": self.control,
            "laser": {
                "a0": self.a0,
                "wavelength": float(laser.wavelength),
                "fwhm_duration": float(laser.pulse_duration_fwhm),
                "waist": float(laser.waist),
                "polarization": self.polarization,
                "envelope": "gaussian",
            },
            "beam": {
                "species": species,
                "collision_angle": float(getattr(laser, "collision_angle", 0.0) or 0.0),
                "from_hdf5": {
                    "file": os.path.basename(self.input_file),
                    "distance_bt_ips": 0.0,
                    "auto_timing": True,
                },
            },
            "output": {
                "ident": self.objectname,
                "dump_all_particles": "hdf5",
                "coordinate_system": "beam",
                "units": "auto",
            },
        }
        if not config["control"]:
            del config["control"]
        with open(os.path.join(self.global_parameters["master_subdir"], f"{self.objectname}.yml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def run(self) -> None:
        """Run Ptarmigan on ``<objectname>.yml``, logging to ``<objectname>.log``."""
        if self.remote_setup:
            return self.run_remote()
        subdir = self.global_parameters["master_subdir"]
        command = self.executables.build_command(
            self.executables[self.code] + [f"{self.objectname}.yml"], os.path.abspath(subdir)
        )
        self.run_command(command, os.path.join(subdir, f"{self.objectname}.log"), cwd=subdir)

    def postProcess(self) -> None:
        """
        Read Ptarmigan's final particles and drift them to every screen, marker and BPM
        downstream of the laser, and to the end of the line, writing each: every species
        in one file, the beam first. Those upstream of the laser are not written.
        """
        super().postProcess()
        subdir = self.global_parameters["master_subdir"]
        particles = rbf.ptarmigan.read_ptarmigan_output(
            os.path.join(subdir, f"{self.objectname}_particles.h5")
        )
        injected = self.global_parameters["beam"]
        s_values = self.getSValues(as_dict=True)
        planes = {
            e.name: float(e.physical.middle.z) for e in self.screens_and_markers_and_bpms
            if float(e.physical.middle.z) > self.interaction_z
        }
        planes[self.end] = float(self.endObject.physical.end.z)
        upstream = [e.name for e in self.screens_and_markers_and_bpms
                    if e.name not in planes and e.name != self.start]
        if upstream:
            warn(f"{self.objectname}: no beam is written at {upstream}, upstream of the laser")
        for name, z in sorted(planes.items(), key=lambda plane: plane[1]):  # the end last
            beam = rbf.beam()
            rbf.ptarmigan.ptarmigan_particles_to_beam(
                beam, particles, primary=injected.species,
                z_out=z - self.interaction_z, t_offset=self._laser_peak_time,
                laser_offset=self._laser_offset,
            )
            beam.set_z_from_t(z)
            beam.s = self.entrance_s + s_values[name]
            self.write_beam_file(beam, name)
        self.global_parameters["beam"] = beam
