"""OPAL beam generator."""
import os
import subprocess
import numpy as np
from typing import Any, Dict
from warnings import warn

from ...Modules import constants
from ...FrameworkHelperFunctions import saveFile
from ...Codes.OPAL.OPAL import update_globals
from .Generators import (
    frameworkGenerator,
    aliases,
    opal_generator_keywords,
)
from ...Modules import Beams as rbf


class OPALGenerator(frameworkGenerator):
    """
    Generates a cathode beam by emission in OPAL.
    """

    opalglobal: Dict = {}
    """Global settings for OPAL"""

    breakstr: str = "//----------------------------------------------------------------------------"
    """Section separator in the input file."""

    MIN_PARTICLES_PER_EMISSION_STEP: int = 64
    """Fewest particles an emission step may emit before
    :meth:`capped_emission_steps` reduces the step count."""

    MIN_EMISSION_STEPS: int = 50
    """Floor on the emission step count, so a small bunch still resolves the
    shape of the laser pulse."""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.apply_alias_and_multiplier(aliases, "opal")
        self.code = "opal"
        self.opalglobal = update_globals({}, beamlen=self.particles)

    def run(self) -> None:
        """
        Run OPAL on the input file in `master_subdir`.
        """
        command = self.executables[self.code] + [self.objectname + ".in"]
        workdir = os.path.abspath(self.global_parameters["master_subdir"])
        command = self.executables.build_command(command, workdir)
        with open(os.devnull, "w") as f:
            subprocess.call(
                command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    @property
    def initial_gamma(self) -> float:
        """
        Initial Lorentz factor from :attr:`thermal_kinetic_energy`.
        """
        if self.species not in list(aliases["aliases"]["opal"].keys()):
            raise NotImplementedError(f"{self.species} is not current implemented for OPAL")
        ke_therm = self.thermal_kinetic_energy
        mass_eV = (
            self.particle_mass * (constants.speed_of_light**2) / constants.elementary_charge
        )
        return (ke_therm + mass_eV) / mass_eV

    def _get_bunch_length(self) -> float:
        if self.distribution_type_z in ["p", "plateau", "flattop"]:
            return sum([self.plateau_bunch_length + self.plateau_fall_time + self.plateau_rise_time])
        else:
            return self.sigma_t * self.gaussian_cutoff_z

    def _get_elemedge(self) -> float:
        if self.distribution_type_z in ["p", "plateau", "flattop"]:
            return self._get_bunch_length() * 1e6
        else:
            return (self.sigma_t * self.gaussian_cutoff_z) * 1e6

    def capped_emission_steps(self, requested: int) -> int:
        """
        Lower `requested` emission steps so each emits a useful number of particles.

        Too few particles per step make a slice too thin for OPAL's space-charge
        solver, which then adds spurious emittance at the cathode.

        Parameters
        ----------
        requested: int
            Configured number of emission steps

        Returns
        -------
        int
            Number of emission steps to use
        """
        ceiling = max(
            int(self.particles // self.MIN_PARTICLES_PER_EMISSION_STEP),
            self.MIN_EMISSION_STEPS,
        )
        if requested <= ceiling:
            return int(requested)
        warn(
            f"emission_steps={requested} would emit only "
            f"{self.particles / requested:.1f} particles per step for "
            f"{self.particles} particles; capping at {ceiling}. Raise the "
            f"particle count to use finer emission stepping."
        )
        return ceiling

    def _write_distribution(self) -> str:
        """
        OPAL ``DISTRIBUTION`` command from the allowed attributes, under their OPAL aliases.
        """
        output = "//DISTRIBUTION\n"
        output += "DIST: DISTRIBUTION"
        dist_dict = {}
        if self.distribution_type_z in ["p", "flattop"]:
            dist_dict.update({"TYPE": "FLATTOP"})
            if not self.plateau_bunch_length > 0:
                raise ValueError("plateau_bunch_length must be defined for flattop longitudinal distribution")
            rise_time = self.plateau_rise_time
            fall_time = self.plateau_fall_time or rise_time
            dist_dict.update({aliases["aliases"]["opal"]["plateau_rise_time"]["alias"]: rise_time})
            dist_dict.update({aliases["aliases"]["opal"]["plateau_fall_time"]["alias"]: fall_time})
            sigma_rise = rise_time / 1.6869
            sigma_fall = fall_time / 1.6869
            tpulsefwhm = self.plateau_bunch_length + np.sqrt(2 * np.log(2)) * (
                sigma_rise + sigma_fall
            )
            dist_dict.update({aliases["aliases"]["opal"]["plateau_bunch_length"]["alias"]: tpulsefwhm})
        else:
            if not self.sigma_t > 0:
                raise ValueError("sigma_t must be defined for flattop longitudinal distribution")
            dist_dict.update({"TYPE": "GAUSS"})
            dist_dict.update({aliases["aliases"]["opal"]["sigma_t"]["alias"]: self.sigma_t})
        for k, v in self.__dict__.items():
            disallowed = opal_generator_keywords["disallowed"]
            if k not in disallowed:
                dist = True
                if k in list(aliases["aliases"]["opal"].keys()):
                    dist = aliases["aliases"]["opal"][k]["type"] == "distribution"
                    k = aliases["aliases"]["opal"][k]["alias"]
                if (getattr(self, k) is not None) and dist and (k.lower() != "type"):
                    dist_dict.update({k: v})
        if dist_dict.get("TYPE") == "FLATTOP":
            for attr in ("sigma_x", "sigma_y"):
                key = aliases["aliases"]["opal"][attr]["alias"]
                if dist_dict.get(key):
                    dist_dict[key] = 2 * dist_dict[key]
        key = aliases["aliases"]["opal"]["emission_steps"]["alias"]
        if dist_dict.get(key):
            dist_dict[key] = self.capped_emission_steps(dist_dict[key])
        allowed = {a.upper() for a in opal_generator_keywords.get("allowed", [])}
        for k, v in dist_dict.items():
            if allowed and k.upper() not in allowed:
                continue
            if k.upper() == "EKIN" and self.emission_model == "ASTRA":
                continue
            output += f",\n\t {k.upper()} = {v}"
        if self.emission_model == "ASTRA":
            output += f",\n\t EKIN = {self.thermal_kinetic_energy}"
        if self.emission_model == "NONEQUIL":
            for key in opal_generator_keywords["NONEQUIL"]:
                if not hasattr(self, key):
                    raise KeyError(f"Generator does not have {key} attribute required for NONEQUIL emission")
                else:
                    if getattr(self, key) < 0:
                        raise ValueError(f"{key} not defined correctly, required for NONEQUIL emission")
        output += ";\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_globals(self):
        """
        OPAL global ``REAL`` variables.
        """
        output = "//GLOBAL PARAMETERS\n"
        output += f"REAL rf_freq = {float(self.bfreq)};\n"
        output += f"REAL n_particles = {int(self.particles)};\n"
        output += f"REAL beam_bunch_charge = {float(self.charge) * 1e6};\n"
        output += f"REAL GAMMA = {self.initial_gamma};\n"
        output += (
            f"REAL MINSTEPFORREBIN = {self.opalglobal['global']['MINSTEPFORREBIN']};\n"
        )
        output += (
            f"REAL MINBINEMITTED = {self.opalglobal['global']['MINBINEMITTED']};\n"
        )
        output += f"{self.breakstr}\n"
        return output

    def _write_options(self):
        """
        OPAL ``OPTION`` commands from :attr:`opalglobal`.
        """
        output = "//OPTIONS\n"
        for name, val in self.opalglobal["option"].items():
            output += f"OPTION, {name} = {val};\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_line(self):
        """
        OPAL line holding just the emission monitor.
        """
        output = "//EMISSION MONITOR\n"
        output += f"MONI: MONITOR, OUTFN=\"MONI\", TYPE=TEMPORAL, ELEMEDGE={str(self._get_elemedge())};\n"
        output += "EMISSION: LINE = (MONI);\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_field_solver(self):
        """
        OPAL ``FIELDSOLVER`` command, of type NONE: no space charge here.
        """
        output = "//FIELD SOLVER\n"
        output += "FS: FIELDSOLVER "
        self.opalglobal["fieldsolver"].update({"FSTYPE": "NONE"})
        self.opalglobal["fieldsolver"].update({"MX": 1, "MY": 1, "MT": 1})
        for name, val in self.opalglobal["fieldsolver"].items():
            output += f",\n\t {name} = {val}"
        output += ";\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_beam(self):
        """
        OPAL ``BEAM`` command.
        """
        if self.species not in list(aliases["aliases"]["opal"].keys()):
            raise NotImplementedError(f"{self.species} is not currently implemented for OPAL")
        output = "//BEAM\n"
        output += "BEAM1: BEAM,\n"
        output += f"\tPARTICLE = {aliases['aliases']['opal'][self.species]['alias']},\n"
        output += "\tGAMMA = GAMMA,\n"
        output += "\tNPART = n_particles,\n"
        output += "\tBFREQ = 1,\n"
        output += "\tBCURRENT = beam_bunch_charge,\n"
        output += f"\tCHARGE = {int(self.charge_sign)};\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_track(self):
        """
        OPAL ``TRACK`` command, stopping just after emission (it cannot stop at z=0).
        """
        output = "//TRACK\n"
        output += "TRACK, \n"
        output += "\tLINE = EMISSION,\n"
        output += "\tBEAM = BEAM1,\n"
        output += "\tMAXSTEPS = 100000,\n"
        output += "\tDT = {" + str(self.tstep) + "},\n"
        output += "\tZSTOP = {" + str(self._get_bunch_length()*constants.speed_of_light*0.1) + "};\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_run(self):
        """
        OPAL ``RUN`` command and the end of the file.
        """
        output = "//RUN\n"
        output += "RUN, \n"
        output += '\tMETHOD = "PARALLEL-T",\n'
        output += "\tBEAM = BEAM1,\n"
        output += "\tFIELDSOLVER = FS,\n"
        output += "\tDISTRIBUTION = DIST;\n"
        output += "ENDTRACK;\n"
        output += "QUIT;\n"
        return output

    def write(self):
        """
        Write the OPAL input file to `master_subdir`.
        """
        self.apply_alias_and_multiplier(aliases, "opal")
        output = ""
        output += f"{self.breakstr}\n"
        output += self._write_options()
        output += self._write_globals()
        output += self._write_line()
        output += self._write_distribution()
        output += self._write_field_solver()
        output += self._write_beam()
        output += self._write_track()
        output += self._write_run()
        saveFile(
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in",
            output,
        )

    def postProcess(self):
        """
        Convert OPAL's ``MONI.h5`` to ``laser.hdf5`` (names hardcoded).
        """
        opalbeamfilename = "MONI.h5"
        rbf.opal.read_opal_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + opalbeamfilename,
            step=0,
        )
        self.global_parameters["beam"].z = [0 for _ in range(len(self.global_parameters["beam"].x))]
        HDF5filename = "laser.hdf5"
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=opalbeamfilename,
        )
