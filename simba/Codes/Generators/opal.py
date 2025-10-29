"""
This file generates an OPAL beam file from the provided parameters.

It includes methods to run the OPAL generator, write the input file, and post-process the generated beam data.
"""
import os
import subprocess
from typing import Any, Dict

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
    A class to generate an OPAL beam file from the provided parameters.

    :param executables: Dictionary containing the paths to the executables.
    :param global_parameters: Dictionary containing global parameters for the simulation.
    :param generator_keywords: Dictionary containing keywords for the generator.
    :param kwargs: Additional keyword arguments for the generator.
    :ivar filename: Name of the output file.
    :ivar code: Code identifier for the generator.
    """

    opalglobal: Dict = {}
    """Global settings for OPAL"""

    breakstr: str = "//----------------------------------------------------------------------------"
    """String to indicate a new section of the generator txt file"""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.apply_alias_and_multiplier(aliases, "opal")
        self.code = "opal"
        self.opalglobal = update_globals({}, beamlen=self.particles)

    def run(self) -> None:
        """
        Run the OPAL generator to create the beam file.
        This method constructs the command to run the OPAL executable with the input file
        and executes it in the specified working directory.
        :return: None
        """
        command = self.executables[self.code] + [self.objectname + ".in"]
        with open(os.devnull, "w") as f:
            subprocess.call(
                command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    @property
    def initial_gamma(self) -> float:
        """
        Calculate the initial Lorentz factor (gamma) of the particles based on their thermal kinetic energy.

        :return: The initial Lorentz factor (gamma) of the particles.
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

    def _write_distribution(self) -> str:
        """
        Write the OPAL distribution input file with the parameters defined in the class.
        Global opal parameters are used to generate the input file base on the `globals_opal.yaml` file.

        Base attributes of :class:`~simba.Codes.Generators.frameworkGenerator`
        are used to generate the input file, with the appropriate aliases and multipliers applied for the OPAL code.

        This method constructs the distribution parameters based on the attributes of the class,
        including the type of distribution, thermal kinetic energy, and other relevant parameters.

        :return: A string representation of the OPAL distribution input parameters.
        """
        output = "//DISTRIBUTION\n"
        output += f"DIST: DISTRIBUTION"
        dist_dict = {}
        if self.distribution_type_z in ["p", "flattop"]:
            dist_dict.update({"TYPE": "FLATTOP"})
            if not getattr(self, "plateau_bunch_length") > 0:
                raise ValueError(f"plateau_bunch_length must be defined for flattop longitudinal distribution")
            dist_dict.update({aliases["aliases"]["opal"]["plateau_bunch_length"]["alias"]: self.plateau_bunch_length})
            dist_dict.update({aliases["aliases"]["opal"]["plateau_rise_time"]["alias"]: self.plateau_rise_time})
            dist_dict.update({aliases["aliases"]["opal"]["plateau_fall_time"]["alias"]: self.plateau_fall_time})
        else:
            if not getattr(self, "sigma_t") > 0:
                raise ValueError(f"sigma_t must be defined for flattop longitudinal distribution")
            dist_dict.update({"TYPE": "GAUSS"})
            dist_dict.update({aliases["aliases"]["opal"]["sigma_t"]["alias"]: self.sigma_t})
        for k, v in self.__dict__.items():
            disallowed = opal_generator_keywords["disallowed"]
            if k not in disallowed:
                dist = True
                if k in list(aliases["aliases"]["opal"].keys()):
                    dist = (
                        True
                        if aliases["aliases"]["opal"][k]["type"] == "distribution"
                        else False
                    )
                    k = aliases["aliases"]["opal"][k]["alias"]
                if (getattr(self, k) is not None) and dist and (k.lower() != "type"):
                    dist_dict.update({k: v})
        for k, v in dist_dict.items():
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
        Write the global parameters for the OPAL input file with the parameters defined in the class.
        Global opal parameters are used to generate the input file base on the `globals_opal.yaml` file.

        :return: A string representation of the OPAL global parameters input.
        """
        output = "//GLOBAL PARAMETERS\n"
        output += f"REAL rf_freq = {float(self.bfreq)};\n"
        output += f"REAL n_particles = {int(self.particles)};\n"
        output += f"REAL beam_bunch_charge = {float(self.charge) * 1e6};\n"
        output += f"REAL GAMMA = {self.initial_gamma};\n"
        output += (
            f"REAL MINSTEPFOREBIN = {self.opalglobal['global']['MINSTEPFOREBIN']};\n"
        )
        output += (
            f"REAL MINBINEMITTED = {self.opalglobal['global']['MINBINEMITTED']};\n"
        )
        output += f"{self.breakstr}\n"
        return output

    def _write_options(self):
        """
        Write the options for the OPAL input file with the parameters defined in the class.
        Global opal parameters are used to generate the input file base on the `globals_opal.yaml` file.

        :return: A string representation of the OPAL options input.
        """
        output = "//OPTIONS\n"
        for name, val in self.opalglobal["option"].items():
            output += f"OPTION, {name} = {val};\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_line(self):
        """
        Write the lattice line for the OPAL input file with the parameters defined in the class (a simple monitor).

        :return: A string representation of the OPAL line input.
        """
        output = "//EMISSION MONITOR\n"
        output += f"MONI: MONITOR, OUTFN=\"MONI\", TYPE=TEMPORAL, ELEMEDGE={str(self._get_elemedge())};\n"
        output += f"EMISSION: LINE = (MONI);\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_field_solver(self):
        """
        Write the field solver parameters for the OPAL input file with the parameters defined in the class.
        Global opal parameters are used to generate the input file base on the `globals_opal.yaml` file.

        Space charge is not used in this case, so the field solver type is set to NONE.

        :return: A string representation of the OPAL field solver input.
        """
        output = "//FIELD SOLVER\n"
        output += f"FS: FIELDSOLVER "
        self.opalglobal["fieldsolver"].update({"FSTYPE": "NONE"})
        self.opalglobal["fieldsolver"].update({"MX": 1, "MY": 1, "MT": 1})
        for name, val in self.opalglobal["fieldsolver"].items():
            output += f",\n\t {name} = {val}"
        output += ";\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_beam(self):
        """
        Write the beam parameters for the OPAL input file with the parameters defined in the class.

        The beam is defined with the particle type, gamma, number of particles, bunch frequency,
        bunch charge, and charge sign.
        :return: A string representation of the OPAL beam input.
        """
        if self.species not in list(aliases["aliases"]["opal"].keys()):
            raise NotImplementedError(f"{self.species} is not currently implemented for OPAL")
        output = "//BEAM\n"
        output += f"BEAM1: BEAM,\n"
        output += f"\tPARTICLE = {aliases['aliases']['opal'][self.species]['alias']},\n"
        output += f"\tGAMMA = GAMMA,\n"
        output += f"\tNPART = n_particles,\n"
        output += f"\tBFREQ = 1,\n"
        output += f"\tBCURRENT = beam_bunch_charge,\n"
        output += f"\tCHARGE = {int(self.charge_sign)};\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_track(self):
        """
        Write the track command for the OPAL input file with the parameters defined in the class.
        By default we stop a short time after particle generation (cannot do it at z=0).

        :return: A string representation of the OPAL track command.
        """
        output = "//TRACK\n"
        output += f"TRACK, \n"
        output += f"\tLINE = EMISSION,\n"
        output += f"\tBEAM = BEAM1,\n"
        output += f"\tMAXSTEPS = 100000,\n"
        output += "\tDT = {" + str(self.tstep) + "},\n"
        output += "\tZSTOP = {" + str(self._get_bunch_length()*constants.speed_of_light*0.1) + "};\n"
        output += f"{self.breakstr}\n"
        return output

    def _write_run(self):
        """
        Write the run command for the OPAL input file with the parameters defined in the class.
        This uses properties defined earlier in the file.

        :return: A string representation of the OPAL track command.
        """
        output = "//RUN\n"
        output += f"RUN, \n"
        output += f'\tMETHOD = "PARALLEL-T",\n'
        output += f"\tBEAM = BEAM1,\n"
        output += f"\tFIELDSOLVER = FS,\n"
        output += f"\tDISTRIBUTION = DIST;\n"
        output += f"ENDTRACK;\n"
        output += f"QUIT;\n"
        return output

    def write(self):
        """
        Write the OPAL commands to an input file.

        :return: None
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
        Convert the output from OPAL to standard HDF5 format.

        #TODO filename is hardcoded and it shouldn't be.
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
