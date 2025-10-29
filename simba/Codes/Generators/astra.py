"""
This file generates an ASTRA beam file from the provided parameters.

It includes methods to run the ASTRA generator, write the input file, and post-process the generated beam data.
"""
import os
import subprocess
from ...FrameworkHelperFunctions import saveFile
from .Generators import (
    frameworkGenerator,
    aliases,
    astra_generator_keywords,
)
from typing import Dict, Any, List
from ...Modules import Beams as rbf


class ASTRAGenerator(frameworkGenerator):
    """
    A class to generate an ASTRA beam file from the provided parameters.

    :param executables: Dictionary containing the paths to the executables.
    :param global_parameters: Dictionary containing global parameters for the simulation.
    :param generator_keywords: Dictionary containing keywords for the generator.
    :param kwargs: Additional keyword arguments for the generator.
    :ivar objectname: Name of the object to be generated.
    :ivar filename: Name of the output file.
    :ivar thermal_kinetic_energy: Thermal kinetic energy of the particles.
    :ivar aliases: Aliases for the parameters.
    :ivar code: Code identifier for the generator.
    :ivar apply_alias_and_multiplier: Method to apply aliases and multipliers to the parameters.
    """
    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.apply_alias_and_multiplier(aliases, "astra")
        self.code = "ASTRA"

    def run(self):
        """
        Run the ASTRA generator to create the beam file.
        """
        command = self.executables["ASTRAgenerator"] + [self.objectname + ".in"]
        with open(os.devnull, "w") as f:
            subprocess.call(
                command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    def _write_ASTRA(self):
        """
        Write the ASTRA input file with the parameters defined in the class.
        Base attributes of :class:`~simba.Codes.Generators.frameworkGenerator`
        are used to generate the input file, with the appropriate aliases and multipliers applied for the ASTRA code.
        :return: A string representation of the ASTRA input parameters.
        """
        output = ""
        self.apply_alias_and_multiplier(aliases, "ASTRA")
        for k, v in self.__dict__.items():
            disallowed = astra_generator_keywords["disallowed"]
            if k not in disallowed:
                key = k
                val = v
                if k in list(aliases["aliases"]["astra"].keys()):
                    key = aliases["aliases"]["astra"][k]["alias"]
                    val = v
                    if "multiplier" in list(aliases["aliases"]["astra"][k].keys()):
                        val = v * aliases["aliases"]["astra"][k]["multiplier"]
                if key in ["le"]:
                    val = 1e-3 * self.thermal_kinetic_energy
                if isinstance(v, str):
                    if v == "electron":
                        val = "electrons"
                    param_string = key + " = '" + str(val) + "',\n"
                else:
                    param_string = key + " = " + str(val) + ",\n"
                if len((output + param_string).splitlines()[-1]) > 70:
                    output += "\n"
                output += param_string
        return output[:-2]

    def write(self):
        """
        Write the ASTRA input file to the specified directory.
        #TODO Filenames are hardcoded for simplicity and they shouldn't be.
        """
        output = "&INPUT\n"
        if self.filename is None:
            self.filename = "generator.txt"
        output += self._write_ASTRA()
        output += "\n/\n"
        saveFile(
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in",
            output,
        )

    def postProcess(self):
        """
        Post-process the generated ASTRA beam file to create an HDF5 beam file.
        This method reads the ASTRA beam file and writes it to an HDF5 format.
        #TODO Filenames are hardcoded for simplicity and they shouldn't be.
        """
        astrabeamfilename = "generator.txt"
        rbf.astra.read_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
            keepLost=True,
        )
        HDF5filename = "laser.hdf5"
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            os.path.join(self.global_parameters["master_subdir"], HDF5filename),
            centered=False,
            sourcefilename=astrabeamfilename,
            cathode=True
        )

