"""ASTRA beam generator."""
import os
import subprocess
from ...FrameworkHelperFunctions import saveFile
from .Generators import (
    frameworkGenerator,
    aliases,
    astra_generator_keywords,
)
from typing import Any
from ...Modules import Beams as rbf


class ASTRAGenerator(frameworkGenerator):
    """
    Generates a beam with ASTRA's ``generator``.
    """

    Lprompt: bool = False

    filename: str = "generator.txt"
    """Name of the output file."""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.apply_alias_and_multiplier(aliases, "astra")
        self.code = "ASTRA"

    def run(self):
        """
        Run the ASTRA generator to create the beam file.
        """
        command = self.executables["ASTRAgenerator"] + [self.objectname + ".in"]
        workdir = os.path.abspath(self.global_parameters["master_subdir"])
        command = self.executables.build_command(command, workdir)
        with open(os.devnull, "w") as f:
            subprocess.call(
                command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    def _write_ASTRA(self):
        """
        The ``&INPUT`` body: allowed attributes under their ASTRA aliases and units.
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
                if val == "electron":
                    val = "electrons"
                if val == "proton":
                    val = "protons"
                if val == "positron":
                    val = "positrons"
                if isinstance(v, str):
                    param_string = key + " = '" + str(val) + "',\n"
                else:
                    param_string = key + " = " + str(val) + ",\n"
                if len((output + param_string).splitlines()[-1]) > 70:
                    output += "\n"
                output += param_string
        return output[:-2]

    def write(self):
        """
        Write the ASTRA input file to `master_subdir`.
        """
        output = "&INPUT\n"
        self.filename = self.filename.replace(".openpmd.hdf5", ".txt")
        output += self._write_ASTRA()
        output += "\n/\n"
        saveFile(
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in",
            output,
        )

    def postProcess(self):
        """
        Convert the ASTRA beam file to ``laser.openpmd.hdf5`` (name hardcoded).
        """
        self.global_parameters["beam"] = rbf.beam()
        rbf.astra.read_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + self.filename,
            normaliseZ=False,
            keepLost=True,
        )
        if self.cathode:
            HDF5filename = "laser.openpmd.hdf5"
        else:
            HDF5filename = "laser.openpmd.hdf5" #self.filename.replace(".txt", ".openpmd.hdf5")
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )


