import sys, os

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../../"))
import numpy as np
from SimulationFramework.Modules import Beams as rbf
import argparse, subprocess
from SimulationFramework.Modules import constants

parser = argparse.ArgumentParser(description="Convert SDDS file to SimFrame HDF5.")
parser.add_argument("filename")


class convertSDDSToGDF:

    def __init__(self, directory="."):
        super().__init__()
        self.global_parameters = {}
        self.global_parameters["master_subdir"] = directory
        self.beam = self.global_parameters["beam"] = rbf.beam()

    def sdds_to_gdf(self, filename, charge=None, mass=None):
        charge = -1 * constants.elementary_charge if charge is None else charge
        mass = constants.electron_mass if mass is None else mass
        self.output_filename = filename
        beamfilename = self.output_filename.strip('"')
        self.beam.read_SDDS_beam_file(
            self.global_parameters["master_subdir"] + "/" + beamfilename
        )
        gdftxtfilename = self.output_filename.replace(".sdds", ".txt").strip('"')
        self.beam.write_gdf_beam_file(
            self.global_parameters["master_subdir"] + "/" + gdftxtfilename,
            charge=charge,
            mass=mass,
        )
        gdffilename = self.output_filename.replace(".sdds", ".gdf").strip('"')
        subprocess.call(
            [
                r"C:\Program Files\General Particle Tracer\bin\asci2gdf",
                "-o",
                gdffilename,
                gdftxtfilename,
            ],
            cwd=self.global_parameters["master_subdir"],
        )


if __name__ == "__main__":
    args = parser.parse_args()
    converter = convertSDDSToGDF()
    converter.sdds_to_gdf(
        args.filename, constants.elementary_charge, constants.proton_mass
    )
