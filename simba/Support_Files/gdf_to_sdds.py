import sys, os

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../"))
import numpy as np
from Modules import Beams as rbf
import argparse

parser = argparse.ArgumentParser(description="Convert SDDS file to SimFrame HDF5.")
parser.add_argument("filename")


class convertGDFToSDDS:

    def __init__(self, directory="."):
        super().__init__()
        self.global_parameters = {}
        self.global_parameters["master_subdir"] = directory
        self.beam = self.global_parameters["beam"] = rbf.beam()

    def gdf_to_sdds(self, filename, middle=0, end=0):
        self.output_filename = filename
        beamfilename = self.output_filename.strip('"')
        self.beam.read_gdf_beam_file(
            self.global_parameters["master_subdir"] + "/" + beamfilename
        )
        SDDSfilename = self.output_filename.replace(".gdf", ".sdds").strip('"')
        self.beam.write_SDDS_beam_file(
            self.global_parameters["master_subdir"] + "/" + SDDSfilename
        )


if __name__ == "__main__":
    args = parser.parse_args()
    converter = convertGDFToSDDS()
    converter.gdf_to_sdds(args.filename, [0, 0, 0], [0, 0, 0])
