# import sys
# import os

# sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../../"))
import numpy as np
from SimulationFramework.Modules import Beams as rbf
import argparse

parser = argparse.ArgumentParser(description="Convert SDDS file to SimFrame HDF5.")
parser.add_argument("filename")


class convertSDDS:

    def __init__(self, directory="."):
        super().__init__()
        self.global_parameters = {}
        self.global_parameters["master_subdir"] = directory
        self.global_parameters["beam"] = rbf.beam()

    def sdds_to_hdf5(self, filename, middle=0, end=0):
        self.output_filename = filename
        elegantbeamfilename = self.output_filename.replace(".sdds", ".SDDS").strip('"')
        rbf.sdds.read_SDDS_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + elegantbeamfilename,
        )
        HDF5filename = (
            self.output_filename.replace(".sdds", ".hdf5")
            .replace(".SDDS", ".hdf5")
            .strip('"')
        )
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=elegantbeamfilename,
            pos=middle,
            zoffset=end,
            toffset=(-1 * np.mean(self.global_parameters["beam"].t)),
        )


if __name__ == "__main__":
    args = parser.parse_args()
    converter = convertSDDS()
    converter.sdds_to_hdf5(args.filename, [0, 0, 0], [0, 0, 0])
