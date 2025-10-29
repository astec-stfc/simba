import sys
import os

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../../"))
import numpy as np
from SimulationFramework.Modules import Beams as rbf
import argparse

parser = argparse.ArgumentParser(description="Convert GDF file to SimFrame HDF5.")
parser.add_argument("filename")
parser.add_argument("--middle", type=list, default=[0, 0, 0])
parser.add_argument("--end", type=list, default=[0, 0, 0])
parser.add_argument("--longitudinal_reference", type=str, default="z")


class convertGDFToHDF5:

    def __init__(self, directory="."):
        super().__init__()
        self.beam = rbf.beam()
        self.beam.longitudinal_reference = args.longitudinal_reference

    def gdf_to_hdf5(self, filename, middle=0, end=0):
        beamfilename = filename
        self.beam.read_gdf_beam_file(beamfilename)
        HDF5filename = beamfilename.replace(".gdf", ".hdf5")
        self.beam.write_HDF5_beam_file(HDF5filename)


if __name__ == "__main__":
    args = parser.parse_args()
    converter = convertGDFToHDF5()
    converter.gdf_to_hdf5(args.filename, args.middle, args.end)
