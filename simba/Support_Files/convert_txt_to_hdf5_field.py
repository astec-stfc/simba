import csv
import numpy as np
import h5py

with open(
    r"C:\Users\jkj62.CLRC\Documents\GitHub\masterlattice\MasterLattice\Data_Files\SzSx20um25mmxband.csv",
    "r",
) as infile:
    reader = csv.reader(infile, delimiter=",")
    data = np.array([[float(value) for value in row] for row in reader])

z, Wx, Wy, Wz = data.T

attrs = {
    "/": {"type": "3DWake"},
    "/Wx": {"units": "V/C/m"},
    "/Wy": {"units": "V/C/m"},
    "/Wz": {"units": "V/c"},
    "/z": {"units": "m"},
}
dataFormat = {
    "Wx": np.float64,
    "Wy": np.float64,
    "Wz": np.float64,
    "z": np.float64,
}

# Save the data to an HDF5 file
output_file = r"C:\Users\jkj62.CLRC\Documents\GitHub\masterlattice\MasterLattice\Data_Files\SzSx20um25mmxband.hdf5"
with h5py.File(output_file, "w") as hdf:
    # Create datasets for each data array
    hdf.create_dataset("z", data=z, dtype=dataFormat["z"])
    hdf.create_dataset("Wx", data=Wx, dtype=dataFormat["Wx"])
    hdf.create_dataset("Wy", data=Wy, dtype=dataFormat["Wy"])
    hdf.create_dataset("Wz", data=Wz, dtype=dataFormat["Wz"])

    # Add attributes to the datasets and root
    for key, value in attrs.items():
        if key == "/":
            for attr_name, attr_value in value.items():
                hdf.attrs[attr_name] = attr_value
        else:
            for attr_name, attr_value in value.items():
                hdf[key].attrs[attr_name] = attr_value

print(f"Data successfully saved to {output_file}")
