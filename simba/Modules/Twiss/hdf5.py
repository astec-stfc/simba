import h5py
import numpy as np
from ..units import UnitValue


def read_hdf_summary(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    f = h5py.File(filename, "r")
    xemit = f.get("Xemit")
    yemit = f.get("Yemit")
    zemit = f.get("Zemit")
    for item, params in sorted(xemit.items()):
        self.interpret_astra_data(
            np.array(xemit.get(item)),
            np.array(yemit.get(item)),
            np.array(zemit.get(item)),
        )


def write_HDF5_twiss_file(self, filename, sourcefilename=None, version=2):
    with h5py.File(filename, "w") as f:
        inputgrp = f.create_group("Parameters")
        if sourcefilename is not None:
            inputgrp["Source"] = sourcefilename
        inputgrp["Version"] = str(version)
        twissgrp = f.create_group("twiss")
        if str(version) == "1":
            twissgrp["columns"] = np.array(list(self.properties.keys()), dtype="S")
            twissgrp["units"] = np.array(list(self.properties.values()), dtype="S")
            array = np.array(
                [
                    (
                        self[k]
                        if not k == "element_name" and not k == "lattice_name"
                        else np.array(self[k], dtype="S")
                    )
                    for k in self.properties.keys()
                    if len(self[k]) > 0
                ]
            ).transpose()
            twissgrp.create_dataset("twiss", data=array)
        if str(version) == "2":
            self.sort("z")
            for name, unit in self.properties.items():
                if len(getattr(self, name).val) > 0:
                    array = (
                        getattr(self, name).val
                        if not name == "element_name" and not name == "lattice_name"
                        else np.array(getattr(self, name).val, dtype="S")
                    )
                    dataset = twissgrp.create_dataset(name, data=array)
                    dataset.attrs.create("Units", str(unit.unit))


def read_HDF5_twiss_file(self, filename):
    with h5py.File(filename, "r") as h5file:
        if not hasattr(h5file, "version") or h5file["Version"] < "2":
            cols = list(h5file.get("twiss/columns").asstr())
            twiss = np.array(h5file.get("twiss/twiss")).transpose()
            units = list(h5file.get("twiss/units").asstr())
            [
                setattr(self, c, UnitValue(v, units=u, dtype=self.properties[c].dtype))
                for c, v, u in zip(cols, twiss, units)
            ]
        elif h5file["Version"] == "2":
            for name, data in h5file["twiss"].items():
                setattr(
                    self,
                    name,
                    UnitValue(
                        data,
                        units=data.attrs["Units"],
                        dtype=self.properties[name].dtype,
                    ),
                )
