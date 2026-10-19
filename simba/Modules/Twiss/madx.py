"""
Save and load the Twiss summary files of :class:`~simba.Codes.MADX.MADX.madxLattice` runs.

These are HDF5 files (``*_twiss.madx.hdf5``) with one dataset per
:class:`~simba.Modules.Twiss.twiss` parameter, same names and units.
"""

import os
import numpy as np
import h5py


def save_madx_twiss_hdf(self, filename: str, twiss: dict = {}) -> None:
    """
    Save MAD-X Twiss and beam-statistics arrays to an HDF5 file.

    Parameters
    ----------
    filename: str
        Output filename
    twiss: dict
        Twiss parameter name to array; entries h5py can't store are skipped
    """
    with h5py.File(filename, "w") as f:
        for grp_name in twiss:
            try:
                f.create_dataset(grp_name, data=twiss[grp_name])
            except Exception:
                pass


def read_madx_twiss_files(self, filename, reset=True):
    """
    Read MAD-X Twiss summary files into a :class:`~simba.Modules.Twiss.twiss` object.

    Parameters
    ----------
    filename: str or list
        File(s) to read
    reset: bool
        Reset the twiss object first
    """
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_madx_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        fdat = {}
        with h5py.File(filename, "r") as data:
            for key in data:
                try:
                    fdat[key] = np.array(data[key])
                except ValueError as e:
                    print(f"Failed to interpret {key} for {filename}, {e}")
        interpret_madx_data(self, lattice_name, fdat)


def interpret_madx_data(self, lattice_name, fdat):
    """
    Append a MAD-X Twiss summary file's data to a :class:`~simba.Modules.Twiss.twiss` object.

    Missing parameters are zero-filled so every array stays the same length.
    """
    if "s" not in fdat:
        return
    nrows = len(fdat["s"])
    cls = self.__class__
    for key in cls.model_fields:
        param = getattr(self, key)
        # only twissParameter-like fields
        if not hasattr(param, "val"):
            continue
        if key == "lattice_name":
            if key in fdat:
                values = np.array([_decode(v) for v in fdat[key]])
            else:
                values = np.full(nrows, lattice_name)
        elif key == "element_name":
            if key in fdat:
                values = np.array([_decode(v) for v in fdat[key]])
            else:
                values = np.full(nrows, "")
        elif key in fdat:
            values = np.array(fdat[key], dtype=float)
        else:
            values = np.zeros(nrows)
        param.val = np.append(param.val, values)


def _decode(value):
    """Decode HDF5 byte-strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
