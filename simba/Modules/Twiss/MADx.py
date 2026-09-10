"""
Simframe Twiss MAD-X Module

Functions for saving and loading Twiss summary files produced by the
:class:`~simba.Codes.MADX.MADX.madxLattice` tracking runs. The files are
plain HDF5 files (``*_twiss.madx.hdf5``) with one dataset per Twiss
parameter, using the same parameter names (and units) as the
:class:`~simba.Modules.Twiss.twiss` object, so that they can be interpreted
in the same way as the output of the other tracking codes.
"""

import os
import numpy as np
import h5py


def save_madx_twiss_hdf(self, filename: str, twiss: dict = {}) -> None:
    """
    Save a dictionary of MAD-X Twiss/beam-statistics arrays to an HDF5 file.

    Parameters
    ----------
    filename: str
        Name of the file to write
    twiss: dict
        Dictionary of arrays keyed by Twiss parameter name
    """
    with h5py.File(filename, "w") as f:
        for grp_name in twiss:
            try:
                f.create_dataset(grp_name, data=twiss[grp_name])
            except Exception:
                pass


def read_madx_twiss_files(self, filename, reset=True):
    """
    Read one or more MAD-X Twiss summary files (``*_twiss.madx.hdf5``) into a
    :class:`~simba.Modules.Twiss.twiss` object.

    Parameters
    ----------
    filename: str or list
        Name(s) of the file(s) to read
    reset: bool
        If True, reset the twiss object before reading
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
            for key in data.keys():
                try:
                    fdat[key] = np.array(data[key])
                except ValueError as e:
                    print(f"Failed to interpret {key} for {filename}, {e}")
        interpret_madx_data(self, lattice_name, fdat)


def interpret_madx_data(self, lattice_name, fdat):
    """
    Append the data loaded from a MAD-X Twiss summary file to the arrays of a
    :class:`~simba.Modules.Twiss.twiss` object. Every Twiss parameter is
    appended (missing parameters are zero-filled) so that all arrays remain
    the same length and can be sorted and interpolated consistently.
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
    """Decode HDF5 byte-strings"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
