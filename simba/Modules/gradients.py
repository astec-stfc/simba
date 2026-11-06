"""
Derivatives of a line's tracking with respect to lattice parameters, as written by codes
that can differentiate (``supports_gradients``) to ``<line>_gradients.<code>.hdf5``::

    parameters          str (P,)        ELEMENT:keyword, LAURA names
    parameter_values    (P,)
    Twiss/<key>         (n_s, P)        d key / d parameter, at Twiss/s and Twiss/z
    Beams/<name>        (n, 6, P)       d particle / d parameter at each screen and the
                                        line end; columns are the ``coordinates`` attribute
"""

import h5py
import numpy as np


def write_gradients(
    filename: str,
    parameters: list,
    parameter_values: list,
    twiss: dict,
    beams: dict,
    **attrs,
) -> None:
    """
    Write a line's derivatives; see the module docstring for the layout.

    Parameters
    ----------
    filename: str
    parameters: list
        ``ELEMENT:keyword`` names, one per column.
    parameter_values: list
        Their values when tracked.
    twiss: dict
        Twiss key -> (n_s, P) array, plus ``s`` and ``z`` -> (n_s,).
    beams: dict
        Screen or line-end name -> (n, 6, P) array.
    attrs:
        File attributes, e.g. ``code``, ``lattice``, ``coordinates``.
    """
    with h5py.File(filename, "w") as f:
        f.attrs.update(attrs)
        f.create_dataset("parameters", data=np.array(parameters, dtype=h5py.string_dtype()))
        f.create_dataset("parameter_values", data=np.asarray(parameter_values, dtype=float))
        for group, data in (("Twiss", twiss), ("Beams", beams)):
            grp = f.create_group(group)
            for key, val in data.items():
                grp.create_dataset(key, data=np.asarray(val))


def read_gradients(filename: str) -> dict:
    """
    Read a file written by :func:`write_gradients`.

    Returns
    -------
    dict
        ``parameters`` (list of str), ``parameter_values``, ``Twiss`` and ``Beams``
        (dicts of arrays) and ``attrs``.
    """
    with h5py.File(filename, "r") as f:
        return {
            "parameters": f["parameters"].asstr()[()].tolist(),
            "parameter_values": f["parameter_values"][()],
            "Twiss": {k: v[()] for k, v in f["Twiss"].items()},
            "Beams": {k: v[()] for k, v in f["Beams"].items()},
            "attrs": dict(f.attrs),
        }
