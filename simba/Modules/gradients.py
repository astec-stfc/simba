"""
Derivatives of a line's tracking with respect to lattice parameters, as written by codes
that can differentiate (``supports_gradients``) to ``<line>_gradients.<code>.hdf5``::

    parameters          str (P,)        ELEMENT:keyword, LAURA names
    parameter_values    (P,)
    Twiss/<key>         (n_s, P)        d key / d parameter, at Twiss/s and Twiss/z
    Beams/<name>        (n, 6, P)       d particle / d parameter at each screen and the
                                        line end; columns are the ``coordinates`` attribute
"""

import glob
import os

import h5py
import numpy as np

lab_coordinates = ("x", "cpx", "y", "cpy", "t", "cpz")
"""Columns of a tangent, in m, eV/c and s."""


def lab_array(beam) -> np.ndarray:
    """A :class:`~simba.Modules.Beams.beam`'s particles as an (n, 6) array of :data:`lab_coordinates`."""
    return np.stack([getattr(beam, c).val for c in lab_coordinates], -1)


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


def saved_tangent(beam_file: str) -> dict | None:
    """
    The tangent a gradients file next to ``beam_file`` holds for it (``Beams/<stem>``), the
    newest if more than one does, or None.

    Returns
    -------
    dict or None
        ``parameters``, ``values`` and ``tangent`` (n, 6, P), as handed from line to line,
        plus ``file``, the gradients file it came from.
    """
    stem = os.path.basename(beam_file).removesuffix(".hdf5").removesuffix(".openpmd")
    files = sorted(glob.glob(os.path.join(os.path.dirname(beam_file), "*_gradients.*.hdf5")), key=os.path.getmtime)
    for filename in reversed(files):
        with h5py.File(filename, "r") as f:
            if stem in f.get("Beams", {}):
                return {
                    "parameters": f["parameters"].asstr()[()].tolist(),
                    "values": f["parameter_values"][()].tolist(),
                    "tangent": f["Beams"][stem][()],
                    "file": filename,
                }
    return None
