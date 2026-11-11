"""HiPACE++ beam input: an openPMD file its ``from_file`` injection reads.

This file and HiPACE++'s diagnostics are both read by
:func:`~simba.Modules.Beams.fbpic.read_fbpic_beam_file`.
"""

import h5py
import numpy as np

from ..units import UnitValue

_LENGTH = [1.0, 0, 0, 0, 0, 0, 0]
_MOMENTUM = [1.0, 1, -1, 0, 0, 0, 0]
_NONE = [0.0] * 7
_CHARGE = [0.0, 0, 1, 1, 0, 0, 0]
_MASS = [0.0, 1, 0, 0, 0, 0, 0]


def _val(value) -> np.ndarray:
    return np.asarray(value.val if isinstance(value, UnitValue) else value, dtype=float)


def write_hipace_beam_file(
    self, filename: str, zstart: float = 0.0, dtype: type = np.float64
) -> tuple[float, float]:
    """
    Write the beam as species ``beam`` of an openPMD file for HiPACE++'s
    ``beam.injection_type = from_file``, or WarpX's ``injection_style = external_file``.
    Both read through openPMD-api, which loads arrays only as their stored type.

    Parameters
    ----------
    self: :class:`~simba.Modules.Beams.beam`
        The beam to write
    filename: str
        File to write
    zstart: float
        Lattice z placed at the box's z = 0
    dtype: type
        Array type: the reader's particle precision, ``np.float32`` for a
        single-precision WarpX

    Returns
    -------
    tuple[float, float]
        Particle charge [C] and mass [kg], which HiPACE++ takes as ``beam.charge``
        and ``beam.mass`` rather than from the file

    Raises
    ------
    ValueError
        If the beam mixes particle species; HiPACE++ gives one charge and mass per beam
    """
    charge, mass = _val(self._beam.particle_charge), _val(self._beam.particle_mass)
    if np.ptp(charge) or np.ptp(mass):
        raise ValueError("A HiPACE++ beam has one particle species; this beam mixes several.")
    position = {"x": _val(self._beam.x), "y": _val(self._beam.y), "z": _val(self._beam.z) - zstart}
    momentum = {k: _val(getattr(self._beam, f"p{k}")) for k in "xyz"}  # kg m/s
    weighting = np.abs(_val(self._beam.charge) / charge)
    with h5py.File(filename, "w") as f:
        f.attrs.update({
            "openPMD": np.bytes_("1.1.0"), "openPMDextension": np.uint32(0),
            "basePath": np.bytes_("/data/%T/"), "particlesPath": np.bytes_("particles/"),
            "iterationEncoding": np.bytes_("groupBased"),
            "iterationFormat": np.bytes_("/data/%T/"),
        })
        iteration = f.create_group("data/0")
        iteration.attrs.update({"dt": 1.0, "time": 0.0, "timeUnitSI": 1.0})
        species = iteration.create_group("particles/beam")
        for name, components, dimension in (
            ("position", position, _LENGTH),
            ("momentum", momentum, _MOMENTUM),
        ):
            record = species.create_group(name)
            record.attrs.update({"unitDimension": np.array(dimension), "timeOffset": np.float32(0)})
            for axis, data in components.items():
                record.create_dataset(axis, data=data.astype(dtype)).attrs["unitSI"] = 1.0
        record = species.create_dataset("weighting", data=weighting.astype(dtype))
        record.attrs.update(
            {"unitSI": 1.0, "unitDimension": np.array(_NONE), "timeOffset": np.float32(0)}
        )
        # WarpX reads positionOffset unconditionally.
        record = species.create_group("positionOffset")
        record.attrs.update({"unitDimension": np.array(_LENGTH), "timeOffset": np.float32(0)})
        for axis in "xyz":
            record.create_group(axis).attrs.update({
                "value": 0.0, "shape": np.array([len(charge)], dtype=np.uint64), "unitSI": 1.0,
            })
        # Constant records, as HiPACE++ writes them; it weights by `weighting` first.
        for name, value, dimension in (("charge", charge[0], _CHARGE), ("mass", mass[0], _MASS)):
            species.create_group(name).attrs.update({
                "value": float(value), "shape": np.array([len(charge)], dtype=np.uint64),
                "unitSI": 1.0, "unitDimension": np.array(dimension), "timeOffset": np.float32(0),
            })
    return float(charge[0]), float(mass[0])
