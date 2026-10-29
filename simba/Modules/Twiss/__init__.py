"""Read and manipulate Twiss parameters from various simulation codes."""

from __future__ import annotations
import os
import math
import warnings
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    ValidationInfo,
    model_validator,
    Field,
)
import numpy as np
from typing import Dict, List, Callable

from .. import constants
import munch
import glob
from . import hdf5
from . import gpt
from . import astra
from . import elegant
from . import ocelot
from . import cheetah
from . import opal
from . import xsuite
from . import genesis
from . import madx
from . import bmad

try:
    from . import plot

    use_matplotlib = True
except ImportError:
    use_matplotlib = False

from ..units import UnitValue

codes = {
    "elegant": elegant.read_elegant_twiss_files,
    "gpt": gpt.read_gdf_twiss_files,
    "astra": astra.read_astra_twiss_files,
    "ocelot": ocelot.read_ocelot_twiss_files_hdf,
    "opal": opal.read_opal_twiss_files,
    "cheetah": cheetah.read_cheetah_twiss_files,
    "xsuite": xsuite.read_xsuite_twiss_files,
    "genesis": genesis.read_genesis_twiss_files,
    "madx": madx.read_madx_twiss_files,
    "bmad": bmad.read_bmad_twiss_files,
}

code_signatures = [
    ["elegant", ".twi"],
    ["elegant", ".flr"],
    ["elegant", ".sig"],
    ["gpt", "emit.gdf"],
    ["astra", "Xemit.001"],
    ["ocelot", "_twiss.npz"],
    ["opal", "opal_twiss.h5"],
    ["ocelot", "_twiss.oh5"],
    ["cheetah", "_twiss.cheetah.hdf5"],
    ["genesis", ".out.h5"],
    ["xsuite", "_twiss.csv"],
    ["madx", "_twiss.madx.hdf5"],
    ["bmad", "_twiss.bmad.hdf5"],
]

twiss_defaults = {
    "z": {"name": "z", "unit": "m"},
    "s": {"name": "s", "unit": "m"},
    "t": {"name": "t", "unit": "s"},
    "kinetic_energy": {"name": "kinetic_energy", "unit": "eV"},
    "gamma": {"name": "gamma", "unit": ""},
    "cp": {"name": "cp", "unit": "eV/c"},
    "p": {"name": "p", "unit": "kg*m/s"},
    "ex": {"name": "ex", "unit": "m-rad"},
    "enx": {"name": "enx", "unit": "m-rad"},
    "ecnx": {"name": "ecnx", "unit": "m-rad"},
    "ey": {"name": "ey", "unit": "m-rad"},
    "eny": {"name": "eny", "unit": "m-rad"},
    "ecny": {"name": "ecny", "unit": "m-rad"},
    "ez": {"name": "ez", "unit": "eV*s"},
    "enz": {"name": "enz", "unit": "eV*s"},
    "ecnz": {"name": "ecnz", "unit": "eV*s"},
    "beta_x": {"name": "beta_x", "unit": "m"},
    "gamma_x": {"name": "gamma_x", "unit": ""},
    "alpha_x": {"name": "alpha_x", "unit": ""},
    "beta_y": {"name": "beta_y", "unit": "m"},
    "gamma_y": {"name": "gamma_y", "unit": ""},
    "alpha_y": {"name": "alpha_y", "unit": ""},
    "beta_z": {"name": "beta_z", "unit": "m"},
    "gamma_z": {"name": "gamma_z", "unit": ""},
    "alpha_z": {"name": "alpha_z", "unit": ""},
    "sigma_x": {"name": "sigma_x", "unit": "m"},
    "sigma_xp": {"name": "sigma_xp", "unit": "rad"},
    "sigma_y": {"name": "sigma_y", "unit": "m"},
    "sigma_yp": {"name": "sigma_yp", "unit": "rad"},
    "sigma_t": {"name": "sigma_t", "unit": "s"},
    "sigma_z": {"name": "sigma_z", "unit": "m"},
    "sigma_p": {"name": "sigma_p", "unit": "kg*m/s"},
    "sigma_cp": {"name": "sigma_cp", "unit": "eV/c"},
    "mean_x": {"name": "mean_x", "unit": "m"},
    "mean_y": {"name": "mean_y", "unit": "m"},
    "mean_cp": {"name": "mean_cp", "unit": "eV/c"},
    "mux": {"name": "mux", "unit": "2 pi"},
    "muy": {"name": "muy", "unit": "2 pi"},
    "eta_x": {"name": "eta_x", "unit": "m"},
    "eta_xp": {"name": "eta_xp", "unit": "rad"},
    "eta_y": {"name": "eta_y", "unit": "m"},
    "eta_yp": {"name": "eta_yp", "unit": "rad"},
    "element_name": {"name": "element_name", "unit": "", "dtype": "U"},
    "lattice_name": {"name": "lattice_name", "unit": "", "dtype": "U"},
    "turn": {"name": "turn", "unit": "", "dtype": "i"},
    "eta_x_beam": {"name": "eta_x_beam", "unit": "m"},
    "eta_xp_beam": {"name": "eta_xp_beam", "unit": "rad"},
    "eta_y_beam": {"name": "eta_y_beam", "unit": "m"},
    "eta_yp_beam": {"name": "eta_yp_beam", "unit": "rad"},
    "beta_x_beam": {"name": "beta_x_beam", "unit": "m"},
    "alpha_x_beam": {"name": "alpha_x_beam", "unit": ""},
    "beta_y_beam": {"name": "beta_y_beam", "unit": "m"},
    "alpha_y_beam": {"name": "alpha_y_beam", "unit": ""},
}


class twissParameter(BaseModel):
    """A named Twiss column with its unit and values."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """Parameter name, e.g. 'z', 'beta_x'."""

    unit: str
    """Unit string, e.g. 'm', 's', 'eV'."""

    val: List = []
    """Values."""

    label: str = Field(default=None, validate_default=True)
    """Display label; defaults to `name`."""

    dtype: str = "f"
    """Numpy dtype code."""

    @field_validator("label", mode="before")
    @classmethod
    def default_label(cls, v: str, info: ValidationInfo):
        if v is None:
            return info.data["name"]
        return v

    def min(self) -> float:
        return min(self.val)

    def max(self) -> float:
        return max(self.val)

    def __len__(self) -> int:
        return len(self.val)


class initialTwiss(BaseModel):
    """Initial Twiss parameters of a beam."""

    alpha_x: float
    """Horizontal alpha."""

    beta_x: float
    """Horizontal beta."""

    alpha_y: float
    """Vertical alpha."""

    beta_y: float
    """Vertical beta."""

    ex: float
    """Horizontal emittance."""

    ey: float
    """Vertical emittance."""

    enx: float
    """Normalised horizontal emittance."""

    eny: float
    """Normalised vertical emittance."""

    eta_x: float
    """Horizontal dispersion."""

    eta_xp: float
    """Horizontal dispersion derivative."""

    eta_y: float
    """Vertical dispersion."""

    eta_yp: float
    """Vertical dispersion derivative."""


class twiss(BaseModel):
    """Twiss parameters and beam statistics along a lattice, read from any supported code."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    z: "twissParameter" = None
    """Longitudinal position [m]."""

    s: "twissParameter" = None
    """Path length [m]."""

    t: "twissParameter" = None
    """Time [s]."""

    kinetic_energy: "twissParameter" = None
    """Kinetic energy [eV]."""

    gamma: "twissParameter" = None
    """Lorentz factor."""

    cp: "twissParameter" = None
    """Momentum [eV/c]."""

    p: "twissParameter" = None
    """Momentum [kg*m/s]."""

    enx: "twissParameter" = None
    """Normalised horizontal emittance [m-rad]."""

    ex: "twissParameter" = None
    """Horizontal emittance [m-rad]."""

    eny: "twissParameter" = None
    """Normalised vertical emittance [m-rad]."""

    ey: "twissParameter" = None
    """Vertical emittance [m-rad]."""

    enz: "twissParameter" = None
    """Normalised longitudinal emittance [eV*s]."""

    ez: "twissParameter" = None
    """Longitudinal emittance [eV*s]."""

    beta_x: "twissParameter" = None
    """Horizontal beta [m]."""

    gamma_x: "twissParameter" = None
    """Horizontal Twiss gamma."""

    alpha_x: "twissParameter" = None
    """Horizontal alpha."""

    beta_y: "twissParameter" = None
    """Vertical beta [m]."""

    gamma_y: "twissParameter" = None
    """Vertical Twiss gamma."""

    alpha_y: "twissParameter" = None
    """Vertical alpha."""

    beta_z: "twissParameter" = None
    """Longitudinal beta [m]."""

    gamma_z: "twissParameter" = None
    """Longitudinal Twiss gamma."""

    alpha_z: "twissParameter" = None
    """Longitudinal alpha."""

    sigma_x: "twissParameter" = None
    """RMS x [m]."""

    sigma_xp: "twissParameter" = None
    """RMS x' [rad]."""

    sigma_y: "twissParameter" = None
    """RMS y [m]."""

    sigma_yp: "twissParameter" = None
    """RMS y' [rad]."""

    sigma_z: "twissParameter" = None
    """RMS z [m]."""

    sigma_t: "twissParameter" = None
    """RMS t [s]."""

    sigma_p: "twissParameter" = None
    """RMS momentum [kg*m/s]."""

    sigma_cp: "twissParameter" = None
    """RMS momentum [eV/c]."""

    mean_x: "twissParameter" = None
    """Mean x [m]."""

    mean_y: "twissParameter" = None
    """Mean y [m]."""

    mean_cp: "twissParameter" = None
    """Mean momentum [eV/c]."""

    mux: "twissParameter" = None
    """Horizontal phase advance [2 pi]."""

    muy: "twissParameter" = None
    """Vertical phase advance [2 pi]."""

    eta_x: "twissParameter" = None
    """Horizontal dispersion [m]."""

    eta_xp: "twissParameter" = None
    """Horizontal dispersion derivative [rad]."""

    eta_y: "twissParameter" = None
    """Vertical dispersion [m]."""

    eta_yp: "twissParameter" = None
    """Vertical dispersion derivative [rad]."""

    element_name: "twissParameter" = None
    """Element name at each row."""

    lattice_name: "twissParameter" = None
    """Lattice name at each row."""

    turn: "twissParameter" = None
    """Turn of a multi-turn run each row was measured on, 1-based."""

    ecnx: "twissParameter" = None
    """Normalised horizontal emittance with the dispersive part removed [m-rad]."""

    ecny: "twissParameter" = None
    """Normalised vertical emittance with the dispersive part removed [m-rad]."""

    eta_x_beam: "twissParameter" = None
    """Horizontal dispersion from the tracked particles [m]."""

    eta_xp_beam: "twissParameter" = None
    """Horizontal dispersion derivative from the tracked particles [rad]."""

    eta_y_beam: "twissParameter" = None
    """Vertical dispersion from the tracked particles [m]."""

    eta_yp_beam: "twissParameter" = None
    """Vertical dispersion derivative from the tracked particles [rad]."""

    beta_x_beam: "twissParameter" = None
    """Horizontal beta from the tracked particles [m]."""

    beta_y_beam: "twissParameter" = None
    """Vertical beta from the tracked particles [m]."""

    alpha_x_beam: "twissParameter" = None
    """Horizontal alpha from the tracked particles."""

    alpha_y_beam: "twissParameter" = None
    """Vertical alpha from the tracked particles."""

    rest_mass: float | None = None
    """Particle rest mass [kg]."""

    codes: Dict = codes
    """Twiss reader function for each code name."""

    code_signatures: List[List[str]] = code_signatures
    """[code, filename suffix] pairs used to identify a Twiss file's code."""

    sddsindex: int = 0
    """Index for SDDS files."""

    q_over_c: float = constants.e / constants.speed_of_light
    """Elementary charge over c, for eV/c to kg*m/s conversion."""

    E0: float = constants.m_e * constants.speed_of_light**2
    """Particle rest energy [J]; electron by default."""

    E0_eV: float = E0 / constants.elementary_charge
    """Particle rest energy [eV]; electron by default."""

    elegantTwiss: Dict = {}
    """Raw ELEGANT Twiss data."""

    elegantData: Dict = {}
    """Raw ELEGANT data."""

    def __init__(
        self,
        rest_mass=None,
    ):
        twiss.rest_mass = rest_mass
        super().__init__(rest_mass=rest_mass)
        self.reset_dicts()
        self.sddsindex = 0
        self.codes = dict(codes)
        self.code_signatures = code_signatures

    @model_validator(mode="before")
    def validate_fields(cls, values):
        return values

    @property
    def properties(self):
        keys = twiss.model_fields.keys()
        return {
            k: getattr(self, k)
            for k in keys
            if isinstance(getattr(self, k), twissParameter)
        }

    def set_E0(self, value) -> None:
        """
        Set :attr:`E0` and :attr:`E0_eV` from the particle rest mass.

        Parameters
        ----------
        value: float
            Rest mass [kg]
        """
        self.E0 = value * constants.speed_of_light**2
        self.E0_eV = self.E0 / constants.elementary_charge

    def read_astra_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return astra.read_astra_twiss_files(self, *args, **kwargs)

    def read_elegant_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return elegant.read_elegant_twiss_files(self, *args, **kwargs)

    def read_gdf_twiss_files(self, *args, **kwargs) -> None:
        return self.read_GPT_twiss_files(*args, **kwargs)

    def read_GPT_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gpt.read_gdf_twiss_files(self, *args, **kwargs)

    def read_ocelot_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ocelot.read_ocelot_twiss_files_hdf(self, *args, **kwargs)

    def read_ocelot_twiss_files_hdf(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ocelot.read_ocelot_twiss_files_hdf(self, *args, **kwargs)

    def read_opal_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return opal.read_opal_twiss_files(self, *args, **kwargs)

    def read_xsuite_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xsuite.read_xsuite_twiss_files(self, *args, **kwargs)

    def read_genesis_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return genesis.read_genesis_twiss_files(self, *args, **kwargs)

    def read_madx_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return madx.read_madx_twiss_files(self, *args, **kwargs)

    def read_bmad_twiss_files(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bmad.read_bmad_twiss_files(self, *args, **kwargs)

    def save_HDF5_twiss_file(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return hdf5.write_HDF5_twiss_file(self, *args, **kwargs)

    def read_cheetah_twiss_files(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cheetah.read_cheetah_twiss_files(self, *args, **kwargs)

    def read_HDF5_twiss_file(self, *args, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return hdf5.read_HDF5_twiss_file(self, *args, **kwargs)

    def __repr__(self):
        return repr({k: getattr(self, k) for k in self.model_fields_set})

    def stat(self, key) -> twissParameter:
        """
        Get a Twiss parameter by name.

        Parameters
        ----------
        key: str
            Parameter name, e.g. 'z', 'beta_x'

        Returns
        -------
        twissParameter
        """
        return getattr(self, key)

    def find_nearest_idx(self, array: List, value: float) -> int:
        """
        Find the index of the element nearest to `value` in a sorted array.

        Parameters
        ----------
        array: List
            Sorted array
        value: float
            Value to look for

        Returns
        -------
        int
            Index of the nearest element, clamped to the array bounds
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (
            idx == len(array)
            or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
        ):
            return idx - 1
        else:
            return idx

    def find_nearest(self, array: List, value: float) -> float:
        """
        Find the element nearest to `value` in a sorted array.

        Parameters
        ----------
        array: List
            Sorted array
        value: float
            Value to look for

        Returns
        -------
        float
            Nearest element
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (
            idx == len(array)
            or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
        ):
            return array[idx - 1]
        else:
            return array[idx]

    def reset_dicts(self) -> None:
        """Reset every Twiss parameter to empty and clear :attr:`elegantTwiss`."""
        self.sddsindex = 0
        for name in twiss.model_fields:
            if name in list(twiss_defaults.keys()):
                setattr(self, name, twissParameter(**twiss_defaults[name]))
        self.elegantTwiss = {}

    def sort(self, key: str = "s", reverse: bool = False) -> None:
        """
        Sort every Twiss parameter by the values of one of them.

        Parameters
        ----------
        key: str
            Parameter to sort by
        reverse: bool, optional
            Sort in descending order
        """
        flat = np.array(getattr(self, key).val).flatten()
        index = flat.argsort()
        cls = self.__class__
        for k in cls.model_fields:
            if isinstance(getattr(self, k), twissParameter):
                if len(getattr(self, k).val) > 0:
                    try:
                        flat = np.array(getattr(self, k).val).flatten()
                    except Exception:
                        flat = getattr(self, k).val
                    if reverse:
                        getattr(self, k).val = flat[index[::-1]]
                    else:
                        getattr(self, k).val = flat[index[::1]]

    def append(self, array: str, data: List | np.ndarray) -> None:
        """
        Append data to a Twiss parameter.

        Parameters
        ----------
        array: str
            Parameter name
        data: List | np.ndarray
            Data to append
        """
        getattr(self, array).val = np.append(getattr(self, array).val, data)

    def append_columns(self, n: int, **columns) -> None:
        """
        Append `n` rows to several twiss parameter arrays at once.

        Parameters
        ----------
        n: int
            Number of rows being appended
        **columns:
            Parameter name and its `n` values; a scalar is repeated, so
            ``ez=0.0`` zero-fills a quantity the code doesn't output
        """
        for name, data in columns.items():
            self.append(name, np.full(n, data) if np.ndim(data) == 0 else data)

    def _which_code(self, name: str) -> Callable | None:
        """
        Get the Twiss reader for a code name.

        Parameters
        ----------
        name: str
            Code name, case-insensitive

        Returns
        -------
        callable | None
            Reader function, or None if the code is unknown
        """
        if name.lower() in self.codes:
            return self.codes[name.lower()]
        return None

    def _determine_code(self, filename: str) -> Callable | None:
        """
        Get the Twiss reader for a file from its suffix (see :attr:`code_signatures`).

        Parameters
        ----------
        filename: str
            Twiss filename

        Returns
        -------
        callable | None
            Reader function, or None if no signature matches
        """
        for k, v in self.code_signatures:
            cutl = -len(v)
            if v == filename[cutl:]:
                return self.codes[k]
        return None

    def interpolate(self, z=None, value="z", index="z") -> float:
        """
        Interpolate one Twiss parameter against another.

        Parameters
        ----------
        z: float, optional
            Position at which to interpolate
        value: str, optional
            Parameter to interpolate
        index: str, optional
            Parameter to interpolate against

        Returns
        -------
        float
            Interpolated value; 1e6 if `z` is beyond the end of `index`
        """
        if z is None:
            return np.interp(z, getattr(self, index), getattr(self, value).val)
        else:
            if z > np.max(getattr(self, index).val):
                return 10**6
            else:
                return float(np.interp(z, getattr(self, index).val, getattr(self, value).val))

    def extract_values(self, name: str, start: float, end: float) -> np.ndarray:
        """
        Extract a Twiss parameter between two z positions, inclusive.

        Parameters
        ----------
        name: str
            Parameter name
        start: float
            Initial z [m]
        end: float
            Final z [m]

        Returns
        -------
        np.ndarray
        """
        startidx = self.find_nearest_idx(self.z.val, start)
        endidx = self.find_nearest_idx(self.z.val, end) + 1
        return getattr(self, name).val[startidx:endidx]

    def get_parameter_at_z(self, param: str, z: UnitValue, tol: float = 1e-3) -> float:
        """
        Get a Twiss parameter at a z position.

        Parameters
        ----------
        param: str
            Parameter name
        z: float
            z position [m]
        tol: float, optional
            Use the nearest row if it is within this distance of `z` [m]

        Returns
        -------
        float
            Value at the nearest row within `tol`, otherwise interpolated
        """
        if z in self.z.val:
            idx = list(self.z.val).index(z)
            return getattr(self, param).val[idx]
        else:
            nearest_z = self.find_nearest(self.z.val, z)
            if abs(nearest_z - z) < tol:
                idx = list(self.z.val).index(nearest_z)
                return getattr(self, param).val[idx]
            else:
                return self.interpolate(z=float(z), value=param, index="z")

    def get_parameter_at_element(self, param: str, element_name: str) -> float | None:
        """
        Get a Twiss parameter at a named element.

        Parameters
        ----------
        param: str
            Parameter name
        element_name: str
            Element name

        Returns
        -------
        float | None
            Value at the element's first row, or None if the element is not found
        """
        idx = self._element_row(element_name)
        return None if idx is None else getattr(self, param).val[idx]

    def _element_row(self, element_name: str) -> int | None:
        """First row of `element_name`, or None if it is not in the table."""
        rows = np.flatnonzero(np.asarray(self.element_name.val) == element_name)
        return int(rows[0]) if rows.size else None

    def get_twiss_dict(self, idx: int) -> Dict[str, float]:
        """
        Get every Twiss parameter at a row index.

        Parameters
        ----------
        idx: int
            Row index

        Returns
        -------
        Dict[str, float]
            Parameter name to value
        """
        twissdict = {}
        for param in self.model_fields:
            try:
                twissdict[param] = getattr(self, param).val[idx]
            except Exception:
                pass
        return twissdict

    def get_twiss_at_element(
        self, element_name: str, before: bool = False
    ) -> Dict[str, float] | None:
        """
        Get every Twiss parameter at a named element.

        Parameters
        ----------
        element_name: str
            Element name
        before: bool, optional
            Use the row before the element

        Returns
        -------
        Dict[str, float] | None
            Parameter name to value, or None if the element is not found
        """
        idx = self._element_row(element_name)
        if idx is None:
            return None
        return self.get_twiss_dict(idx - 1 if before else idx)

    def get_twiss_at_z(self, z: float, tol: float = 1e-3) -> Dict[str, float]:
        """
        Get every Twiss parameter at a z position.

        Parameters
        ----------
        z: float
            z position [m]
        tol: float, optional
            Use the nearest row if it is within this distance of `z` [m]

        Returns
        -------
        Dict[str, float]
            Parameter name to value; float parameters are interpolated if no row is within `tol`
        """
        if z in self.z.val:
            idx = list(self.z.val).index(z)
            return self.get_twiss_dict(idx)
        else:
            nearest_z = self.find_nearest(self.z.val, z)
            if abs(nearest_z - z) < tol:
                idx = list(self.z.val).index(nearest_z)
                return self.get_twiss_dict(idx)
            else:
                twissdict = {}
                for param in [
                    k for k in self.model_fields if isinstance(getattr(self, k), twissParameter) and getattr(self, k).dtype == "f"
                ]:
                    twissdict[param] = self.interpolate(z=z, value=param, index="z")
                return twissdict

    if use_matplotlib:

        def plot(self, *args, **kwargs):
            return plot.plot(self, *args, **kwargs)

    def covariance(self, u: np.ndarray, up: np.ndarray) -> float:
        """
        Covariance of two arrays.

        Parameters
        ----------
        u, up: array-like
            Arrays to correlate

        Returns
        -------
        float
        """
        u2 = u - np.mean(u)
        up2 = up - np.mean(up)
        return np.mean(u2 * up2) - np.mean(u2) * np.mean(up2)

    def read_sdds_file(self, filename: str, ascii: bool = False) -> Dict[str, float]:
        """
        Read the raw columns of an ELEGANT SDDS file, without touching this object.

        Broken: ``Twiss.elegant`` no longer has a ``read_sdds_file``.

        Parameters
        ----------
        filename: str
            SDDS filename
        ascii: bool, optional
            Read as ASCII SDDS

        Returns
        -------
        Dict[str, np.ndarray]
            Column name to values
        """
        sddsobject = munch.Munch()
        sddsobject.sddsindex = 0
        sddsobject.elegantTwiss = munch.Munch()
        elegant.read_sdds_file(sddsobject, filename, ascii)
        return sddsobject.elegantTwiss

    def load_directory(
        self,
        directory: str = ".",
        types: Dict = {
            "elegant": ".twi",
            "GPT": "emit.gdf",
            "ASTRA": "Xemit.001",
            "ocelot": "_twiss.oh5",
            "opal": "opal_twiss.h5",
            "cheetah": "_twiss.cheetah.hdf5",
            "xsuite": "_twiss.csv",
            "genesis": ".out.h5",
            "madx": "_twiss.madx.hdf5",
            "bmad": "_twiss.bmad.hdf5",
        },
        preglob: str = "*",
        verbose: bool = False,
        sortkey: str = "z",
    ) -> "twiss":
        """
        Reset this object and load every Twiss file in a directory into it.

        Parameters
        ----------
        directory: str
            Directory to load
        types: Dict[str, str]
            Code name to Twiss filename suffix
        preglob: str
            Glob pattern prepended to each suffix
        verbose: bool, optional
            Print progress
        sortkey: str, optional
            Parameter to sort by

        Returns
        -------
        twiss
            This object
        """
        if verbose:
            print("Directory:", directory)
        self.reset_dicts()
        for code, string in types.items():
            twiss_files = glob.glob(directory + "/" + preglob + string)
            if verbose:
                print(code, preglob + string, [os.path.basename(t) for t in twiss_files])
            if self._which_code(code) is not None and len(twiss_files) > 0:
                self._which_code(code)(self, twiss_files, reset=False)
        self.sort(key=sortkey)
        return self

    @classmethod
    def initialise_directory(cls, *args, **kwargs):
        t = cls()
        t.load_directory(*args, **kwargs)
        return t


def load_directory(
    directory=".",
    types={
        "elegant": ".twi",
        "GPT": "emit.gdf",
        "ASTRA": "Xemit.001",
        "opal": "opal_twiss.h5",
        "ocelot": "_twiss.oh5",
        "cheetah": "_twiss.cheetah.hdf5",
        "xsuite": "_twiss.csv",
        "genesis": ".out.h5",
        "madx": "_twiss.madx.hdf5",
        "bmad": "_twiss.bmad.hdf5",
    },
    preglob="*",
    verbose=False,
    sortkey="z",
) -> twiss:
    """
    Load every Twiss file in a directory into a new :class:`~simba.Modules.Twiss.twiss`.

    Parameters
    ----------
    directory: str
        Directory to load
    types: Dict
        Code name to Twiss filename suffix
    preglob: str
        Glob pattern prepended to each suffix
    verbose: bool
        Print progress
    sortkey: str
        Parameter to sort by

    Returns
    -------
    :class:`~simba.Modules.Twiss.twiss`
    """
    t = twiss()
    if verbose:
        print("Directory:", directory)
    for code, string in types.items():
        twiss_files = glob.glob(directory + "/" + preglob + string)
        if verbose:
            print(code, [os.path.basename(t) for t in twiss_files])
        if t._which_code(code) is not None and len(twiss_files) > 0:
            t._which_code(code)(t, twiss_files, reset=False)
    t.sort(key=sortkey)
    return t


def load_file(filename, *args, **kwargs):
    twissobject = twiss()
    code = twissobject._determine_code(filename)
    if code is not None:
        code(twissobject, filename, reset=False)
    return twissobject
