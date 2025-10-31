"""
Simframe Twiss Module

Twiss module for reading and manipulating twiss parameters from various simulation codes.

Classes:
  - :class:`~SimulationFramework.Modules.Twiss.twiss`: Twiss object class
"""

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
    "ocelot": ocelot.read_ocelot_twiss_files,
    "opal": opal.read_opal_twiss_files,
    "cheetah": cheetah.read_cheetah_twiss_files,
    "xsuite": xsuite.read_xsuite_twiss_files,
    "genesis": genesis.read_genesis_twiss_files,
}

code_signatures = [
    ["elegant", ".twi"],
    ["elegant", ".flr"],
    ["elegant", ".sig"],
    ["GPT", "emit.gdf"],
    ["astra", "Xemit.001"],
    ["ocelot", "_twiss.npz"],
    ["opal", ".stat"],
    ["cheetah", "_twiss.cheetah.hdf5"],
    ["genesis", ".out.h5"],
    ["xsuite", "_twiss.csv"],
]

twiss_defaults = {
    "z": {"name": "z", "unit": "m"},
    "s": {"name": "s", "unit": "m"},
    "t": {"name": "t", "unit": "s"},
    "kinetic_energy": {"name": "kinetic_energy", "unit": "eV"},
    "gamma": {"name": "gamma", "unit": ""},
    "cp": {"name": "cp", "unit": "eV/c"},
    "cp_eV": {"name": "cp_eV", "unit": "eV/c"},
    "p": {"name": "p", "unit": "kg*m/s"},
    "ex": {"name": "ex", "unit": "m-rad"},
    "enx": {"name": "enx", "unit": "m-rad"},
    "ecnx": {"name": "ecnx", "unit": "m-rad"},
    "ey": {"name": "ey", "unit": "m-rad"},
    "eny": {"name": "eny", "unit": "m-rad"},
    "ecny": {"name": "ecny", "unit": "m-rad"},
    "ez": {"name": "ey", "unit": "eV*s"},
    "enz": {"name": "eny", "unit": "eV*s"},
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
    """
    A class to represent a twiss parameter with its name, unit, value, label, and data type.
    This class is used to store and validate twiss parameters in the simulation framework.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """The name of the twiss parameter, e.g., 'z', 'beta_x', etc."""

    unit: str
    """The unit of the twiss parameter, e.g., 'm', 's', 'eV', etc."""

    val: List = []
    """The value of the twiss parameter, stored as a list."""

    label: str = Field(default=None, validate_default=True)
    """A label for the twiss parameter, used for plotting or display purposes."""

    dtype: str = "f"
    """The data type of the twiss parameter, default is 'f' (float)."""

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
    """
    A class to represent the initial twiss parameters of a beam.
    """

    alpha_x: float
    """The alpha parameter in the x-direction."""

    beta_x: float
    """The beta parameter in the x-direction."""

    alpha_y: float
    """The alpha parameter in the y-direction."""

    beta_y: float
    """The beta parameter in the y-direction."""

    ex: float
    """The horizontal emittance."""

    ey: float
    """The vertical emittance."""

    enx: float
    """The normalized horizontal emittance."""

    eny: float
    """The normalized vertical emittance."""

    eta_x: float
    """The horizontal dispersion."""

    eta_xp: float
    """The horizontal dispersion derivative."""

    eta_y: float
    """The vertical dispersion."""

    eta_yp: float
    """The vertical dispersion derivative."""


class twiss(BaseModel):
    """
    A class to represent the twiss parameters of a beam in a simulation framework.
    This class includes various twiss parameters such as position, time, kinetic energy,
    momentum, emittance, beta functions, and dispersion parameters.

    It also provides methods to read twiss data from different simulation codes
    (e.g., ELEGANT, GPT, ASTRA, Ocelot),
    save twiss data to HDF5 files, and perform various operations such as interpolation,
    sorting, and extracting values.
    The class is designed to be flexible and extensible, allowing for the addition of
    new parameters and methods as needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    z: "twissParameter" = None
    """The longitudinal position of the beam in the simulation."""

    s: "twissParameter" = None
    """The longitudinal position of the beam in the simulation."""

    t: "twissParameter" = None
    """The time coordinate of the beam in the simulation."""

    kinetic_energy: "twissParameter" = None
    """The kinetic energy of the beam."""

    gamma: "twissParameter" = None
    """The Lorentz factor of the beam, defined as E/mc^2."""

    cp: "twissParameter" = None
    """The momentum of the beam in eV/c."""

    cp_eV: "twissParameter" = None
    """The momentum of the beam in eV/c, specifically for energy calculations."""

    p: "twissParameter" = None
    """The momentum of the beam in kg*m/s, calculated as cp * q_over_c."""

    enx: "twissParameter" = None
    """The normalized horizontal emittance of the beam."""

    ex: "twissParameter" = None
    """The horizontal emittance of the beam."""

    eny: "twissParameter" = None
    """The normalized vertical emittance of the beam."""

    ey: "twissParameter" = None
    """The vertical emittance of the beam."""

    enz: "twissParameter" = None
    """The normalized longitudinal emittance of the beam, typically in eV*s."""

    ez: "twissParameter" = None
    """The longitudinal emittance of the beam, typically in eV*s."""

    beta_x: "twissParameter" = None
    """The beta function in the x-direction."""

    gamma_x: "twissParameter" = None
    """The twiss gamma function in the x-direction."""

    alpha_x: "twissParameter" = None
    """The alpha function in the x-direction."""

    beta_y: "twissParameter" = None
    """The beta function in the y-direction."""

    gamma_y: "twissParameter" = None
    """The twiss gamma function in the y-direction."""

    alpha_y: "twissParameter" = None
    """The alpha function in the y-direction."""

    beta_z: "twissParameter" = None
    """The beta function in the z-direction."""

    gamma_z: "twissParameter" = None
    """The twiss gamma function in the z-direction."""

    alpha_z: "twissParameter" = None
    """The alpha function in the z-direction."""

    sigma_x: "twissParameter" = None
    """The standard deviation of the beam in the x-direction."""

    sigma_xp: "twissParameter" = None
    """The standard deviation of the beam in the xp-direction."""

    sigma_y: "twissParameter" = None
    """The standard deviation of the beam in the y-direction."""

    sigma_yp: "twissParameter" = None
    """The standard deviation of the beam in the yp-direction."""

    sigma_z: "twissParameter" = None
    """The standard deviation of the beam in the z-direction."""

    sigma_t: "twissParameter" = None
    """The standard deviation of the beam in time."""

    sigma_p: "twissParameter" = None
    """The standard deviation of the beam momentum in kg*m/s."""

    sigma_cp: "twissParameter" = None
    """The standard deviation of the beam momentum in eV/c."""

    mean_x: "twissParameter" = None
    """The mean position of the beam in the x-direction."""

    mean_y: "twissParameter" = None
    """The mean position of the beam in the y-direction."""

    mean_cp: "twissParameter" = None
    """The mean value of the beam momentum in eV/c."""

    mux: "twissParameter" = None
    """The horizontal phase advance of the beam, in units of 2 pi."""

    muy: "twissParameter" = None
    """The vertical phase advance of the beam, in units of 2 pi."""

    eta_x: "twissParameter" = None
    """The horizontal dispersion of the beam."""

    eta_xp: "twissParameter" = None
    """The horizontal dispersion derivative of the beam."""

    eta_y: "twissParameter" = None
    """The vertical dispersion of the beam."""

    eta_yp: "twissParameter" = None
    """The vertical dispersion derivative of the beam."""

    element_name: "twissParameter" = None
    """The name of the element in the simulation."""

    lattice_name: "twissParameter" = None
    """The name of the lattice in the simulation."""

    ecnx: "twissParameter" = None
    """The normalized horizontal emittance of the beam, in m-mrad."""

    ecny: "twissParameter" = None
    """The normalized vertical emittance of the beam, in m-mrad."""

    eta_x_beam: "twissParameter" = None
    """The horizontal dispersion of the beam, specifically for beam parameters."""

    eta_xp_beam: "twissParameter" = None
    """The horizontal dispersion derivative of the beam, specifically for beam parameters."""

    eta_y_beam: "twissParameter" = None
    """The vertical dispersion of the beam, specifically for beam parameters."""

    eta_yp_beam: "twissParameter" = None
    """The vertical dispersion derivative of the beam, specifically for beam parameters."""

    beta_x_beam: "twissParameter" = None
    """The beta function in the x-direction, specifically for beam parameters."""

    beta_y_beam: "twissParameter" = None
    """The beta function in the y-direction, specifically for beam parameters."""

    alpha_x_beam: "twissParameter" = None
    """The alpha function in the x-direction, specifically for beam parameters."""

    alpha_y_beam: "twissParameter" = None
    """The alpha function in the y-direction, specifically for beam parameters."""

    rest_mass: float | None = None
    """The rest mass of the particle, in kg. If None, it will be set to the electron rest mass."""

    codes: Dict = codes
    """A dictionary of functions to read twiss data from different simulation codes."""

    code_signatures: Dict = code_signatures
    """A list of code signatures to identify twiss files from different simulation codes."""

    sddsindex: int = 0
    """An index for SDDS files, used to track the current file being processed."""

    q_over_c: float = constants.e / constants.speed_of_light
    """The charge over the speed of light, used for momentum calculations."""

    E0: float = constants.m_e * constants.speed_of_light**2
    """The rest energy of the particle, in Joules. Default is the electron rest mass energy."""

    E0_eV: float = E0 / constants.elementary_charge
    """The rest energy of the particle, in eV. Default is the electron rest mass energy in eV."""

    elegantTwiss: Dict = {}
    """A dictionary to store ELEGANT twiss data."""

    elegantData: Dict = {}
    """A dictionary to store ELEGANT data."""

    def __init__(
        self,
        rest_mass=None,
    ):
        twiss.rest_mass = rest_mass
        super(
            twiss,
            self,
        ).__init__(
            rest_mass=rest_mass,
        )
        self.reset_dicts()
        self.sddsindex = 0
        self.codes = {
            "elegant": elegant.read_elegant_twiss_files,
            "gpt": gpt.read_gdf_twiss_files,
            "astra": astra.read_astra_twiss_files,
            "ocelot": ocelot.read_ocelot_twiss_files,
            "opal": opal.read_opal_twiss_files,
            "cheetah": cheetah.read_cheetah_twiss_files,
            "xsuite": xsuite.read_xsuite_twiss_files,
            "genesis": genesis.read_genesis_twiss_files,
        }
        self.code_signatures = [
            ["elegant", ".twi"],
            ["elegant", ".flr"],
            ["elegant", ".sig"],
            ["GPT", "emit.gdf"],
            ["astra", "Xemit.001"],
            ["ocelot", "_twiss.npz"],
            ["opal", ".stat"],
            ["cheetah", "_twiss.cheetah.hdf5"],
            ["xsuite", "_twiss.csv"],
            ["genesis", ".out.h5"],
        ]

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
        Set the rest energy of the particle.

        Parameters
        ----------
        value: float
            The rest energy in Joules to set for the particle.
        Returns:
        -----------
        None
        """
        self.E0 = value * constants.speed_of_light**2
        self.E0_eV = self.E0 / constants.elementary_charge

    # def __getitem__(self, key):
    #     if key in super(twiss, self).__getitem__('data') and super(twiss, self).__getitem__('data') is not None:
    #         return self.get(key)
    #     else:
    #         return super(twiss, self).__getitem__(key)

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
            return ocelot.read_ocelot_twiss_files(self, *args, **kwargs)

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
        Get the value of a twiss parameter by its key.

        Parameters
        ----------
        key: str
            The key of the twiss parameter to retrieve, e.g., 'z', 'beta_x', etc.

        Returns:
        -----------
        twissParameter:
            The value of the twiss parameter associated with the given key.
        """
        return getattr(self, key)

    def find_nearest_idx(self, array: List, value: float) -> int:
        """
        Find the index of the nearest value in a sorted array.

        Parameters
        ----------
        array: List
            A sorted array to search within.
        value: float
            The value to find the nearest index for in the array.

        Returns
        -------
        int:
            The index of the nearest value in the array.
            If the value is exactly equal to an element, it returns that index.
            If the value is less than the first element, it returns 0.
            If the value is greater than the last element, it returns the last index.
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
        Find the nearest value in a sorted array to a given value.

        Parameters
        ----------
        array: List
            A sorted array to search within.
        value: float
            The value to find the nearest element for in the array.

        Returns
        -------
        float:
            The value in the array
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
        """
        Reset the twiss parameters to their initial state.
        This method initializes all twiss parameters to their default values
        and clears the elegantTwiss dictionary.
        This is useful for starting fresh with a new set of twiss parameters or when reloading data.
        """
        self.sddsindex = 0
        for name in twiss.model_fields:
            if name in list(twiss_defaults.keys()):
                setattr(self, name, twissParameter(**twiss_defaults[name]))
        self.elegantTwiss = {}

    def sort(self, key: str = "s", reverse: bool = False) -> None:
        """
        Sort the twiss parameters based on a specified key.
        This method sorts the twiss parameters in ascending order by default,
        or in descending order if `reverse` is set to True.
        The sorting is done based on the values of the specified key,
        which should be one of the twiss parameters (e.g., 'z', 'beta_x', etc.).
        If the key is not found, it raises an AttributeError.

        Parameters
        ----------
        key: str
            The key by which to sort all the Twiss parameters
        reverse: bool, optional
            Reverse the Twiss parameter arrays
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
        Append data to a specified twiss parameter array.

        Parameters
        ----------
        array: str
            Name of existing Twiss parameter array
        data: List | np.ndarray
            Data to append
        """
        newval = UnitValue(
            np.concatenate([getattr(self, array), data]),
            units=getattr(self, array).units,
        )
        setattr(self, array, newval)

    def _which_code(self, name: str) -> Callable | None:
        """
        Determine the function associated with a specific simulation code name.

        Parameters
        ----------
        name: str
            The name of the code which produced the Twiss file

        Returns
        -------
        callable | None:
            The function associated with the specified simulation code name, or None if not found.
        """
        if name.lower() in self.codes.keys():
            return self.codes[name.lower()]
        return None

    def _determine_code(self, filename: str) -> Callable | None:
        """
        Determine the simulation code based on the filename.

        Parameters
        ----------
        filename: str
            Based on the filename, determine the code which produced it.

        Returns
        -------
        callable | None:
            The function associated with the simulation code if found, otherwise None.
        """
        for k, v in self.code_signatures:
            cutl = -len(v)
            if v == filename[cutl:]:
                return self.codes[k]
        return None

    def interpolate(self, z=None, value="z", index="z") -> float:
        """
        Interpolate a value at a given z position based on the twiss parameters.

        Parameters
        ----------
        z: float or None, optional
        value: str, optional
        index: str, optional

        Returns
        -------
        float:
            The interpolated value at the specified z position.
            If z is None, it returns the interpolated value for the entire range.
            If z is greater than the maximum value in the index, it returns a large number (10^6).
            Otherwise, it returns the interpolated value at the specified z position.
        """
        if z is None:
            return np.interp(z, getattr(self, index), getattr(self, value))
        else:
            if z > max(self[index]):
                return 10**6
            else:
                return float(np.interp(z, self[index], self[value]))

    def extract_values(self, name: str, start: float, end: float) -> np.ndarray:
        """
        Extract values from a specified twiss parameter array between two z positions.

        Parameters
        ----------
        name: str
            Name of Twiss parameter
        start: float
            Initial z position
        end: float
            Final z position

        Returns
        -------
        np.ndarray:
            An array of values from the specified twiss parameter array between the start and end z positions.
        """
        startidx = self.find_nearest_idx(getattr(self, "z"), start)
        endidx = self.find_nearest_idx(getattr(self, "z"), end) + 1
        return getattr(self, name)[startidx:endidx]

    def get_parameter_at_z(self, param: str, z: UnitValue, tol: float = 1e-3) -> float:
        """
        Get the value of a twiss parameter at a specific z position.

        Parameters
        ----------
        param: str
            The name of the Twiss parameter
        z: float
            The z position of interest
        tol: float, optional
            The z-position tolerance

        Returns
        -------
        float:
            The value of the specified twiss parameter at the given z position.
            If z is exactly in the list of z positions, it returns the corresponding value.
            If z is not found, it finds the nearest z position and checks if it's within the tolerance.
            If it is, it returns the corresponding value; otherwise, it interpolates the value.
        """
        if z in self.z.val:
            idx = list(self.z.val).index(z)
            return getattr(self, param)[idx]
        else:
            nearest_z = self.find_nearest(self.z, z)
            if abs(nearest_z - z) < tol:
                idx = list(self.z.val).index(nearest_z)
                return getattr(self, param).val[idx]
            else:
                # print('interpolate!', z, self['z'])
                return self.interpolate(z=z.val, value=param, index="z")

    def get_parameter_at_element(self, param: str, element_name: str) -> float | None:
        """
        Get the value of a twiss parameter at a specific element name.

        Parameters
        ----------
        param: str
            The Twiss parameter of interest
        element_name: str
            The element name

        Returns
        -------
        float | None:
            The value of the specified twiss parameter at the given element name.
            If the element name is found, it returns the corresponding value; otherwise, it returns None.
        """
        if element_name in self.element_name:
            idx = list(self.element_name).index(element_name)
            return self[param][idx]
        return None

    def get_twiss_dict(self, idx: int) -> Dict[str, float]:
        """
        Get a dictionary of twiss parameters at a specific index.

        Parameters
        ----------
        idx: int
            The index in the Twiss parameter list

        Returns
        -------
        Dict[str, float]:
            A dictionary containing the twiss parameters at the specified index.
            The keys are the parameter names, and the values are the corresponding values at that index.
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
        Get the twiss parameters at a specific element name.

        Parameters
        ----------
        element_name: str
            The name of the element
        before:
            Get the parameters before the specified element

        Returns
        -------
        Dict[str, float] | None:
            A dictionary of twiss parameters at the specified element name.
            If the element name is found, it returns the corresponding twiss parameters;
            otherwise, it returns None.
            If `before` is True, it returns the parameters before the specified element.
        """
        if element_name in self.element_name:
            idx = list(self.element_name).index(element_name)
            if before:
                idx = idx - 1
            return self.get_twiss_dict(idx)
        return None

    def get_twiss_at_z(self, z: float, tol: float = 1e-3) -> Dict[str, float]:
        """
        Get the twiss parameters at a specific z position.

        Parameters
        ----------
        z: float
            The z-position of interest
        tol: float, optional
            Tolerance on the z-position

        Returns
        -------
        Dict[str, float]:
            A dictionary of twiss parameters at the specified z position.
            If z is exactly in the list of z positions, it returns the corresponding twiss parameters.
            If z is not found, it finds the nearest z position and checks if it's within the tolerance.
            If it is, it returns the corresponding twiss parameters; otherwise, it interpolates the values.
        """
        if z in self.z:
            idx = list(self.z).index(z)
            return self.get_twiss_dict(idx)
        else:
            nearest_z = self.find_nearest(self.z, z)
            if abs(nearest_z - z) < tol:
                idx = list(self.z).index(nearest_z)
                return self.get_twiss_dict(idx)
            else:
                twissdict = {}
                for param in [
                    k for k in self.model_fields if getattr(self, k).dtype == "f"
                ]:
                    twissdict[param] = self.interpolate(z=z, value=param, index="z")
                return twissdict

    if use_matplotlib:

        def plot(self, *args, **kwargs):
            return plot.plot(self, *args, **kwargs)

    def covariance(self, u: np.ndarray, up: np.ndarray) -> float:
        """
        Calculate the covariance between two twiss parameters.

        Parameters
        ----------
        u: array-like
            First Twiss parameter set
        up: array-like
            Second Twiss parameter set

        Returns
        -------
        float:
            The covariance between the two twiss parameters.
            The covariance is calculated as the mean of the product of the deviations from their means.
            If the input arrays are empty, it returns NaN.
        """
        u2 = u - np.mean(u)
        up2 = up - np.mean(up)
        return np.mean(u2 * up2) - np.mean(u2) * np.mean(up2)

    def read_sdds_file(self, filename: str, ascii: bool = False) -> Dict[str, float]:
        """
        Read an SDDS file and extract the twiss parameters.
        #TODO deprecated????

        Parameters
        ----------
        filename: str
            SDDS filename
        ascii: bool, optional
            Convert to ascii

        Returns
        -------
        Dict[str, float]:
            A dictionary containing the twiss parameters extracted from the SDDS file.
            The dictionary is stored in the `elegantTwiss` attribute of the twiss object.
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
            "ocelot": "_twiss.npz",
            "opal": ".stat",
            "cheetah": "_twiss.cheetah.hdf5",
            "xsuite": "_twiss.csv",
            "genesis": ".out.h5",
        },
        preglob: str = "*",
        verbose: bool = False,
        sortkey: str = "z",
    ) -> "twiss":
        """
        Load twiss files from a specified directory based on the provided types and preglob pattern.

        Parameters
        ----------
        directory: str
            The directory to load
        types: Dict[str, str]
            Keys for codes and the Twiss file extensions that they produce
        preglob: str
            Territorial globbing
        verbose: bool, optional
            Print out status of loading
        sortkey: str, optional
            Sort Twiss data by the key provided

        Returns
        -------
        twiss:
            The twiss object with the loaded twiss parameters.
            The method reads twiss files from the specified directory, processes them based on the types,
            and sorts the parameters based on the specified sortkey.
        """
        if verbose:
            print("Directory:", directory)
        self.reset_dicts()
        for code, string in types.items():
            twiss_files = glob.glob(directory + "/" + preglob + string)
            if verbose:
                print(code, [os.path.basename(t) for t in twiss_files])
            if self._which_code(code) is not None and len(twiss_files) > 0:
                self._which_code(code)(self, twiss_files, reset=False)
        self.sort(key=sortkey)
        return self

    @classmethod
    def initialise_directory(cls, *args, **kwargs):
        t = cls()
        t.load_directory(*args, **kwargs)
        return t

    # @property
    # def cp_eV(self):
    #     return self['cp']
    # @property
    # def cp_MeV(self):
    #     return self['cp'] / 1e6


def load_directory(
    directory=".",
    types={
        "elegant": ".twi",
        "GPT": "emit.gdf",
        "ASTRA": "Xemit.001",
        "ocelot": "_twiss.npz",
        "opal": ".stat",
        "cheetah": "_twiss.cheetah.hdf5",
        "xsuite": "_twiss.csv",
        "genesis": ".out.h5",
    },
    preglob="*",
    verbose=False,
    sortkey="z",
) -> twiss:
    """
    Load in all Twiss output files from a directory and create a
    :class:`~SimulationFramework.Modules.Twiss.twiss` object.

    Parameters
    ----------
    directory: str
        Directory from which to load the files
    types: Dict
        Codes and their file extensions
    preglob: str
        String for file pattern matching
    verbose: bool
        If true, print progress
    sortkey: str
        Key by which to sort Twiss parameters

    Returns
    -------
    :class:`~SimulationFramework.Modules.Twiss.twiss`
        A new `twiss` object.
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
