"""
SIMBA Objects Module

Various objects and functions to handle simulation lattices, commands, and elements.

Classes:
    - :class:`~simba.Framework_objects.RunSetup`: Defines simulation run settings, allowing for single runs, element scans or jitter/error studies.

    - :class:`~simba.Framework_objects.FrameworkObject`: Base class for generic objects in SIMBA, including lattice elements and simulation code commands.

    - :class:`~simba.Framework_objects.FrameworkLattice`: Base class for simulation lattices, consisting of a line of `LAURA` elements.

    - :class:`~simba.Framework_objects.FrameworkCounter`: Used for counting elements of the same type in ASTRA and CSRTrack

    - :class:`~simba.Framework_objects.FrameworkGroup`: Used for grouping elements together and controlling them all simultaneously.

    - :class:`~simba.Framework_objects.ElementGroup`: Subclass of :class:`~simba.Framework_objects.FrameworkGroup` for grouping elements.
    # TODO is this ever used?

    - :class:`~simba.Framework_objects.R56Group`: Subclass of :class:`~simba.Framework_objects.FrameworkGroup` for grouping elements with an R56.
    # TODO is this ever used?

    - :class:`~simba.Framework_objects.Chicane`: Subclass of :class:`~simba.Framework_objects.FrameworkGroup` for a 4-dipole bunch compressor chicane.

    - :class:`~simba.Framework_objects.GetGrids`: Used for determining the appropriate number of space charge grids given a number of particles.
"""

from simba._compat import DeprecatedMethodAliases
import os
import subprocess
from warnings import warn
import stat
import yaml
from copy import deepcopy
import time

from laura import LAURA
from laura.models.element_list import SectionLattice, ElementList
from laura.models.physical import Position
from laura.models.element import PhysicalBaseElement, Quadrupole, Sextupole, Octupole
from laura.translator.converters.section import SectionLatticeTranslator

from .modules.merge_two_dicts import merge_two_dicts
from .modules.math_parser import MathParser
from .framework_settings import FrameworkSettings
from .framework_helper_functions import expand_substitution
from .modules.fields import FieldMap
from .modules import beams as rbf
from .codes import executables as exes
from .modules.constants import speed_of_light

try:
    import numpy as np
except ImportError:
    np = None
from pydantic import (
    BaseModel,
    field_validator,
    PositiveInt,
    computed_field,
    ConfigDict,
    Field,
)
from typing import (
    Dict,
    List,
    Any,
)

if os.name == "nt":
    # from .Modules.symmlinks import has_symlink_privilege
    def has_symlink_privilege():
        return False

else:

    def has_symlink_privilege():
        return True


with open(
    os.path.dirname(os.path.abspath(__file__)) + "/codes/type_conversion_rules.yaml",
    "r",
) as infile:
    type_conversion_rules = yaml.safe_load(infile)
    type_conversion_rules_elegant = type_conversion_rules["elegant"]
    type_conversion_rules_names = type_conversion_rules["name"]
    type_conversion_rules_opal = type_conversion_rules["opal"]

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/codes/elegant/commands_Elegant.yaml",
    "r",
) as infile:
    commandkeywords_elegant = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/codes/opal/commands_Opal.yaml",
    "r",
) as infile:
    commandkeywords_opal = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/codes/genesis/commands_Genesis.yaml",
    "r",
) as infile:
    commandkeywords_genesis = yaml.safe_load(infile)

commandkeywords = commandkeywords_elegant | commandkeywords_opal
commandkeywords = commandkeywords | commandkeywords_genesis

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/elementkeywords.yaml", "r"
) as infile:
    elementkeywords = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/codes/elegant/keyword_conversion_rules_elegant.yaml",
    "r",
) as infile:
    keyword_conversion_rules_elegant = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/codes/elegant/elements_Elegant.yaml",
    "r",
) as infile:
    elements_elegant = yaml.safe_load(infile)


class RunSetup(DeprecatedMethodAliases, object):
    """
    Class defining settings for simulations that include multiple runs
    such as error studies or parameter scans.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "loadElementErrors": "load_element_errors",
        "setElementScan": "set_element_scan",
        "setNRuns": "set_n_runs",
        "setSeedValue": "set_seed_value",
    }

    def __init__(self):
        # define the number of runs and the random number seed
        self.nruns = 1
        self.seed = 0

        # init errorElement and elementScan settings as None
        self.element_errors = None
        self.elementScan = None

    def set_n_runs(self, nruns: int | float) -> None:
        """
        Sets the number of simulation runs to a new value.

        Parameters
        -----------
        nruns : int or float
            The number of runs to set. If a float is passed, it will be converted to an integer.

        Raises
        ------
        TypeError
            If `nruns` is not an integer or float.
        """
        # enforce integer argument type
        if isinstance(nruns, (int, float)):
            self.nruns = int(nruns)
        else:
            raise TypeError(
                "Argument nruns passed to runSetup instance must be an integer"
            )

    def set_seed_value(self, seed: int | float) -> None:
        """
        Sets the random number seed to a new value for all lattice objects

        Parameters
        -----------
        seed : int or float
            The random number seed to set. If a float is passed, it will be converted to an integer.

        Raises
        ------
        TypeError
            If `seed` is not an integer or float.
        """
        # enforce integer argument type
        if isinstance(seed, (int, float)):
            self.seed = int(seed)
        else:
            raise TypeError("Argument seed passed to runSetup must be an integer")

    def load_element_errors(self, file: str | dict) -> None:
        """
        Load error definitions from a file or dictionary and assign them to the elementErrors attribute.
        This method can handle both a YAML file and a dictionary containing error definitions.

        Parameters
        -----------
        file: str or dict
            - str: Path to a YAML file containing error definitions.
            - dict: A dictionary containing error definitions.
        """
        # load error definitions from markup file
        error_setup = None
        if isinstance(file, str) and (".yaml" in file):
            with open(file, "r") as inputfile:
                error_setup = dict(yaml.safe_load(inputfile))
        # define errors from dictionary
        elif isinstance(file, dict):
            error_setup = file
        else:
            warn("error_setup must be a str or dict")

        if error_setup is not None and "elements" in list(error_setup.keys()):
            # assign the element error definitions
            self.element_errors = error_setup["elements"]
            self.elementScan = None

            # set the number of runs and random number seed, if available
            if "nruns" in error_setup:
                self.set_n_runs(error_setup["nruns"])
            if "seed" in error_setup:
                self.set_seed_value(error_setup["seed"])

    def set_element_scan(
        self,
        name: str,
        item: str,
        scanrange: list | tuple | np.ndarray,
        multiplicative: bool = False,
    ) -> None:
        """
        Define a parameter scan for a single parameter of a given machine element

        Parameters
        -----------
        name : str
            Name of the machine element to be scanned.
        item : str
            Name of the item (parameter) to be scanned within the machine element.
        scanrange : list or tuple or np.ndarray
            A list or tuple containing two floats, representing the minimum and maximum values of the scan range.
        multiplicative : bool, optional
            If True, the scan will be multiplicative; otherwise, it will be additive. Default is False.
        """
        if not (isinstance(name, str) and isinstance(item, str)):
            raise TypeError(
                "Machine element name and item (parameter) must be defined as strings"
            )

        if (
            isinstance(scanrange, (list, tuple, np.ndarray))
            and (len(scanrange) == 2)
            and all([isinstance(x, (float, int)) for x in scanrange])
        ):
            minval, maxval = scanrange
        else:
            raise TypeError("Scan range (min. and max.) must be defined as floats")

        if not isinstance(multiplicative, bool):
            raise ValueError(
                "Argument multiplicative passed to runSetup.setElementScan must be a boolean"
            )

        # if no type errors were raised, build an assign a dictionary
        self.elementScan = {
            "name": name,
            "item": item,
            "min": minval,
            "max": maxval,
            "multiplicative": multiplicative,
        }
        self.element_errors = None


class FrameworkObject(DeprecatedMethodAliases, BaseModel):
    """
    Class defining a framework object, which is the base class for all elements
    in a simulation lattice. It provides methods to add properties, validate parameters,
    and handle various simulation-specific functionalities.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "change_Parameter": "change_parameter",
    }

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True,
    )

    objectname: str = Field(alias="name")
    """Name of the object, used as a unique identifier in the simulation."""

    objecttype: str = Field(alias="type")
    """Type of the object, which determines its behavior and properties in the simulation."""

    objectdefaults: Dict = {}
    """Default values for the object's properties, used when no specific value is provided."""

    allowedkeywords: List | Dict = {}
    """List of allowed keywords for the object, which defines what properties can be set."""

    global_parameters: Dict = {}
    """Global parameters to be cascaded through all objects."""

    def model_post_init(self, __context):
        extra_fields = {
            k: v for k, v in self.model_dump().items()
            if k not in self.__annotations__
        }
        for k, v in extra_fields.items():
            setattr(self, k, v)
        if self.objecttype in commandkeywords:
            self.allowedkeywords = commandkeywords[self.objecttype]
        elif self.objecttype in elementkeywords:
            self.allowedkeywords = elementkeywords[self.objecttype]["keywords"] | elementkeywords["common"]["keywords"]
            if "framework_keywords" in elementkeywords[self.objecttype]:
                self.allowedkeywords = merge_two_dicts(
                    self.allowedkeywords,
                    elementkeywords[self.objecttype]["framework_keywords"],
                )
        else:
            raise NameError(f"Unknown type = {self.objecttype}")
        self.allowedkeywords = [x.lower() for x in self.allowedkeywords]
        # for key, value in list(kwargs.items()):
        #     self.add_property(key, value)

    @field_validator("objectname", mode="before")
    @classmethod
    def validate_objectname(cls, value: str) -> str:
        """Validate the objectname to ensure it is a string."""
        if not isinstance(value, str):
            raise ValueError("objectname must be a string.")
        return value

    @field_validator("objecttype", mode="before")
    @classmethod
    def validate_objecttype(cls, value: str) -> str:
        """Validate the objecttype to ensure it is a string."""
        if not isinstance(value, str):
            raise ValueError("objecttype must be a string.")
        return value

    # def __setattr__(self, name, value):
    #     # Let Pydantic set known fields normally
    #     if name in frameworkObject.model_fields:
    #         return super().__setattr__(name, value)
    #     object.__setattr__(self, name, value)

    def change_parameter(self, key: str, value: Any) -> None:
        """
        Change a parameter of the object by setting an attribute.

        Parameters
        ----------
        key: str
            The name of the parameter to change.
        value: Any
            The new value to set for the parameter.
        """
        setattr(self, key, value)

    def add_property(self, key: str, value: Any) -> None:
        """
        Add a property to the object by setting an attribute if the key is allowed.

        Parameters
        ----------
        key: str
            The name of the property to add.
        value: Any
            The value to set for the property.
        """
        key = key.lower()
        if key in self.allowedkeywords:
            try:
                setattr(self, key, value)
            except Exception as e:
                warn(f"add_property error: ({self.objecttype} [{key}]: {e}")

    def add_properties(self, **keyvalues: dict) -> None:
        """
        Add multiple properties to the object by setting attributes for each key-value pair.

        Parameters
        ----------
        **keyvalues: dict
            A dictionary of key-value pairs where keys are property names
            and values are the corresponding values to set.
        """
        for key, value in keyvalues.items():
            key = key.lower()
            if key in self.allowedkeywords:
                try:
                    setattr(self, key, value)
                except Exception as e:
                    warn(f"add_properties error: ({self.objecttype} [{key}]: {e}")

    def add_default(self, key: str, value: Any) -> None:
        """
        Add a default value for a property of the object, updating `objectdefaults`.

        Parameters
        ----------
        key: str
            The name of the property to set a default value for.
        value: Any
            The name of the property to set a default value for and the value to set.
        """
        self.objectdefaults[key] = value

    @property
    def parameters(self) -> list:
        """
        Returns a list of all parameters (keys) of the object.

        Returns
        -------
        list
            A list of keys representing the parameters of the object.
        """
        return list(self.keys())

    @property
    def objectproperties(self):
        """
        Returns a dictionary of the object's properties, excluding disallowed keywords.

        Returns
        -------
        frameworkObject
            The object itself, allowing for method chaining.
        """
        cls = self.__class__
        return {key: getattr(self, key) for key in cls.model_fields} | {key: getattr(self, key) for key in cls.model_computed_fields}

    # def __getitem__(self, key):
    #     lkey = key.lower()
    #     defaults = self.objectdefaults
    #     if lkey in defaults:
    #         try:
    #             return getattr(self, lkey)
    #         except Exception:
    #             return defaults[lkey]
    #     else:
    #         try:
    #             return getattr(self, lkey)
    #         except Exception:
    #             try:
    #                 return getattr(self, key)
    #             except Exception:
    #                 return None

    def __repr__(self):
        string = ""
        for k in self.model_fields_set:
            if k in self.allowedkeywords:
                string += f"{k} = {getattr(self, k)}" + "\n"
        return string


class FrameworkLattice(DeprecatedMethodAliases, BaseModel):
    """
    Class defining a framework lattice object, which contains all elements and groups
    of elements in a simulation lattice. It also contains methods to manipulate and
    retrieve information about the elements and groups, as well as methods to run
    simulations and process results.

    See :ref:`getting-started` and :ref:`loading-a-lattice`.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "allElements": "all_elements",
        "createDrifts": "create_drifts",
        "elementObjects": "element_objects",
        "endObject": "end_object",
        "findS": "find_s",
        "getElement": "get_element",
        "getElementType": "get_element_type",
        "getElems": "get_elems",
        "getInitialTwiss": "get_initial_twiss",
        "getNames": "get_names",
        "getSNames": "get_s_names",
        "getSNamesElems": "get_s_names_elems",
        "getSValues": "get_s_values",
        "getZNamesElems": "get_z_names_elems",
        "getZValues": "get_z_values",
        "globalSettings": "global_settings",
        "groupObjects": "group_objects",
        "groupSettings": "group_settings",
        "postProcess": "post_process",
        "preProcess": "pre_process",
        "runSettings": "run_settings",
        "setElementType": "set_element_type",
        "startObject": "start_object",
        "updateRunSettings": "update_run_settings",
    }

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    name: str
    """Name of the lattice, used as a prefix for output files and commands."""

    objectname: str | None = ""
    """Name of the lattice, used as a prefix for output files and commands."""

    objecttype: str | None = ""
    """Type of the lattice, used as a prefix for output files and commands."""

    file_block: Dict
    """File block containing input and output settings for the lattice."""

    machine: LAURA
    """LAURA model of the lattice"""

    element_objects: Dict
    """Dictionary of element objects, where keys are element names and values are element instances."""

    group_objects: Dict
    """Dictionary of group objects, where keys are group names and values are group instances."""

    run_settings: RunSetup
    """Run settings for the lattice, including number of runs and random seed."""

    settings: FrameworkSettings
    """Instance of :class:`~simba.Framework_Settings.FrameworkSettings`"""

    executables: exes.Executables
    """Executable commands for running simulations, defined in the Executables class.
    See :class:`~simba.Framework.Codes.Executables.Executables` for more details."""

    global_parameters: Dict
    """Global parameters for the lattice, including master subdirectory and other configuration settings."""

    global_settings: Dict
    """Global settings for the lattice."""

    allow_negative_drifts: bool = False
    """If True, allows negative drifts in the lattice."""

    _lsc_enable: bool = True
    """Flag to enable LSC drifts in the lattice."""

    _csr_enable: bool = True
    """Flag to enable CSR drifts in the lattice."""

    _lsc_bins: int = 20
    """Number of bins for LSC drifts."""

    _csr_bins: int = 20
    """Number of bins for CSR calculations"""

    lsc_high_frequency_cutoff_start: float = -1
    """Spatial frequency at which smoothing filter begins. If not positive, no frequency filter smoothing is done. 
    See `Elegant manual LSC drift`_
    
    .. _Elegant manual LSC drift: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu168.html#x179-18000010.58"""

    lsc_high_frequency_cutoff_end: float = -1
    """Spatial frequency at which smoothing filter is 0. See `Elegant manual LSC drift`_"""

    lsc_low_frequency_cutoff_start: float = -1
    """Highest spatial frequency at which low-frequency cutoff filter is zero. See `Elegant manual LSC drift`_"""

    lsc_low_frequency_cutoff_end: float = -1
    """Lowest spatial frequency at which low-frequency cutoff filter is 1. See `Elegant manual LSC drift`_"""

    sample_interval: int = 1
    """Sample interval for downsampling particles, in units of 2**(3*sample_interval)"""

    global_settings: Dict = {"charge": None}
    """Global settings for the lattice, including charge and other parameters."""

    group_settings: Dict = {}
    """Group settings for the lattice, including group-specific parameters."""

    all_elements: List = []
    """List of all element names in the lattice."""

    initial_twiss: Dict = {}
    """Initial Twiss parameters for the lattice, used for tracking and analysis."""

    _section: SectionLatticeTranslator = None
    """LAURA SectionLatticeTranslator object"""

    remote_setup: Dict = {}
    """Dictionary containing parameters for running executables remotely."""

    files: List = []
    """List of all files needed to run the lattice."""

    def model_post_init(self, __context):
        # super().model_post_init(__context)
        for key, value in list(self.element_objects.items()):
            setattr(self, key, value)
        self.all_elements = list(self.element_objects.keys())
        self.objectname = self.name
        self.remote_setup = {}
        self.files = []

        # define settings for simulations with multiple runs
        self.update_run_settings(self.run_settings)
        if not isinstance(self.file_block, dict):
            raise ValueError("file_block must be a dictionary.")
        if "groups" in self.file_block:
            if self.file_block["groups"] is not None:
                self.group_settings = self.file_block["groups"]
        if "input" in self.file_block:
            if "sample_interval" in self.file_block["input"]:
                self.sample_interval = self.file_block["input"]["sample_interval"]
        else:
            self.file_block.update({"input": {}})
        self.global_settings = self.settings["global"]
        self.update_groups()

    # @field_validator("file_block", mode="before")
    # @classmethod
    # def validate_file_block(cls, value: Dict) -> Dict:
    #     """
    #     Validate the file_block dictionary to ensure it has the required structure.
    #     This method checks if the file_block is a dictionary and contains the necessary keys.
    #
    #     Raises
    #     ------
    #     ValueError
    #         If the file_block is not a dictionary or does not contain the required keys.
    #     """
    #     if not isinstance(value, dict):
    #         raise ValueError("file_block must be a dictionary.")
    #     if "groups" in value:
    #         if value["groups"] is not None:
    #             cls.groupSettings = value["groups"]
    #     if "input" in value:
    #         if "sample_interval" in value["input"]:
    #             cls.sample_interval = value["input"]["sample_interval"]
    #     return value
    #
    # @field_validator("settings", mode="before")
    # @classmethod
    # def validate_settings(cls, value: Dict) -> Dict:
    #     """
    #     Validate the settings dictionary to ensure it has the required structure.
    #     This method checks if the settings is a dictionary and contains the necessary keys.
    #
    #     Raises
    #     ------
    #     ValueError
    #         If the settings is not a dictionary or does not contain the required keys.
    #
    #     """
    #     if not isinstance(value, dict):
    #         raise ValueError("settings must be a dictionary.")
    #     if "global" in value:
    #         if value["global"] is not None:
    #             cls.globalSettings = value["global"]
    #     return value

    def __setattr__(self, name, value):
        # Let Pydantic set known fields normally
        if name in FrameworkLattice.model_fields:
            return super().__setattr__(name, value)
        object.__setattr__(self, name, value)

    def insert_element(self, index: int, element: "PhysicalBaseElement") -> None:
        """
        Insert an element at a specific index in the elements dictionary.

        Parameters
        ----------
        index: int
            The index at which to insert the element.
        element: Element
            The element to insert into the elements dictionary.

        """
        for i, _ in enumerate(range(len(self.elements))):
            k, v = self.elements.popitem(False)
            self.elements[element.name if i == index else k] = element

    @property
    def csr_enable(self) -> bool:
        """
        Property to get or set the CSR enable flag.
        """
        return self._csr_enable

    @csr_enable.setter
    def csr_enable(self, csr: bool) -> None:
        self._csr_enable = csr
        self.section.csr_enable = csr
        for elem in self.element_objects.values():
            try:
                elem.simulation.csr_enable = csr
            except ValueError:
                pass
            except AttributeError:
                pass

    @property
    def csr_bins(self) -> int:
        """
        Property to get or set the number of bins for CSR calculations.
        """
        return self._csr_bins

    @csr_bins.setter
    def csr_bins(self, csr: int) -> None:
        self._csr_bins = csr
        for elem in self.element_objects.values():
            try:
                elem.simulation.csr_bins = csr
            except ValueError:
                pass
            except AttributeError:
                pass

    @property
    def lsc_enable(self) -> bool:
        """
        Property to get or set the LSC enable flag.
        """
        return self._lsc_enable

    @lsc_enable.setter
    def lsc_enable(self, lsc: bool) -> None:
        self._lsc_enable = lsc
        self.section.lsc_enable = lsc
        for elem in self.element_objects.values():
            try:
                elem.simulation.lsc_enable = lsc
            except ValueError:
                pass
            except AttributeError:
                pass

    @property
    def lsc_bins(self) -> int:
        """
        Property to get or set the number of bins for LSC calculations.
        """
        return self._lsc_bins

    @lsc_bins.setter
    def lsc_bins(self, lsc: int) -> None:
        self._lsc_bins = lsc
        self.section.lsc_bins = lsc
        for elem in self.element_objects.values():
            try:
                elem.simulation.lsc_bins = lsc
            except ValueError:
                pass
            except AttributeError:
                pass

    def get_prefix(self) -> str:
        """
        Get the prefix from the input file block.

        Returns
        -------
        str
            The prefix string used in the input file block.
        """
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "prefix" not in self.file_block["input"]:
            self.file_block["input"]["prefix"] = self.global_parameters["master_subdir"] + "/"
        return self.file_block["input"]["prefix"]

    def set_prefix(self, prefix: str) -> None:
        """
        Set the prefix for the input file block.

        Parameters
        ----------
        prefix: str
            The prefix string used in the input file block.
        """
        if not hasattr(self, "file_block") or self.file_block is None:
            self.file_block = {}
        if "input" not in self.file_block or self.file_block["input"] is None:
            self.file_block["input"] = {}
        self.file_block["input"]["prefix"] = prefix

    @computed_field
    @property
    def prefix(self) -> str:
        return self.get_prefix()

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self.set_prefix(prefix)

    def read_input_file(self, prefix, particle_definition, read_file=True):
        filepath = ""
        HDF5filename = prefix + particle_definition + ".openpmd.hdf5"
        if "$" in particle_definition:
            if os.path.isfile(expand_substitution(self, particle_definition + ".openpmd.hdf5")):
                filepath = expand_substitution(self, particle_definition + ".openpmd.hdf5")
            else:
                filepath = expand_substitution(self, particle_definition + ".openpmd.hdf5")
        elif os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        elif os.path.isfile(self.global_parameters["master_subdir"] + "/" + HDF5filename):
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        if os.path.isfile(filepath):
            if read_file:
                rbf.openpmd.read_openpmd_beam_file(
                    self.global_parameters["beam"],
                    os.path.abspath(filepath),
                )
                self.get_initial_twiss()
            return filepath
        HDF5filename = prefix + particle_definition + ".hdf5"
        if "$" in particle_definition:
            if os.path.isfile(expand_substitution(self, particle_definition + ".hdf5")):
                filepath = expand_substitution(self, particle_definition + ".hdf5")
            else:
                filepath = expand_substitution(self, particle_definition + ".hdf5")
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        elif os.path.isfile(self.global_parameters["master_subdir"] + "/" + HDF5filename):
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        if os.path.isfile(filepath):
            if read_file:
                rbf.hdf5.read_hdf5_beam_file(
                    self.global_parameters["beam"],
                    os.path.abspath(filepath),
                )
            return filepath
        raise Exception(
            f'HDF5 input file {expand_substitution(self, prefix + particle_definition)}.[openpmd.].hdf5 does not exist!')

    def update_groups(self) -> None:
        """
        Update the group objects in the lattice with their settings.
        """
        for g in list(self.group_settings.keys()):
            if g in self.group_objects:
                setattr(self, g, self.group_objects[g])
                if self.group_settings[g] is not None:
                    self.group_objects[g].update(**self.group_settings[g])

    def get_element(self, element: str, param: str = None) -> dict | PhysicalBaseElement:
        """
        Get an element or group object by its name and optionally a specific parameter.
        This method checks if the element exists in the allElements dictionary or in the groupObjects dictionary.
        If the element exists, it returns the element object or the specified parameter of the element.

        Parameters
        ----------
        element: str
        param: str, optional
            The parameter to retrieve from the element object. If None, returns the entire element object.

        Returns
        -------
        dict | :class:`~laura.models.element.Element`
            The element object or the specified parameter of the element.
        """
        if element in self.elements:
            if param is not None:
                return getattr(self.element_objects[element], param.lower())
            else:
                return self.element_objects[element]
        elif element in list(self.group_objects.keys()):
            if param is not None:
                return getattr(self.group_objects[element], param.lower())
            else:
                return self.group_objects[element]
        else:
            warn(f"WARNING: Element {element} does not exist")
            return {}

    def get_element_type(
        self,
        typ: list | tuple | str,
        param: list | tuple | str = None,
    ) -> list | tuple | zip:
        """
        Get all elements of a specific type or types from the lattice.

        Parameters
        ----------
        typ: list, tuple, or str
            The type or types of elements to retrieve.
            If a list or tuple is provided, it retrieves elements of all specified types.
        param: list, tuple, or str, optional
            The specific parameter to retrieve from each element.

        Returns
        -------
        list | tuple | zip
            A list or tuple of elements of the specified type(s), or a zip object if multiple parameters are specified.
            If `param` is provided, it returns the specified parameter for each element.
        """
        if isinstance(typ, (list, tuple)):
            return [self.get_element_type(t, param=param) for t in typ]
        if isinstance(param, (list, tuple)):
            return zip(*[self.get_element_type(typ, param=p) for p in param])
        return [
            self.elements[element] if param is None else getattr(self.elements[element], param)
            for element in list(self.elements.keys())
            if self.elements[element].hardware_type.lower() == typ.lower()
        ]

    def set_element_type(
        self, typ: list | tuple | str, setting: str, values: list | tuple | Any
    ) -> None:
        """
        Set a specific setting for all elements of a specific type or types in the lattice.

        Parameters
        ----------
        typ: list, tuple, or str
            The type or types of elements to set the setting for.
        setting: str
            The setting to be updated for the elements. This can be a single setting or a list of settings.
        values: list, tuple, or Any
            The values to set for the specified setting.

        Raises
        ------
        ValueError
            If the number of elements of the specified type does not match the number of values provided.
        """
        elems = self.get_element_type(typ)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                e[setting] = v
        else:
            raise ValueError

    @property
    def quadrupoles(self) -> list:
        """
        Property to get all quadrupole elements in the lattice.

        Returns
        -------
        list
            A list of quadrupole elements in the lattice.
        """
        return self.get_element_type("quadrupole")

    @property
    def cavities(self) -> list:
        """
        Property to get all cavity elements in the lattice.

        Returns
        -------
        list
            A list of cavity elements in the lattice.
        """
        return self.get_element_type("cavity")

    @property
    def solenoids(self) -> list:
        """
        Property to get all solenoid elements in the lattice.

        Returns
        -------
        list
            A list of solenoid elements in the lattice.
        """
        return self.get_element_type("solenoid")

    @property
    def dipoles(self) -> list:
        """
        Property to get all dipole elements in the lattice.

        Returns
        -------
        list
            A list of dipole elements in the lattice.
        """
        return self.get_element_type("dipole")

    @property
    def kickers(self) -> list:
        """
        Property to get all kicker elements in the lattice.

        Returns
        -------
        list
            A list of kicker elements in the lattice.
        """
        return self.get_element_type("kicker")

    @property
    def dipoles_and_kickers(self) -> list:
        """
        Property to get all dipole and kicker elements in the lattice.

        Returns
        -------
        list
            A list of dipole and kicker elements in the lattice.
        """
        return sorted(
            self.get_element_type("dipole") + self.get_element_type("kicker"),
            key=lambda x: x.physical.end.z,
        )

    @property
    def wakefields(self) -> list:
        """
        Property to get all wakefield elements in the lattice.

        Returns
        -------
        list
            A list of wakefield elements in the lattice.
        """
        return self.get_element_type("wakefield")

    @property
    def wakefields_and_cavity_wakefields(self) -> list:
        """
        Property to get all wakefield and cavity wakefield elements in the lattice.

        Returns
        -------
        list
            A list of wakefield and cavity wakefield elements in the lattice.
        """
        cavities = [
            cav
            for cav in self.get_element_type("cavity")
            if (
                isinstance(cav.simulation.wakefield_definition, FieldMap)
                or cav.simulation.wakefield_definition != ""
            )
        ]
        wakes = self.get_element_type("wakefield")
        return cavities + wakes

    @property
    def screens(self) -> list:
        """
        Property to get all screen elements in the lattice.

        Returns
        -------
        list
            A list of screen elements in the lattice.
        """
        return self.get_element_type("screen")

    @property
    def screens_and_bpms(self) -> list:
        """
        Property to get all screen and BPM elements in the lattice.

        Returns
        -------
        list
            A list of screen and BPM elements in the lattice.
        """
        return sorted(
            self.get_element_type("screen")
            + self.get_element_type("beam_position_monitor"),
            key=lambda x: x.physical.start.z,
        )

    @property
    def screens_and_markers_and_bpms(self) -> list:
        """
        Property to get all screen and BPM and marker elements in the lattice.

        Returns
        -------
        list
            A list of screen and BPM and marker elements in the lattice.
        """
        return sorted(
            self.get_element_type("screen")
            + self.get_element_type("marker")
            + self.get_element_type("beam_position_monitor"),
            key=lambda x: x.physical.start.z,
        )

    @property
    def apertures(self) -> list:
        """
        Property to get all aperture and collimator elements in the lattice.

        Returns
        -------
        list
            A list of aperture and collimator elements in the lattice.
        """
        return sorted(
            self.get_element_type("aperture") + self.get_element_type("collimator"),
            key=lambda x: x.physical.start.z,
        )

    @property
    def wigglers(self) -> list:
        """
        Property to get all wiggler elements in the lattice.

        Returns
        -------
        list
            A list of wiggler elements in the lattice.
        """
        return self.get_element_type("wiggler")

    @property
    def lines(self) -> list:
        """
        Property to get all lines in the lattice.

        Returns
        -------
        list
            A list of lines in the lattice.
        """
        return list(self.lineObjects.keys())

    @property
    def start(self) -> str:
        """
        Property to get the name of the starting element of the lattice.
        This method checks if the file block contains a "start_element" key or a "zstart" key.
        If "start_element" is present, it returns the corresponding element.
        If "zstart" is present, it iterates through the elementObjects to find the element
        with the matching start position. If no match is found, it returns the first element in the elementObjects.


        Returns
        -------
        str
            The name of the starting element of the lattice.
        """
        if "start_element" in self.file_block["output"]:
            return self.file_block["output"]["start_element"]
        elif "zstart" in self.file_block["output"]:
            for name, elem in self.element_objects.items():
                if isinstance(elem, PhysicalBaseElement):
                    if (
                        np.isclose(elem.physical.start.z,
                        self.file_block["output"]["zstart"], atol=1e-2)
                    ) and not elem.subelement:
                        return name
            return list(self.element_objects.keys())[0]
        else:
            return list(self.element_objects.keys())[0]

    @property
    def start_object(self) -> "PhysicalBaseElement":
        """
        Property to get the starting element of the lattice.
        See :func:`start` for more details.


        Returns
        -------
        Element
            The starting element of the lattice.
        """
        return self.element_objects[self.start]

    @property
    def end(self) -> str:
        """
        Property to get the name of the ending element of the lattice.
        This method checks if the file block contains an "end_element" key or a "zstop" key.
        If "end_element" is present, it returns the corresponding element.
        If "zstop" is present, it iterates through the elementObjects to find the element
        with the matching end position. If no match is found, it returns the last element in the elementObjects.


        Returns
        -------
        str
            The name of final element of the lattice.
        """
        if "end_element" in self.file_block["output"]:
            return self.file_block["output"]["end_element"]
        elif "zstop" in self.file_block["output"]:
            endelems = []
            for name, elem in self.element_objects.keys():
                if isinstance(elem, PhysicalBaseElement):
                    if (
                        np.isclose(elem.physical.end.z,
                        self.file_block["output"]["zstop"], atol=1e-2)
                    ) and not elem.subelement:
                        endelems.append(name)
                    elif (
                        elem.physical.end.z
                        > self.file_block["output"]["zstop"]
                        and len(endelems) == 0
                    ) and not elem.subelement:
                        endelems.append(name)
            return endelems[-1]
        else:
            return list(self.element_objects.keys())[-1]

    @property
    def end_object(self) -> "PhysicalBaseElement":
        """
        Property to get the final element of the lattice.
        See :func:`end` for more details.


        Returns
        -------
        Element
            The final element of the lattice.
        """
        return self.element_objects[self.end]

    @computed_field
    @property
    def section(self) -> SectionLatticeTranslator:
        """
        Property to get the lattice elements as a `SectionLatticeTranslator`.

        Returns
        -------
        SectionLatticeTranslator
            LAURA `SectionLatticeTranslator`
        """
        if not isinstance(self._section, SectionLatticeTranslator):
            keys = self.machine.elements_between(start=self.start, end=self.end)
            vals = {k: self.machine.get_element(k) for k in keys if isinstance(self.machine.get_element(k), PhysicalBaseElement)}
            section = SectionLattice(
                order=keys,
                elements=ElementList(elements=vals),
                name=self.objectname,
                master_lattice=self.global_parameters["master_lattice"],
            )
            slt = SectionLatticeTranslator.from_section(section)
            slt.lsc_enable = self.lsc_enable
            slt.csr_enable = self.csr_enable
            slt.lsc_bins = self.lsc_bins
            slt.directory = self.global_parameters["master_subdir"]
            self._section = slt
            return slt
        return self._section

    @property
    def elements(self) -> dict:
        """
        Property to get a dictionary of elements in the lattice.

        Returns
        -------
        dict
            A dictionary where keys are element names and values are the corresponding element objects.
        """
        return self.section.elements.elements

    def write(self):
        pass

    def run(self) -> None:
        """
        Run the code with input 'filename'
        This method constructs the command to run the simulation using the specified executable
        and the name of the lattice. It redirects the output to a log file in the master subdirectory.

        If  :attr:`~remote_setup` is set, then :func:`~run_remote` will be called instead.

        Raises
        ------
        FileNotFoundError
            If the executable for the specified code is not found in the executables dictionary.
        """
        if self.remote_setup:
            self.run_remote()
        else:
            command = self.executables[self.code] + [self.name]
            with open(
                os.path.relpath(
                    self.global_parameters["master_subdir"] + "/" + self.name + ".log",
                    ".",
                ),
                "w",
            ) as f:
                subprocess.call(
                    command, stdout=f, cwd=self.global_parameters["master_subdir"]
                )

    def run_remote(self) -> None:
        """
        Run the simulation on a remote server using SSH and SFTP, following these steps:

        1. Connect to the remote server using :func:`~connect_remote`.

        2. Create a subdirectory on the remote server with the same name as `master_subdir`.

        3. Send the required files (simulation input file(s), initial beam distribution file,
        field/wakefield files).

        4. Execute the simulation and wait for completion.

        5. Retrieve all output files created since the start of the simulation back into `master_subdir`
        """
        ssh = self.connect_remote()
        subdir = self.global_parameters["master_subdir"]
        cod = self.code.lower() if self.code.lower() != "elegant" else "sdds"
        for e in self.elements.values():
            if hasattr(e.simulation, "field_definition") and isinstance(e.simulation.field_definition, str):
                fn = e.simulation.field_definition.split('/')[-1].split('\\')[-1]
                filename = os.path.splitext(fn)[0]
                if cod in ["opal", "gpt", "astra"]:
                    self.files.append(f'{subdir}/{filename}.{cod.lower()}')
            if hasattr(e.simulation, "wakefield_definition") and  isinstance(e.simulation.wakefield_definition, str):
                fn = e.simulation.wakefield_definition.split('/')[-1].split('\\')[-1]
                filename = os.path.splitext(fn)[0]
                self.files.append(f'{subdir}/{filename}.{cod.lower()}')
        starttime = time.time()
        subdir = self.global_parameters["master_subdir"]
        rel_subdir = f"/home/{self.remote_setup['username']}/{os.path.basename(subdir)}"
        cmd = f"mkdir -p {rel_subdir}"
        ssh.exec_command(f"mkdir -p {rel_subdir}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        stdout.channel.recv_exit_status()
        sent = []
        for file in self.files:
            remote_file = os.path.join(rel_subdir, os.path.basename(file))
            if file not in sent:
                with ssh.open_sftp() as sftp:
                    sftp.put(file, remote_file)
            sent.append(file)
        suffix = ".ele" if self.code.lower() == "elegant" else ".in"
        command = self.objectname + suffix
        full_command = ""
        if self.code.lower() == "elegant":
            full_command += f'export RPN_DEFNS={self.remote_setup["host"]["rpn"]} && '
        full_command += f"cd {rel_subdir} && "
        full_command +=  f"{' '.join(self.executables[self.code])} {command}"
        stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
        stdout.channel.recv_exit_status()

        with ssh.open_sftp() as sftp:
            for attr in sftp.listdir_attr(rel_subdir):

                # Skip directories
                if stat.S_ISDIR(attr.st_mode):
                    continue

                # Only download files modified since starttime
                if attr.st_mtime >= starttime:
                    remote_path = os.path.join(rel_subdir, attr.filename)
                    local_path = os.path.join(self.global_parameters["master_subdir"], attr.filename)
                    sftp.get(remote_path, local_path)

        sftp.close()
        cmd = f"rm -rf '{rel_subdir}'"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        stdout.channel.recv_exit_status()
        ssh.close()

    def connect_remote(self) -> Any:
        """
        Set up an SSH connection to a remote server using the parameters defined in `remote_setup`.
        These keys must include `host`, `username`, and `password`.

        Returns
        -------
        paramiko.SSHClient
            The SSH client for the established connection.

        Raises
        ------
        KeyError
            If the `remote_setup` attribute of this class does not contain the required keys.
        paramiko.AuthenticationException
            If the SSH authentication fails (i.e. due to incorrect credentials).
        TimeoutError
            If the SSH connection fails, for example if the server is unreachable.
        """
        if not all(name in self.remote_setup for name in ["host", "username", "password"]):
            raise KeyError("remote_setup must contain 'host', 'username' and 'password'")
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                self.remote_setup["host"]["address"],
                username=self.remote_setup["username"],
                password=self.remote_setup["password"],
            )
            return ssh
        except paramiko.SSHException:
            ssh.connect(
                self.remote_setup["host"]["address"],
                username=self.remote_setup["username"],
                password=self.remote_setup["password"],
                allow_agent=False, look_for_keys=False
            )
            return ssh
        except TimeoutError as e:
            raise TimeoutError(f"Connection to {self.remote_setup['host']} timed out") from e

    def get_initial_twiss(self) -> dict:
        """
        Get the initial Twiss parameters from the file block
        This method checks if the file block contains an "input" key with a "twiss" subkey.
        If the "twiss" subkey exists and contains values, it retrieves the alpha, beta, and normalized emittance
        parameters for both horizontal and vertical planes.

        Returns
        -------
        dict
            A dictionary containing the initial Twiss parameters for horizontal and vertical planes.
            If the parameters are not found, it returns False for each parameter.
        """
        if (
            "input" in self.file_block
            and "twiss" in self.file_block["input"]
            and self.file_block["input"]["twiss"]
        ):
            alpha_x = (
                self.file_block["input"]["twiss"]["alpha_x"]
                if "alpha_x" in self.file_block["input"]["twiss"]
                else False
            )
            alpha_y = (
                self.file_block["input"]["twiss"]["alpha_y"]
                if "alpha_y" in self.file_block["input"]["twiss"]
                else False
            )
            beta_x = (
                self.file_block["input"]["twiss"]["beta_x"]
                if "beta_x" in self.file_block["input"]["twiss"]
                else False
            )
            beta_y = (
                self.file_block["input"]["twiss"]["beta_y"]
                if "beta_y" in self.file_block["input"]["twiss"]
                else False
            )
            nemit_x = (
                self.file_block["input"]["twiss"]["nemit_x"]
                if "nemit_x" in self.file_block["input"]["twiss"]
                else False
            )
            nemit_y = (
                self.file_block["input"]["twiss"]["nemit_y"]
                if "nemit_y" in self.file_block["input"]["twiss"]
                else False
            )
            return {
                "horizontal": {
                    "alpha": alpha_x,
                    "beta": beta_x,
                    "nEmit": nemit_x,
                },
                "vertical": {
                    "alpha": alpha_y,
                    "beta": beta_y,
                    "nEmit": nemit_y,
                },
            }
        else:
            return {
                "horizontal": {
                    "alpha": False,
                    "beta": False,
                    "nEmit": False,
                },
                "vertical": {
                    "alpha": False,
                    "beta": False,
                    "nEmit": False,
                },
            }

    def longitudinal_match(self, settings) -> None:
        harmonics = {}
        harm_number = 0
        if "cavities" in settings:
            cavs = [c for c in self.cavities if c.name in settings["cavities"]]
            freq = list(set([c.cavity.frequency for c in cavs]))
            if len(freq) > 1:
                raise ValueError("All accelerating cavities must have the same frequency")
            freq = freq[0]
        else:
            raise KeyError("settings must contain `cavities` key containing names of cavities")
        if "harmonics" in settings:
            harmonics = [c for c in self.cavities if c.name in settings["harmonics"]]
            harm_freq = list(set([c.cavity.frequency for c in harmonics]))
            if len(harm_freq) > 1:
                raise ValueError("All harmonic cavities must have the same frequency")
            harm_freq = harm_freq[0]
            if not harm_freq % freq == 0:
                raise ValueError("Harmonic cavity frequency is not a harmonic of the main frequency")
            harm_number = int(harm_freq / freq)
        if "chirp" in settings:
            chirp = settings["chirp"]
        else:
            raise ValueError("Chirp must be defined")
        curvature = settings["curvature"] if "curvature" in settings else 0
        skewness = settings["skewness"] if "skewness" in settings else 0

        k = 2 * np.pi * freq / speed_of_light
        M = np.array(
            [
                [1, 0, 1, 0],
                [0, -k, 0, -(harm_number * k)],
                [-k ** 2, 0, -(harm_number * k) ** 2, 0],
                [0, k ** 3, 0, (harm_number * k) ** 3]
            ]
        )

        initial_energy = self.global_parameters["beam"].centroids.mean_cpz.val * 1e-9
        final_energy = self.global_parameters["beam"].centroids.mean_cpz.val * 1e-9
        for cav in cavs:
            final_energy += (cav.simulation.field_amplitude * np.cos(cav.cavity.phase)) * 1e-9
        if harmonics:
            for harm in harmonics:
                final_energy += (harm.simulation.field_amplitude * np.cos(harm.cavity.phase)) * 1e-9

        chirps = self.global_parameters["beam"].slice.get_chirp_coeffs()

        energy_gain = final_energy - initial_energy
        r = np.array(
            [
                energy_gain,
                chirp * final_energy - (initial_energy * chirps["order_1"]),
                curvature * final_energy - ((initial_energy * chirps["order_2"]) / 2),
                skewness * final_energy - ((initial_energy * chirps["order_3"]) / 6),
            ]
        )

        if not harmonics:
            M = np.array([[1, 0],
                          [0, -k]])
            r = np.array(
                [
                    energy_gain,
                    chirp * final_energy - (initial_energy * chirps["order_1"]),
                ]
            )
        rf = np.dot(np.linalg.inv(M), r)
        X1 = rf[0]
        Y1 = rf[1]
        rad2deg = 180 / np.pi
        v1 = np.sqrt(X1 ** 2 + Y1 ** 2) * 1e9
        phi1 = (np.arctan(Y1 / X1) + np.pi / 2 * (1 - np.sign(X1))) * rad2deg
        for cav in cavs:
            cav.simulation.field_amplitude = v1
            cav.cavity.phase = ((-phi1 + 180) % 360)# - 180
        print(f"Longitudinal matching gave cavity phase of {phi1} and field amplitude of {v1}")
        if harmonics:
            X13 = rf[2]
            Y13 = rf[3]
            vh = np.sqrt(X13 ** 2 + Y13 ** 2) * 1e9
            phih = (np.arctan(Y13 / X13) + np.pi / 2 * (1 - np.sign(X13)) - 2 * np.pi) * rad2deg
            for harm in harmonics:
                harm.simulation.field_amplitude = vh
                harm.cavity.phase = ((-phih + 180) % 360)# - 180
                print(f"Longitudinal matching gave harmonic phase of {phi1} and field amplitude of {v1}")

    def pre_process(self) -> None:
        """
        Pre-process the lattice before running the simulation.
        This method initializes the initial Twiss parameters by calling the `getInitialTwiss` method.

        Returns
        -------
        None
        """
        ast = self.section.astra_headers.copy()
        self.initial_twiss = self.get_initial_twiss()
        if "match" in self.file_block:
            domatch = True
            if "enable" in self.file_block["match"]:
                if not self.file_block["match"]["enable"]:
                    domatch = False
            if domatch:
                self.match(self.file_block["match"])
            # if matchtwiss:
            #     self.elementObjects = matchtwiss
        if "longitudinal_match" in self.file_block:
            self.longitudinal_match(self.file_block["longitudinal_match"])
        self.section.astra_headers = ast

    def post_process(self):
        pass

    def __repr__(self):
        return self.elements

    def __str__(self):
        str = self.name + " = ("
        for e in self.elements:
            if len((str + e).splitlines()[-1]) > 60:
                str += "&\n"
            str += e + ", "
        return str + ")"

    def create_drifts(
        self, drift_elements: tuple = ("screen", "beam_position_monitor")
    ) -> dict:
        """
        Insert drifts into a sequence of 'elements'.
        This method creates drifts for elements that are not subelements and have a length greater than zero.
        It calculates the start and end positions of each element and creates drift elements accordingly.

        Parameters
        ----------
        drift_elements: tuple, optional
            A tuple of element types for which drifts should be created.
            Default is ("screen", "beam_position_monitor").

        Returns
        -------
        dict
            A dictionary containing the new drift elements created for the lattice.
            The keys are the names of the new drift elements, and the values are the corresponding drift objects.
        """
        return self.section.create_drifts()

    def get_s_values(
        self,
        as_dict: bool = False,
        at_entrance: bool = False,
        drifts: bool = True,
    ) -> list | dict:
        """
        Get the S values for the elements in the lattice.
        This method calculates the cumulative length of the elements in the lattice,
        starting from the entrance or the first element, depending on the `at_entrance` parameter.
        It returns a list or dict of S values, which represent the positions of the elements along the lattice.

        Parameters
        ----------
        as_dict: bool, optional
            If True, returns a dictionary with element names as keys and their S values as values.
        at_entrance: bool, optional
            If True, calculates S values starting from the entrance of the lattice.
            If False, calculates S values starting from the first element.
        drifts: bool, optional
            If True, include s-values for drift elements

        Returns
        -------
        list | dict
            A list or dictionary of S values for the elements in the lattice.
            If `as_dict` is True, returns a dictionary with element names as keys and their S values as values.
            If `as_dict` is False, returns a list of S values.
        """
        elems = self.create_drifts() if drifts else self.elements
        s = [0]
        for e in list(elems.values()):
            s.append(s[-1] + e.physical.length)
        s = s[:-1] if at_entrance else s[1:]
        if as_dict:
            return dict(zip([e.name for e in elems.values()], s))
        return list(s)

    def get_z_values(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
        """
        Get the Z values for the elements in the lattice.
        This method calculates the cumulative length of the elements in the lattice,
        starting from the entrance or the first element, depending on the `at_entrance` parameter.
        It returns a list or dict of S values, which represent the positions of the elements along the lattice.

        Parameters
        ----------
        drifts: bool, optional
            If True, includes drift elements in the calculation.
            If False, only considers the main elements in the lattice.
        as_dict: bool, optional
            If True, returns a dictionary with element names as keys and their Z values as values.

        Returns
        -------
        list | dict
            A list or dictionary of Z values for the elements in the lattice.
            If `as_dict` is True, returns a dictionary with element names as keys and their Z values as values.
            If `as_dict` is False, returns a list of Z values.
        """
        if drifts:
            elems = self.create_drifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.name: [e.physical.start.z, e.physical.end.z] for e in elems.values()}
        return [[e.physical.start.z, e.physical.end.z] for e in elems.values()]

    def get_names(self, drifts: bool = True) -> list:
        """
        Get the names of the elements in the lattice.

        Parameters
        ----------
        drifts: bool, optional
            If True, includes drift elements in the list of names.

        Returns
        -------
        list
            A list of names of the elements in the lattice.
            If `drifts` is True, includes drift elements; otherwise, only includes main elements.
        """
        if drifts:
            elems = self.create_drifts()
        else:
            elems = self.elements
        return [e.name for e in list(elems.values())]

    def get_elems(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
        """
        Get the elements in the lattice.

        Parameters
        ----------
        drifts: bool, optional
            If True, includes drift elements in the list of elements.
        as_dict: bool, optional
            If True, returns a dictionary with element names as keys and their corresponding element objects as values.

        Returns
        -------
        list | dict
            A list or dictionary of elements in the lattice.
        """
        if drifts:
            elems = self.create_drifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.name: e for e in list(elems.values())}
        return [e for e in list(elems.values())]

    def get_s_names(self) -> list:
        """
        Get the names and S values of the elements in the lattice.

        Returns
        -------
        list
            A list of tuples, where each tuple contains the name of an element and its corresponding S value.
        """
        s = self.get_s_values()
        names = self.get_names()
        return list(zip(names, s))

    def get_s_names_elems(self) -> tuple:
        """
        Get the names, elements, and S values of the elements in the lattice.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - A list of names of the elements.
            - A list of element objects.
            - A list of S values corresponding to the elements.
        """
        s = self.get_s_values()
        names = self.get_names()
        elems = self.get_elems()
        return names, elems, s

    def get_z_names_elems(self) -> tuple:
        """
        Get the names, elements, and Z values of the elements in the lattice.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - A list of names of the elements.
            - A list of element objects.
            - A list of Z values corresponding to the elements.
        """
        z = self.get_z_values()
        names = self.get_names()
        elems = self.get_elems()
        return names, elems, z

    def find_s(self, elem) -> list:
        """
        Find the S values for a specific element in the lattice.

        Parameters
        ----------
        elem: str
            The name of the element to find in the lattice.


        Returns
        -------
        list
            A list of tuples, where each tuple contains the name of the element and its corresponding S value.
            If the element does not exist in the lattice, returns an empty list.
        """
        if elem in self.all_elements:
            sNames = self.get_s_names()
            return [a for a in sNames if a[0] == elem]
        return []

    def update_run_settings(self, runSettings: RunSetup) -> None:
        """
        Update the run settings for the lattice.

        Parameters
        ----------
        runSettings: runSetup
            An instance of runSetup containing the new run settings.

        Raises
        ------
        TypeError
            If the `runSettings` argument is not an instance of `runSetup`.

        """
        if isinstance(runSettings, RunSetup):
            self.run_settings = runSettings
        else:
            raise TypeError(
                "runSettings argument passed to frameworkLattice.updateRunSettings is not a runSetup instance"
            )

    def setup_xsuite_line(self) -> tuple:
        """
        Set up an Xsuite Line object from the current lattice elements.

        Returns
        -------
        tuple (xt.Line, rbf.beam, List)
            * An Xsuite Line object representing the current lattice.
            * An rbf.beam object containing the beam parameters.
            * A list of element names in the Xsuite Line.
        """
        prefix = self.get_prefix()
        self.read_input_file(prefix, self.particle_definition)
        import xtrack as xt
        beam = self.global_parameters["beam"]
        particle_ref = xt.particles(
            p0c=[beam.centroids.mean_cp.val],
            mass0=[beam.particle_rest_energy_eV.val],
            q0=-1,
            zeta=0.0,
        )
        line = self.section.to_xsuite(
            beam_length=len(self.global_parameters["beam"].x.val),
            particle_ref=particle_ref,
        )
        beam = deepcopy(self.global_parameters["beam"])
        return line, beam, self.get_names()

    def r_matrix(
            self,
            start: str = None,
            end: str = None,
            element_by_element: bool = True,
    ) -> np.ndarray:
        """
        Compute the one-turn transfer matrix for the lattice using Xsuite.
        This method sets up an Xsuite Line object from the current lattice elements
        and computes the one-turn transfer matrix using finite differences.

        Parameters
        ----------
        start: str, optional
            The first element from which to compute the transfer matrix (first element by default).
        end: str, optional
            The last element from which to compute the transfer matrix (last element by default).
        element_by_element: bool, optional
            Return the element-by-element transfer matrices if True; if not return the full
            transfer matrix for the entire line

        Returns
        -------
        np.ndarray
            Transfer matrix (or matrices) as a NumPy array.
        """
        line, beam, names = self.setup_xsuite_line()
        matrix = line.compute_one_turn_matrix_finite_differences(
            start=start,
            end=end,
            particle_on_co=line.particle_ref,
            element_by_element=True
        )
        if element_by_element:
            return matrix["R_matrix_ebe"]
        return matrix["R_matrix"]

    def match(self, params: Dict) -> None:
        """
        Perform transverse matching of the lattice using Ocelot's built-in matching algorithm.

        The `params` dictionary should contain the following
        keys:
            - "variables": A list of element names (magnets only).
            - "targets": A dictionary where keys are element names and values are dictionaries
              with keys corresponding to Twiss parameters ("beta_x", "beta_y", "alpha_x",
              "alpha_y", "eta_x", "eta_y", "eta_xp", "eta_yp", "mux", "muy") and their target values.
            - "start": (optional) The name of the starting element for matching. Defaults to the first element.
            - "end": (optional) The name of the ending element for matching. Defaults to the last element.

        The matching dictionary should have this structure within the lattice file block:

        .. code-block:: yaml

            files:
              line:
                <.....>
                match:
                  variables:
                    Q1
                    Q2
                    S1
                  targets:
                    SCR1: {beta_x: 10.0, alpha_x: 0.0}
                    SCR2: {beta_y: 12.0, alpha_y: 0.0}
                    SCR3: {beta_x: {mode: greaterthan, value: 8.0}}
                  start: Q1
                  end: SCR3

        Parameters
        ----------
        params: Dict
            Dictionary containing matching variables, targets, and optional start and end elements.

        Returns
        -------
        Dict | None
            Updated elementObjects if matching is successful, None otherwise.

        Raises
        ------
        ValueError
            If required keys are missing in the `params` dictionary or
            if specified elements are not found in the lattice.
        RuntimeError
            If the matching process fails.
        """
        if "variables" not in params:
            raise ValueError("No matching variables provided")
        if "targets" not in params:
            raise ValueError("No matching targets provided")
        from .framework_lattices import OcelotLattice
        from ocelot.cpbd.beam import Twiss
        from ocelot.cpbd.match import match as match_oce
        latcopy = deepcopy(self)
        lat = OcelotLattice(
            name=f"{latcopy.name}_match",
            file_block=latcopy.file_block,
            machine=latcopy.machine,
            element_objects=latcopy.element_objects,
            group_objects=latcopy.group_objects,
            run_settings=latcopy.run_settings,
            executables=latcopy.executables,
            global_parameters=latcopy.global_parameters,
            settings=latcopy.settings,
        )
        prefix = lat.get_prefix()
        prefix = prefix if lat.track_beam else prefix + lat.particle_definition
        lat.read_input_file(prefix, lat.particle_definition)
        lat.ref_s = self.global_parameters["beam"].s
        lat.ref_idx = self.global_parameters["beam"].reference_particle_index
        lat.hdf5_to_npz(prefix)
        lat.write_elements()
        beam = lat.global_parameters["beam"]
        twsobj = Twiss(
            beta_x=beam.twiss.beta_x.val,
            beta_y=beam.twiss.beta_y.val,
            alpha_x=beam.twiss.alpha_x.val,
            alpha_y=beam.twiss.alpha_y.val,
            # Dx=beam.twiss.eta_x.val,
            # Dy=beam.twiss.eta_y.val,
            # Dxp=beam.twiss.eta_xp.val,
            # Dyp=beam.twiss.eta_yp.val,
            E=beam.centroids.mean_cp.val * 1e-9
        )
        matchelems = [e for e in lat.lat_obj.sequence if e.id in params["targets"].keys()]
        constr = {e: params["targets"][e.id] for e in matchelems}
        if "global" in params["targets"]:
            constr.update({"global": params["targets"]["global"]})
        varelems = []
        for p in params["variables"]:
            if p in self.elements.keys():
                if type(self.elements[p]) in [Quadrupole, Sextupole, Octupole]:
                    varelems.append([e for e in lat.lat_obj.sequence if e.id == p][0])
        try:
            max_iter = params["max_iterations"]
        except KeyError:
            max_iter = 10000
        if len(varelems) == 0:
            raise ValueError("No variables added; make sure quadrupoles/sextupoles/octupoles are used for matching")
        res = match_oce(lat=lat.lat_obj, constr=constr, vars=varelems, tw=twsobj, verbose=False, max_iter=max_iter)
        print("Matching results:")
        for i, r in enumerate(res):
            magnetic_order = self.element_objects[params["variables"][i]].magnetic.order
            magnetic_length = self.element_objects[params["variables"][i]].magnetic.length
            setattr(self.element_objects[params["variables"][i]], f"k{magnetic_order}l", r * magnetic_length)
            print("\t", self.element_objects[params["variables"][i]].name, f"k{magnetic_order}l =", r * magnetic_length)

class GlobalError(FrameworkObject):
    """
    Class defining a global error element.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "_write_ASTRA": "_write_astra",
        "_write_GPT": "_write_gpt",
        "add_Error": "add_error",
    }

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(GlobalError, self).__init__(
            *args,
            **kwargs,
        )

    def add_error(self, type, sigma):
        if type in global_Error_Types:
            self.add_property(type, sigma)

    def _write_astra(self):
        return self._write_ASTRA_dictionary(
            dict([[key, {"value": value}] for key, value in self._errordict])
        )

    def _write_gpt(self, Brho, ccs="wcs", *args, **kwargs):
        relpos, relrot = ccs.relative_position(self.middle, [0, 0, 0])
        coord = self.gpt_coordinates(relpos, relrot)
        output = (
            str(self.objecttype)
            + "( "
            + ccs.name
            + ", "
            + coord
            + ", "
            + str(self.length)
            + ", "
            + str(Brho * self.k1)
            + ");\n"
        )
        return output

class FrameworkCommand(FrameworkObject):
    """
    Class defining a framework command, which is used to generate commands used in setup files
    for various simulation codes.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "write_Elegant": "write_elegant",
        "write_Genesis": "write_genesis",
        "write_MAD8": "write_mad8",
    }

    def model_post_init(self, __context):
        if self.objecttype not in commandkeywords:
            raise NameError("Command '%s' does not exist" % self.objecttype)
        super().model_post_init(__context)

    def write_elegant(self) -> str:
        """
        Writes the command string for ELEGANT.

        Returns
        -------
        str
            String representation of the command for ELEGANT
        """
        string = "&" + self.objecttype + "\n"
        for key in commandkeywords[self.objecttype]:
            if (
                key.lower() in self.allowedkeywords
                and not key == "objectname"
                and not key == "objecttype"
                and hasattr(self, key)
            ):
                if getattr(self, key.lower()) is not None:
                    string += "\t" + key + " = " + str(getattr(self, key.lower())) + "\n"
        string += "&end\n"
        return string

    def write_mad8(self) -> str:
        """
        Writes the command string for MAD8.
        # TODO deprecated?

        Returns
        -------
        str
            String representation of the command for MAD8
        """
        string = self.objecttype
        # print(self.objecttype, self.objectproperties)
        for key in commandkeywords[self.objecttype]:
            if (
                    key.lower() in self.objectproperties
                    and not key == "name"
                    and not key == "type"
                    and not self.objectproperties[key.lower()] is None
            ):
                e = "," + key + "=" + str(self.objectproperties[key.lower()])
                if len((string + e).splitlines()[-1]) > 79:
                    string += ",&\n"
                string += e
        string += ";\n"
        return string

    def write_genesis(self) -> str:
        """
        Writes the command string for Genesis.
        # TODO deprecated?

        Returns
        -------
        str
            String representation of the command for Genesis
        """
        string = "&" + self.objecttype + "\n"
        for key in commandkeywords_genesis[self.objecttype]:
            if (
                key.lower() in self.allowedkeywords
                and not key == "objectname"
                and not key == "objecttype"
                and hasattr(self, key)
            ):
                val = getattr(self, key.lower())
                val = int(val) if isinstance(val, bool) else val
                if val is not None:
                    string += "\t" + key + " = " + str(val) + "\n"
        string += "&end\n"
        return string


class FrameworkGroup(DeprecatedMethodAliases, object):
    """
    Class defining a framework group, which is used to group together elements to perform coordinated
    actions on them.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "allElementObjects": "all_element_objects",
        "allGroupObjects": "all_group_objects",
        "change_Parameter": "change_parameter",
        "get_Parameter": "get_parameter",
    }

    def __init__(self, name, framework, type, elements, **kwargs):
        super(FrameworkGroup, self).__init__()
        self.objectname = name
        self.type = type
        self.framework = framework
        self.elements = elements

    @property
    def all_element_objects(self):
        return self.framework.element_objects

    @property
    def all_group_objects(self):
        return self.framework.group_objects

    def update(self, **kwargs):
        pass

    def get_parameter(self, p: str) -> Any:
        """
        Get a specific parameter associated with the group, i.e. bunch compressor angle

        Parameters
        ----------
        p: str
            A parameter associated with the group

        Returns
        -------
        Any
            The parameter, if defined.
        """
        try:
            isinstance(type(getattr(self, p)), p)
            return getattr(self, p)
        except Exception:
            if self.elements[0] in self.all_group_objects:
                return getattr(self.all_group_objects[self.elements[0]], p)
            return getattr(self.all_element_objects[self.elements[0]], p)

    def change_parameter(self, p: Any, v: Any) -> None:
        """
        Set a parameter on all elements in the group.

        Parameters
        ----------
        p: str
            The parameter to be set
        v: Any
            The value to be set.
        """
        try:
            getattr(self, p)
            setattr(self, p, v)
            if p == "angle":
                self.set_angle(v)
            # print ('Changing group ', self.objectname, ' ', p, ' = ', v, '  result = ', self.get_Parameter(p))
        except Exception:
            for e in self.elements:
                setattr(self.all_element_objects[e], p, v)
                # print ('Changing group elements ', self.objectname, ' ', p, ' = ', v, '  result = ', self.allElementObjects[self.elements[0]].objectname, self.get_Parameter(p))

    # def __getattr__(self, p):
    #     return self.get_Parameter(p)

    def __repr__(self):
        return str([self.all_element_objects[e].name for e in self.elements])

    def __str__(self):
        return str([self.all_element_objects[e].name for e in self.elements])

    def __getitem__(self, key):
        return self.get_parameter(key)

    def __setitem__(self, key, value):
        return self.change_parameter(key, value)


class ElementGroup(FrameworkGroup):
    """
    Class defining a group of elements, which is used to group together elements to perform coordinated
    actions on them.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)

    def __str__(self):
        return str([self.all_element_objects[e] for e in self.elements])


class R56Group(FrameworkGroup):
    """
    Class defining a group of elements with a total R56.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "get_Parameter": "get_parameter",
        "updateElements": "update_elements",
    }

    def __init__(self, name, elementObjects, type, elements, ratios, keys, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = ratios
        self.keys = keys
        self._r56 = None

    def __str__(self):
        return str({e: k for e, k in zip(self.elements, self.keys)})

    def get_parameter(self, p: str) -> Any:
        """
        Get a parameter associated with the group.

        Parameters
        ----------
        p: str
            The parameter to be retrieved.

        Returns
        -------
        Any
            The parameter.
        """
        if str(p) == "r56":
            return self.r56
        else:
            return super().get_parameter(p)

    @property
    def r56(self) -> float:
        """
        Get the R56 of the group of elements

        Returns
        -------
        float
            The R56 pararmeter
        """
        return self._r56

    @r56.setter
    def r56(self, r56: float) -> None:
        """
        Set the R56 of the group of elements

        Parameters
        ----------
        r56: float
            The R56 to be set
        """
        # print('Changing r56!', self._r56)
        self._r56 = r56
        data = {"r56": self._r56}
        parser = MathParser(data)
        values = [parser.parse(e) for e in self.ratios]
        # print('\t', list(zip(self.elements, self.keys, values)))
        for e, k, v in zip(self.elements, self.keys, values):
            self.update_elements(e, k, v)

    def update_elements(self, element: str | list | tuple, key: str, value: Any) -> None:
        """
        Update one or more elements in the group.

        Parameters
        ----------
        element: str, list or tuple
            The element(s) to be updated
        key: str
            The parameter in the element or group of elements to be changed
        value: Any
            The value to which the parameter should be set
        """
        # print('R56 : updateElements', element, key, value)
        if isinstance(element, (list, tuple)):
            [self.update_elements(e, key, value) for e in self.elements]
        else:
            if element in self.all_element_objects:
                # print('R56 : updateElements : element', element, key, value)
                self.all_element_objects[element].change_parameter(key, value)
            if element in self.all_group_objects:
                # print('R56 : updateElements : group', element, key, value)
                self.all_group_objects[element].change_parameter(key, value)


class Chicane(FrameworkGroup):
    """
    Class defining a 4-dipole chicane.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(Chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (1, -1, -1, 1)

    def update(self, **kwargs) -> None:
        """
        Update the bending angle and/or dipole width and/or dipole gap of all magnets in the chicane.

        Parameters
        ----------
        **kwargs: Dict
            Dictionary containing parameters to be updated -- must be in ["dipoleangle", "width", "gap"]
        """
        if "dipoleangle" in kwargs:
            self.set_angle(kwargs["dipoleangle"])
        if "width" in kwargs:
            self.change_parameter("width", kwargs["width"])
        if "gap" in kwargs:
            self.change_parameter("gap", kwargs["gap"])
        return None

    @property
    def angle(self) -> float:
        """
        Bending angle of the chicane

        Returns
        -------
        float
            The bending angle
        """
        obj = [self.all_element_objects[e] for e in self.elements]
        return float(obj[0].angle)

    @angle.setter
    def angle(self, theta: float) -> None:
        """
        Set the bending angle of the chicane; see :func:`~simba.Framework_objects.chicane.set_angle`.

        Parameters
        -----------
        theta: float
            Chicane bending angle
        """
        self.set_angle(theta)

    def set_angle(self, a: float) -> None:
        """
        Set the chicane bending angle, including updating the inter-dipole drift lengths.

        Parameters
        ----------
        a: float
            The angle to be set
        """
        indices = list(
            sorted([list(self.all_element_objects).index(e) for e in self.elements])
        )
        dipole_objs = [self.all_element_objects[e] for e in self.elements]
        obj = dipole_objs
        dipole_number = 0
        ref_pos = None
        ref_angle = None
        for i in range(len(obj)):
            if dipole_number > 0:
                adj = obj[i].physical.middle.z - ref_pos.z
                obj[i].physical.middle = Position(
                    x=ref_pos.x + np.tan(-1.0 * ref_angle) * adj,
                    y=0,
                    z=obj[i].physical.middle.z,
                )
                obj[i].physical.global_rotation.theta = ref_angle
            if obj[i] in dipole_objs:
                ref_pos = deepcopy(obj[i].physical.middle)
                obj[i].magnetic.angle = a * self.ratios[dipole_number]
                ref_angle = obj[i].physical.global_rotation.theta + obj[i].magnetic.angle
                obj[i].physical.physical_angle = obj[i].magnetic.angle
                dipole_number += 1

    def __str__(self):
        return str(
            [
                [
                    self.all_element_objects[e].name,
                    self.all_element_objects[e].magnetic.angle,
                    self.all_element_objects[e].physical.global_rotation.z,
                    self.all_element_objects[e].physical.start,
                    self.all_element_objects[e].physical.end,
                ]
                for e in self.elements
            ]
        )



class SChicane(Chicane):
    """
    Class defining an s-type chicane; in this case the bending ratios for
    :func:`~simba.Framework_objects.chicane.set_angle` are different.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(SChicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (-1, 2, -2, 1)


class FrameworkCounter(dict):
    """
    Class defining a counter object, used for numbering elements of the same type in ASTRA and CSRTrack
    """

    def __init__(self, sub={}):
        super(FrameworkCounter, self).__init__()
        self.sub = sub

    def counter(self, typ: str) -> int:
        """
        Increment count of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type

        Returns
        -------
        int
            The updated number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            return 1
        return self[typ] + 1

    def value(self, typ: str) -> int:
        """
        Number of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type

        Returns
        -------
        int
            The number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            return 1
        return self[typ]

    def add(self, typ: str, n: PositiveInt = 1) -> int:
        """
        Add to count of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type
        n: PositiveInt, optional
            Add more than one element at a time

        Returns
        -------
        int
            The number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            self[typ] = n
        else:
            self[typ] += n
        return self[typ]

    def subtract(self, typ: str) -> int:
        """
        Reduce count of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type

        Returns
        -------
        int
            The updated number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            self[typ] = 0
        else:
            self[typ] = self[typ] - 1 if self[typ] > 0 else 0
        return self[typ]


class GetGrids(DeprecatedMethodAliases, object):
    """
    Class defining the appropriate number of space charge bins given the number of particles,
    defined as the closest power of 8 to the cube root of the number of particles.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "getGridSizes": "get_grid_sizes",
    }

    def __init__(self):
        self.powersof8 = np.asarray([2**j for j in range(1, 20)])

    def get_grid_sizes(self, x: PositiveInt) -> int:
        """
        Calculate the 3D space charge grid size given the number of particles, minimum of 4

        Parameters
        ----------
        x: PositiveInt
            Number of particles

        Returns
        -------
        int
            The number of space charge grids
        """
        self.x = abs(x)
        self.cuberoot = int(round(self.x ** (1.0 / 3)))
        return max([4, self.find_nearest(self.powersof8, self.cuberoot)])

    def find_nearest(self, array: np.ndarray | list, value: int) -> int:
        """
        Get the nearest value in an array to the value provided; in this case the array should be a list of
        powers of 8.

        Parameters
        ----------
        array: np.ndarray or list
            Array of values to be checked
        value: Value to be found in the array

        Returns
        -------
        int
            The closest value in `array` to `value`
        """
        self.array = array
        self.value = value
        self.idx = (np.abs(self.array - self.value)).argmin()
        return self.array[self.idx]


from simba._compat import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(
    __name__,
    globals(),
    {
        "chicane": "Chicane",
        "element_group": "ElementGroup",
        "elements_Elegant": "elements_elegant",
        "frameworkCommand": "FrameworkCommand",
        "frameworkCounter": "FrameworkCounter",
        "frameworkGroup": "FrameworkGroup",
        "frameworkLattice": "FrameworkLattice",
        "frameworkObject": "FrameworkObject",
        "getGrids": "GetGrids",
        "global_error": "GlobalError",
        "r56_group": "R56Group",
        "runSetup": "RunSetup",
        "s_chicane": "SChicane",
        "type_conversion_rules_Elegant": "type_conversion_rules_elegant",
        "type_conversion_rules_Names": "type_conversion_rules_names",
        "type_conversion_rules_Opal": "type_conversion_rules_opal",
    },
)
