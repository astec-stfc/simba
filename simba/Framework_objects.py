"""
SIMBA Objects Module

Various objects and functions to handle simulation lattices, commands, and elements.

Classes:
    - :class:`~simba.Framework_objects.runSetup`: Defines simulation run settings, allowing for single runs, element scans or jitter/error studies.

    - :class:`~simba.Framework_objects.frameworkObject`: Base class for generic objects in SIMBA, including lattice elements and simulation code commands.

    - :class:`~simba.Framework_objects.frameworkLattice`: Base class for simulation lattices, consisting of a line of `LAURA` elements.

    - :class:`~simba.Framework_objects.frameworkCounter`: Used for counting elements of the same type in ASTRA and CSRTrack

    - :class:`~simba.Framework_objects.frameworkGroup`: Used for grouping elements together and controlling them all simultaneously.

    - :class:`~simba.Framework_objects.element_group`: Subclass of :class:`~simba.Framework_objects.frameworkGroup` for grouping elements.
      # TODO is this ever used?

    - :class:`~simba.Framework_objects.r56_group`: Subclass of :class:`~simba.Framework_objects.frameworkGroup` for grouping elements with an R56.
      # TODO is this ever used?

    - :class:`~simba.Framework_objects.chicane`: Subclass of :class:`~simba.Framework_objects.frameworkGroup` for a 4-dipole bunch compressor chicane.

    - :class:`~simba.Framework_objects.getGrids`: Used for determining the appropriate number of space charge grids given a number of particles.
"""

import math
import os
import shutil
import subprocess
from pathlib import Path
from warnings import warn
import stat
import yaml
from copy import deepcopy
import time

from laura import LAURA
from laura.models.element_list import (
    SectionLattice,
    ElementList,
    flatten_occurrence,
)
from laura.models.magnetic import brho as laura_brho
from laura.models.physical import Position
from laura.models.element import PhysicalBaseElement, Quadrupole, Sextupole, Octupole
from laura.translator.converters.section import SectionLatticeTranslator

from . import exceptions
from .Modules.DeviceProgram import DeviceProgram
from .Modules.EnergyRamp import (
    RF_MODES,
    EnergyRamp,
    beta_from_p0c,
    rf_phase_slip,
    wrap_phase,
)
from .Modules.MathParser import MathParser
from .Framework_Settings import FrameworkSettings
from .FrameworkHelperFunctions import expand_substitution
from .Modules.Fields import field
from .Modules import Beams as rbf
from .Codes import Executables as exes
from .Modules import constants
from .Modules.constants import speed_of_light

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
    ClassVar,
    Dict,
    List,
    Any,
    Set,
)

OUTPUT_TURN_SEPARATOR = "-t"
"""Separates an element name from a turn index in an output beam filename,
multi-turn only."""

OUTPUT_LINE_SEPARATOR = "-"
"""Separates a line name from an element name in an output beam filename."""

if os.name == "nt":
    # from .Modules.symmlinks import has_symlink_privilege
    def has_symlink_privilege():
        return False

else:

    def has_symlink_privilege():
        return True


with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/type_conversion_rules.yaml",
    "r",
) as infile:
    type_conversion_rules = yaml.safe_load(infile)
    type_conversion_rules_Elegant = type_conversion_rules["elegant"]
    type_conversion_rules_Names = type_conversion_rules["name"]
    type_conversion_rules_Opal = type_conversion_rules["opal"]

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/Elegant/commands_Elegant.yaml",
    "r",
) as infile:
    commandkeywords_elegant = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/OPAL/commands_Opal.yaml",
    "r",
) as infile:
    commandkeywords_opal = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/Genesis/commands_Genesis.yaml",
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
    + "/Codes/Elegant/keyword_conversion_rules_elegant.yaml",
    "r",
) as infile:
    keyword_conversion_rules_elegant = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/Elegant/elements_Elegant.yaml",
    "r",
) as infile:
    elements_Elegant = yaml.safe_load(infile)


class runSetup(object):
    """
    Class defining settings for simulations that include multiple runs
    such as error studies or parameter scans.
    """

    def __init__(self):
        # define the number of runs and the random number seed
        self.nruns = 1
        self.seed = 0

        # init errorElement and elementScan settings as None
        self.elementErrors = None
        self.elementScan = None

    def setNRuns(self, nruns: int | float) -> None:
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

    def setSeedValue(self, seed: int | float) -> None:
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

    def loadElementErrors(self, file: str | dict) -> None:
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
            self.elementErrors = error_setup["elements"]
            self.elementScan = None

            # set the number of runs and random number seed, if available
            if "nruns" in error_setup:
                self.setNRuns(error_setup["nruns"])
            if "seed" in error_setup:
                self.setSeedValue(error_setup["seed"])

    def setElementScan(
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
        self.elementErrors = None


class frameworkObject(BaseModel):
    """
    Class defining a framework object, which is the base class for all elements
    in a simulation lattice. It provides methods to add properties, validate parameters,
    and handle various simulation-specific functionalities.
    """

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
                self.allowedkeywords = self.allowedkeywords | elementkeywords[self.objecttype]["framework_keywords"]
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

    def change_Parameter(self, key: str, value: Any) -> None:
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


class frameworkLattice(BaseModel):
    """
    Class defining a framework lattice object, which contains all elements and groups
    of elements in a simulation lattice. It also contains methods to manipulate and
    retrieve information about the elements and groups, as well as methods to run
    simulations and process results.

    See :ref:`getting-started` and :ref:`loading-a-lattice`.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    name: str
    """Name of the lattice, used as a prefix for output files and commands."""

    objectname: str = ""
    """Name of the lattice, used as a prefix for output files and commands."""

    objecttype: str = ""
    """Type of the lattice, used as a prefix for output files and commands."""

    file_block: Dict
    """File block containing input and output settings for the lattice."""

    colliding_outputs: Set[str] = set()
    """Element names another line in this run also writes an output file for.

    Set by :meth:`~simba.Framework.Framework.track` before anything is
    written. See :meth:`output_basename`."""

    machine: LAURA
    """LAURA model of the lattice"""

    elementObjects: Dict
    """Dictionary of element objects, where keys are element names and values are element instances."""

    groupObjects: Dict
    """Dictionary of group objects, where keys are group names and values are group instances."""

    runSettings: runSetup
    """Run settings for the lattice, including number of runs and random seed."""

    settings: FrameworkSettings
    """Instance of :class:`~simba.Framework_Settings.FrameworkSettings`"""

    executables: exes.Executables
    """Executable commands for running simulations, defined in the Executables class.
    See :class:`~simba.Framework.Codes.Executables.Executables` for more details."""

    global_parameters: Dict
    """Global parameters for the lattice, including master subdirectory and other configuration settings."""

    globalSettings: Dict
    """Global settings for the lattice."""

    allow_negative_drifts: bool = False
    """If True, allows negative drifts in the lattice."""

    _lsc_enable: bool = False
    """Flag to enable LSC drifts in the lattice. Off by default, as in LAURA, so
    that every code models the same physics unless asked for more; set
    ``lsc_enable: true`` on the line to turn it on."""

    _csr_enable: bool = True
    """Flag to enable CSR drifts in the lattice."""

    _wakefield_enable: bool = True
    """Flag to enable structure wakefields in the lattice."""

    _lsc_bins: int = 20
    """Number of bins for LSC drifts."""

    _csr_bins: int | None = None
    """Number of bins for CSR calculations, or None if nobody has chosen one."""

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
    """Downsampling of the incoming beam: every code tracks every
    ``sample_interval``-th particle, with the total charge kept. The beam is
    sampled once, as it is read (:meth:`load_input_beam`)."""

    globalSettings: Dict = {"charge": None}
    """Global settings for the lattice, including charge and other parameters."""

    groupSettings: Dict = {}
    """Group settings for the lattice, including group-specific parameters."""

    allElements: List = []
    """List of all element names in the lattice."""

    initial_twiss: Dict = {}
    """Initial Twiss parameters for the lattice, used for tracking and analysis."""

    ref_idx: int | None = None
    """Index of the incoming beam's reference particle; see :func:`load_input_beam`."""

    native_time: ClassVar[tuple[str, str]] = ("t", "s")
    """The code's own longitudinal time coordinate, as ``(name, units)``; see
    :meth:`native_time_scale`."""

    _reference_clock: tuple | None = None
    """Fixed when the input beam is read; see :attr:`reference_clock`."""

    _input_reference: dict | None = None
    """The incoming beam's means, before sampling; see :attr:`reference_p0c`."""

    _section: SectionLatticeTranslator = None
    """LAURA SectionLatticeTranslator object"""

    _start_s: float = None
    """Cached s position of the start of the lattice; see :func:`start_s`."""

    remote_setup: Dict = {}
    """Dictionary containing parameters for running executables remotely."""

    files: List = []
    """List of all files needed to run the lattice."""

    code: str = None
    """Code to run the lattice."""

    supports_turns: ClassVar[bool] = False
    """Whether this code can track a line more than once. Which codes can is
    :meth:`codes_that_can` ``("supports_turns")``, as for every flag here."""

    supports_periodic: ClassVar[bool] = False
    """Whether this code can be asked for the *periodic* (closed) optics solution
    rather than propagating the incoming beam's Twiss."""

    supports_frequency_map: ClassVar[bool] = False
    """Whether this code can produce a tune footprint over a tracked grid; see
    :meth:`run_frequency_map`."""

    supports_single_particle: ClassVar[bool] = False
    """Whether this code can run in :meth:`single_particle` mode -- tracking
    13 probes and carrying the distribution through the map they measure,
    rather than tracking every macroparticle."""

    supports_dynamic_aperture: ClassVar[bool] = False
    """Whether this code can run a dynamic-aperture scan; see
    :meth:`run_dynamic_aperture`."""

    supports_nsuperperiods: ClassVar[bool] = False
    """Whether this code can track one sector of an N-fold-symmetric ring N
    times per turn; see :meth:`nsuperperiods`."""

    radiates_by_default: ClassVar[bool] = False
    """Whether this code radiates with no asking."""

    supports_radiation: ClassVar[bool] = False
    """Whether simba can switch synchrotron radiation on for this code; see
    :meth:`check_radiation_supported`."""

    supports_programs: ClassVar[bool] = False
    """Whether this code can vary an element's strength from turn to turn;
    see :class:`~simba.Modules.DeviceProgram.DeviceProgram`. Any code with a
    per-turn loop of its own can, which is most of the ring codes."""

    supports_ramp: ClassVar[bool] = False
    """Whether this code can track an energy ramp, the reference momentum
    changing from turn to turn; see :class:`~simba.Modules.EnergyRamp.EnergyRamp`."""

    electrons_only: ClassVar[bool] = False
    """Whether this code tracks electrons and nothing else; any other beam is
    refused as it is read, see :meth:`check_species`."""

    native_rf: ClassVar[str | None] = None
    """How this code's cavities keep time over many passes left to themselves,
    for :meth:`rf_phase_corrections` to subtract: ``fixed`` or ``synchronous``. 
    ``None`` where simba cannot move a cavity's phase pass
    by pass, so cannot impose :attr:`rf_mode`."""

    rf_phase_sign: ClassVar[float] = 1.0
    """The code's cavity phase moves by this times a phase the reference
    sees; measured per code. See :meth:`rf_phase_shifts`."""

    rf_phase_per_radian: ClassVar[float] = 180 / math.pi
    """The code's cavity phase units per radian: degrees, unless the code
    says otherwise."""

    _rf_corrections: dict | None = None
    """This run's :meth:`rf_phase_corrections`; see :meth:`begin_rf_phases`."""

    _rf_phase0: dict | None = None
    """Each moved cavity's phase as the code was given it this run, by name;
    see :meth:`apply_rf_phases`."""

    otm_convention: ClassVar[str] = ""
    """The coordinate order :attr:`one_turn_map` is written in, as this code
    writes it. :meth:`one_turn_map_canonical` converts it.

    * The transverse blocks need no conversion at all.
    * **The longitudinal block is three different problems.** Ocelot differs
      from MAD-X by a sign, Xsuite by a factor ``beta0**2``, and elegant by an
      *additive* term -- its fifth coordinate is geometric path length, so a
      drift has ``R56 = 0``.
    """

    otm_longitudinal_sign: ClassVar[int] = 1
    """Sign of this code's fifth coordinate against the canonical one (Xsuite, Bmad)."""

    otm_longitudinal_scale: ClassVar[int | None] = None
    """Power of ``beta0`` in the magnitude of the longitudinal conversion; see
    :meth:`one_turn_map_canonical`. ``None`` means the conversion is not a rescale (elegant)."""

    optics_summary: Any = None
    """The code's own tune and chromaticity; see :meth:`read_optics_summary`."""

    dynamic_aperture: Any = None
    """Last :meth:`run_dynamic_aperture` result, so a scan survives the call
    that produced it the way :attr:`one_turn_map` does."""

    frequency_map: Any = None
    """Last :meth:`run_frequency_map` result."""

    closed_orbit: Any = None
    """The periodic orbit at the start of the line, as a 6-vector in this
    code's :attr:`otm_convention`; see :meth:`read_closed_orbit`."""

    one_turn_map: Any = None
    """The 6x6 linear map of one turn, in :attr:`otm_convention` coordinates.

    ``None`` unless the line is :meth:`periodic` and the code can produce one.
    Read after the run by :meth:`read_one_turn_map`.
    """

    program_attributes: ClassVar[dict] = {}
    """``(horizontal, vertical)`` attribute an unqualified program sets, per
    element type as this code names it; see :meth:`program_attribute`."""

    def model_post_init(self, __context):
        # super().model_post_init(__context)
        for key, value in list(self.elementObjects.items()):
            setattr(self, key, value)
        self.allElements = list(self.elementObjects.keys())
        self.objectname = self.name
        self.remote_setup = {}
        self.files = []

        # define settings for simulations with multiple runs
        self.updateRunSettings(self.runSettings)
        if not isinstance(self.file_block, dict):
            raise ValueError("file_block must be a dictionary.")
        if "groups" in self.file_block:
            if self.file_block["groups"] is not None:
                self.groupSettings = self.file_block["groups"]
        if "input" in self.file_block:
            if "sample_interval" in self.file_block["input"]:
                self.sample_interval = self.file_block["input"]["sample_interval"]
        else:
            self.file_block.update({"input": {}})
        self._apply_collective_settings()
        self.globalSettings = self.settings["global"]
        self.update_groups()

    def __setattr__(self, name, value):
        if name in frameworkLattice.model_fields or name in self.__private_attributes__:
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

    def _apply_collective_settings(self) -> None:
        """Take ``csr_enable`` / ``lsc_enable`` from this line's settings block.

        Whether CSR and LSC are worth modelling is a property of the line, not
        of the elements in it.
        """
        for flag in ("csr_enable", "lsc_enable"):
            stated = self.file_block.get(flag)
            if stated is not None:
                setattr(self, flag, bool(stated))

    def _apply_radiation_to_section(self) -> None:
        """Push :meth:`radiation` onto LAURA's ``sr_enable``/``isr_enable``."""
        model = self.radiation
        if model is None:
            return
        for element in self.elementObjects.values():
            try:
                element.simulation.sr_enable = model != "off"
                element.simulation.isr_enable = model == "quantum"
            except (AttributeError, ValueError):
                pass
        try:
            self.section.sr_enable = model != "off"
            self.section.isr_enable = model == "quantum"
        except (AttributeError, ValueError):
            pass

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
        for elem in self.elementObjects.values():
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

        Reads 20 until somebody chooses otherwise, either here or on the machine
        section this lattice cuts.
        """
        if self._csr_bins is not None:
            return self._csr_bins
        stated = getattr(self._machine_space_charge(), "number_of_bins", None)
        return 20 if stated is None else stated

    @csr_bins.setter
    def csr_bins(self, csr: int) -> None:
        self._csr_bins = csr
        for elem in self.elementObjects.values():
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
        for elem in self.elementObjects.values():
            try:
                elem.simulation.lsc_enable = lsc
            except ValueError:
                pass
            except AttributeError:
                pass

    @property
    def lsc_in_use(self) -> bool:
        """Whether any element in this lattice actually asks the code for LSC."""
        return any(
            getattr(getattr(elem, "simulation", None), "lsc_enable", False)
            for elem in self.elementObjects.values()
        )

    @property
    def wakefield_enable(self) -> bool:
        """
        Property to get or set the wakefield enable flag. When False, the
        structure wakefields of accelerating cavities are not applied.
        The wakefield definitions themselves are
        left intact, so the flag can be toggled back on.
        """
        return self._wakefield_enable

    @wakefield_enable.setter
    def wakefield_enable(self, wake: bool) -> None:
        self._wakefield_enable = wake
        self.section.wakefield_enable = wake
        for elem in self.elementObjects.values():
            try:
                elem.simulation.wakefield_enable = wake
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
        for elem in self.elementObjects.values():
            try:
                elem.simulation.lsc_bins = lsc
            except ValueError:
                pass
            except AttributeError:
                pass

    @property
    def turns(self) -> int:
        """How many times this line is tracked; ``1`` unless the settings say.

        A turn count is a tracking setting rather than lattice data, so it is
        read from the ``files:`` block::

            files:
              RING:
                code: elegant
                tracking: {turns: 1000}
        """
        tracking = self.file_block.get("tracking") or {}
        return int(tracking.get("turns", 1))

    @property
    def periodic(self) -> bool:
        """
        Whether to ask the code for the closed (periodic) optics solution.

        Unlike :meth:`turns`, this is **not** primarily a tracking setting:
        whether the reference orbit closes is a fact about the lattice, and
        LAURA already records it as section ``geometry``
        (:meth:`_machine_geometry`).

        The ``tracking`` block overrides that either way, which is what an
        injection-mismatch study on a real ring needs::

            files:
              RING:
                code: elegant
                tracking: {turns: 1000, periodic: false}

        Returns
        -------
        bool
            True if section geometry is closed
        """
        tracking = self.file_block.get("tracking") or {}
        if "periodic" in tracking:
            return bool(tracking["periodic"])
        return self.closed_geometry

    @property
    def closed_geometry(self) -> bool:
        """Whether LAURA records this line's reference orbit as closing.
        :meth:`periodic` is this plus an override, because asking for the
        closed optics solution is a *choice*; whether the machine is a ring
        is not.

        Returns
        -------
        bool
            True if LAURA's section geometry is closed
        """
        geometry = self._machine_geometry()
        return getattr(geometry, "value", geometry) == "closed"

    @property
    def radiation(self) -> str | None:
        """
        Synchrotron-radiation model for this line, or None for none.

        ``mean`` gives damping and the energy loss; ``quantum`` adds the
        excitation, and only with both does an equilibrium emittance exist;
        ``None`` means not stated, and leaves every
        code on its own default -- which is not the same default everywhere.

        A tracking setting, like :meth:`turns`::

            files:
              RING:
                code: xsuite
                tracking: {turns: 100000, radiation: quantum}
        """
        tracking = self.file_block.get("tracking") or {}
        if "radiation" not in tracking:
            return None
        model = tracking["radiation"]
        if model in (None, False):
            return "off"
        return "mean" if model is True else str(model)

    def check_radiation(self, turns_that_matter: int = 10000) -> None:
        """
        Warn when a long lepton ring is tracked with radiation switched off.

        Parameters
        ----------
        turns_that_matter: int
            How many turns are enough for the beam to reach equilibrium if
            radiation is on
        """
        if self.radiation is not None or not self.periodic:
            return
        if self.radiates_by_default:
            return
        if self.turns < turns_that_matter:
            return
        warn(exceptions.NoRadiationWarning(self.objectname, self.turns))

    @classmethod
    def codes_that_can(cls, flag: str) -> str:
        """
        The codes with `flag` set, as the sentence a warning that this one
        cannot ends on.

        Parameters
        ----------
        flag: str
            A ``supports_*`` class flag

        Returns
        -------
        str
        """
        from . import Framework_lattices  # noqa: F401 -- defines every code's class

        def subclasses(klass):
            for sub in klass.__subclasses__():
                yield sub
                yield from subclasses(sub)

        names = sorted({
            sub.model_fields["code"].default
            for sub in subclasses(frameworkLattice)
            if sub.__module__.startswith("simba.Codes.")
            and getattr(sub, flag, False)
            and sub.model_fields["code"].default
        })
        if not names:
            return "No code can."
        if len(names) == 1:
            return f"{names[0]} is the code that can."
        return f"{', '.join(names[:-1])} and {names[-1]} are the codes that can."

    def check_periodic_supported(self) -> None:
        """Warn when the periodic solution was asked for and this code cannot."""
        if self.periodic and not self.supports_periodic:
            warn(exceptions.PeriodicUnsupportedWarning(
                self.objectname, self.code, self.codes_that_can("supports_periodic")
            ))

    def check_radiation_supported(self) -> None:
        """Warn when a radiation model was asked for and this code has no switch
        for it; :meth:`radiation` is then not applied at all."""
        if self.radiation is None or self.supports_radiation:
            return
        warn(exceptions.RadiationUnsupportedWarning(
            self.objectname, self.code, self.radiation, self.radiates_by_default,
            self.codes_that_can("supports_radiation"),
        ))

    @property
    def write_turns(self) -> bool:
        """
        Whether a multi-turn run writes a beam file per turn.
        Off by default, and a multi-turn run writes what a single-turn run writes:
        one file per screen, holding the last turn. Turn-resolved output
        is then something you ask for::

            files:
              RING:
                code: xsuite
                tracking: {turns: 1000, write_turns: true}

        Has no effect on a single-turn run, which was always one file per
        screen.
        """
        tracking = self.file_block.get("tracking") or {}
        return bool(tracking.get("write_turns", False))

    @property
    def programs(self) -> list:
        """
        Elements whose strength is a program over turn number.

        A tracking setting, like :meth:`turns`, and the half of R19 that is
        the *study* rather than the hardware -- when a kicker fires and at
        what amplitude::

            files:
              RING:
                code: elegant
                tracking:
                  turns: 10
                  programs:
                    - element: KICK1
                      turns:  [1, 4, 5]
                      values: [0.0, 1.0e-3, 0.0]
                      interpolation: hold

        Turns are 1-based, ``values`` are in the element attribute's own
        units, and the default rule is ``hold`` and is no code's default:
        see :mod:`simba.Modules.DeviceProgram`, which is where all three of
        those are argued.

        Returns
        -------
        list
            :class:`~simba.Modules.DeviceProgram.DeviceProgram`, one per
            entry. An entry simba cannot read warns and is dropped, rather
            than taking the run down with it.
        """
        tracking = self.file_block.get("tracking") or {}
        entries = tracking.get("programs") or []
        if isinstance(entries, dict):
            entries = [entries]
        programs = []
        for entry in entries:
            try:
                programs.append(DeviceProgram.from_dict(entry))
            except ValueError as error:
                warn(exceptions.UnreadableSettingWarning(self.objectname, error))
        return programs

    def check_programs_supported(self) -> None:
        """Warn when a device program was asked for and this code cannot run one."""
        if not self.programs or self.supports_programs:
            return
        warn(exceptions.ProgramsUnsupportedWarning(
            self.objectname, self.code, [p.element for p in self.programs],
            self.codes_that_can("supports_programs"),
        ))

    def check_programs_fit(self) -> None:
        """
        Warn when a program and the run do not cover the same turns.

        Two ways to author a pulse that is not the one intended, both of
        which track perfectly: knots past the last turn, and a last knot the
        run then sits on for thousands of turns.
        """
        for program in self.programs:
            if program.last_turn > self.turns:
                warn(exceptions.ProgramOverrunWarning(
                    self.objectname, program.element, program.last_turn, self.turns
                ))
            elif program.last_turn < self.turns and program.values[-1]:
                warn(exceptions.ProgramHeldWarning(
                    self.objectname, program.element, program.last_turn,
                    program.values[-1], self.turns - program.last_turn,
                ))

    @property
    def ramp(self) -> EnergyRamp | None:
        """
        The reference momentum as a program over turn number, if there is one.

        A tracking setting, like :meth:`programs`::

            files:
              RING:
                code: xsuite
                tracking:
                  turns: 2000
                  ramp:
                    turns: [1, 1000]
                    momentum: [1.0e9, 2.0e9]

        :mod:`simba.Modules.EnergyRamp` sets out the model every backend
        follows.

        Returns
        -------
        :class:`~simba.Modules.EnergyRamp.EnergyRamp` | None
            The ramp, or None if there is none. A ramp simba cannot read
            warns and is ignored
        """
        tracking = self.file_block.get("tracking") or {}
        entry = tracking.get("ramp")
        if not entry:
            return None
        try:
            return EnergyRamp.from_dict(entry)
        except ValueError as error:
            warn(exceptions.UnreadableSettingWarning(self.objectname, error))
            return None

    @property
    def ramped(self) -> bool:
        """Whether this run changes the reference momentum turn by turn: a
        :meth:`ramp`, a code that can follow one, and more than one turn."""
        return self.supports_ramp and self.turns > 1 and self.ramp is not None

    @property
    def fixed_reference(self) -> bool:
        """
        Whether the reference momentum is the line's own rather than the beam's:
        under a :meth:`ramp`, which owns it, and in a ring, whose reference is the
        design momentum.
        Then a cavity accelerates particles and leaves the reference alone -- an
        off-crest ring cavity drives synchrotron motion about the reference, it
        does not carry the reference with it.
        """
        return self.ramped or self.periodic or self.closed_geometry

    def _apply_fixed_reference_to_section(self) -> None:
        """Push :attr:`fixed_reference` onto LAURA's cavity ``change_p0``."""
        if not self.fixed_reference:
            return
        for element in self.elementObjects.values():
            simulation = getattr(element, "simulation", None)
            if simulation is not None and "change_p0" in type(simulation).model_fields:
                simulation.change_p0 = 0

    @property
    def rest_energy(self) -> float:
        """The beam's rest energy in eV."""
        beam = (self.global_parameters or {}).get("beam")
        for name, scale in (
            ("particle_rest_energy_eV", 1.0),
            ("particle_mass", constants.speed_of_light**2 / constants.elementary_charge),
        ):
            value = getattr(beam, name, None) if beam is not None else None
            value = getattr(value, "val", value)
            if value is not None and np.size(value):
                return float(np.mean(value)) * scale
        return constants.m_e * constants.speed_of_light**2 / constants.elementary_charge

    @property
    def reference_charge(self) -> int:
        """The tracked species' charge, in units of e."""
        beam = (self.global_parameters or {}).get("beam")
        charge = getattr(getattr(beam, "particle_charge", None), "val", None)
        if charge is not None and np.size(charge):
            sign = int(np.sign(np.mean(charge)))
            if sign:
                return sign
        return -1

    def check_species(self) -> None:
        """
        Refuse a beam if :attr:`electrons_only`.

        Raises
        ------
        :class:`~simba.exceptions.WrongSpeciesError`
            A ``ValueError``, if the beam's rest energy is not an electron's,
            or its charge is not negative.
        """
        if not self.electrons_only:
            return
        electron = constants.m_e * constants.speed_of_light**2 / constants.elementary_charge
        if self.reference_charge != -1 or not np.isclose(self.rest_energy, electron, rtol=1e-6):
            raise exceptions.WrongSpeciesError(
                self.objectname, self.code, self.rest_energy, self.reference_charge
            )

    def ramp_p0c(self, turn: int) -> float | None:
        """
        The reference momentum on `turn` under :meth:`ramp`.

        Parameters
        ----------
        turn: int
            Turn number, 1-based

        Returns
        -------
        float | None
            ``p0c`` in eV, or None unless this run is :meth:`ramped`
        """
        if not self.ramped:
            return None
        return self.ramp.p0c_at(turn, self.rest_energy)

    def ramp_clock(self, pass_length: float | None = None):
        """
        Seconds at the start of each turn of a ramped run.

        Parameters
        ----------
        pass_length: float | None
            Length of one pass in metres; this line's length if not given.

        Returns
        -------
        :class:`~simba.Modules.EnergyRamp.RampClock` | None
            The clock, or None unless this run is :meth:`ramped`
        """
        if not self.ramped:
            return None
        if pass_length is None:
            pass_length = self.pass_length
        return self.ramp.clock(
            self.turns, pass_length, self.rest_energy, self.passes_per_turn
        )

    @property
    def rf_mode(self) -> str:
        """
        How the RF keeps time from pass to pass, the same in every code::

            files:
              RING:
                code: madx
                tracking:
                  turns: 1000
                  rf: follow    # or fixed

        * ``follow``, the default: each cavity's frequency scales with the
          reference speed, as a booster's RF programme tracks the revolution
          frequency up a ramp;
        * ``fixed``: each cavity runs at its own frequency throughout.

        :func:`~simba.Modules.EnergyRamp.rf_phase_slip`
        has the detail, and :meth:`rf_phase_corrections` what each code needs.

        Returns
        -------
        str
            ``follow`` or ``fixed``; anything else warns and is ``follow``
        """
        tracking = self.file_block.get("tracking") or {}
        mode = str(tracking.get("rf", "follow")).lower()
        if mode not in RF_MODES:
            warn(exceptions.UnknownRFModeWarning(self.objectname, mode, RF_MODES))
            return "follow"
        return mode

    def pass_p0c(self) -> np.ndarray:
        """
        The reference momentum on every pass of the run: the :meth:`ramp`'s,
        or else the entering beam's mean throughout; see :attr:`reference_clock`.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn`` values of ``p0c``, in eV
        """
        clock = getattr(self, "_reference_clock", None)
        if clock is not None:
            return clock[3]
        passes = self.turns * self.passes_per_turn
        if self.ramped:
            return self.ramp.p0c_per_pass(
                self.turns, self.rest_energy, self.passes_per_turn
            )[:passes]
        return np.full(passes, self.reference_p0c)

    def _input_mean(self, coord: str) -> float:
        """The incoming beam's mean ``coord``, as :meth:`load_input_beam` read
        it; else the current beam's."""
        if self._input_reference is not None and coord in self._input_reference:
            return self._input_reference[coord]
        beam = self.global_parameters["beam"]
        return float(np.mean(getattr(beam, coord).val))

    @property
    def reference_p0c(self) -> float:
        """
        The incoming beam's reference momentum, in eV/c: its mean ``cp``,
        taken before :meth:`sample_beam`.

        Every code takes its reference from here rather than from the beam it
        is handed. A sampled beam's mean is a different number, by about
        :math:`\\sigma_\\delta/\\sqrt{N}`, and a magnet's field is its
        strength times the reference rigidity. So a sampled run used to track
        every particle it kept through slightly different magnets.

        Returns
        -------
        float
            ``p0c`` in eV
        """
        return self._input_mean("cp")

    @property
    def reference_energy(self) -> float:
        """Total energy of a particle at :attr:`reference_p0c`, in eV."""
        return float(np.hypot(self.reference_p0c, self.rest_energy))

    @property
    def reference_t0(self) -> float:
        """The incoming beam's mean ``t``, before sampling, in s: the time a
        code without a clock of its own centres the bunch on; see
        :attr:`reference_p0c`."""
        return self._input_mean("t")

    @property
    def reference_z0(self) -> float:
        """The incoming beam's mean ``z``, before sampling, in m; see
        :attr:`reference_t0`."""
        return self._input_mean("z")

    def pass_beta0(self) -> float | np.ndarray:
        """
        Reference speed / ``c`` on every pass; see :meth:`pass_p0c`.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn`` values
        """
        return beta_from_p0c(self.pass_p0c(), self.rest_energy)

    def pass_index(self, turn: int, sector: int = 1) -> int:
        """
        0-based index of a pass: ``sector`` of ``turn``, both 1-based.
        A turn is :attr:`passes_per_turn` passes of this line.
        """
        return (turn - 1) * self.passes_per_turn + sector - 1

    def last_pass(self, turn: int) -> int:
        """0-based index of the last pass of ``turn`` (1-based): the one a turn's
        outputs are recorded on."""
        return turn * self.passes_per_turn - 1

    @property
    def uses_reference_clock(self) -> bool:
        """Whether every code reports ``t`` on :meth:`reference_time`."""
        return self.fixed_reference

    def reset_reference_clock(self) -> None:
        """Fix :meth:`reference_time` for this run, from the beam just read."""
        self._reference_clock = None
        beam = (self.global_parameters or {}).get("beam")
        t = getattr(getattr(beam, "t", None), "val", None)
        if t is None or not np.size(t):
            return
        p0c = self.pass_p0c()
        beta0 = beta_from_p0c(p0c, self.rest_energy)
        starts = np.concatenate(
            ([0.0], np.cumsum(self.pass_length / (beta0 * speed_of_light)))
        )
        self._reference_clock = (self.reference_t0, starts, beta0, p0c)

    @property
    def reference_clock(self) -> tuple:
        """
        ``(t0, starts, beta0, p0c)``: the incoming beam's mean ``t``, then per
        pass its start time after ``t0``, reference speed over ``c`` and
        momentum in eV; see :meth:`reference_time`.
        """
        if self._reference_clock is None:
            self.reset_reference_clock()
        return self._reference_clock

    def reference_time(self, s: float, pass_index: int = 0) -> float:
        """
        Absolute time the reference particle reaches ``s`` on a pass.

        ``T_j(s) = t0 + sum_{k<j} C / (beta_k c) + s / (beta_j c)``, with ``t0``
        the incoming beam's mean ``t``, ``C`` the length of one pass and
        ``beta_k`` the reference speed on pass ``k`` (:meth:`pass_beta0`).
        Normalized for all codes; :meth:`time_to_native` gives the code's reference.

        Parameters
        ----------
        s: float
            Metres from the lattice entrance, within the pass
        pass_index: int
            0-based pass; see :meth:`pass_index`

        Returns
        -------
        float
            Seconds
        """
        t0, starts, beta0, _ = self.reference_clock
        return float(t0 + starts[pass_index] + s / (beta0[pass_index] * speed_of_light))

    def pass_start_times(self) -> np.ndarray:
        """
        The absolute time each pass of the run starts, on :meth:`reference_time`.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn`` values, in seconds
        """
        t0, starts, _, _ = self.reference_clock
        return t0 + starts[: self.turns * self.passes_per_turn]

    @staticmethod
    def pass_staircase(starts, values) -> tuple:
        """
        A per-pass table for an element that reads it against time: flat for
        a quarter pass either side of each pass's start, so a bunch off the
        reference by up to half an RF period still reads its own pass's value.

        Parameters
        ----------
        starts: array-like
            Seconds at the start of each pass
        values: array-like
            One per pass

        Returns
        -------
        tuple
            ``(times, values)``, two knots per pass
        """
        starts = np.asarray(starts, dtype=float)
        periods = np.diff(starts) if len(starts) > 1 else np.ones(1)
        before = np.concatenate(([periods[0]], periods))
        after = np.append(periods, periods[-1])[: len(starts)]
        times, table = [], []
        for start, back, ahead, value in zip(starts, before, after, values):
            times += [start - back / 4, start + ahead / 4]
            table += [value, value]
        return times, table

    @classmethod
    def native_time_scale(cls, beta0: float) -> float | None:
        """
        How the code's own time coordinate (:attr:`native_time`) relates to ``t``:
        ``native = scale * (t - reference_time)``.

        Parameters
        ----------
        beta0: float
            The reference speed over ``c``

        Returns
        -------
        float | None
            ``scale``, or None if the code's own ``t`` is already absolute
            (elegant's is)
        """
        return None

    def time_from_native(
        self, native, s: float, pass_index: int, beta0=None
    ) -> np.ndarray:
        """
        Absolute ``t`` from the code's own time coordinate.

        Parameters
        ----------
        native: array-like
            The code's coordinate, in :attr:`native_time` units
        s: float
            Metres from the lattice entrance, within the pass
        pass_index: int
            0-based pass
        beta0: float | array-like | None
            The reference speed over ``c``, per particle if the code has it;
            the clock's for the pass (:attr:`reference_clock`) if not given

        Returns
        -------
        np.ndarray
            Seconds
        """
        if beta0 is None:
            beta0 = self.reference_clock[2][pass_index]
        scale = self.native_time_scale(beta0)
        native = np.asarray(native, dtype=float)
        if scale is None:
            return native
        return self.reference_time(s, pass_index) + native / scale

    def time_to_native(self, t, s: float, pass_index: int, beta0=None) -> np.ndarray:
        """
        The code's own time coordinate from absolute ``t``; the inverse of
        :meth:`time_from_native`, and the same arguments.
        """
        if beta0 is None:
            beta0 = self.reference_clock[2][pass_index]
        scale = self.native_time_scale(beta0)
        t = np.asarray(t, dtype=float)
        if scale is None:
            return t
        return scale * (t - self.reference_time(s, pass_index))

    def native_times(self, beam, element: str | None = None, turn: int | None = None):
        """
        What the code itself would call the time of each particle in a beam
        SIMBA wrote.

        Parameters
        ----------
        beam:
            A beam this line wrote
        element: str | None
            Where; the end of the line if not given
        turn: int | None
            Which turn; the beam's own :attr:`turn` if not given, else 1

        Returns
        -------
        np.ndarray
            In :attr:`native_time` units
        """
        element = element or self.end
        if turn is None:
            turn = getattr(beam, "turn", None) or 1
        s = self.getSValues(as_dict=True)[element]
        return self.time_to_native(beam.t.val, s, self.last_pass(turn))

    def accelerating_cavities(self) -> dict:
        """
        This line's accelerating cavities; deflecting and crab cavities do not.

        Returns
        -------
        dict
            The elements, by name
        """
        cavities = {}
        for name, element in self.elements.items():
            hardware = str(getattr(element, "hardware_type", "") or "").lower()
            if "cavity" not in hardware or "deflect" in hardware or "crab" in hardware:
                continue
            cavities[name] = element
        return cavities

    def rf_phase_corrections(self) -> dict:
        """
        How far to move each cavity's phase on each pass for this code to run
        :attr:`rf_mode`: the slip the mode asks for less the slip of the code's
        own :attr:`native_rf`, both from
        :func:`~simba.Modules.EnergyRamp.rf_phase_slip`.

        Returns
        -------
        dict
            Radians, one per pass, by cavity name. Empty when nothing needs moving

        Warns
        -----
        :class:`~simba.exceptions.RFPhasesUnsupportedWarning`
            If corrections are needed and this code cannot make them.
        """
        passes = self.turns * self.passes_per_turn
        # a cavity with no voltage or no frequency does nothing at any phase
        cavities = {
            name: element
            for name, element in self.accelerating_cavities().items()
            if self.cavity_voltage(element)
            and getattr(getattr(element, "cavity", None), "frequency", None)
        }
        if passes <= 1 or not cavities:
            return {}
        beta0 = self.pass_beta0()
        s_in = self.getSValues(as_dict=True, at_entrance=True)
        s_out = self.getSValues(as_dict=True)
        corrections = {}
        for name, element in cavities.items():
            frequency = element.cavity.frequency
            s = 0.5 * (s_in[name] + s_out[name])
            args = (float(frequency), s, self.pass_length, beta0)
            correction = wrap_phase(
                rf_phase_slip(self.rf_mode, *args)
                - rf_phase_slip(self.native_rf or "synchronous", *args)
            )
            if np.max(np.abs(correction)) > 1e-9:
                corrections[name] = correction
        if corrections and self.native_rf is None:
            warn(exceptions.RFPhasesUnsupportedWarning(
                self.objectname, self.rf_mode, corrections, self.code
            ))
            return {}
        return corrections

    def cavity_phase(self, name: str) -> float | None:
        """
        A cavity's phase as the code has it now, in its own units (see
        :attr:`rf_phase_per_radian`); for :meth:`apply_rf_phases`.
        Overridden by the codes that move phases pass by pass.

        Parameters
        ----------
        name: str
            The cavity, as simba names it

        Returns
        -------
        float | None
            None if the code's lattice has no such cavity
        """
        return None

    def set_cavity_phase(self, name: str, phase: float) -> None:
        """
        Set a cavity's phase, in the code's own units; the other half of
        :meth:`cavity_phase`.
        """
        raise NotImplementedError(
            f"{self.code} reads cavity phases but cannot set them"
        )

    def rf_phase_shifts(self, correction) -> np.ndarray:
        """
        A correction from :meth:`rf_phase_corrections`, as a move of the
        code's own phase attribute: :attr:`rf_phase_sign` times
        :attr:`rf_phase_per_radian` times the phase the reference sees.
        """
        return self.rf_phase_sign * self.rf_phase_per_radian * np.asarray(correction)

    def begin_rf_phases(self) -> dict:
        """
        Take this run's :meth:`rf_phase_corrections`, and forget every cavity
        phase read on a previous run.

        Returns
        -------
        dict
            The corrections
        """
        self._rf_corrections = self.rf_phase_corrections()
        self._rf_phase0 = {}
        return self._rf_corrections

    def apply_rf_phases(self, pass_index: int | None) -> None:
        """
        Move each cavity's phase for pass `pass_index`, so the code's
        cavities run as :attr:`rf_mode` asks; see :meth:`rf_phase_corrections`.

        Each phase is moved from the one the code was given, read (by
        :meth:`cavity_phase`) the first time the cavity is there to read.

        Parameters
        ----------
        pass_index: int | None
            0-based pass, counting superperiods (:meth:`pass_index`); None
            puts every cavity back as it was given
        """
        if not self._rf_corrections:
            return
        for name, correction in self._rf_corrections.items():
            if name not in self._rf_phase0:
                phase = self.cavity_phase(name)
                if phase is None:
                    continue
                self._rf_phase0[name] = phase
            shift = 0.0 if pass_index is None else self.rf_phase_shifts(correction)[pass_index]
            self.set_cavity_phase(name, self._rf_phase0[name] + shift)

    def run_turns(self, track_pass, start_turn=None) -> None:
        """
        The turn loop of a code SIMBA drives a pass at a time:
        programs set per turn, RF phases moved per pass,
        and the line put back as turn 1 had it at the end
        (:meth:`end_turns`), for the optics and anything run after.

        Parameters
        ----------
        track_pass: callable
            ``track_pass(turn, pass_index, name_turn, record)``: track one
            pass. ``name_turn`` is for :meth:`output_basename`, and ``record``
            is whether this pass's beams are written at all: the last pass of
            a turn :meth:`output_turns` keeps
        start_turn: callable, optional
            ``start_turn(turn)``, called once a turn's programs are set
        """
        wanted = dict(self.output_turns())
        self.begin_rf_phases()
        passes = self.passes_per_turn
        try:
            for turn in range(1, self.turns + 1):
                key = turn if self.turns > 1 else None
                self.apply_programs(turn)
                if start_turn is not None:
                    start_turn(turn)
                for sector in range(1, passes + 1):
                    index = self.pass_index(turn, sector)
                    self.apply_rf_phases(index)
                    track_pass(
                        turn, index, wanted.get(key), sector == passes and key in wanted
                    )
        finally:
            self.end_turns()

    def end_turns(self) -> None:
        """
        Put the line back as turn 1 had it: every cavity's phase as given,
        and every program at turn 1.
        """
        self.apply_rf_phases(None)
        missing = set(self._rf_corrections or {}) - set(self._rf_phase0 or {})
        if missing:
            warn(exceptions.MissingCavitiesWarning(self.objectname, missing, self.code))
        if self.turns > 1:
            self.apply_programs(1)

    @property
    def rf_voltage(self) -> float:
        """
        Total accelerating voltage on one turn, in volts.
        Every accelerating cavity's amplitude, ignoring phase, times the
        passes in a turn.
        """
        total = sum(
            self.cavity_voltage(element)
            for element in self.accelerating_cavities().values()
        )
        return total * self.passes_per_turn

    @staticmethod
    def cavity_voltage(element) -> float:
        """
        A cavity's amplitude, ignoring phase, in volts.

        Parameters
        ----------
        element:
            The cavity

        Returns
        -------
        float
            ``|field_amplitude|``, or 0 if it has none
        """
        simulation = getattr(element, "simulation", None)
        try:
            amplitude = simulation.resolved("field_amplitude")
        except (AttributeError, TypeError, ValueError):
            amplitude = getattr(simulation, "field_amplitude", 0.0)
        return abs(float(amplitude or 0.0))

    def check_ramp(self) -> None:
        """
        Warn about a ramp this run will not track as written.
        The checks that need only the settings: see :meth:`check_ramp_beam`.
        """
        ramp = self.ramp
        if ramp is None:
            return
        if not self.supports_ramp:
            warn(exceptions.RampUnsupportedWarning(
                self.objectname, self.code, self.codes_that_can("supports_ramp")
            ))
            return
        if self.turns <= 1:
            warn(exceptions.RampOneTurnWarning(self.objectname))
            return
        if ramp.last_turn > self.turns:
            warn(exceptions.RampOverrunWarning(self.objectname, ramp.last_turn, self.turns))

    def check_ramp_beam(self) -> None:
        """
        Warn about a ramp the input beam will not follow.
        A beam that does not start on the ramp, and too little RF for the
        beam to follow it.
        """
        if not self.ramped:
            return
        ramp = self.ramp
        beam = (self.global_parameters or {}).get("beam")
        if beam is None:
            return
        rest_energy = self.rest_energy
        start = ramp.p0c_at(1, rest_energy)
        entering = self.reference_p0c
        if entering and abs(start / entering - 1) > 1e-3:
            warn(exceptions.OffRampWarning(self.objectname, start, entering))
        needed = float(np.max(np.abs(ramp.energy_gain_per_turn(self.turns, rest_energy))))
        if not needed:
            return
        voltage = self.rf_voltage
        if not voltage:
            warn(exceptions.RampWithoutRFWarning(self.objectname))
        elif needed > voltage:
            warn(exceptions.RampTooSteepWarning(self.objectname, needed, voltage))

    @property
    def pass_length(self) -> float:
        """Length of one pass of this line, in metres."""
        return float(
            self.machine.get_elements_s_pos(end=self.end)[self.end] - self.entrance_s
        )

    @property
    def revolution_period(self) -> float:
        """
        Seconds for one turn: ``passes * C / (beta0 * c)``.
        ``C`` is this line's length, and :meth:`passes_per_turn` is how
        many times a turn crosses it.

        Returns
        -------
        float
            Seconds per turn, or 0.0 if there is no beam to ask
        """
        beam = (self.global_parameters or {}).get("beam")
        if beam is None:
            return 0.0
        beta = float(np.mean(beam.BetaGamma) / np.mean(beam.gamma))
        if not beta:
            return 0.0
        return self.passes_per_turn * self.pass_length / (beta * speed_of_light)

    def program_is_vertical(self, name: str) -> bool:
        """
        Whether the programmed element steers vertically.

        Parameters
        ----------
        name: str
            Element name, as the lattice names it

        Returns
        -------
        bool
            True for a vertical element, False for a horizontal one or one
            whose type says nothing
        """
        element = self.elements.get(name)
        hardware = str(getattr(element, "hardware_type", "") or "")
        return hardware.lower().startswith("vertical")

    def program_attribute(self, program, element_type: str | None) -> str | None:
        """
        The attribute `program` sets: its own ``parameter``, else the one
        :attr:`program_attributes` gives the element's type, in the element's
        plane (:meth:`program_is_vertical`).

        Parameters
        ----------
        program: :class:`~simba.Modules.DeviceProgram.DeviceProgram`
            The program
        element_type: str | None
            The programmed element's type, as this code names it

        Returns
        -------
        str | None
            The attribute, or None if there is none to set
        """
        if element_type is None:
            warn(exceptions.ProgramMissingElementWarning(
                self.objectname, program.element, self.code
            ))
            return None
        if program.parameter is not None:
            return program.parameter
        planes = self.program_attributes.get(element_type)
        if planes is None:
            warn(exceptions.ProgramNoAttributeWarning(
                self.objectname, program.element, self.code, element_type
            ))
            return None
        return planes[1] if self.program_is_vertical(program.element) else planes[0]

    def apply_programs(self, turn: int) -> None:
        """
        Set each programmed element to its value for `turn`.

        Parameters
        ----------
        turn: int
            Turn number, 1-based
        """
        return None

    def output_turns(self) -> list:
        """
        ``(data_turn, name_turn)`` for each beam file a run should write.

        ``data_turn`` selects which turn's particles to write and
        ``name_turn`` is handed to :meth:`output_basename`, so the
        unsuffixed single-turn name survives when only the last turn is
        being kept.
        """
        if self.turns <= 1:
            return [(None, None)]
        if self.write_turns:
            return [(turn, turn) for turn in range(1, self.turns + 1)]
        return [(self.turns, None)]

    def beam_turn(self, turn: int | None) -> int:
        """
        Which turn a beam from an :meth:`output_turns` entry actually is.

        Parameters
        ----------
        turn: int | None
            Either half of an :meth:`output_turns` pair

        Returns
        -------
        int
            A 1-based turn number, never ``None``
        """
        return self.turns if turn is None else turn

    def link_handoff_beam(self) -> None:
        """
        Make sure the end of the line (or the last turn) has an unsuffixed beam file.
        """
        if self.turns <= 1 or not self.write_turns:
            return
        directory = (self.global_parameters or {}).get("master_subdir")
        if not directory:
            return
        stem = self.output_basename(self.end)
        handoff = Path(directory) / f"{stem}.openpmd.hdf5"
        if handoff.is_file():
            return
        last = Path(directory) / (
            f"{self.output_basename(self.end, turn=self.turns)}.openpmd.hdf5"
        )
        if not last.is_file():
            return
        shutil.copyfile(last, handoff)

    @property
    def da_settings(self) -> dict:
        """
        Grid for a dynamic-aperture scan, from the ``tracking`` block::

            files:
              RING:
                code: ocelot
                tracking:
                  turns: 1000
                  dynamic_aperture: {nx: 20, ny: 10, x_max: 0.02, y_max: 0.01}

        Returns
        -------
        dict
            Dictionary containing `dynamic_aperture` settings from the `tracking`
            block for this section.
        """
        tracking = self.file_block.get("tracking") or {}
        return tracking.get("dynamic_aperture") or {}

    def da_grid(self):
        """
        ``(xs, ys)`` starting amplitudes for a dynamic-aperture scan.

        Both start one step off zero rather than at it; see :meth:`da_settings`.

        Returns
        -------
        list
            Two numpy `linspace` with grid settings for the DA scan.
        """
        settings = self.da_settings
        nx = max(1, int(settings.get("nx", 10)))
        ny = max(1, int(settings.get("ny", 1)))
        x_max = float(settings.get("x_max", 0.01))
        y_max = float(settings.get("y_max", 0.001))
        return (
            np.linspace(x_max / nx, x_max, nx),
            np.linspace(y_max / ny, y_max, ny),
        )

    def dynamic_aperture_boundary(self, results) -> list:
        """
        Largest surviving ``x`` at each ``y``, from :meth:`run_dynamic_aperture`.

        A particle counts as surviving if it reached the last turn. Note the
        off-by-one.

        Parameters
        ----------
        results : list
            Results produced by :meth:`run_dynamic_aperture`.

        Returns
        -------
        list
            ``(y, x_max_surviving)`` pairs, ascending in ``y``. A ``y`` row
            where nothing survived is absent rather than zero.
        """
        survived = {}
        for x, y, turn in results:
            if turn >= self.turns - 1:
                survived[y] = max(survived.get(y, 0.0), x)
        return sorted(survived.items())

    @staticmethod
    def tune_from_harmonic(line_position: float, reference_tune: float) -> float:
        """
        Rebuild a tune from the harmonic position.

        ``freq_analysis`` reports ``|nearest integer - Q|``, not the
        fractional tune. The reference tune says which side of the
        integer to come back on.

        Parameters
        ----------
        line_position : float
            The harmonic position reported by `freq_analysis`, which is the
            absolute difference between the nearest integer and the tune.
        reference_tune : float
            The reference tune, which indicates which side of the nearest integer
            the tune should be reconstructed from.

        Returns
        -------
        float
            The reconstructed tune, taking into account the harmonic position
            and the reference tune.
        """
        nearest = round(reference_tune)
        return (
            nearest - line_position
            if reference_tune < nearest
            else nearest + line_position
        )

    @property
    def single_particle(self) -> bool:
        """
        Track 13 probes instead of the whole bunch, and carry the
        distribution through the map they measure.

        A tracking setting, like :meth:`turns`::

            files:
              RING:
                code: madx
                tracking: {single_particle: true}

        A linear reconstruction, buying a speed-up on some codes.

        Returns
        -------
        bool
            True if `single_particle` was asked, False otherwise.
        """
        tracking = self.file_block.get("tracking") or {}
        return bool(tracking.get("single_particle", False))

    @property
    def nsuperperiods(self) -> int:
        """
        How many times the line is traversed per turn.
        One sector of an N-fold-symmetric ring is a *superperiod*: it is open
        on its own and closes after N of them::

            files:
              RING:
                code: ocelot
                tracking: {turns: 1000, nsuperperiods: 4}

        A turn stays a turn: those settings track 4000 passes through the
        sector, and the beam files, the turn suffixes and anything else
        counted per turn still count 1000 of them.

        Returns
        -------
        int
            The declared count, or 1 -- which is the same as not asking.
        """
        tracking = self.file_block.get("tracking") or {}
        value = tracking.get("nsuperperiods", 1)
        try:
            count = int(value)
        except (TypeError, ValueError):
            warn(exceptions.BadSuperperiodsWarning(self.objectname, value))
            return 1
        if count < 1:
            warn(exceptions.BadSuperperiodsWarning(self.objectname, count))
            return 1
        return count

    @property
    def passes_per_turn(self) -> int:
        """
        How many times this code will actually traverse the line per turn.

        :meth:`nsuperperiods` is what was *asked* for; this is what will
        *happen*, so it is 1 on a code that cannot repeat the line.
        """
        return self.nsuperperiods if self.supports_nsuperperiods else 1

    def check_nsuperperiods_supported(self) -> None:
        """Warn when superperiods were asked for and cannot be given."""
        if self.nsuperperiods <= 1 or self.supports_nsuperperiods:
            return
        warn(exceptions.SuperperiodsUnsupportedWarning(
            self.objectname, self.code, self.nsuperperiods,
            self.codes_that_can("supports_nsuperperiods"),
        ))

    def check_single_particle_supported(self) -> None:
        """Warn when single-particle mode was asked for and cannot be given."""
        if self.single_particle and not self.supports_single_particle:
            warn(exceptions.SingleParticleUnsupportedWarning(
                self.objectname, self.code,
                self.codes_that_can("supports_single_particle"),
            ))

    def track_reference_particle(self) -> dict:
        """
        The reference particle's trajectory, turn by turn.

        What a ring study usually wants and bunch tracking does not give:
        not a distribution at each screen, but where one particle is on
        every turn.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py`` arrays of length :meth:`turns`, or
            ``{}`` if this code cannot produce one.
        """
        return {}

    def normalisation_twiss(self) -> dict:
        """
        Periodic Twiss and closed orbit for Courant-Snyder normalisation.

        Handed to :func:`~simba.Modules.Matrices.tune_diffusion` so a
        frequency map works in normalised coordinates.

        Returns
        -------
        dict
            Empty if the lattice has no one-turn map, in which case the
            tunes are taken from raw coordinates instead.
        """
        parameters = self.ring_parameters()
        wanted = (
            "beta_x",
            "alpha_x",
            "beta_y",
            "alpha_y",
            "closed_orbit_x",
            "closed_orbit_px",
            "closed_orbit_y",
            "closed_orbit_py",
        )
        twiss = {k: parameters[k] for k in wanted if k in parameters}
        if not all(k in twiss for k in ("beta_x", "alpha_x", "beta_y", "alpha_y")):
            return {}
        return twiss

    def run_frequency_map(self) -> list:
        """
        Tune per starting amplitude, over the same grid as the aperture scan.

        This asks at what tune particles survive.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y)`` per surviving grid point. Lost
            particles are absent. Empty, with a warning, if this code
            cannot do it.
        """
        warn(exceptions.FrequencyMapUnsupportedWarning(
            self.objectname, self.code, self.codes_that_can("supports_frequency_map")
        ))
        return []

    def run_dynamic_aperture(self) -> list:
        """
        Track a grid of single particles and see which survive.

        The standard nonlinear ring study:
        one particle per grid point, each tracked for :meth:`turns`.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per grid point. Empty, with a
            warning, if this code cannot do it.
        """
        warn(exceptions.DynamicApertureUnsupportedWarning(
            self.objectname, self.code, self.codes_that_can("supports_dynamic_aperture")
        ))
        return []

    def check_turns_supported(self) -> None:
        """Warn when turns were asked for and this code cannot do them."""
        if self.turns > 1 and not self.supports_turns:
            warn(exceptions.TurnsUnsupportedWarning(
                self.objectname, self.code, self.turns,
                self.codes_that_can("supports_turns"),
            ))

    def check_turns_closed(self, tolerance: float = 1e-4) -> None:
        """Warn when a line that does not close is treated as though it did.

        Closure is tested on LAURA's geometry: the first
        element's entrance against the last element's exit, to ``tolerance``
        relative to the path length.

        A **superperiod** is the legitimate exception -- one sector of an
        N-fold-symmetric ring is open on its own and closes after N of them.
        """
        if self.turns <= 1 and not self.periodic:
            return
        if self.nsuperperiods > 1:
            self.check_superperiods_close()
            return
        try:
            entrance = self.startObject.physical.start
            exit_ = self.endObject.physical.end
        except (AttributeError, TypeError):
            return
        gap = math.dist(
            (entrance.x, entrance.y, entrance.z), (exit_.x, exit_.y, exit_.z)
        )
        length = sum(
            e.physical.length or 0.0
            for e in self.elements.values()
            if getattr(e, "physical", None) is not None
        )
        if not length or gap <= tolerance * length:
            return
        warn(exceptions.NotClosedWarning(
            self.objectname, self.turns, gap, length, self.net_bend_angle
        ))

    def check_superperiods_close(self) -> None:
        """
        Warn when the declared superperiod count and the geometry disagree.
        """
        count = self.nsuperperiods
        angle = abs(self.net_bend_angle)
        if angle <= 1e-9:
            return
        turns_of_bend = count * angle / (2 * math.pi)
        if abs(round(turns_of_bend) - turns_of_bend) < 1e-3:
            return
        warn(exceptions.SuperperiodsDoNotCloseWarning(self.objectname, count, angle))

    @property
    def net_bend_angle(self) -> float:
        """Total bending angle of the line, in radians.

        ``2*pi`` for a closed planar ring, or an even fraction for one
        superperiod, see :meth:`check_turns_closed`.
        """
        total = 0.0
        for element in self.elements.values():
            magnetic = getattr(element, "magnetic", None)
            if magnetic is None:
                continue
            try:
                total += float(magnetic.KnL(0))
            except (TypeError, ValueError, KeyError):
                continue
        return total

    def check_pass_rigidity(self, brho: float, tolerance: float = 0.01) -> None:
        """Warn if the tracked beam disagrees with this pass's stated momentum.

        Codes that take a field rather than a normalised strength get ``Brho``
        from the beam actually loaded.
        The ``k`` came from the layout, resolved at the momentum that
        pass states.

        A warning rather than a refusal.
        """
        stated = None
        layout = None
        machine = getattr(self, "machine", None)
        if machine is not None:
            layout = machine.lattices.get(machine.default_path)
        if layout is not None:
            stated = layout.pass_momentum(self.start)
        if stated is None or not brho:
            return
        expected = laura_brho(stated)
        if abs(brho - expected) > tolerance * expected:
            warn(exceptions.RigidityMismatchWarning(
                self.objectname, brho, self.start, stated, expected
            ))

    def output_basename(self, name: str, turn: int | None = None) -> str:
        """Filename stem for ``name``'s output beam file, qualified if needed.

        Output beam files are named by element alone, so they must not clash for
        multi-turn tracking. Qualifies only what actually collides:
        :attr:`colliding_outputs` is empty unless ``Framework.track`` found the
        same name written by more than one line. All colliding occurrences are
        qualified, including the first, so the name follows from the settings file.

        ``turn`` qualifies the other axis. It is ignored on a single-turn run,
        so nothing changes for a lattice that does not ask for turns.
        """
        qualified = name in self.colliding_outputs
        name = flatten_occurrence(name)
        if qualified:
            name = f"{self.objectname}{OUTPUT_LINE_SEPARATOR}{name}"
        if turn is not None and self.turns > 1:
            name = f"{name}{OUTPUT_TURN_SEPARATOR}{turn:0{len(str(self.turns))}d}"
        return name

    def sampled_index(self, index: int | None) -> int | None:
        """
        A particle's index once the beam is sampled to every
        :attr:`sample_interval`-th particle, or None if sampling drops it.
        """
        if index is None:
            return None
        interval = max(1, int(self.sample_interval))
        return int(index) // interval if int(index) % interval == 0 else None

    def sample_beam(self, bm):
        """
        Every :attr:`sample_interval`-th particle of a beam, conserving the
        total charge; see :meth:`sampled_index`.
        """
        from .Modules.units import UnitValue

        interval = max(1, int(self.sample_interval))
        newbeam = deepcopy(bm)
        nb = newbeam._beam
        idx = slice(None, None, interval)
        for key in ["x", "y", "z", "t", "px", "py", "pz", "nmacro", "status",
                    "particle_mass", "particle_rest_energy",
                    "particle_rest_energy_eV", "particle_charge"]:
            if hasattr(nb, key) and getattr(nb, key) is not None:
                val = getattr(nb, key)
                try:
                    setattr(nb, key, UnitValue(np.array(val.val)[idx], units=val.units))
                except Exception:
                    pass
        nb.set_total_charge(np.sum(np.array(bm.charge.val)))
        newbeam.reference_particle_index = self.sampled_index(bm.reference_particle_index)
        return newbeam

    def writes_output(self, name: str, turn: int | None = None) -> bool:
        """
        Whether a beam recorded at ``name`` gets a file of its own, under
        :meth:`output_basename`.
        Every recorded element does, except the start of the line under its
        bare name: that file is the input beam, the previous line's end.

        Parameters
        ----------
        name: str
            The element
        turn: int | None
            The turn, as :meth:`output_turns` names it
        """
        return name != self.start or (turn is not None and self.turns > 1)

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
                self.getInitialTwiss()
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
                rbf.hdf5.read_HDF5_beam_file(
                    self.global_parameters["beam"],
                    os.path.abspath(filepath),
                )
            return filepath
        raise Exception(
            f'HDF5 input file {expand_substitution(self, prefix + particle_definition)}.[openpmd.].hdf5 does not exist!')

    @property
    def input_particle_definition(self) -> str:
        """
        The input beam's file name, without its extension: ``input:
        particle_definition``, where ``initial_distribution`` is the
        generator's ``laser``; else this line's start.
        """
        stated = (self.file_block.get("input") or {}).get("particle_definition")
        if stated is None:
            return self.start
        return "laser" if stated == "initial_distribution" else stated

    def load_input_beam(self, prefix: str, particle_definition: str) -> str:
        """
        Read the incoming beam and make it the beam this lattice should see.
        The ``s`` a code reports is anchored to :attr:`entrance_s`.

        Every code does the same things once its input is read, so they
        are done here rather than in each ``preProcess``:

        * record the beam's reference, :attr:`reference_p0c` and
          :attr:`reference_t0`, which every code takes as its own;
        * keep every :attr:`sample_interval`-th particle, conserving the total
          charge (:meth:`sample_beam`);
        * refuse a beam the code cannot track, :meth:`check_species`;
        * rematch to ``input: twiss`` if it is given;
        * take :attr:`ref_idx` from the beam;
        * run the beam-dependent ramp checks, :func:`check_ramp_beam`;
        * fix the reference clock every code reports ``t`` on,
          :meth:`reference_time`.

        Parameters
        ----------
        prefix: str
            Prefix of the input beam file
        particle_definition: str
            Name of the input beam file, without its extension

        Returns
        -------
        str
            Path of the file that was read; see :func:`read_input_file`
        """
        filepath = self.read_input_file(prefix, particle_definition)
        full = self.global_parameters["beam"]
        self._input_reference = {
            coord: float(np.mean(getattr(full, coord).val)) for coord in ("cp", "t", "z")
        }
        if int(self.sample_interval) > 1:
            self.global_parameters["beam"] = self.sample_beam(self.global_parameters["beam"])
        self.check_species()
        beam = self.global_parameters["beam"]
        beam.beam.rematchXPlane(**self.initial_twiss["horizontal"])
        beam.beam.rematchYPlane(**self.initial_twiss["vertical"])
        self.ref_idx = beam.reference_particle_index
        self.check_ramp_beam()
        self.reset_reference_clock()
        return filepath

    def update_groups(self) -> None:
        """
        Update the group objects in the lattice with their settings.
        """
        for g in list(self.groupSettings.keys()):
            if g in self.groupObjects:
                setattr(self, g, self.groupObjects[g])
                if self.groupSettings[g] is not None:
                    self.groupObjects[g].update(**self.groupSettings[g])

    def getElement(self, element: str, param: str = None) -> dict | PhysicalBaseElement:
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
                return getattr(self.elementObjects[element], param.lower())
            else:
                return self.elementObjects[element]
        elif element in list(self.groupObjects.keys()):
            if param is not None:
                return getattr(self.groupObjects[element], param.lower())
            else:
                return self.groupObjects[element]
        else:
            warn(exceptions.MissingElementWarning(element))
            return {}

    def getElementType(
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
            return [self.getElementType(t, param=param) for t in typ]
        if isinstance(param, (list, tuple)):
            return zip(*[self.getElementType(typ, param=p) for p in param])
        return [
            self.elements[element] if param is None else getattr(self.elements[element], param)
            for element in list(self.elements.keys())
            if self.elements[element].hardware_type.lower() == typ.lower()
        ]

    def setElementType(
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
        elems = self.getElementType(typ)
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
        return self.getElementType("quadrupole")

    @property
    def cavities(self) -> list:
        """
        Property to get all cavity elements in the lattice.

        Returns
        -------
        list
            A list of cavity elements in the lattice.
        """
        return self.getElementType("cavity")

    @property
    def solenoids(self) -> list:
        """
        Property to get all solenoid elements in the lattice.

        Returns
        -------
        list
            A list of solenoid elements in the lattice.
        """
        return self.getElementType("solenoid")

    @property
    def dipoles(self) -> list:
        """
        Property to get all dipole elements in the lattice.

        Returns
        -------
        list
            A list of dipole elements in the lattice.
        """
        return self.getElementType("dipole")

    @property
    def kickers(self) -> list:
        """
        Property to get all kicker elements in the lattice.

        Returns
        -------
        list
            A list of kicker elements in the lattice.
        """
        return self.getElementType("kicker")

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
            self.getElementType("dipole") + self.getElementType("kicker"),
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
        return self.getElementType("wakefield")

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
            for cav in self.getElementType("cavity")
            if (
                isinstance(cav.simulation.wakefield_definition, field)
                or cav.simulation.wakefield_definition != ""
            )
        ]
        wakes = self.getElementType("wakefield")
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
        return self.getElementType("screen")

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
            self.getElementType("screen")
            + self.getElementType("beam_position_monitor"),
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
            self.getElementType("screen")
            + self.getElementType("marker")
            + self.getElementType("beam_position_monitor"),
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
            self.getElementType("aperture") + self.getElementType("collimator"),
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
        return self.getElementType("wiggler")

    @property
    def photon_monitors(self) -> list:
        """
        Property to get all photon monitor elements in the lattice.

        Returns
        -------
        list
            A list of photon monitor elements in the lattice.
        """
        return self.getElementType("photon_monitor")

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
        beam_path = self.machine.elements_between(end=self.end)
        if "zstart" in self.file_block["output"]:
            zstart = self.file_block["output"]["zstart"]
            candidates = [
                name
                for name in beam_path
                if isinstance(self.elementObjects.get(name), PhysicalBaseElement)
                and not self.elementObjects[name].subelement
                and np.isclose(self.elementObjects[name].physical.start.z, zstart, atol=1e-2)
            ]
            for name in candidates:
                if self.elementObjects[name].physical.length > 0:
                    return name
            if candidates:
                return candidates[0]
        return beam_path[0]

    @property
    def startObject(self) -> "PhysicalBaseElement":
        """
        Property to get the starting element of the lattice.
        See :func:`start` for more details.


        Returns
        -------
        Element
            The starting element of the lattice.
        """
        return self.elementObjects[self.start]

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
            for name, elem in self.elementObjects.keys():
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
            return list(self.elementObjects.keys())[-1]

    @property
    def endObject(self) -> "PhysicalBaseElement":
        """
        Property to get the final element of the lattice.
        See :func:`end` for more details.


        Returns
        -------
        Element
            The final element of the lattice.
        """
        return self.elementObjects[self.end]

    @property
    def start_s(self) -> float:
        """
        Property to get the s position of the start of the lattice, measured along
        the reference trajectory from the start of the machine.

        This is what every tracking code should anchor its reported s to. It is not
        the same as ``startObject.physical.start.z`` -- anything that bends (a
        chicane, a dogleg) makes the path longer than its projection onto z, and
        that difference has to carry forward into every downstream lattice.

        Returns
        -------
        float
            S position of the first element of the lattice.
        """
        if self._start_s is None:
            self._start_s = self.machine.get_elements_s_pos(end=self.start)[self.start]
        return self._start_s

    @property
    def entrance_s(self) -> float:
        """
        Property to get the s position of the lattice entrance.

        :attr:`start_s` is the s at the *exit* of the first element, so its length has to
        come back off. The two are equal for the usual case of a lattice starting on a
        zero-length marker, and differ for one starting on a real element.

        This is the anchor for per-element s positions, which are measured from the
        lattice entrance.

        Returns
        -------
        float
            S position of the entrance of the lattice.
        """
        return float(self.start_s - self.startObject.physical.length)

    def _machine_space_charge(self):
        """
        The collective-field resolution of the machine section this lattice cuts.
        `csr_bins` set on this lattice still wins, being applied after.

        Returns
        -------
        SpaceChargeSettings | None
            The settings to run this lattice with, or None to leave each code on
            its own defaults.
        """
        for section in (getattr(self.machine, "sections", None) or {}).values():
            if self.start in getattr(section, "order", ()):
                return section.space_charge
        return None

    def _machine_geometry(self):
        """
        Whether LAURA says the reference orbit of this lattice's section closes.

        ``geometry`` is section metadata in LAURA (``open``/``closed``, mirroring
        Bmad's ``parameter[geometry]``), so a ring already says so in the layout
        and does not need saying again here. Bmad is the backend that reads it
        straight through; see :meth:`periodic`.

        Returns
        -------
        LatticeGeometryEnum | None
            The section's geometry, or None when the layout does not say.
        """
        for section in (getattr(self.machine, "sections", None) or {}).values():
            if self.start in getattr(section, "order", ()):
                return getattr(section, "geometry", None)
        return None

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
            layout = self.machine.lattices.get(self.machine.default_path)
            order, vals = [], {}
            for key in keys:
                element = layout.element_on_pass(key) if layout is not None else None
                if element is None:
                    element = self.machine.get_element(key)
                if not isinstance(element, PhysicalBaseElement):
                    continue
                flat = flatten_occurrence(key)
                order.append(flat)
                vals[flat] = element
            section = SectionLattice(
                order=order,
                elements=ElementList(elements=vals),
                name=self.objectname,
                master_lattice=self.global_parameters["master_lattice"],
                functional_definitions=self.settings["functional_definitions"],
                resolve_functional=self.settings["resolve_functional"],
                space_charge=self._machine_space_charge(),
                geometry=self._machine_geometry(),
            )
            slt = SectionLatticeTranslator.from_section(section)
            slt.lsc_enable = self.lsc_enable
            slt.csr_enable = self.csr_enable
            slt.lsc_bins = self.lsc_bins
            slt.directory = self.global_parameters["master_subdir"]
            self._section = slt
            return slt
        self._section.directory = self.global_parameters["master_subdir"]
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

    def run_command(self, command: list, logfile: str, **kwargs) -> None:
        """
        Run a simulation code, logging to `logfile`, and raise if the code says it failed.

        A code that gives up part-way still looks like a successful run to everything
        downstream, which then dies reading output that was never written -- so the
        exit status is read here, once, for every code that runs a subprocess.

        Parameters
        ----------
        command: list
            The command to run, as passed to :mod:`subprocess`
        logfile: str
            Where the code's output is written; its tail is quoted if the code fails
        kwargs:
            Passed through to :func:`subprocess.call` (``cwd``, ``env``, ...)

        Raises
        ------
        RuntimeError
            If the code exits with a non-zero status.
        """
        with open(logfile, "w") as f:
            status = subprocess.call(
                command, stdout=f, stderr=subprocess.STDOUT, **kwargs
            )
        if status == 0:
            return
        try:
            with open(logfile, "r") as f:
                tail = "".join(f.readlines()[-20:]).strip()
        except OSError:
            tail = ""
        raise RuntimeError(
            f"{self.code} exited with status {status} running {self.objectname}.\n"
            f"Last lines of {logfile}:\n{tail}"
        )

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
        RuntimeError
            If the code exits with a non-zero status.
        """
        if self.remote_setup:
            self.run_remote()
        else:
            command = self.executables[self.code] + [self.name]
            workdir = os.path.abspath(self.global_parameters["master_subdir"])
            command = self.executables.build_command(command, workdir)
            self.run_command(
                command,
                os.path.relpath(
                    self.global_parameters["master_subdir"] + "/" + self.name + ".log",
                    ".",
                ),
                cwd=self.global_parameters["master_subdir"],
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
            if hasattr(e.simulation, "wakefield_definition") and isinstance(e.simulation.wakefield_definition, str):
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

    def getInitialTwiss(self) -> dict:
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
            final_energy += (cav.simulation.resolved("field_amplitude") * np.cos(cav.cavity.resolved("phase"))) * 1e-9
        if harmonics:
            for harm in harmonics:
                final_energy += (harm.simulation.resolved("field_amplitude") * np.cos(harm.cavity.resolved("phase"))) * 1e-9

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

    def preProcess(self) -> None:
        """
        Pre-process the lattice before running the simulation.
        This method initializes the initial Twiss parameters by calling the `getInitialTwiss` method.

        Returns
        -------
        None
        """
        self.check_turns_supported()
        self.check_periodic_supported()
        self.check_radiation()
        self.check_radiation_supported()
        self.check_single_particle_supported()
        self.check_nsuperperiods_supported()
        self.check_programs_supported()
        self.check_programs_fit()
        self.check_ramp()
        self._apply_radiation_to_section()
        self._apply_fixed_reference_to_section()
        self.check_turns_closed()
        ast = self.section.astra_headers.copy()
        self.initial_twiss = self.getInitialTwiss()
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

    def read_closed_orbit(self):
        """
        The orbit that closes on itself, at the start of the line.

        Everything else in a ring is defined about the closed orbit.
        On a perfectly aligned lattice it is identically zero, which is why
        a test of it needs a steering error to mean anything.

        Returns
        -------
        numpy.ndarray | None
            6 components in this code's :attr:`otm_convention` order, or None
            if the code did not give one.
        """
        return None

    def read_optics_summary(self) -> dict:
        """
        The code's own tune and chromaticity, from its periodic solution.

        Not derived here, instead read back from the code, giving the integer
        part of the tune, and the chromaticity.

        Returns
        -------
        dict
            Any of ``tune_x_total``, ``tune_y_total``, ``chromaticity_x``,
            ``chromaticity_y``. Empty when the code reports none of them.
        """
        return {}

    def read_one_turn_map(self):
        """This code's 6x6 one-turn map, or None if it has none to give.

        Overridden by the backends that can; every ring code has a native call
        for this, so none of them need the map rebuilding by hand.

        Returns
        -------
        numpy.ndarray | None
            A 6x6 matrix in :attr:`otm_convention` coordinates.
        """
        return None

    def one_turn_map_canonical(
        self, beta0: float | None = None, magnitude: bool = True
    ):
        """
        :attr:`one_turn_map` in one common convention, so codes compare.

        Canonical here is Xsuite's ``(x, px, y, py, zeta, delta)``. The
        conversion is a *diagonal similarity* ``D R D^-1`` with
        ``D = diag(1, 1, 1, 1, d5, d6)``. Normalising cannot invent or
        destroy a tune; it can only fix the longitudinal block.

        With ``magnitude=True`` (the default) it returns ``None`` where the
        conversion is not a rescale. Elegant's
        fifth coordinate is geometric path length, not time of flight, so a
        drift has ``R56 = 0`` where the other codes have
        ``L / (beta0 * gamma0)**2``.

        ``magnitude=False`` applies :attr:`otm_longitudinal_sign` alone.

        Parameters
        ----------
        beta0: float | None
            Reference ``v/c``. Read from the tracked beam when not given, and
            not needed at all when ``magnitude`` is False.
        magnitude: bool
            Convert sizes as well as signs.

        Returns
        -------
        numpy.ndarray | None
            The 6x6 map in canonical coordinates, or None when ``magnitude``
            was asked for and this code's conversion is not a rescale.
        """
        matrix = self.one_turn_map
        if matrix is None or np is None:
            return None
        ratio = float(self.otm_longitudinal_sign)
        if magnitude:
            if self.otm_longitudinal_scale is None:
                return None
            if beta0 is None:
                try:
                    betagamma = float(
                        np.mean(self.global_parameters["beam"].BetaGamma)
                    )
                    beta0 = betagamma / math.sqrt(1.0 + betagamma**2)
                except (KeyError, TypeError, AttributeError, ValueError):
                    return None
            if not beta0:
                return None
            ratio *= beta0**self.otm_longitudinal_scale
        matrix = np.asarray(matrix, dtype=float)
        diagonal = np.array([1.0, 1.0, 1.0, 1.0, ratio, 1.0])
        return (diagonal[:, None] * matrix) / diagonal[None, :]

    def ring_parameters(self) -> dict:
        """
        Tune, periodic Twiss and momentum compaction, from the one-turn map.
        The first two come from the raw map, and the latter comes from the
        canonical map.

        Chromaticity is deliberately absent: it is not in a single one-turn
        map. It needs maps at two momenta, or the code's own periodic Twiss.

        Returns
        -------
        dict
            ``{}`` if there is no map. Otherwise ``tune_x``/``tune_y``
            (fractional), ``beta_x``/``alpha_x``/``gamma_x`` and the ``y``
            equivalents, ``stable_x``/``stable_y``, and ``slip_factor`` /
            ``momentum_compaction`` where the convention allows.
        """
        from .Modules.Matrices import (
            fractional_tune,
            is_stable,
            momentum_compaction,
            periodic_twiss,
            slip_factor,
        )

        matrix = self.one_turn_map
        if matrix is None or np is None:
            return {}
        result = {}
        for plane in ("x", "y"):
            result[f"stable_{plane}"] = is_stable(matrix, plane)
            result[f"tune_{plane}"] = fractional_tune(matrix, plane)
            for key, value in periodic_twiss(matrix, plane).items():
                result[f"{key}_{plane}"] = value
        result.update(self.optics_summary or {})
        if self.closed_orbit is not None:
            orbit = np.asarray(self.closed_orbit, dtype=float)
            for index, name in enumerate(
                ("x", "px", "y", "py", "zeta", "delta")[: len(orbit)]
            ):
                result[f"closed_orbit_{name}"] = float(orbit[index])
        canonical = self.one_turn_map_canonical()
        circumference = sum(
            e.physical.length or 0.0
            for e in self.elements.values()
            if getattr(e, "physical", None) is not None
        )
        if canonical is not None and circumference:
            result["slip_factor"] = slip_factor(canonical, circumference)
            try:
                betagamma = float(np.mean(self.global_parameters["beam"].BetaGamma))
                gamma0 = math.sqrt(1.0 + betagamma**2)
            except (KeyError, TypeError, AttributeError, ValueError):
                gamma0 = None
            if gamma0:
                result["momentum_compaction"] = momentum_compaction(
                    canonical, circumference, gamma0
                )
        return result

    def check_one_turn_map(self, tolerance: float = 1e-3) -> None:
        """
        Warn when the map that came back cannot be a one-turn map.

        ``det(R) == 1`` for a linear map that neither creates nor destroys
        phase-space volume.
        """
        matrix = self.one_turn_map
        if matrix is None or np is None:
            return
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (6, 6):
            warn(exceptions.BadOneTurnMapWarning(
                self.objectname, self.code, shape=matrix.shape
            ))
            return
        determinant = float(np.linalg.det(matrix))
        if abs(determinant - 1.0) > tolerance:
            warn(exceptions.BadOneTurnMapWarning(
                self.objectname, self.code, determinant=determinant
            ))

    def postProcess(self):
        """
        Read back whatever the run produced that is not a beam file.

        For a ring that means the one-turn map, which every capable code
        offers natively; see :attr:`supports_periodic`.
        """
        if self.periodic and self.supports_periodic:
            self.one_turn_map = self.read_one_turn_map()
            self.check_one_turn_map()
            self.optics_summary = self.read_optics_summary()
            self.closed_orbit = self.read_closed_orbit()

    def __repr__(self):
        return self.elements

    def __str__(self):
        str = self.name + " = ("
        for e in self.elements:
            if len((str + e).splitlines()[-1]) > 60:
                str += "&\n"
            str += e + ", "
        return str + ")"

    def createDrifts(
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
        return self.section.createDrifts()

    def getSValues(
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
        elems = self.createDrifts() if drifts else self.elements
        s = [0]
        for e in list(elems.values()):
            s.append(s[-1] + e.physical.length)
        s = s[:-1] if at_entrance else s[1:]
        if as_dict:
            return dict(zip([e.name for e in elems.values()], s))
        return list(s)

    def getZValues(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
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
            elems = self.createDrifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.name: [e.physical.start.z, e.physical.end.z] for e in elems.values()}
        return [[e.physical.start.z, e.physical.end.z] for e in elems.values()]

    def getNames(self, drifts: bool = True) -> list:
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
            elems = self.createDrifts()
        else:
            elems = self.elements
        return [e.name for e in list(elems.values())]

    def getElems(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
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
            elems = self.createDrifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.name: e for e in list(elems.values())}
        return [e for e in list(elems.values())]

    def getSNames(self) -> list:
        """
        Get the names and S values of the elements in the lattice.

        Returns
        -------
        list
            A list of tuples, where each tuple contains the name of an element and its corresponding S value.
        """
        s = self.getSValues()
        names = self.getNames()
        return list(zip(names, s))

    def getSNamesElems(self) -> tuple:
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
        s = self.getSValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, s

    def getZNamesElems(self) -> tuple:
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
        z = self.getZValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, z

    def findS(self, elem) -> list:
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
        if elem in self.allElements:
            sNames = self.getSNames()
            return [a for a in sNames if a[0] == elem]
        return []

    def updateRunSettings(self, runSettings: runSetup) -> None:
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
        if isinstance(runSettings, runSetup):
            self.runSettings = runSettings
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
        particle_ref = xt.Particles(
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
        return line, beam, self.getNames()

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

        The `params` dictionary should contain the following keys:

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
        from .Framework_lattices import ocelotLattice
        from ocelot.cpbd.beam import Twiss
        from ocelot.cpbd.match import match as match_oce
        latcopy = deepcopy(self)
        lat = ocelotLattice(
            name=f"{latcopy.name}_match",
            file_block=latcopy.file_block,
            machine=latcopy.machine,
            elementObjects=latcopy.elementObjects,
            groupObjects=latcopy.groupObjects,
            runSettings=latcopy.runSettings,
            executables=latcopy.executables,
            global_parameters=latcopy.global_parameters,
            settings=latcopy.settings,
        )
        prefix = lat.get_prefix()
        prefix = prefix if lat.trackBeam else prefix + lat.particle_definition
        lat.read_input_file(prefix, lat.particle_definition)
        lat.ref_idx = self.global_parameters["beam"].reference_particle_index
        lat.hdf5_to_npz(prefix)
        lat.writeElements()
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
            magnetic_order = self.elementObjects[params["variables"][i]].magnetic.order
            magnetic_length = self.elementObjects[params["variables"][i]].magnetic.length
            setattr(self.elementObjects[params["variables"][i]], f"k{magnetic_order}l", r * magnetic_length)
            print("\t", self.elementObjects[params["variables"][i]].name, f"k{magnetic_order}l =", r * magnetic_length)

class global_error(frameworkObject):
    """
    Class defining a global error element.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(global_error, self).__init__(
            *args,
            **kwargs,
        )

    def add_Error(self, type, sigma):
        if type in global_Error_Types:
            self.add_property(type, sigma)

    def _write_ASTRA(self):
        return self._write_ASTRA_dictionary(
            dict([[key, {"value": value}] for key, value in self._errordict])
        )

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
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

class frameworkCommand(frameworkObject):
    """
    Class defining a framework command, which is used to generate commands used in setup files
    for various simulation codes.
    """

    def model_post_init(self, __context):
        if self.objecttype not in commandkeywords:
            raise NameError("Command '%s' does not exist" % self.objecttype)
        super().model_post_init(__context)

    def write_Elegant(self) -> str:
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

    def write_MAD8(self) -> str:
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

    def write_Genesis(self) -> str:
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


class frameworkGroup(object):
    """
    Class defining a framework group, which is used to group together elements to perform coordinated
    actions on them.
    """

    def __init__(self, name, framework, type, elements, **kwargs):
        super(frameworkGroup, self).__init__()
        self.objectname = name
        self.type = type
        self.framework = framework
        self.elements = elements

    @property
    def allElementObjects(self):
        return self.framework.elementObjects

    @property
    def allGroupObjects(self):
        return self.framework.groupObjects

    def update(self, **kwargs):
        pass

    def get_Parameter(self, p: str) -> Any:
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
            if self.elements[0] in self.allGroupObjects:
                return getattr(self.allGroupObjects[self.elements[0]], p)
            return getattr(self.allElementObjects[self.elements[0]], p)

    def change_Parameter(self, p: Any, v: Any) -> None:
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
                setattr(self.allElementObjects[e], p, v)
                # print ('Changing group elements ', self.objectname, ' ', p, ' = ', v, '  result = ', self.allElementObjects[self.elements[0]].objectname, self.get_Parameter(p))

    # def __getattr__(self, p):
    #     return self.get_Parameter(p)

    def __repr__(self):
        return str([self.allElementObjects[e].name for e in self.elements])

    def __str__(self):
        return str([self.allElementObjects[e].name for e in self.elements])

    def __getitem__(self, key):
        return self.get_Parameter(key)

    def __setitem__(self, key, value):
        return self.change_Parameter(key, value)


class element_group(frameworkGroup):
    """
    Class defining a group of elements, which is used to group together elements to perform coordinated
    actions on them.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)

    def __str__(self):
        return str([self.allElementObjects[e] for e in self.elements])


class r56_group(frameworkGroup):
    """
    Class defining a group of elements with a total R56.
    """

    def __init__(self, name, elementObjects, type, elements, ratios, keys, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = ratios
        self.keys = keys
        self._r56 = None

    def __str__(self):
        return str({e: k for e, k in zip(self.elements, self.keys)})

    def get_Parameter(self, p: str) -> Any:
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
            return super().get_Parameter(p)

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
            self.updateElements(e, k, v)

    def updateElements(self, element: str | list | tuple, key: str, value: Any) -> None:
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
            [self.updateElements(e, key, value) for e in self.elements]
        else:
            if element in self.allElementObjects:
                # print('R56 : updateElements : element', element, key, value)
                self.allElementObjects[element].change_Parameter(key, value)
            if element in self.allGroupObjects:
                # print('R56 : updateElements : group', element, key, value)
                self.allGroupObjects[element].change_Parameter(key, value)


class chicane(frameworkGroup):
    """
    Class defining a 4-dipole chicane.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (1, -1, -1, 1)
        self.elementObjects = [self.allElementObjects[e] for e in self.elements]

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
            self.change_Parameter("width", kwargs["width"])
        if "gap" in kwargs:
            self.change_Parameter("gap", kwargs["gap"])
        return None

    @property
    def drift_d1_to_d2(self) -> float:
        """
        Drift length between dipole 1 and dipole 2

        Returns
        -------
        float
            The drift length between dipole 1 and dipole 2
        """
        e1 = self.elementObjects[0]
        e2 = self.elementObjects[1]
        return np.sqrt(np.sum([(getattr(e2.start, d) - getattr(e1.end, d)) ** 2 for d in ["x", "y", "z"]]))

    @property
    def r56(self) -> float:
        """
        R56 of the chicane

        Returns
        -------
        float
            R56 = 2 * angle^2 * (L1 + 2/3 * L2)
        """
        e1 = self.elementObjects[0]
        ld = self.drift_d1_to_d2
        return 2 * self.angle ** 2 * (ld * (2 * e1.magnetic.length) / 3)

    @property
    def delay(self) -> float:
        """
        Delay (longitudinal slippage) of the chicane

        Returns
        -------
        float
            Delay = 2 * R56
        """
        return 2 * self.r56

    @property
    def angle(self) -> float:
        """
        Bending angle of the chicane

        Returns
        -------
        float
            The bending angle
        """
        obj = [self.allElementObjects[e] for e in self.elements]
        return float(obj[0].magnetic.KnL(0))

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
        rotation, theta0, origin = self._design_axis
        across, _, along = rotation.T

        def spos(e):
            # Not every element in the machine is on the beamline (lasers, etc.)
            try:
                return float(np.dot(np.asarray(e.physical.middle.array) - origin, along))
            except AttributeError:
                return None

        def to_global(u, s):
            p = origin + rotation @ np.array([u, 0.0, s])
            return Position(x=p[0], y=p[1], z=p[2])

        dipole_names = list(self.elements)
        dipoles = [self.allElementObjects[e] for e in dipole_names]
        ss = [spos(d) for d in dipoles]
        between = [
            (s, e)
            for s, e in ((spos(e), e) for e in self.allElementObjects.values())
            if s is not None and min(ss) <= s <= max(ss)
        ]
        obj = [e for _, e in sorted(between, key=lambda se: se[0])]

        z_extents = [self._z_extent(d, i) for i, d in enumerate(dipoles)]

        x, phi, s_cursor = 0.0, 0.0, ss[0] - z_extents[0] / 2.0
        dipole_number = 0
        for e in obj:
            s_here = spos(e)
            if e.name in dipole_names:
                lz = z_extents[dipole_number]
                ang = a * self.ratios[dipole_number]
                x += (s_here - lz / 2.0 - s_cursor) * np.tan(phi)
                p0, p1 = phi, phi + ang
                if abs(ang) > 1e-12:
                    # radius fixed by having to cross `lz` in z while turning p0 -> p1
                    r = lz / (np.sin(p1) - np.sin(p0))
                    dx, arc = r * (np.cos(p0) - np.cos(p1)), r * (p1 - p0)
                else:
                    dx, arc = 0.0, lz
                e.physical.middle = to_global(x + dx / 2.0, s_here)
                e.physical.global_rotation.theta = theta0
                e.magnetic.angle = ang
                e.physical.set_physical_angle(np.copysign(ang, dx) if dx else ang)
                e.magnetic.length = arc
                e.physical.length = arc
                x, phi, s_cursor = x + dx, p1, s_here + lz / 2.0
                dipole_number += 1
            elif dipole_number > 0:
                x_here = x + (s_here - s_cursor) * np.tan(phi)
                e.physical.middle = to_global(x_here, s_here)
                e.physical.global_rotation.theta = theta0 + phi

    @property
    def _design_axis(self) -> tuple:
        """
        The axis the beam arrives on: LAURA's orientation matrix for the first dipole,
        the yaw that goes back onto the elements, and the dipole's entrance, which sits
        on the axis whatever the angle. Everything :func:`set_angle` lays out is
        measured in this frame.

        Cached on first use, because :func:`set_angle` moves the entrance it is read
        from.

        Returns
        -------
        tuple
            ``(rotation_matrix, theta, entrance)``.
        """
        if not hasattr(self, "_axis"):
            d0 = self.allElementObjects[self.elements[0]].physical
            self._axis = (
                np.array(d0.rotation_matrix),
                float(d0.global_rotation.theta),
                np.asarray(d0.start.array, dtype=float),
            )
        return self._axis

    def _z_extent(self, dipole, index: int) -> float:
        """
        The z that the magnet spans, which a variable chicane holds fixed while the angle
        changes -- the magnets translate but never rotate, so their faces stay
        perpendicular to the 0mm axis.

        This is the length in the lattice, which is the zero-angle case where the arc and
        the z extent coincide. Cached on first use because :func:`set_angle` overwrites
        the element's length with the (longer) arc.
        """
        if not hasattr(self, "_design_z_extents"):
            self._design_z_extents = {}
        if dipole.name not in self._design_z_extents:
            self._design_z_extents[dipole.name] = float(dipole.magnetic.length)
        return self._design_z_extents[dipole.name]

    def __str__(self):
        return str(
            [
                [
                    self.allElementObjects[e].name,
                    self.allElementObjects[e].magnetic.angle,
                    self.allElementObjects[e].physical.global_rotation.z,
                    self.allElementObjects[e].physical.start,
                    self.allElementObjects[e].physical.end,
                ]
                for e in self.elements
            ]
        )



class s_chicane(chicane):
    """
    Class defining an s-type chicane; in this case the bending ratios for
    :func:`~simba.Framework_objects.chicane.set_angle` are different.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(s_chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (-1, 2, -2, 1)


class frameworkCounter(dict):
    """
    Class defining a counter object, used for numbering elements of the same type in ASTRA and CSRTrack
    """

    def __init__(self, sub={}):
        super(frameworkCounter, self).__init__()
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


class getGrids(object):
    """
    Class defining the appropriate number of space charge bins given the number of particles,
    defined as the closest power of 8 to the cube root of the number of particles.
    """

    def __init__(self):
        self.powersof8 = np.asarray([2**j for j in range(1, 20)])

    def getGridSizes(self, x: PositiveInt) -> int:
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
