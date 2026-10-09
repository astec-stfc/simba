"""
SIMBA lattices, commands and element groups.

Classes:
    - :class:`~simba.Framework_objects.runSetup`: single runs, element scans or error studies.
    - :class:`~simba.Framework_objects.frameworkObject`: base class for code commands.
    - :class:`~simba.Framework_objects.frameworkLattice`: base class for a line of LAURA elements tracked by one code.
    - :class:`~simba.Framework_objects.frameworkGroup`: elements controlled together.
    - :class:`~simba.Framework_objects.element_group`: a plain group. # TODO is this ever used?
    - :class:`~simba.Framework_objects.r56_group`: a group with an R56. # TODO is this ever used?
    - :class:`~simba.Framework_objects.chicane`: a 4-dipole bunch compressor chicane.
    - :class:`~simba.Framework_objects.getGrids`: number of space-charge grids for a number of particles.
"""

import math
import os
import subprocess
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
"""Separates an element name from a turn index in an output beam filename (multi-turn only)."""

OUTPUT_LINE_SEPARATOR = "-"
"""Separates a line name from an element name in an output beam filename."""

with open(os.path.dirname(os.path.abspath(__file__)) + "/Codes/Elegant/commands_Elegant.yaml") as infile:
    commandkeywords_elegant = yaml.safe_load(infile)

with open(os.path.dirname(os.path.abspath(__file__)) + "/Codes/OPAL/commands_Opal.yaml") as infile:
    commandkeywords_opal = yaml.safe_load(infile)

with open(os.path.dirname(os.path.abspath(__file__)) + "/Codes/Genesis/commands_Genesis.yaml") as infile:
    commandkeywords_genesis = yaml.safe_load(infile)

commandkeywords = commandkeywords_elegant | commandkeywords_opal
commandkeywords = commandkeywords | commandkeywords_genesis

with open(os.path.dirname(os.path.abspath(__file__)) + "/elementkeywords.yaml") as infile:
    elementkeywords = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/Codes/Elegant/keyword_conversion_rules_elegant.yaml"
) as infile:
    keyword_conversion_rules_elegant = yaml.safe_load(infile)


class runSetup:
    """Settings for multi-run simulations such as error studies or parameter scans."""

    def __init__(self):
        self.nruns = 1
        self.seed = 0

        self.elementErrors = None
        self.elementScan = None

    def setNRuns(self, nruns: int | float) -> None:
        """
        Set the number of runs.

        Parameters
        ----------
        nruns : int or float
            Number of runs; truncated to an integer.

        Raises
        ------
        TypeError
            If ``nruns`` is not a number.
        """
        if isinstance(nruns, (int, float)):
            self.nruns = int(nruns)
        else:
            raise TypeError(
                "Argument nruns passed to runSetup instance must be an integer"
            )

    def setSeedValue(self, seed: int | float) -> None:
        """
        Set the random number seed.

        Parameters
        ----------
        seed : int or float
            Seed; truncated to an integer.

        Raises
        ------
        TypeError
            If ``seed`` is not a number.
        """
        if isinstance(seed, (int, float)):
            self.seed = int(seed)
        else:
            raise TypeError("Argument seed passed to runSetup must be an integer")

    def loadElementErrors(self, file: str | dict) -> None:
        """
        Load element error definitions (and optional ``nruns`` and ``seed``) into ``elementErrors``.

        Parameters
        ----------
        file: str or dict
            YAML file path, or the definitions themselves.
        """
        error_setup = None
        if isinstance(file, str) and (".yaml" in file):
            with open(file) as inputfile:
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
        Scan one parameter of one element.

        Parameters
        ----------
        name : str
            Element name.
        item : str
            Parameter to scan.
        scanrange : list or tuple or np.ndarray
            ``(min, max)`` of the scan.
        multiplicative : bool, optional
            Values multiply the original rather than add to it.
        """
        if not (isinstance(name, str) and isinstance(item, str)):
            raise TypeError(
                "Machine element name and item (parameter) must be defined as strings"
            )

        if (
            isinstance(scanrange, (list, tuple, np.ndarray))
            and (len(scanrange) == 2)
            and all(isinstance(x, (float, int)) for x in scanrange)
        ):
            minval, maxval = scanrange
        else:
            raise TypeError("Scan range (min. and max.) must be defined as floats")

        if not isinstance(multiplicative, bool):
            raise ValueError(
                "Argument multiplicative passed to runSetup.setElementScan must be a boolean"
            )

        self.elementScan = {
            "name": name,
            "item": item,
            "min": minval,
            "max": maxval,
            "multiplicative": multiplicative,
        }
        self.elementErrors = None


class frameworkObject(BaseModel):
    """Base class for code commands, whose allowed keywords come from the command and element keyword files."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True,
    )

    objectname: str = Field(alias="name")
    """Unique name of the object."""

    objecttype: str = Field(alias="type")
    """Type of the object, which sets its allowed keywords."""

    objectdefaults: Dict = {}
    """Default property values."""

    allowedkeywords: List | Dict = {}
    """Keywords that can be set as properties."""

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

    @field_validator("objectname", mode="before")
    @classmethod
    def validate_objectname(cls, value: str) -> str:
        """Require a string objectname."""
        if not isinstance(value, str):
            raise ValueError("objectname must be a string.")
        return value

    @field_validator("objecttype", mode="before")
    @classmethod
    def validate_objecttype(cls, value: str) -> str:
        """Require a string objecttype."""
        if not isinstance(value, str):
            raise ValueError("objecttype must be a string.")
        return value

    def change_Parameter(self, key: str, value: Any) -> None:
        """
        Set an attribute.

        Parameters
        ----------
        key: str
            Parameter name.
        value: Any
            New value.
        """
        setattr(self, key, value)

    def add_property(self, key: str, value: Any) -> None:
        """
        Set an attribute if ``key`` (case-insensitive) is in :attr:`allowedkeywords`.

        Parameters
        ----------
        key: str
            Property name.
        value: Any
            Value to set.
        """
        key = key.lower()
        if key in self.allowedkeywords:
            try:
                setattr(self, key, value)
            except Exception as e:
                warn(f"add_property error: ({self.objecttype} [{key}]: {e}")

    def add_properties(self, **keyvalues: dict) -> None:
        """
        Set several properties; see :meth:`add_property`.

        Parameters
        ----------
        **keyvalues: dict
            Property names and values.
        """
        for key, value in keyvalues.items():
            key = key.lower()
            if key in self.allowedkeywords:
                try:
                    setattr(self, key, value)
                except Exception as e:
                    warn(f"add_properties error: ({self.objecttype} [{key}]: {e}")

    def __repr__(self):
        string = ""
        for k in self.model_fields_set:
            if k in self.allowedkeywords:
                string += f"{k} = {getattr(self, k)}" + "\n"
        return string


class frameworkLattice(BaseModel):
    """
    A line of elements and groups tracked by one code: writing, running and post-processing it.

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
    """Same as :attr:`name`."""

    objecttype: str = ""
    """Lattice class name, e.g. ``elegantLattice``."""

    file_block: Dict
    """This line's ``files:`` entry: input, output and tracking settings."""

    colliding_outputs: Set[str] = set()
    """Element names another line in this run also writes an output for.

    Set by :meth:`~simba.Framework.Framework.track`; see :meth:`output_basename`."""

    machine: LAURA
    """LAURA model of the lattice."""

    elementObjects: Dict
    """Element objects, by name."""

    groupObjects: Dict
    """Group objects, by name."""

    runSettings: runSetup
    """Number of runs, seed, errors and scans."""

    settings: FrameworkSettings
    """The framework settings."""

    executables: exes.Executables
    """Commands for running the codes; see :class:`~simba.Codes.Executables.Executables`."""

    global_parameters: Dict
    """Global parameters, including ``master_subdir``."""

    globalSettings: Dict = {"charge": None}
    """Global settings, including charge."""

    allow_negative_drifts: bool = False
    """Allow negative drifts in the lattice."""

    _lsc_enable: bool = False
    """Enable LSC drifts; off by default, as in LAURA, so every code models the same physics."""

    _csr_enable: bool = True
    """Enable CSR drifts."""

    _wakefield_enable: bool = True
    """Enable structure wakefields."""

    _lsc_bins: int = 20
    """Number of bins for LSC drifts."""

    _csr_bins: int | None = None
    """Number of bins for CSR, or None if nobody has chosen one."""

    lsc_high_frequency_cutoff_start: float = -1
    """Spatial frequency at which smoothing filter begins; if not positive, no smoothing. See `Elegant manual LSC drift`_

    .. _Elegant manual LSC drift: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu168.html#x179-18000010.58"""

    lsc_high_frequency_cutoff_end: float = -1
    """Spatial frequency at which smoothing filter is 0. See `Elegant manual LSC drift`_"""

    lsc_low_frequency_cutoff_start: float = -1
    """Highest spatial frequency at which low-frequency cutoff filter is zero. See `Elegant manual LSC drift`_"""

    lsc_low_frequency_cutoff_end: float = -1
    """Lowest spatial frequency at which low-frequency cutoff filter is 1. See `Elegant manual LSC drift`_"""

    sample_interval: int = 1
    """Track every ``sample_interval``-th incoming particle, keeping the total charge.

    Sampled once, as the beam is read (:meth:`load_input_beam`)."""

    groupSettings: Dict = {}
    """Group settings for this line."""

    allElements: List = []
    """All element names in the lattice."""

    initial_twiss: Dict = {}
    """Initial Twiss parameters."""

    ref_idx: int | None = None
    """Index of the incoming beam's reference particle; see :meth:`load_input_beam`."""

    native_time: ClassVar[tuple[str, str]] = ("t", "s")
    """The code's own longitudinal time coordinate, as ``(name, units)``; see :meth:`native_time_scale`."""

    _reference_clock: tuple | None = None
    """Fixed when the input beam is read; see :attr:`reference_clock`."""

    _input_reference: dict | None = None
    """The incoming beam's means, before sampling; see :attr:`reference_p0c`."""

    _input_particle: dict | None = None
    """The incoming reference particle's ``t`` and ``z``, if any; see :attr:`reference_t0`."""

    _section: SectionLatticeTranslator = None
    """LAURA section translator."""

    _start_s: float = None
    """Cached s at the start of the lattice; see :attr:`start_s`."""

    remote_setup: Dict = {}
    """Settings for running executables remotely."""

    files: List = []
    """Files needed to run the lattice."""

    code: str = None
    """Code to run the lattice."""

    supports_turns: ClassVar[bool] = False
    """Whether this code can track a line more than once.

    :meth:`codes_that_can` lists the codes with any of these flags."""

    supports_periodic: ClassVar[bool] = False
    """Whether this code can give the periodic (closed) optics rather than propagate the incoming Twiss."""

    supports_frequency_map: ClassVar[bool] = False
    """Whether this code can produce a tune footprint; see :meth:`run_frequency_map`."""

    supports_single_particle: ClassVar[bool] = False
    """Whether this code can track 13 probes and carry the distribution through their map; see :attr:`single_particle`."""

    supports_dynamic_aperture: ClassVar[bool] = False
    """Whether this code can run a dynamic-aperture scan; see :meth:`run_dynamic_aperture`."""

    supports_nsuperperiods: ClassVar[bool] = False
    """Whether this code can track one sector of an N-fold ring N times per turn; see :attr:`nsuperperiods`."""

    radiates_by_default: ClassVar[bool] = False
    """Whether this code radiates unasked."""

    supports_radiation: ClassVar[bool] = False
    """Whether simba can switch radiation on for this code; see :meth:`check_radiation_supported`."""

    supports_programs: ClassVar[bool] = False
    """Whether this code can vary an element from turn to turn; see :class:`~simba.Modules.DeviceProgram.DeviceProgram`."""

    supports_ramp: ClassVar[bool] = False
    """Whether this code can track an energy ramp; see :class:`~simba.Modules.EnergyRamp.EnergyRamp`."""

    electrons_only: ClassVar[bool] = False
    """Whether this code tracks only electrons; other beams are refused by :meth:`check_species`."""

    native_rf: ClassVar[str | None] = None
    """How this code's cavities keep time left to themselves, ``fixed`` or ``synchronous``.

    Subtracted by :meth:`rf_phase_corrections`; None where simba cannot move a
    cavity's phase pass by pass, so cannot impose :attr:`rf_mode`."""

    rf_phase_sign: ClassVar[float] = 1.0
    """Sign of the code's cavity phase against the phase the reference sees, measured per code; see :meth:`rf_phase_shifts`."""

    rf_phase_per_radian: ClassVar[float] = 180 / math.pi
    """The code's cavity phase units per radian (degrees by default)."""

    _rf_corrections: dict | None = None
    """This run's :meth:`rf_phase_corrections`; see :meth:`begin_rf_phases`."""

    _rf_phase0: dict | None = None
    """Each moved cavity's phase as given to the code this run, by name; see :meth:`apply_rf_phases`."""

    otm_convention: ClassVar[str] = ""
    """The coordinate order of :attr:`one_turn_map` as this code writes it; see :meth:`one_turn_map_canonical`.

    Only the longitudinal block differs: Ocelot by a sign from MAD-X, Xsuite by
    ``beta0**2``, and elegant by an additive term, as its fifth coordinate is path length.
    """

    otm_longitudinal_sign: ClassVar[int] = 1
    """Sign of this code's fifth coordinate against the canonical one (Xsuite, Bmad)."""

    otm_longitudinal_scale: ClassVar[int | None] = None
    """Power of ``beta0`` in the longitudinal conversion; None if it is not a rescale (elegant)."""

    optics_summary: Any = None
    """The code's own tune and chromaticity; see :meth:`read_optics_summary`."""

    dynamic_aperture: Any = None
    """Last :meth:`run_dynamic_aperture` result."""

    frequency_map: Any = None
    """Last :meth:`run_frequency_map` result."""

    closed_orbit: Any = None
    """The periodic orbit at the start of the line, a 6-vector in :attr:`otm_convention`; see :meth:`read_closed_orbit`."""

    one_turn_map: Any = None
    """The 6x6 linear one-turn map in :attr:`otm_convention`, read by :meth:`read_one_turn_map`.

    None unless the line is :attr:`periodic` and the code can produce one.
    """

    program_attributes: ClassVar[dict] = {}
    """``(horizontal, vertical)`` attribute an unqualified program sets, per code element type; see :meth:`program_attribute`."""

    def model_post_init(self, __context):
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
        if "groups" in self.file_block and self.file_block["groups"] is not None:
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

    def _apply_collective_settings(self) -> None:
        """Take ``csr_enable`` / ``lsc_enable`` from this line's settings block; a ring defaults CSR off."""
        if self.file_block.get("csr_enable") is None and (
            self.closed_geometry or self.periodic
        ):
            self.csr_enable = False
        for flag in ("csr_enable", "lsc_enable"):
            stated = self.file_block.get(flag)
            if stated is not None:
                setattr(self, flag, bool(stated))

    def _apply_radiation_to_section(self) -> None:
        """Push :attr:`radiation` onto LAURA's ``sr_enable``/``isr_enable``."""
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
        """Enable CSR, on the section and every element."""
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
        """Number of CSR bins; 20 unless set here or on the machine section."""
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
        """Enable LSC, on the section and every element."""
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
        """Apply cavity structure wakefields; turning it off keeps their definitions."""
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
        """Number of LSC bins."""
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
        """How many times this line is tracked; 1 unless the ``tracking`` block says::

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

        Defaults to :attr:`closed_geometry`; the ``tracking`` block overrides it
        either way, e.g. for an injection-mismatch study::

            files:
              RING:
                code: elegant
                tracking: {turns: 1000, periodic: false}
        """
        tracking = self.file_block.get("tracking") or {}
        if "periodic" in tracking:
            return bool(tracking["periodic"])
        return self.closed_geometry

    @property
    def closed_geometry(self) -> bool:
        """Whether LAURA's section ``geometry`` is closed; :attr:`periodic` can override it."""
        geometry = self._machine_geometry()
        return getattr(geometry, "value", geometry) == "closed"

    @property
    def radiation(self) -> str | None:
        """
        Synchrotron-radiation model: ``off``, ``mean`` or ``quantum``; None if not stated.

        ``mean`` gives damping and energy loss; ``quantum`` adds excitation, needed for an
        equilibrium emittance. None leaves each code on its own (differing) default::

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
            Turns enough to reach equilibrium with radiation on.
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
        The codes with ``flag`` set, as the closing sentence of a warning.

        Parameters
        ----------
        flag: str
            A ``supports_*`` class flag.

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
        """Warn when :attr:`radiation` was asked for and this code has no switch for it."""
        if self.radiation is None or self.supports_radiation:
            return
        warn(exceptions.RadiationUnsupportedWarning(
            self.objectname, self.code, self.radiation, self.radiates_by_default,
            self.codes_that_can("supports_radiation"),
        ))

    @property
    def write_turns(self) -> bool:
        """
        Whether a multi-turn run keeps every turn's beams; off by default (see :attr:`bundles_turns`)::

            files:
              RING:
                code: xsuite
                tracking: {turns: 1000, write_turns: true}
        """
        tracking = self.file_block.get("tracking") or {}
        return bool(tracking.get("write_turns", False))

    @property
    def bundles_turns(self) -> bool:
        """
        Whether each screen's file holds every turn (multi-turn with :attr:`write_turns`); see :meth:`write_beam_file`.

        Read one with ``beam.read_beam_file(filename, turn=...)``.
        """
        return self.turns > 1 and self.write_turns

    @property
    def programs(self) -> list:
        """
        Elements whose strength is a program over turn number, from the ``tracking`` block::

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

        See :mod:`simba.Modules.DeviceProgram`.

        Returns
        -------
        list
            One :class:`~simba.Modules.DeviceProgram.DeviceProgram` per entry;
            an unreadable entry warns and is dropped.
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
        """Warn about knots past the last turn, or a non-zero last knot held for the rest of the run."""
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
        The reference momentum over turn number, from the ``tracking`` block::

            files:
              RING:
                code: xsuite
                tracking:
                  turns: 2000
                  ramp:
                    turns: [1, 1000]
                    momentum: [1.0e9, 2.0e9]

        See :mod:`simba.Modules.EnergyRamp`.

        Returns
        -------
        :class:`~simba.Modules.EnergyRamp.EnergyRamp` | None
            None if there is none; an unreadable ramp warns and is ignored.
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
        """Whether this run ramps: a :attr:`ramp`, a code that supports one, and more than one turn."""
        return self.supports_ramp and self.turns > 1 and self.ramp is not None

    @property
    def fixed_reference(self) -> bool:
        """
        Whether the reference momentum is the line's own (under a :attr:`ramp`, or in a ring) rather than the beam's.

        Cavities then accelerate particles but leave the reference alone.
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
            If the beam's rest energy or charge is not an electron's.
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
        The reference momentum on ``turn`` under :attr:`ramp`.

        Parameters
        ----------
        turn: int
            Turn number, 1-based.

        Returns
        -------
        float | None
            ``p0c`` in eV, or None unless this run is :attr:`ramped`.
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
            None unless this run is :attr:`ramped`.
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

        ``follow`` (default) scales each cavity's frequency with the reference speed;
        ``fixed`` keeps it constant. See :func:`~simba.Modules.EnergyRamp.rf_phase_slip`
        and :meth:`rf_phase_corrections`.

        Returns
        -------
        str
            ``follow`` or ``fixed``; anything else warns and is ``follow``.
        """
        tracking = self.file_block.get("tracking") or {}
        mode = str(tracking.get("rf", "follow")).lower()
        if mode not in RF_MODES:
            warn(exceptions.UnknownRFModeWarning(self.objectname, mode, RF_MODES))
            return "follow"
        return mode

    def pass_p0c(self) -> np.ndarray:
        """
        The reference momentum on every pass: the :attr:`ramp`'s, else :attr:`reference_p0c`; see :attr:`reference_clock`.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn`` values of ``p0c``, in eV.
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
        """The incoming beam's mean ``coord`` as :meth:`load_input_beam` read it; else the current beam's."""
        if self._input_reference is not None and coord in self._input_reference:
            return self._input_reference[coord]
        beam = self.global_parameters["beam"]
        return float(np.mean(getattr(beam, coord).val))

    @property
    def design_p0c(self) -> float | None:
        """
        A ring's design ``p0c`` in eV, from its section's ``reference_energy``.

        None on an open line, under a :attr:`ramp`, or when the section does not say.
        """
        if self.ramped or not (self.periodic or self.closed_geometry):
            return None
        energy = self._machine_reference_energy()
        if energy is None:
            return None
        return float(np.sqrt(energy**2 - self.rest_energy**2))

    @property
    def reference_p0c(self) -> float:
        """The reference ``p0c`` in eV: :attr:`design_p0c` if any, else the incoming mean ``cp`` before :meth:`sample_beam`."""
        design = self.design_p0c
        if design is not None:
            return design
        return self._input_mean("cp")

    @property
    def reference_energy(self) -> float:
        """Total energy of a particle at :attr:`reference_p0c`, in eV."""
        return float(np.hypot(self.reference_p0c, self.rest_energy))

    @property
    def reference_t0(self) -> float:
        """
        The time the reference particle enters, in s, which cavities are phased to.

        On an open line, the incoming mean ``t`` before sampling. With a
        :attr:`fixed_reference`, where the mean would absorb an injection timing
        error, the ``tracking`` setting comes first::

            files:
              RING:
                tracking:
                  reference_t0: 1.0e-9   # s

        then the beam's reference particle, if it has one, then the mean.
        """
        if self.fixed_reference:
            tracking = self.file_block.get("tracking") or {}
            if tracking.get("reference_t0") is not None:
                return float(tracking["reference_t0"])
            if self._input_particle is not None:
                return self._input_particle["t"]
        return self._input_mean("t")

    @property
    def reference_z0(self) -> float:
        """
        The reference particle's ``z`` as it enters, in m, following :attr:`reference_t0`.

        A ``reference_t0`` setting shifts the mean ``z`` by the distance the reference travels in between.
        """
        if self.fixed_reference:
            tracking = self.file_block.get("tracking") or {}
            if tracking.get("reference_t0") is not None:
                beta0 = beta_from_p0c(self.reference_p0c, self.rest_energy)
                return self._input_mean("z") + beta0 * speed_of_light * (
                    self._input_mean("t") - self.reference_t0
                )
            if self._input_particle is not None:
                return self._input_particle["z"]
        return self._input_mean("z")

    def pass_beta0(self) -> float | np.ndarray:
        """
        Reference speed over ``c`` on every pass; see :meth:`pass_p0c`.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn`` values.
        """
        return beta_from_p0c(self.pass_p0c(), self.rest_energy)

    def pass_index(self, turn: int, sector: int = 1) -> int:
        """0-based index of pass ``sector`` of ``turn``, both 1-based; see :attr:`passes_per_turn`."""
        return (turn - 1) * self.passes_per_turn + sector - 1

    def last_pass(self, turn: int) -> int:
        """0-based index of the last pass of ``turn`` (1-based), on which its outputs are recorded."""
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
        ``(t0, starts, beta0, p0c)``: :attr:`reference_t0`, then per pass its start after ``t0``,
        reference speed over ``c`` and momentum in eV; see :meth:`reference_time`.
        """
        if self._reference_clock is None:
            self.reset_reference_clock()
        return self._reference_clock

    def reference_time(self, s: float, pass_index: int = 0) -> float:
        """
        Absolute time the reference particle reaches ``s`` on a pass.

        ``T_j(s) = t0 + sum_{k<j} C / (beta_k c) + s / (beta_j c)``, with ``t0`` the
        :attr:`reference_t0`, ``C`` the pass length and ``beta_k`` from :meth:`pass_beta0`.
        The same for all codes; :meth:`time_to_native` converts to the code's own.

        Parameters
        ----------
        s: float
            Metres from the lattice entrance, within the pass.
        pass_index: int
            0-based pass; see :meth:`pass_index`.

        Returns
        -------
        float
            Seconds.
        """
        t0, starts, beta0, _ = self.reference_clock
        return float(t0 + starts[pass_index] + s / (beta0[pass_index] * speed_of_light))

    def pass_start_times(self) -> np.ndarray:
        """
        The absolute time each pass starts, on :meth:`reference_time`.

        Returns
        -------
        np.ndarray
            ``turns * passes_per_turn`` values, in seconds.
        """
        t0, starts, _, _ = self.reference_clock
        return t0 + starts[: self.turns * self.passes_per_turn]

    @staticmethod
    def pass_staircase(starts, values) -> tuple:
        """
        A per-pass table against time, flat for a quarter pass either side of each pass's start.

        So a bunch off the reference still reads its own pass's value.

        Parameters
        ----------
        starts: array-like
            Seconds at the start of each pass.
        values: array-like
            One per pass.

        Returns
        -------
        tuple
            ``(times, values)``, two knots per pass.
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
        ``scale`` in ``native = scale * (t - reference_time)``, for the code's :attr:`native_time`.

        Parameters
        ----------
        beta0: float
            Reference speed over ``c``.

        Returns
        -------
        float | None
            None if the code's own ``t`` is already absolute (elegant's is).
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
            The code's coordinate, in :attr:`native_time` units.
        s: float
            Metres from the lattice entrance, within the pass.
        pass_index: int
            0-based pass.
        beta0: float | array-like | None
            Reference speed over ``c``, per particle if the code has it;
            defaults to the pass's from :attr:`reference_clock`.

        Returns
        -------
        np.ndarray
            Seconds.
        """
        if beta0 is None:
            beta0 = self.reference_clock[2][pass_index]
        scale = self.native_time_scale(beta0)
        native = np.asarray(native, dtype=float)
        if scale is None:
            return native
        return self.reference_time(s, pass_index) + native / scale

    def time_to_native(self, t, s: float, pass_index: int, beta0=None) -> np.ndarray:
        """The code's own time coordinate from absolute ``t``; the inverse of :meth:`time_from_native`."""
        if beta0 is None:
            beta0 = self.reference_clock[2][pass_index]
        scale = self.native_time_scale(beta0)
        t = np.asarray(t, dtype=float)
        if scale is None:
            return t
        return scale * (t - self.reference_time(s, pass_index))

    def native_times(self, beam, element: str | None = None, turn: int | None = None):
        """
        Each particle's time in the code's own coordinate, for a beam this line wrote.

        Parameters
        ----------
        beam: :class:`~simba.Modules.Beams.beam`
            A beam this line wrote.
        element: str | None
            Where; defaults to the end of the line.
        turn: int | None
            Defaults to the beam's own ``turn``, else 1.

        Returns
        -------
        np.ndarray
            In :attr:`native_time` units.
        """
        element = element or self.end
        if turn is None:
            turn = getattr(beam, "turn", None) or 1
        s = self.getSValues(as_dict=True)[element]
        return self.time_to_native(beam.t.val, s, self.last_pass(turn))

    def accelerating_cavities(self) -> dict:
        """
        This line's accelerating cavities, excluding deflecting and crab cavities.

        Returns
        -------
        dict
            The elements, by name.
        """
        cavities = {}
        for name, element in self.elements.items():
            hardware = str(getattr(element, "hardware_type", "") or "").lower()
            if "cavity" not in hardware or "deflect" in hardware or "crab" in hardware:
                continue
            cavities[name] = element
        return cavities

    def live_cavities(self) -> dict:
        """The :meth:`accelerating_cavities` with a voltage and a frequency, so whose phase matters."""
        return {
            name: element
            for name, element in self.accelerating_cavities().items()
            if self.cavity_voltage(element)
            and getattr(getattr(element, "cavity", None), "frequency", None)
        }

    def rf_phase_corrections(self) -> dict:
        """
        Phase moves per cavity and pass so this code runs :attr:`rf_mode`.

        The :attr:`rf_mode` slip less the :attr:`native_rf` slip, both from
        :func:`~simba.Modules.EnergyRamp.rf_phase_slip`.

        Returns
        -------
        dict
            Radians, one per pass, by cavity name; empty when nothing needs moving.

        Warns
        -----
        :class:`~simba.exceptions.RFPhasesUnsupportedWarning`
            If corrections are needed and this code cannot make them.
        """
        passes = self.turns * self.passes_per_turn
        cavities = self.live_cavities()
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
            if np.max(np.abs(correction)) > 1e-4:
                corrections[name] = correction
        if corrections and self.native_rf is None:
            warn(exceptions.RFPhasesUnsupportedWarning(
                self.objectname, self.rf_mode, corrections, self.code
            ))
            return {}
        return corrections

    def cavity_phase(self, name: str) -> float | None:
        """
        A cavity's current phase in the code's units (:attr:`rf_phase_per_radian`), for :meth:`apply_rf_phases`.

        Overridden by the codes that move phases pass by pass.

        Parameters
        ----------
        name: str
            Cavity name, as simba names it.

        Returns
        -------
        float | None
            None if the code's lattice has no such cavity.
        """
        return None

    def set_cavity_phase(self, name: str, phase: float) -> None:
        """Set a cavity's phase in the code's own units; the inverse of :meth:`cavity_phase`."""
        raise NotImplementedError(
            f"{self.code} reads cavity phases but cannot set them"
        )

    def rf_phase_shifts(self, correction) -> np.ndarray:
        """An :meth:`rf_phase_corrections` entry in the code's phase units and sign convention."""
        return self.rf_phase_sign * self.rf_phase_per_radian * np.asarray(correction)

    def begin_rf_phases(self) -> dict:
        """
        Store and return this run's :meth:`rf_phase_corrections`, forgetting phases read on earlier runs.
        """
        self._rf_corrections = self.rf_phase_corrections()
        self._rf_phase0 = {}
        return self._rf_corrections

    def apply_rf_phases(self, pass_index: int | None) -> None:
        """
        Move each cavity's phase for ``pass_index``; see :meth:`rf_phase_corrections`.

        Shifts are from the phase first read by :meth:`cavity_phase`.

        Parameters
        ----------
        pass_index: int | None
            0-based pass (:meth:`pass_index`); None restores the given phases.
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
        Turn loop for codes SIMBA drives a pass at a time, restoring turn 1 after (:meth:`end_turns`).

        Parameters
        ----------
        track_pass: callable
            ``track_pass(turn, pass_index, name_turn, record)`` tracks one pass;
            ``record`` is True on the last pass of a turn :meth:`output_turns` keeps.
        start_turn: callable, optional
            ``start_turn(turn)``, called once a turn's programs are set.
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
        """Restore every cavity phase as given and every program to turn 1."""
        self.apply_rf_phases(None)
        missing = set(self._rf_corrections or {}) - set(self._rf_phase0 or {})
        if missing:
            warn(exceptions.MissingCavitiesWarning(self.objectname, missing, self.code))
        if self.turns > 1:
            self.apply_programs(1)

    @property
    def rf_voltage(self) -> float:
        """Total accelerating cavity amplitude on one turn, in volts, ignoring phase."""
        total = sum(
            self.cavity_voltage(element)
            for element in self.accelerating_cavities().values()
        )
        return total * self.passes_per_turn

    @staticmethod
    def cavity_voltage(element) -> float:
        """A cavity's ``|field_amplitude|`` in volts, or 0 if it has none."""
        simulation = getattr(element, "simulation", None)
        try:
            amplitude = simulation.resolved("field_amplitude")
        except (AttributeError, TypeError, ValueError):
            amplitude = getattr(simulation, "field_amplitude", 0.0)
        return abs(float(amplitude or 0.0))

    def check_ramp(self) -> None:
        """Warn about a ramp this run will not track as written; beam checks are in :meth:`check_ramp_beam`."""
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

    def check_design_energy(self) -> None:
        """Warn when a ring's beam enters more than 1 % from its :attr:`design_p0c`."""
        design = self.design_p0c
        if design is None or self._input_reference is None:
            return
        entering = self._input_mean("cp")
        if abs(entering / design - 1) > 1e-2:
            warn(exceptions.OffDesignEnergyWarning(self.objectname, design, entering))

    def check_ramp_beam(self) -> None:
        """Warn if the input beam is off the ramp's start, or the RF is too weak to follow it."""
        if not self.ramped:
            return
        ramp = self.ramp
        beam = (self.global_parameters or {}).get("beam")
        if beam is None:
            return
        rest_energy = self.rest_energy
        start = ramp.p0c_at(1, rest_energy)
        entering = self._input_mean("cp")
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
        """Seconds per turn, ``passes_per_turn * C / (beta0 * c)``, or 0.0 without a beam."""
        beam = (self.global_parameters or {}).get("beam")
        if beam is None:
            return 0.0
        beta = float(np.mean(beam.BetaGamma) / np.mean(beam.gamma))
        if not beta:
            return 0.0
        return self.passes_per_turn * self.pass_length / (beta * speed_of_light)

    def program_is_vertical(self, name: str) -> bool:
        """Whether element ``name``'s ``hardware_type`` marks it vertical."""
        element = self.elements.get(name)
        hardware = str(getattr(element, "hardware_type", "") or "")
        return hardware.lower().startswith("vertical")

    def program_attribute(self, program, element_type: str | None) -> str | None:
        """
        The attribute ``program`` sets: its ``parameter``, else :attr:`program_attributes` for the element's plane.

        Parameters
        ----------
        program: :class:`~simba.Modules.DeviceProgram.DeviceProgram`
        element_type: str | None
            The programmed element's type, as this code names it.

        Returns
        -------
        str | None
            None if there is none to set.
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
        """Set each programmed element to its value for ``turn`` (1-based)."""

    def output_turns(self) -> list:
        """
        ``(data_turn, name_turn)`` for each beam a run should write.

        ``name_turn`` is passed to :meth:`write_beam_file` and :meth:`output_basename`.
        """
        if self.turns <= 1:
            return [(None, None)]
        if self.write_turns:
            return [(turn, turn) for turn in range(1, self.turns + 1)]
        return [(self.turns, None)]

    def beam_turn(self, turn: int | None) -> int:
        """The 1-based turn for either half of an :meth:`output_turns` pair; None means the last."""
        return self.turns if turn is None else turn

    def output_beam_file(self, name: str) -> str:
        """The openPMD file a beam recorded at ``name`` goes to, every turn of it."""
        return os.path.join(
            self.global_parameters["master_subdir"],
            f"{self.output_basename(name)}.openpmd.hdf5",
        )

    def write_beam_file(self, beam, name: str, turn: int | None = None) -> bool:
        """
        Write ``beam``, recorded at ``name``, to :meth:`output_beam_file`.

        ``turn`` is an :meth:`output_turns` ``name_turn``. When the run
        :attr:`bundles_turns`, turns go into one file and must come in order;
        no turn then means the last.

        Returns
        -------
        bool
            Whether it was written (see :meth:`writes_output`).
        """
        if not self.writes_output(name):
            return False
        rbf.openpmd.write_openpmd_beam_file(
            beam, self.output_beam_file(name),
            turn=self.beam_turn(turn) if self.bundles_turns else None,
        )
        return True

    @property
    def da_settings(self) -> dict:
        """
        Grid for a dynamic-aperture scan, from the ``tracking`` block::

            files:
              RING:
                code: ocelot
                tracking:
                  turns: 1000
                  dynamic_aperture: {nx: 20, ny: 10, x_max: 0.02, y_max: 0.01, n_lines: 11}

        The aperture is searched along ``n_lines`` rays (:meth:`da_rays`), the
        frequency map over the ``nx`` by ``ny`` grid (:meth:`da_grid`).
        """
        tracking = self.file_block.get("tracking") or {}
        return tracking.get("dynamic_aperture") or {}

    def da_grid(self):
        """``(xs, ys)`` starting amplitudes, each from one step off zero; see :attr:`da_settings`."""
        settings = self.da_settings
        nx = max(1, int(settings.get("nx", 10)))
        ny = max(1, int(settings.get("ny", 1)))
        x_max = float(settings.get("x_max", 0.01))
        y_max = float(settings.get("y_max", 0.001))
        return (
            np.linspace(x_max / nx, x_max, nx),
            np.linspace(y_max / ny, y_max, ny),
        )

    def da_rays(self) -> list:
        """
        ``(x, y)`` starts along the rays of elegant's ``find_aperture`` ``n-line`` mode.

        ``n_lines`` rays from +x round to -x, each with ``nx - 1`` points out to the
        ellipse through ``(x_max, 0)`` and ``(0, y_max)``, so every code tracks the same starts.
        """
        settings = self.da_settings
        nx = max(2, int(settings.get("nx", 10)))
        n_lines = max(2, int(settings.get("n_lines", 11)))
        x_max = float(settings.get("x_max", 0.01))
        y_max = float(settings.get("y_max", 0.001))
        return [
            (x_max * j / (nx - 1) * math.cos(angle), y_max * j / (nx - 1) * math.sin(angle))
            for angle in (math.pi * k / (n_lines - 1) for k in range(n_lines))
            for j in range(1, nx)
        ]

    def dynamic_aperture_boundary(self, results) -> list:
        """
        The last survivor before the first loss on each ray, as elegant's ``find_aperture`` reports it.

        See :func:`~simba.Modules.plotting.ring.aperture_boundary` for the survival rule.

        Parameters
        ----------
        results : list
            From :meth:`run_dynamic_aperture`.

        Returns
        -------
        list
            ``(x, y)`` from +x round to -x; a ray lost at its first point is absent.
        """
        from .Modules.plotting.ring import aperture_boundary

        return aperture_boundary(results, self.turns)

    @staticmethod
    def tune_from_harmonic(line_position: float, reference_tune: float) -> float:
        """
        Rebuild a tune from a ``freq_analysis`` harmonic position.

        ``freq_analysis`` reports ``|nearest integer - Q|``; the reference tune picks the side.

        Parameters
        ----------
        line_position : float
            ``|nearest integer - Q|``.
        reference_tune : float
            A tune on the right side of the nearest integer.

        Returns
        -------
        float
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
        Track 13 probes and carry the distribution through the linear map they measure.

        A tracking setting, like :attr:`turns`::

            files:
              RING:
                code: madx
                tracking: {single_particle: true}
        """
        tracking = self.file_block.get("tracking") or {}
        return bool(tracking.get("single_particle", False))

    @property
    def nsuperperiods(self) -> int:
        """
        How many times the line is traversed per turn, for one sector of an N-fold-symmetric ring::

            files:
              RING:
                code: ocelot
                tracking: {turns: 1000, nsuperperiods: 4}

        This tracks 4000 passes, but outputs still count 1000 turns. Defaults to 1.
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
        """:attr:`nsuperperiods` if this code supports it, else 1."""
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

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py`` arrays of length :attr:`turns`, or
            ``{}`` if this code cannot produce one.
        """
        return {}

    def normalisation_twiss(self) -> dict:
        """
        Periodic Twiss and closed orbit for :func:`~simba.Modules.Matrices.tune_diffusion`.

        Returns
        -------
        dict
            Empty without a one-turn map, so tunes come from raw coordinates.
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
        Tune per starting amplitude over :meth:`da_grid`.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y)`` per surviving grid point; empty, with a
            warning, if this code cannot do it.
        """
        warn(exceptions.FrequencyMapUnsupportedWarning(
            self.objectname, self.code, self.codes_that_can("supports_frequency_map")
        ))
        return []

    def _footprint(self, tracks) -> list:
        """
        Tunes of tracked survivors, stored as :attr:`frequency_map`.

        Parameters
        ----------
        tracks: iterable
            ``(x, y, xs, pxs, ys, pys)`` per surviving start: its grid point and
            turn-by-turn coordinates.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y, diffusion)`` for each start that gave a tune.
        """
        from .Modules.Matrices import tune_diffusion

        twiss = self.normalisation_twiss()
        footprint = []
        for x, y, xs, pxs, ys, pys in tracks:
            tune_x, tune_y, diffusion = tune_diffusion(xs, pxs, ys, pys, twiss=twiss)
            if not math.isnan(tune_x):
                footprint.append((float(x), float(y), tune_x, tune_y, diffusion))
        if not footprint:
            warn(exceptions.NoFootprintWarning(self.objectname))
        self.frequency_map = footprint
        return footprint

    def run_dynamic_aperture(self) -> list:
        """
        Track one particle per grid point for :attr:`turns` and record which survive.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per grid point; empty, with a warning,
            if this code cannot do it.
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

        Compares LAURA's first entrance and last exit, to ``tolerance`` relative to
        the path length; superperiods go to :meth:`check_superperiods_close`.
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
        """Warn when the declared superperiod count and the geometry disagree."""
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
        """Total bending angle of the line, in radians; ``2*pi`` for a closed planar ring."""
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
        """Warn if the beam's ``brho`` disagrees with this pass's stated momentum.

        Field-based codes take ``Brho`` from the loaded beam, but ``k`` was resolved at the stated momentum.
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

        Prefixed with the line name only if in :attr:`colliding_outputs`, and
        suffixed with ``turn`` when given on a multi-turn run.
        """
        qualified = name in self.colliding_outputs
        name = flatten_occurrence(name)
        if qualified:
            name = f"{self.objectname}{OUTPUT_LINE_SEPARATOR}{name}"
        if turn is not None and self.turns > 1:
            name = f"{name}{OUTPUT_TURN_SEPARATOR}{turn:0{len(str(self.turns))}d}"
        return name

    def sampled_index(self, index: int | None) -> int | None:
        """A particle's index after :meth:`sample_beam`, or None if sampling drops it."""
        if index is None:
            return None
        interval = max(1, int(self.sample_interval))
        return int(index) // interval if int(index) % interval == 0 else None

    def sample_beam(self, bm):
        """Every :attr:`sample_interval`-th particle of a beam, conserving the total charge."""
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

    def writes_output(self, name: str) -> bool:
        """
        Whether a beam recorded at ``name`` gets an :meth:`output_beam_file`.

        Not at the start of the line: that is the input beam, the previous line's end.
        """
        return name != self.start

    def get_prefix(self) -> str:
        """The ``input: prefix`` of the file block, defaulting to the master subdirectory."""
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "prefix" not in self.file_block["input"]:
            self.file_block["input"]["prefix"] = self.global_parameters["master_subdir"] + "/"
        return self.file_block["input"]["prefix"]

    def set_prefix(self, prefix: str) -> None:
        """Set the ``input: prefix`` of the file block."""
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
        The input beam's file stem: ``input: particle_definition`` (``initial_distribution`` maps to ``laser``), else :attr:`start`.
        """
        stated = (self.file_block.get("input") or {}).get("particle_definition")
        if stated is None:
            return self.start
        return "laser" if stated == "initial_distribution" else stated

    def load_input_beam(self, prefix: str, particle_definition: str) -> str:
        """
        Read the incoming beam and prepare it for this lattice, shared by every code's ``preProcess``.

        Records the reference for :attr:`reference_p0c` and :attr:`reference_t0`,
        samples (:meth:`sample_beam`), runs :meth:`check_species`, rematches to
        ``input: twiss``, sets :attr:`ref_idx`, runs :meth:`check_design_energy` and
        :meth:`check_ramp_beam`, and resets the :meth:`reference_time` clock.

        Parameters
        ----------
        prefix: str
            Prefix of the input beam file.
        particle_definition: str
            Input beam file name, without its extension.

        Returns
        -------
        str
            Path of the file read by :meth:`read_input_file`.
        """
        filepath = self.read_input_file(prefix, particle_definition)
        full = self.global_parameters["beam"]
        self._input_reference = {
            coord: float(np.mean(getattr(full, coord).val)) for coord in ("cp", "t", "z")
        }
        index = full.reference_particle_index
        self._input_particle = (
            {coord: float(getattr(full, coord).val[index]) for coord in ("t", "z")}
            if index is not None and 0 <= index < len(full.t.val)
            else None
        )
        if int(self.sample_interval) > 1:
            self.global_parameters["beam"] = self.sample_beam(self.global_parameters["beam"])
        self.check_species()
        beam = self.global_parameters["beam"]
        beam.beam.rematchXPlane(**self.initial_twiss["horizontal"])
        beam.beam.rematchYPlane(**self.initial_twiss["vertical"])
        self.ref_idx = beam.reference_particle_index
        self.check_design_energy()
        self.check_ramp_beam()
        self.reset_reference_clock()
        return filepath

    def update_groups(self) -> None:
        """Update the group objects in the lattice with their settings."""
        for g in list(self.groupSettings.keys()):
            if g in self.groupObjects:
                setattr(self, g, self.groupObjects[g])
                if self.groupSettings[g] is not None:
                    self.groupObjects[g].update(**self.groupSettings[g])

    def getElement(self, element: str, param: str = None) -> dict | PhysicalBaseElement:
        """
        Get an element or group by name, or one of its parameters.

        Parameters
        ----------
        element: str
        param: str, optional
            Parameter to return instead of the object.

        Returns
        -------
        dict | :class:`~laura.models.element.Element`
            Empty dict, with a warning, if the name is unknown.
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
        Get all elements of a hardware type, or their parameters.

        Parameters
        ----------
        typ: list, tuple, or str
            Type(s); a sequence gives one list per type.
        param: list, tuple, or str, optional
            Parameter(s) to return instead of the elements.

        Returns
        -------
        list | zip
            A zip of per-parameter values when ``param`` is a sequence.
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
        Set ``setting`` on every element of a hardware type, one value each.

        Parameters
        ----------
        typ: list, tuple, or str
        setting: str
        values: list, tuple, or Any
            One per element.

        Raises
        ------
        ValueError
            If the element and value counts differ.
        """
        elems = self.getElementType(typ)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                setattr(e, setting, v)
                if e.hardware_type.lower() == "dipole" and setting == "angle":
                    e.magnetic.multipoles.K0L.normal = v
        else:
            raise ValueError

    @property
    def quadrupoles(self) -> list:
        """All quadrupoles in the lattice."""
        return self.getElementType("quadrupole")

    @property
    def cavities(self) -> list:
        """All RF cavities in the lattice."""
        return self.getElementType("RFCavity")

    @property
    def solenoids(self) -> list:
        """All solenoids in the lattice."""
        return self.getElementType("solenoid")

    @property
    def dipoles(self) -> list:
        """All dipoles in the lattice."""
        return self.getElementType("dipole")

    @property
    def kickers(self) -> list:
        """All horizontal, vertical and combined correctors in the lattice."""
        return sum(
            (self.getElementType(t) for t in ("Horizontal_Corrector", "Vertical_Corrector", "Combined_Corrector")),
            [],
        )

    @property
    def dipoles_and_kickers(self) -> list:
        """All dipoles and kickers, sorted by end ``z``."""
        return sorted(
            self.dipoles + self.kickers,
            key=lambda x: x.physical.end.z,
        )

    @property
    def wakefields(self) -> list:
        """All wakefield elements in the lattice."""
        return self.getElementType("wakefield")

    @property
    def wakefields_and_cavity_wakefields(self) -> list:
        """Cavities with a wakefield definition, then wakefield elements."""
        cavities = [
            cav
            for cav in self.cavities
            if cav.simulation.wakefield_definition
        ]
        wakes = self.getElementType("wakefield")
        return cavities + wakes

    @property
    def screens(self) -> list:
        """All screens in the lattice."""
        return self.getElementType("screen")

    @property
    def screens_and_bpms(self) -> list:
        """All screens and BPMs, sorted by start ``z``."""
        return sorted(
            self.getElementType("screen")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.physical.start.z,
        )

    @property
    def screens_and_markers_and_bpms(self) -> list:
        """All screens, markers and BPMs, sorted by start ``z``."""
        return sorted(
            self.getElementType("screen")
            + self.getElementType("marker")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.physical.start.z,
        )

    @property
    def apertures(self) -> list:
        """All apertures and collimators, sorted by start ``z``."""
        return sorted(
            self.getElementType("aperture") + self.getElementType("collimator"),
            key=lambda x: x.physical.start.z,
        )

    @property
    def wigglers(self) -> list:
        """All wigglers in the lattice."""
        return self.getElementType("wiggler")

    @property
    def photon_monitors(self) -> list:
        """All photon monitors in the lattice."""
        return self.getElementType("photon_monitor")

    @property
    def start(self) -> str:
        """
        Name of the lattice's first element.

        ``output: start_element`` if given, else the element starting at ``zstart``
        (preferring one with length), else the first on the beam path.
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
        """The element named by :attr:`start`."""
        return self.elementObjects[self.start]

    @property
    def end(self) -> str:
        """
        Name of the lattice's last element.

        ``output: end_element`` if given, else the element ending at or past ``zstop``,
        else the last element.
        """
        if "end_element" in self.file_block["output"]:
            return self.file_block["output"]["end_element"]
        elif "zstop" in self.file_block["output"]:
            endelems = []
            for name, elem in self.elementObjects.items():
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
        """The element named by :attr:`end`."""
        return self.elementObjects[self.end]

    @property
    def start_s(self) -> float:
        """
        ``s`` at the exit of the first element, along the reference path from the machine start.

        Not ``startObject.physical.start.z``: bends make the path longer than its z projection.
        """
        if self._start_s is None:
            self._start_s = self.machine.get_elements_s_pos(end=self.start)[self.start]
        return self._start_s

    @property
    def entrance_s(self) -> float:
        """
        ``s`` of the lattice entrance: :attr:`start_s` less the first element's length.

        Per-element ``s`` positions are measured from here.
        """
        return float(self.start_s - self.startObject.physical.length)

    def _machine_space_charge(self):
        """
        Space-charge settings of this lattice's LAURA section, or None for code defaults.

        This lattice's own ``csr_bins`` is applied after, so still wins.
        """
        for section in (getattr(self.machine, "sections", None) or {}).values():
            if self.start in getattr(section, "order", ()):
                return section.space_charge
        return None

    def _machine_geometry(self):
        """
        The LAURA section's ``geometry`` (``open``/``closed``, as Bmad's), or None; see :attr:`periodic`.
        """
        for section in (getattr(self.machine, "sections", None) or {}).values():
            if self.start in getattr(section, "order", ()):
                return getattr(section, "geometry", None)
        return None

    def _machine_reference_energy(self) -> float | None:
        """The LAURA section's design total energy in eV, or None; see :attr:`design_p0c`."""
        for section in (getattr(self.machine, "sections", None) or {}).values():
            if self.start in getattr(section, "order", ()):
                energy = getattr(section, "reference_energy", None)
                return float(energy) if energy else None
        return None

    @computed_field
    @property
    def section(self) -> SectionLatticeTranslator:
        """The lattice's elements as a LAURA ``SectionLatticeTranslator``, built once and cached."""
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
        """The lattice's elements, by name."""
        return self.section.elements.elements

    def write(self):
        pass

    def run_command(self, command: list, logfile: str, **kwargs) -> None:
        """
        Run a simulation code, logging to ``logfile``, and raise if it exits non-zero.

        Otherwise a failed run surfaces later as missing output.

        Parameters
        ----------
        command: list
        logfile: str
            Its tail is quoted if the code fails.
        kwargs:
            Passed to :func:`subprocess.call`.

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
            with open(logfile) as f:
                tail = "".join(f.readlines()[-20:]).strip()
        except OSError:
            tail = ""
        raise RuntimeError(
            f"{self.code} exited with status {status} running {self.objectname}.\n"
            f"Last lines of {logfile}:\n{tail}"
        )

    def run(self) -> None:
        """
        Run the code on this lattice's input file, logging to ``<name>.log`` in the master subdirectory.

        Calls :meth:`run_remote` instead if :attr:`remote_setup` is set.

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
        Run the simulation on a remote server over SSH (:meth:`connect_remote`).

        Uploads the input, beam and field files to a directory named after
        ``master_subdir``, runs, then fetches every file modified since the start.
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
                if stat.S_ISDIR(attr.st_mode):
                    continue
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
        Open an SSH connection from :attr:`remote_setup`'s ``host``, ``username`` and ``password``.

        Returns
        -------
        paramiko.SSHClient

        Raises
        ------
        KeyError
            If a required key is missing.
        paramiko.AuthenticationException
            If authentication fails.
        TimeoutError
            If the server is unreachable.
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
        The ``input: twiss`` alpha, beta and normalised emittance per plane.

        Returns
        -------
        dict
            ``horizontal`` and ``vertical`` entries; missing values are False.
        """
        if (
            "input" in self.file_block
            and "twiss" in self.file_block["input"]
            and self.file_block["input"]["twiss"]
        ):
            twiss = self.file_block["input"]["twiss"]
            alpha_x = twiss.get("alpha_x", False)
            alpha_y = twiss.get("alpha_y", False)
            beta_x = twiss.get("beta_x", False)
            beta_y = twiss.get("beta_y", False)
            nemit_x = twiss.get("nemit_x", False)
            nemit_y = twiss.get("nemit_y", False)
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
            freq = list({c.cavity.frequency for c in cavs})
            if len(freq) > 1:
                raise ValueError("All accelerating cavities must have the same frequency")
            freq = freq[0]
        else:
            raise KeyError("settings must contain `cavities` key containing names of cavities")
        if "harmonics" in settings:
            harmonics = [c for c in self.cavities if c.name in settings["harmonics"]]
            harm_freq = list({c.cavity.frequency for c in harmonics})
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
        curvature = settings.get("curvature", 0)
        skewness = settings.get("skewness", 0)

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
                print(f"Longitudinal matching gave harmonic phase of {phih} and field amplitude of {vh}")

    def preProcess(self) -> None:
        """
        Run the pre-tracking checks, apply section settings, read :meth:`getInitialTwiss` and do any matching.
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
            if "enable" in self.file_block["match"] and not self.file_block["match"]["enable"]:
                domatch = False
            if domatch:
                self.match(self.file_block["match"])
        if "longitudinal_match" in self.file_block:
            self.longitudinal_match(self.file_block["longitudinal_match"])
        self.section.astra_headers = ast

    def read_closed_orbit(self):
        """
        The closed orbit at the start of the line; zero on a perfectly aligned lattice.

        Returns
        -------
        numpy.ndarray | None
            6 components in this code's :attr:`otm_convention` order, or None
            if the code did not give one.
        """

    def read_optics_summary(self) -> dict:
        """
        The code's own full tune and chromaticity, from its periodic solution.

        Returns
        -------
        dict
            Any of ``tune_x_total``, ``tune_y_total``, ``chromaticity_x``,
            ``chromaticity_y``; empty if the code reports none.
        """
        return {}

    def read_one_turn_map(self):
        """This code's 6x6 one-turn map, or None; overridden by the backends that can.

        Returns
        -------
        numpy.ndarray | None
            A 6x6 matrix in :attr:`otm_convention` coordinates.
        """

    def one_turn_map_canonical(
        self, beta0: float | None = None, magnitude: bool = True
    ):
        """
        :attr:`one_turn_map` in Xsuite's ``(x, px, y, py, zeta, delta)``, so codes compare.

        A diagonal similarity ``D R D^-1`` that rescales only the longitudinal block,
        so tunes are unchanged. Elegant's fifth coordinate is path length, not time
        of flight, so it cannot be rescaled (a drift has ``R56 = 0``).

        Parameters
        ----------
        beta0: float | None
            Reference ``v/c``; read from the beam if not given. Unused without ``magnitude``.
        magnitude: bool
            Convert sizes as well as signs; False applies only :attr:`otm_longitudinal_sign`.

        Returns
        -------
        numpy.ndarray | None
            None when ``magnitude`` is set and this code's conversion is not a rescale.
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
        Tune and periodic Twiss from the one-turn map, plus momentum compaction from its canonical form.

        Also merges :attr:`optics_summary` and :attr:`closed_orbit`. Chromaticity
        needs more than one map, so comes only from the code's own summary.

        Returns
        -------
        dict
            ``{}`` without a map; else ``stable_*``, fractional ``tune_*``, ``beta_*``,
            ``alpha_*``, ``gamma_*``, ``closed_orbit_*`` and, where the convention
            allows, ``slip_factor`` and ``momentum_compaction``.
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
        """Warn when the one-turn map is not 6x6 or its determinant is not 1 to ``tolerance``."""
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
        """For a periodic run, read back the one-turn map, optics summary and closed orbit."""
        if self.periodic and self.supports_periodic:
            self.one_turn_map = self.read_one_turn_map()
            self.check_one_turn_map()
            self.optics_summary = self.read_optics_summary()
            self.closed_orbit = self.read_closed_orbit()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str = self.name + " = ("
        for e in self.elements:
            if len((str + e).splitlines()[-1]) > 60:
                str += "&\n"
            str += e + ", "
        return str + ")"

    def createDrifts(self) -> dict:
        """The lattice's elements with drifts inserted between them."""
        return self.section.create_drifts()

    def getSValues(
        self,
        as_dict: bool = False,
        at_entrance: bool = False,
        drifts: bool = True,
    ) -> list | dict:
        """
        Cumulative ``s`` of each element from the lattice entrance.

        Parameters
        ----------
        as_dict: bool, optional
            Return ``{name: s}`` instead of a list.
        at_entrance: bool, optional
            Give each element's entrance ``s`` rather than its exit.
        drifts: bool, optional
            Include drifts.

        Returns
        -------
        list | dict
        """
        if drifts:
            lengths = self.section.drift_lengths()
            names, lengths = list(lengths), list(lengths.values())
        else:
            names = [e.name for e in self.elements.values()]
            lengths = [e.physical.length for e in self.elements.values()]
        s = [0]
        for length in lengths:
            s.append(s[-1] + length)
        s = s[:-1] if at_entrance else s[1:]
        if as_dict:
            return dict(zip(names, s))
        return list(s)

    def getZValues(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
        """
        ``[start z, end z]`` of each element.

        Parameters
        ----------
        drifts: bool, optional
            Include drifts.
        as_dict: bool, optional
            Return ``{name: [start, end]}`` instead of a list.

        Returns
        -------
        list | dict
        """
        elems = self.createDrifts() if drifts else self.elements
        if as_dict:
            return {e.name: [e.physical.start.z, e.physical.end.z] for e in elems.values()}
        return [[e.physical.start.z, e.physical.end.z] for e in elems.values()]

    def getNames(self, drifts: bool = True) -> list:
        """
        Names of the elements in the lattice.

        Parameters
        ----------
        drifts: bool, optional
            Include drifts.

        Returns
        -------
        list
        """
        elems = self.createDrifts() if drifts else self.elements
        return [e.name for e in list(elems.values())]

    def getElems(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
        """
        The elements in the lattice.

        Parameters
        ----------
        drifts: bool, optional
            Include drifts.
        as_dict: bool, optional
            Return ``{name: element}`` instead of a list.

        Returns
        -------
        list | dict
        """
        elems = self.createDrifts() if drifts else self.elements
        if as_dict:
            return {e.name: e for e in list(elems.values())}
        return list(elems.values())

    def getSNames(self) -> list:
        """``(name, s)`` for each element, drifts included; see :meth:`getSValues`."""
        s = self.getSValues()
        names = self.getNames()
        return list(zip(names, s))

    def getSNamesElems(self) -> tuple:
        """``(names, elements, s)`` lists, drifts included."""
        s = self.getSValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, s

    def getZNamesElems(self) -> tuple:
        """``(names, elements, z)`` lists, drifts included; ``z`` as in :meth:`getZValues`."""
        z = self.getZValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, z

    def findS(self, elem) -> list:
        """
        ``(name, s)`` entries of :meth:`getSNames` for element ``elem``.

        Parameters
        ----------
        elem: str

        Returns
        -------
        list
            Empty if the element is not in the lattice.
        """
        if elem in self.allElements:
            sNames = self.getSNames()
            return [a for a in sNames if a[0] == elem]
        return []

    def updateRunSettings(self, runSettings: runSetup) -> None:
        """
        Replace the lattice's run settings.

        Parameters
        ----------
        runSettings: :class:`runSetup`

        Raises
        ------
        TypeError
            If ``runSettings`` is not a :class:`runSetup`.
        """
        if isinstance(runSettings, runSetup):
            self.runSettings = runSettings
        else:
            raise TypeError(
                "runSettings argument passed to frameworkLattice.updateRunSettings is not a runSetup instance"
            )

    def setup_xsuite_line(self) -> tuple:
        """
        Build an Xsuite line from this lattice, reading the input beam.

        Returns
        -------
        tuple
            ``(xt.Line, beam copy, element names)``.
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
        Transfer matrix by Xsuite finite differences.

        Parameters
        ----------
        start: str, optional
            First element; defaults to the line start.
        end: str, optional
            Last element; defaults to the line end.
        element_by_element: bool, optional
            Return each element's matrix rather than the whole line's.

        Returns
        -------
        np.ndarray
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
        Transverse matching with Ocelot's ``match``, setting the variables' strengths in place.

        ``variables`` are quadrupole, sextupole or octupole names; ``targets`` maps
        element names (or ``global``) to Ocelot Twiss constraints; ``max_iterations``
        defaults to 10000:

        .. code-block:: yaml

            files:
              line:
                match:
                  variables: [Q1, Q2, S1]
                  targets:
                    SCR1: {beta_x: 10.0, alpha_x: 0.0}
                    SCR2: {beta_y: 12.0, alpha_y: 0.0}
                    SCR3: {beta_x: {mode: greaterthan, value: 8.0}}

        Parameters
        ----------
        params: Dict
            The ``match`` block.

        Raises
        ------
        ValueError
            If ``variables`` or ``targets`` is missing, or no variable is a usable magnet.
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
            E=beam.centroids.mean_cp.val * 1e-9
        )
        matchelems = [e for e in lat.lat_obj.sequence if e.id in params["targets"]]
        constr = {e: params["targets"][e.id] for e in matchelems}
        if "global" in params["targets"]:
            constr.update({"global": params["targets"]["global"]})
        varelems = [
            [e for e in lat.lat_obj.sequence if e.id == p][0]
            for p in params["variables"]
            if p in self.elements and type(self.elements[p]) in [Quadrupole, Sextupole, Octupole]
        ]
        try:
            max_iter = params["max_iterations"]
        except KeyError:
            max_iter = 10000
        if len(varelems) == 0:
            raise ValueError("No variables added; make sure quadrupoles/sextupoles/octupoles are used for matching")
        res = match_oce(lat=lat.lat_obj, constr=constr, vars=varelems, tw=twsobj, verbose=False, max_iter=max_iter)
        print("Matching results:")
        for v, r in zip(varelems, res):
            elem = self.elementObjects[v.id]
            magnetic_order = elem.magnetic.order
            magnetic_length = elem.magnetic.length
            setattr(elem, f"k{magnetic_order}l", r * magnetic_length)
            print("\t", elem.name, f"k{magnetic_order}l =", r * magnetic_length)

class global_error(frameworkObject):
    """A global error element."""

class frameworkCommand(frameworkObject):
    """A command written into a simulation code's setup file."""

    def model_post_init(self, __context):
        if self.objecttype not in commandkeywords:
            raise NameError(f"Command '{self.objecttype}' does not exist")
        super().model_post_init(__context)

    def write_Elegant(self) -> str:
        """The ``&command ... &end`` block for ELEGANT."""
        string = "&" + self.objecttype + "\n"
        for key in commandkeywords[self.objecttype]:
            if (
                key.lower() in self.allowedkeywords
                and key != "objectname"
                and key != "objecttype"
                and hasattr(self, key)
                and getattr(self, key.lower()) is not None
            ):
                string += "\t" + key + " = " + str(getattr(self, key.lower())) + "\n"
        string += "&end\n"
        return string

    def write_Genesis(self) -> str:
        """The ``&command ... &end`` block for Genesis. TODO: deprecated?"""
        string = "&" + self.objecttype + "\n"
        for key in commandkeywords_genesis[self.objecttype]:
            if (
                key.lower() in self.allowedkeywords
                and key != "objectname"
                and key != "objecttype"
                and hasattr(self, key)
            ):
                val = getattr(self, key.lower())
                val = int(val) if isinstance(val, bool) else val
                if val is not None:
                    string += "\t" + key + " = " + str(val) + "\n"
        string += "&end\n"
        return string


class frameworkGroup:
    """A named group of elements acted on together."""

    def __init__(self, name, framework, type, elements, **kwargs):
        super().__init__()
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
        A group parameter (e.g. a chicane's angle), else the first member's.

        Parameters
        ----------
        p: str

        Returns
        -------
        Any
        """
        try:
            return getattr(self, p)
        except Exception:
            if self.elements[0] in self.allGroupObjects:
                return getattr(self.allGroupObjects[self.elements[0]], p)
            return getattr(self.allElementObjects[self.elements[0]], p)

    def change_Parameter(self, p: Any, v: Any) -> None:
        """
        Set a group parameter, else set it on every member.

        Parameters
        ----------
        p: str
        v: Any
        """
        try:
            getattr(self, p)
            setattr(self, p, v)
            if p == "angle":
                self.set_angle(v)
        except Exception:
            for e in self.elements:
                setattr(self.allElementObjects[e], p, v)

    def __repr__(self):
        return str([self.allElementObjects[e].name for e in self.elements])

    def __str__(self):
        return str([self.allElementObjects[e].name for e in self.elements])

    def __getitem__(self, key):
        return self.get_Parameter(key)

    def __setitem__(self, key, value):
        return self.change_Parameter(key, value)


class element_group(frameworkGroup):
    """A plain :class:`frameworkGroup` of elements."""

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)

    def __str__(self):
        return str([self.allElementObjects[e] for e in self.elements])


class r56_group(frameworkGroup):
    """A group whose members' settings follow a total R56 through ``ratios`` expressions."""

    def __init__(self, name, elementObjects, type, elements, ratios, keys, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = ratios
        self.keys = keys
        self._r56 = None

    def __str__(self):
        return str(dict(zip(self.elements, self.keys)))

    def get_Parameter(self, p: str) -> Any:
        """As :meth:`frameworkGroup.get_Parameter`, plus ``r56``."""
        if str(p) == "r56":
            return self.r56
        else:
            return super().get_Parameter(p)

    @property
    def r56(self) -> float:
        """The group's R56; setting it updates each member from ``ratios``."""
        return self._r56

    @r56.setter
    def r56(self, r56: float) -> None:
        """Set the R56 and update each member."""
        self._r56 = r56
        data = {"r56": self._r56}
        parser = MathParser(data)
        values = [parser.parse(e) for e in self.ratios]
        for e, k, v in zip(self.elements, self.keys, values):
            self.updateElements(e, k, v)

    def updateElements(self, element: str | list | tuple, key: str, value: Any) -> None:
        """
        Set ``key`` on one or more elements or groups.

        Parameters
        ----------
        element: str, list or tuple
        key: str
        value: Any
        """
        if isinstance(element, (list, tuple)):
            [self.updateElements(e, key, value) for e in element]
        else:
            if element in self.allElementObjects:
                setattr(self.allElementObjects[element], key, value)
            if element in self.allGroupObjects:
                self.allGroupObjects[element].change_Parameter(key, value)


class chicane(frameworkGroup):
    """A 4-dipole chicane."""

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (1, -1, -1, 1)
        self.elementObjects = [self.allElementObjects[e] for e in self.elements]

    def update(self, **kwargs) -> None:
        """
        Update any of ``dipoleangle``, ``width`` and ``gap`` on every dipole; other keys are ignored.
        """
        if "dipoleangle" in kwargs:
            self.set_angle(kwargs["dipoleangle"])
        if "width" in kwargs:
            self.change_Parameter("width", kwargs["width"])
        if "gap" in kwargs:
            self.change_Parameter("gap", kwargs["gap"])

    @property
    def drift_d1_to_d2(self) -> float:
        """Straight-line distance from dipole 1's exit to dipole 2's entrance."""
        e1 = self.elementObjects[0]
        e2 = self.elementObjects[1]
        return np.sqrt(np.sum([(getattr(e2.start, d) - getattr(e1.end, d)) ** 2 for d in ["x", "y", "z"]]))

    @property
    def r56(self) -> float:
        """R56 of the chicane, ``2 * angle**2 * (drift_d1_to_d2 + 2/3 * dipole length)``."""
        e1 = self.elementObjects[0]
        ld = self.drift_d1_to_d2
        return 2 * self.angle ** 2 * (ld + 2 * e1.magnetic.length / 3)

    @property
    def delay(self) -> float:
        """Delay (longitudinal slippage) of the chicane, ``2 * r56``."""
        return 2 * self.r56

    @property
    def angle(self) -> float:
        """The first dipole's bending angle; setting it calls :meth:`set_angle`."""
        obj = [self.allElementObjects[e] for e in self.elements]
        return float(obj[0].magnetic.KnL(0))

    @angle.setter
    def angle(self, theta: float) -> None:
        """Set the bending angle; see :meth:`set_angle`."""
        self.set_angle(theta)

    def set_angle(self, a: float) -> None:
        """
        Set the chicane bending angle, re-laying out the dipoles and everything between them.

        Dipoles keep their z extent (:meth:`_z_extent`) and lengths become the arc.

        Parameters
        ----------
        a: float
            Bending angle of the first dipole, in radians; the others follow ``ratios``.
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
        ``(rotation_matrix, theta, entrance)`` of the first dipole: the frame :meth:`set_angle` lays out in.

        Cached on first use, because :meth:`set_angle` moves the entrance it is read from.
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
        The z a dipole spans, held fixed as the angle changes: its design (zero-angle) length.

        Cached on first use because :meth:`set_angle` overwrites the length with the arc.
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
    """An s-type chicane: :class:`chicane` with bending ratios ``(-1, 2, -2, 1)``."""

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (-1, 2, -2, 1)


class getGrids:
    """Space-charge grid size per dimension: the power of 2 nearest the cube root of the particle count."""

    def __init__(self):
        self.powersof8 = np.asarray([2**j for j in range(1, 20)])

    def getGridSizes(self, x: PositiveInt) -> int:
        """
        Space-charge grid size for ``x`` particles, at least 4.

        Parameters
        ----------
        x: PositiveInt
            Number of particles.

        Returns
        -------
        int
        """
        self.x = abs(x)
        self.cuberoot = int(round(self.x ** (1.0 / 3)))
        return max([4, self.find_nearest(self.powersof8, self.cuberoot)])

    def find_nearest(self, array: np.ndarray | list, value: int) -> int:
        """
        The entry of ``array`` nearest ``value``.

        Parameters
        ----------
        array: np.ndarray or list
        value: int

        Returns
        -------
        int
        """
        self.array = array
        self.value = value
        self.idx = (np.abs(self.array - self.value)).argmin()
        return self.array[self.idx]
