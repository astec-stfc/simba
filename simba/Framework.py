"""
SIMBA Framework: load lattice settings and track a particle distribution through them.

Classes:
    - :class:`~simba.Framework.Framework`: load, modify and track through lattice settings.
    - :class:`~simba.Framework.frameworkDirectory`: load the Beam and Twiss files of a finished run.
"""

import gc
import os
import pickle
import yaml
import inspect
from typing import Any, Dict, Literal
from pprint import pprint
import numpy as np
from copy import deepcopy
from laura import LAURA
from laura.models.element import PhysicalBaseElement
from laura.models.element_list import flatten_occurrence, split_occurrence
from laura.exporters.yaml_exporter import export_machine, export_elements

from .Modules import Beams as rbf
from .Modules import Twiss as rtf
from .Modules import Wavefronts as rwf
from .Modules import constants
from .Codes import Executables as exes
from .Codes.Generators import (
    ASTRAGenerator,
    GPTGenerator,
    frameworkGenerator,
)
from .Framework_objects import runSetup
from . import Framework_lattices as frameworkLattices
from . import Framework_elements as frameworkElements
from .Framework_Settings import FrameworkSettings
from .FrameworkHelperFunctions import (
    clean_directory,
    convert_numpy_types,
    compare_multiple_models,
    set_deep_attr,
    flatten_changes_dict,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
)
from warnings import warn

try:
    import MasterLattice  # type: ignore

    if MasterLattice.__file__ is not None:
        MasterLatticeLocation = os.path.dirname(MasterLattice.__file__) + "/"
    else:
        MasterLatticeLocation = None
except ImportError:
    MasterLatticeLocation = None
try:
    import SimCodes  # type: ignore

    SimCodesLocation = os.path.dirname(SimCodes.__file__) + "/"
except ImportError:
    SimCodesLocation = None
try:
    import simba.Modules.plotting.plotting as groupplot

    use_matplotlib = True
except ImportError as e:
    print("Import error - plotting disabled. Missing package:", e)
    use_matplotlib = False
from tqdm import tqdm

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(iter(list(data.items())))


def dict_constructor(loader, node):
    return dict(loader.construct_pairs(node))

def numpy_array_representer(dumper, data):
    return dumper.represent_list(data.tolist())

def numpy_scalar_representer(dumper, data):
    if np.issubdtype(type(data), np.floating):
        return dumper.represent_float(float(data))
    elif np.issubdtype(type(data), np.integer):
        return dumper.represent_int(int(data))
    else:
        return dumper.represent_str(str(data))

yaml.SafeDumper.add_representer(np.ndarray, numpy_array_representer)
yaml.SafeDumper.add_representer(np.generic, numpy_scalar_representer)

class NumpySafeDumper(yaml.SafeDumper):
    pass

NumpySafeDumper.add_representer(np.ndarray, numpy_array_representer)
NumpySafeDumper.add_representer(np.generic, numpy_scalar_representer)
NumpySafeDumper.add_representer(np.float64, numpy_scalar_representer)
NumpySafeDumper.add_representer(np.float32, numpy_scalar_representer)
NumpySafeDumper.add_representer(np.int64, numpy_scalar_representer)
NumpySafeDumper.add_representer(np.int32, numpy_scalar_representer)
NumpySafeDumper.add_representer(np.bool_, numpy_scalar_representer)

yaml.add_representer(dict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)

latticeClasses = [
    obj[1] for obj in inspect.getmembers(frameworkLattices) if inspect.isclass(obj[1])
]

with open(os.path.dirname(os.path.abspath(__file__)) + "/hosts.yaml") as infile:
    hosts = yaml.safe_load(infile)

disallowed = [
    "allowedkeywords",
    "conversion_rules",
    "objectdefaults",
    "global_parameters",
    "objectname",
    "subelement",
    "beam",
]

disallowed_changes = [
    "allowedkeywords",
    "conversion_rules",
    "objectdefaults",
    "global_parameters",
    "beam",
    "field_definition",
    "wakefield_definition",
    "generator_keywords",
    "allowedKeyWords",
    "executables",
]


supported_codes = [code.split("Lattice")[0] for code in dir(frameworkLattices) if "lattice" in code.lower()]

class Framework(BaseModel):
    """
    The main class for tracking a particle distribution through a lattice.

    Settings files of one or more `LAURA <https://github.com/astec-stfc/laura/>`_ YAML files
    become :class:`~simba.Framework_objects.frameworkLattice` objects of
    :class:`~laura.models.element.Element` objects. Each lattice is tracked by its code in turn;
    output beams are converted to OpenPMD HDF5, and Twiss and beam summaries are written afterwards.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    directory: str
    """Directory for simulation files."""

    master_lattice: str | None = None
    """Location of the master lattice files; found automatically if the package is installed."""

    simcodes: str | None = None
    """Location of the simulation codes; found automatically if the package is installed."""

    overwrite: bool | None = None
    """Whether existing files are overwritten. #TODO deprecated?"""

    runname: str = "CLARA_240"
    """Name of the run. #TODO deprecated?"""

    clean: bool = False
    """Remove all files already in :attr:`directory`."""

    verbose: bool = True
    """Print status updates during tracking."""

    sddsindex: int = 0
    """Index for SDDS files."""

    delete_output_files: bool = False
    """Delete code output files after tracking."""

    conversion_workers: int = 1
    """Processes for converting screen outputs to openPMD (elegant only, for now); 1 converts them one by one."""

    global_parameters: Dict = {}
    """Global parameters shared with all lattices and elements."""

    elementObjects: Dict = {}
    """All :class:`~laura.models.element.Element` objects, by name."""

    latticeObjects: Dict = {}
    """All :class:`~simba.Framework_objects.frameworkLattice` objects, by name."""

    commandObjects: Dict = {}
    """All :class:`~simba.Framework_objects.frameworkCommand` objects."""

    groupObjects: Dict = {}
    """All :class:`~simba.Framework_objects.frameworkGroup` objects, by name."""

    fileSettings: Dict = {}
    """File settings."""

    globalSettings: Dict = {}
    """Global settings."""

    generatorSettings: Dict = {}
    """Generator settings."""

    _original_elements: Dict | bytes = PrivateAttr(default_factory=dict)
    """Backing store for :attr:`original_elementObjects`; pickled until first read."""

    progress: int | float = 0
    """Current progress of tracking."""

    tracking: bool = False
    """Whether the Framework is tracking."""

    generator: frameworkGenerator | None = None
    """The :class:`~simba.Codes.Generators.Generators.frameworkGenerator`."""

    settings: FrameworkSettings | None = None
    """Settings for the lattice."""

    settingsFilename: str | None = None
    """Lattice settings filename."""

    machine: LAURA = None
    """LAURA model of the lattice."""

    generator_defaults: str | None = None
    """Defaults file for the :class:`~simba.Codes.Generators.Generators.frameworkGenerator`."""

    generator_keywords: Dict = {}
    """Generator keywords loaded from :attr:`generator_defaults` in ``master_lattice``/Generators."""

    username: str = ""
    """Username for remote execution."""

    password: str = ""
    """Password for remote execution."""

    eager_mode: bool = False
    """Bypass lazy loading for LAURA."""

    container_runtime: Literal["docker", "apptainer"] | None = None
    """Container runtime for the codes; if None, executables are looked for locally."""

    executables: exes.Executables | None = None
    """Commands for running the simulation codes."""

    executables_ready: bool = False
    """Whether :attr:`executables` are ready."""

    def model_post_init(self, __context):
        gptlicense = os.environ.get("GPTLICENSE", "")
        astra_use_wsl = os.environ.get("WSL_ASTRA", 1)
        self.global_parameters = {
            "beam": rbf.beam(sddsindex=self.sddsindex),
            "GPTLICENSE": gptlicense,
            "delete_tracking_files": self.delete_output_files,
            "conversion_workers": self.conversion_workers,
            "astra_use_wsl": astra_use_wsl,
            "master_lattice": self.master_lattice,
            "container_runtime": self.container_runtime,
        }
        if self.simcodes is None and self.container_runtime not in ["docker", "apptainer"] and self.verbose:
            warn("Either `simcodes` location or `container_runtime` must be either 'docker' or 'apptainer' to set up executables;"
                 "Cannot run simulations without either of these - please set one or the other")
        self.setSubDirectory(self.directory)
        self.setMasterLatticeLocation(self.master_lattice)
        self.setSimCodesLocation(self.simcodes)
        self.setupGeneratorDefaults(self.generator_defaults)

        self.executables = self.prepare_executables(location=self.container_runtime)

        # object encoding settings for simulations with multiple runs
        self.runSetup = runSetup()

    def __repr__(self) -> repr:
        return repr(
            {
                "master_lattice": self.global_parameters[
                    "master_lattice"
                ],
                "subdirectory": self.subdirectory,
                "settingsFilename": self.settingsFilename,
            }
        )

    def setupLAURA(self) -> None:
        """Set up the LAURA :attr:`machine`."""
        try:
            self.machine = LAURA(layout=self.layout, section=self.section, element_list=self.element_list, eager_mode=self.eager_mode)
        except Exception:
            self.machine = LAURA(layout=self.layout, section=self.section, element_list=self.element_list)

    def prepare_executables(
            self,
            location: str=None,
            ncpu: int = 1,
    ):
        executables = exes.Executables(self.global_parameters)
        executables.define_astra_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_elegant_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_csrtrack_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_gpt_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_genesis_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_opal_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_tao_command(
            override_location=location,
            ncpu=ncpu,
        )
        executables.define_ASTRAgenerator_command(override_location=location)
        self.executables_ready = True
        return executables

    def clear(self) -> None:
        """Clear :attr:`elementObjects`, :attr:`latticeObjects`, :attr:`commandObjects` and :attr:`groupObjects`."""
        self.elementObjects = {}
        self.latticeObjects = {}
        self.commandObjects = {}
        self.groupObjects = {}

    def change_subdirectory(self, *args, **kwargs) -> None:
        """Alias of :meth:`setSubDirectory`."""
        self.setSubDirectory(*args, **kwargs)

    def setSubDirectory(self, direc: str) -> None:
        """
        Set the directory (and ``master_subdir``) for lattice and beam files, emptying it if :attr:`clean`.

        Parameters
        ----------
        direc: str
            Output directory; created if missing.
        """
        self.subdirectory = os.path.abspath(direc)
        self.global_parameters["master_subdir"] = self.subdirectory
        if not os.path.exists(self.subdirectory):
            os.makedirs(self.subdirectory, exist_ok=True)
        else:
            if self.clean is True:
                clean_directory(self.subdirectory)
        if self.overwrite is None:
            self.overwrite = True
        self.updateGlobalParameters()

    def updateGlobalParameters(self):
        for object_lists in [self.latticeObjects, self.groupObjects]:
            for obj in object_lists.values():
                obj.global_parameters = self.global_parameters
                obj.globalSettings = self.globalSettings

    def _resolve_package_location(
            self,
            explicit: str | None,
            cached: str | None,
            candidates: list[str],
            label: str,
            kwarg_name: str,
    ) -> str | None:
        """
        Resolve a package directory: ``explicit``, else ``cached``, else the first existing ``candidates`` entry.

        Parameters
        ----------
        explicit: str, optional
            Path given by the caller; used as-is.
        cached: str, optional
            Location from an earlier ``Framework`` or an installed package.
        candidates: list[str]
            Paths relative to this file, in priority order.
        label: str
            Package name for verbose messages.
        kwarg_name: str
            Keyword argument to suggest if nothing is found.

        Returns
        -------
        str or None
            Absolute location ending in a slash, or None if not found.
        """
        if explicit is not None:
            return os.path.join(os.path.abspath(explicit), "./")
        if cached is not None:
            resolved = cached.replace("\\", "/")
            if self.verbose:
                print(f"Found {label} Package =", resolved)
            return resolved
        here = os.path.dirname(os.path.abspath(__file__))
        for candidate in candidates:
            path = os.path.abspath(os.path.join(here, candidate)) + "/"
            if os.path.isdir(path):
                resolved = path.replace("\\", "/")
                if self.verbose:
                    print(f"Found {label} Directory at '{candidate}' =", resolved)
                return resolved
        if self.verbose:
            print(f"{label} not available - specify using {kwarg_name}=<location>")
        return None

    def setMasterLatticeLocation(self, master_lattice: str | None = None) -> None:
        """
        Set the MasterLattice location, as ``master_lattice`` in :attr:`global_parameters`.

        Parameters
        ----------
        master_lattice: str
            Path to the MasterLattice folder.
        """
        global MasterLatticeLocation
        location = self._resolve_package_location(
            master_lattice,
            MasterLatticeLocation,
            ["../../MasterLattice/MasterLattice", "../MasterLattice/MasterLattice", "../MasterLattice"],
            "MasterLattice",
            "master_lattice",
        )
        self.global_parameters["master_lattice"] = "." if location is None else location
        MasterLatticeLocation = self.global_parameters["master_lattice"]
        self.updateGlobalParameters()

    def setSimCodesLocation(self, simcodes: str | None = None) -> None:
        """
        Set the :ref:`SimCodes` location, as ``simcodes_location`` in :attr:`global_parameters`.

        Parameters
        ----------
        simcodes: str
            Path to the SimCodes folder.
        """
        global SimCodesLocation
        self.global_parameters["simcodes_location"] = self._resolve_package_location(
            simcodes,
            SimCodesLocation,
            ["../../SimCodes/SimCodes", "../SimCodes/SimCodes", "../SimCodes"],
            "SimCodes",
            "simcodes",
        )
        SimCodesLocation = self.global_parameters["simcodes_location"]
        self.updateGlobalParameters()

    def setupGeneratorDefaults(self, generator_defaults: str | None):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/Codes/Generators/keywords.yaml") as infile:
            self.generator_keywords = {"keywords": yaml.safe_load(infile)}

        if not generator_defaults:
            return self.generator_keywords
        if os.path.isfile(self.global_parameters["master_lattice"] + f"Generators/{generator_defaults}"):
            defaults = self.global_parameters["master_lattice"] + f"Generators/{generator_defaults}"
        elif os.path.isfile(generator_defaults):
            defaults = generator_defaults
        else:
            raise FileNotFoundError(f"Could not find generator file {generator_defaults}")
        with open(defaults) as infile:
            fi = yaml.safe_load(infile)
            defaults = fi["defaults"]
            self.generator_keywords.update({"defaults": defaults})
            for k, v in fi.items():
                if k != "defaults":
                    self.generator_keywords.update({k: defaults | (v or {})})

    def loadSettings(
        self,
        filename: str | None = None,
        settings: FrameworkSettings | None = None,
    ) -> None:
        """
        Load lattice settings (lines, their settings, YAML files and global parameters).

        Parameters
        ----------
        filename: str or None
            Settings (.def) file, looked for as given, then in the subdirectory, then in the master lattice.
        settings: FrameworkSettings or None
            Settings to use if no ``filename`` is given.
        """
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            self._load_settings(filename, settings)
        finally:
            if gc_was_enabled:
                gc.enable()

    def _load_settings(
        self, filename: str | None, settings: FrameworkSettings | None
    ) -> None:
        """Body of :meth:`loadSettings`, run with GC off."""
        if isinstance(filename, str):
            self.settingsFilename = filename
            self.settings = FrameworkSettings()
            if os.path.isfile(filename):
                self.settings.loadSettings(filename)
            elif os.path.isfile(os.path.join(self.subdirectory, filename)):
                self.settings.loadSettings(os.path.join(self.subdirectory, filename))
            elif os.path.isfile(os.path.join(self.global_parameters["master_lattice"], filename)):
                self.settings.loadSettings(
                    os.path.join(self.global_parameters["master_lattice"], filename)
                )
            else:
                raise FileNotFoundError(f"Could not find settings file with name {filename}")
        elif isinstance(settings, FrameworkSettings):
            self.settingsFilename = settings.settingsFilename
            self.settings = settings
        else:
            raise ValueError("Could not instantiate lattice settings")

        self.globalSettings = self.settings["global"]
        if "generator" in self.settings and len(self.settings["generator"]) > 0:
            self.generatorSettings = self.settings["generator"]
            self.add_Generator(**self.generatorSettings)
        self.fileSettings = self.settings.get("files", {})
        groups = (
            self.settings["groups"]
            if "groups" in self.settings and self.settings["groups"] is not None
            else {}
        )
        changes = (
            self.settings["changes"]
            if "changes" in self.settings and self.settings["changes"] is not None
            else {}
        )
        if self.settings.layout:
            try:
                self.machine = LAURA(
                    layout=self.settings["layout"],
                    section=self.settings["section"],
                    functional_definitions=self.settings["functional_definitions"],
                    resolve_functional=self.settings["resolve_functional"],
                    element_list=self.settings["element_list"],
                    master_lattice=self.global_parameters["master_lattice"],
                    exclude_keys=["controls", "electrical", "manufacturer", "reference"],
                    eager_mode=self.eager_mode,
                )
            except Exception:
                self.machine = LAURA(
                    layout=self.settings["layout"],
                    section=self.settings["section"],
                    functional_definitions=self.settings["functional_definitions"],
                    resolve_functional=self.settings["resolve_functional"],
                    element_list=self.settings["element_list"],
                    master_lattice=self.global_parameters["master_lattice"],
                    exclude_keys=["controls", "electrical", "manufacturer", "reference"],
                )

            for k in list(self.machine.elements.keys()):
                _ = self.machine.elements[k]

            self.elementObjects = dict(self.machine.elements)

            for name, group in list(groups.items()):
                if "type" in group:
                    group_object = getattr(frameworkElements, group["type"])(
                        name, self, global_parameters=self.global_parameters, **group
                    )
                    self.groupObjects[name] = group_object

            for name, lattice in list(self.fileSettings.items()):
                self.read_Lattice(name, lattice)

            self.apply_changes(changes)
            snapshot = {**self.elementObjects, "generator": self.generator}
            try:
                self._original_elements = pickle.dumps(
                    snapshot, protocol=pickle.HIGHEST_PROTOCOL
                )
            except (pickle.PicklingError, TypeError, AttributeError):
                self._original_elements = deepcopy(snapshot)
            self.updateGlobalParameters()

    @property
    def original_elementObjects(self) -> Dict:
        """All :class:`~laura.models.element.Element` objects as loaded, before changes."""
        if isinstance(self._original_elements, bytes):
            self._original_elements = pickle.loads(self._original_elements)
        return self._original_elements

    @original_elementObjects.setter
    def original_elementObjects(self, value: Dict) -> None:
        self._original_elements = value

    def save_settings(
        self,
        filename: str | None = None,
        directory: str = ".",
        elements: dict | None = None,
    ) -> None:
        """
        Save lattice settings to a file.

        Parameters
        ----------
        filename: str or None
            Defaults to ``settings.def``.
        directory: str
            Output directory.
        elements: dict or None
            Replaces the ``elements`` entry of the saved settings.
        """
        if filename is None:
            filename = "settings.def"
        settings = self.settings.copy()
        if elements is not None:
            settings["elements"] = elements
        settings = convert_numpy_types(settings)
        with open(os.path.join(directory, filename), "w") as yaml_file:
            yaml.default_flow_style = True
            yaml.dump(settings, yaml_file, sort_keys=False, Dumper=NumpySafeDumper)

    def read_Lattice(self, name: str, lattice: dict) -> None:
        """
        Create a ``<code>Lattice`` (see :class:`~simba.Framework_objects.frameworkLattice`) in :attr:`latticeObjects`.

        Parameters
        ----------
        name: str
            Name of the lattice line.
        lattice: dict
            Settings for the lattice line.
        """
        if "code" not in lattice:
            raise KeyError(f"code must be provided for {lattice}")
        code = lattice["code"]
        if code.lower() not in supported_codes:
            raise NotImplementedError(f"code {code} is not supported")
        self.latticeObjects[name] = getattr(
            frameworkLattices, code.lower() + "Lattice"
        )(
            name=name,
            objectname=name,
            objecttype=code.lower() + "Lattice",
            file_block=lattice,
            elementObjects=self.elementObjects,
            groupObjects=self.groupObjects,
            runSettings=self.runSetup,
            settings=self.settings,
            executables=self.executables,
            global_parameters=self.global_parameters,
            machine=self.machine,
            globalSettings=self.globalSettings,
        )
        if "remote" in lattice:
            self.setup_remote_execution(
                lattice=name,
                code=code.lower(),
                **lattice["remote"],
            )

    def detect_changes(
        self,
        elementtype: str | None = None,
        elements: list | None = None,
    ) -> dict:
        """
        Detect changes from the lattice as loaded.

        Parameters
        ----------
        elementtype: str or None
            Element type to check; all if None.
        elements: list or None
            Elements to check; all if None.

        Returns
        -------
        dict
            Changed parameters, by element name.
        """
        changedict = {}
        if elementtype is not None:
            changeelements = self.getElementType(elementtype, "name")
        elif elements is not None:
            changeelements = elements
        else:
            changeelements = ["generator"] + list(self.elementObjects.keys())
        if (
            len(changeelements) > 0
            and isinstance(changeelements[0], (list, tuple, dict))
            and len(changeelements[0]) > 1
        ):
            for ek in changeelements:
                new = None
                e, k = ek[:2]
                if e in self.elementObjects:
                    new = self.elementObjects[e]
                elif e in self.groupObjects:
                    new = self.groupObjects[e]
                if new is not None:
                    if e not in changedict:
                        changedict[e] = {}
                    changedict[e][k] = {
                        "new": convert_numpy_types(getattr(new, k)),
                        "old": convert_numpy_types(
                            getattr(self.original_elementObjects[e], k)
                        ),
                    }
        else:
            for e in changeelements:
                element = None
                if e in self.elementObjects:
                    element = self.elementObjects[e]
                elif e == "generator":
                    element = self.generator
                    e = "generator"
                if isinstance(element, PhysicalBaseElement):
                    orig = self.original_elementObjects[e]
                    pairs = [(orig, element)]

                    changes = compare_multiple_models(pairs)
                    if changes[element.name]:
                        changedict.update(**changes)
                elif isinstance(element, frameworkGenerator):
                    cond = False
                    new = element
                    orig = self.original_elementObjects[e]
                    kval = [k for k in new.model_dump() if k not in disallowed_changes]
                    new_model_fields = {
                        k: v
                        for k, v in new.model_dump().items()
                        if k not in disallowed_changes
                    }
                    orig_model_fields = {
                        k: v
                        for k, v in orig.model_dump().items()
                        if k not in disallowed_changes
                    }
                    for k in kval:
                        if k not in orig_model_fields or new_model_fields[k] != orig_model_fields[k]:
                            cond = True
                    if cond:
                        orig = self.original_elementObjects[e]
                        new = element
                        changedict[e] = {
                            k: convert_numpy_types(getattr(new, k))
                            for k in kval
                            if k not in orig_model_fields
                               or new_model_fields[k] != orig_model_fields[k]
                        }
        return changedict

    def save_changes_file(
        self,
        filename: str | None = None,
        typ: str | None = None,
        elements: dict | None = None,
        dictionary: bool = False,
    ) -> dict | None:
        """
        Save, or return, the changes from the lattice as loaded; see :meth:`detect_changes`.

        Parameters
        ----------
        filename: str or None
            Defaults to ``<settings file stem>_changes.yaml``.
        typ: str or None
            Element type to check; all if None.
        elements: dict or None
            Elements to check; all if None.
        dictionary: bool
            Return the changes instead of saving them.

        Returns
        -------
        dict or None
            The changes if ``dictionary``, otherwise None.
        """
        changedict = self.detect_changes(elementtype=typ, elements=elements)
        if dictionary:
            return changedict
        if filename is None:
            if self.settingsFilename is not None:
                pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
                filename = pre + "_changes.yaml"
            else:
                raise ValueError("settingsFilename not set; cannot determine changes filename")
        with open(filename, "w") as yaml_file:
            yaml.default_flow_style = True
            yaml.dump(changedict, yaml_file, Dumper=NumpySafeDumper)

    def save_lattice(
        self,
        lattice: str | None = None,
        filename: str | None = None,
        directory: str = ".",
    ) -> dict | None:
        """
        Export the machine, or one lattice line's elements, to LAURA YAML.

        Parameters
        ----------
        lattice: str or None
            Lattice line to export; if None, the whole machine.
        filename: str or None
            Its stem names the output, with ``_<lattice>_lattice`` appended for a line;
            defaults to the settings filename.
        directory: str
            Output directory.
        """
        if filename is None:
            if self.settingsFilename is not None:
                pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
            else:
                raise ValueError("settingsFilename not set; cannot determine lattice filename")
        else:
            pre, ext = os.path.splitext(os.path.basename(filename))
        if lattice is None:
            filename = pre
            export_machine(os.path.join(directory, filename), self.machine)
        else:
            if self.latticeObjects[lattice].elements is None:
                warn(f"No elements found in {lattice} to save")
                return
            filename = pre + "_" + lattice + "_lattice"
            export_elements(
                os.path.join(directory, filename),
                list(self.latticeObjects[lattice].section.elementObjects.values()),
            )

    def load_changes_file(
        self,
        filename: str | tuple | list | None = None,
        apply: bool = True,
        verbose: bool = False,
    ) -> dict | list | None:
        """
        Load a changes file and, by default, apply it; see :meth:`apply_changes`.

        Parameters
        ----------
        filename: str or list or tuple or None
            Changes file(s); defaults to ``<settings file stem>_changes.yaml``.
        apply: bool
            Apply the changes.
        verbose: bool
            Print the changes applied.

        Returns
        -------
        dict or list or None
            None if applied, otherwise the changes; a list of results for a list of files.
        """
        if isinstance(filename, (tuple, list)):
            return [self.load_changes_file(c, apply, verbose) for c in filename]
        else:
            if filename is None:
                pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
                filename = pre + "_changes.yaml"
            with open(filename) as infile:
                changes = dict(yaml.safe_load(infile))
            if apply:
                self.apply_changes(changes, verbose=verbose)
                return None
            return changes

    def apply_changes(self, changes: dict, verbose: bool = False) -> None:
        """
        Apply changes to the current lattice.

        Parameters
        ----------
        changes: dict
            Parameters and values to change, by element or group name.
        verbose: bool
            Print each change.
        """
        for e, d in list(changes.items()):
            if e in self.elementObjects or e == "generator":
                flat = flatten_changes_dict(d)
                for param in flat:
                    try:
                        self.modifyElement(e, param[0], param[1])
                        if verbose:
                            print("modifying ", e, "[", param[0], "]", " = ", param[1])
                    except AttributeError as ex:
                        print(f"### ERROR modifying {e} [{param[0]}] = {param[1]}: {ex}")
            if e in self.groupObjects:
                for k, v in list(d.items()):
                    self.groupObjects[e].change_Parameter(k, v)
                    if verbose:
                        print("modifying ", e, "[", k, "]", " = ", v)

    def check_lattice(self, decimals: int = 4) -> bool:
        """
        Check the lattice for positioning errors, printing any found.

        Parameters
        ----------
        decimals: int
            Tolerance, as ``10 ** -decimals``.

        Returns
        -------
        bool
            True if no errors are detected.
        """
        noerror = True
        for elem in self.elementObjects.values():
            if not isinstance(elem, PhysicalBaseElement):
                continue
            physical = elem.physical
            middle = np.array(physical.middle.array)
            end = np.array(physical.end.array)
            angle = physical._physical_angle

            cend = middle + physical.offset_from_middle("end")
            diff = cend - end
            if not np.allclose(diff, 0, atol=10 ** (-decimals)):
                noerror = False
                print(f"check_lattice error: {elem.name}")
                print(f"  Middle: {middle}")
                print(f"  Calculated end: {cend}")
                print(f"  Actual end: {end}")
                print(f"  Difference: {diff}")
                print(f"  Physical angle: {angle}")
                print(
                    f"  Global rotation: [{physical.global_rotation.phi}, {physical.global_rotation.psi}, {physical.global_rotation.theta}]")

            magnet_angle = physical.magnet_angle
            if magnet_angle is not None and not np.isclose(
                    abs(angle), abs(magnet_angle), atol=10 ** (-decimals)):
                noerror = False
                print(f"check_lattice error: {elem.name}")
                print(f"  Layout angle: {angle}")
                print(f"  Magnet angle: {magnet_angle}")

        return noerror

    def change_Lattice_Code(
            self,
            latticename: str,
            code: str,
            exclude: str | list | tuple | None = None,
            nowarn: bool = False,
    ) -> None:
        """
        Change the tracking code for a lattice line.

        Parameters
        ----------
        latticename: str
            Line in :attr:`latticeObjects`, a list of them, or ``All``.
        code: str
            Simulation code.
        exclude: str or list or tuple, optional
            Lines to leave unchanged.
        nowarn: bool
            Suppress the warning that remote execution is reset.
        """
        if latticename == "All":
            [self.change_Lattice_Code(lo, code, exclude, nowarn) for lo in self.latticeObjects]
        elif isinstance(latticename, (tuple, list)):
            [self.change_Lattice_Code(ln, code, exclude, nowarn) for ln in latticename]
        else:
            if latticename != "generator" and not (
                latticename == exclude
                or (isinstance(exclude, (list, tuple)) and latticename in exclude)
            ):
                if code.lower() not in supported_codes:
                    raise NotImplementedError(f"code {code} is not supported")
                currentLattice = self.latticeObjects[latticename]
                if currentLattice.remote_setup and not nowarn:
                    warn(f"Resetting lattice {latticename} to local tracking;"
                         f"call setup_remote_execution to run remotely")
                self.latticeObjects[latticename] = getattr(
                    frameworkLattices, code.lower() + "Lattice"
                )(
                    name=currentLattice.objectname,
                    file_block=currentLattice.file_block,
                    elementObjects=self.elementObjects,
                    groupObjects=self.groupObjects,
                    runSettings=self.runSetup,
                    settings=self.settings,
                    executables=self.executables,
                    global_parameters=self.global_parameters,
                    machine=self.machine,
                    globalSettings=self.globalSettings,
                )

    def getElement(
        self,
        element: str,
        param: str | None = None,
    ) -> dict | Any | PhysicalBaseElement:
        """
        Get an element, or one of its parameters.

        Parameters
        ----------
        element: str
            Element name.
        param: str or None
            Parameter to get; if None, the whole element.

        Returns
        -------
        dict or Any or :class:`~laura.models.element.Element`
            The parameter, or the element (also if it lacks ``param``), or ``{}`` if there is no such element.
        """
        if self.__getitem__(element) is not None:
            if param is not None:
                param = param.lower()
                try:
                    return getattr(self.__getitem__(element), param)
                except AttributeError:
                    warn(f"WARNING: Element {element} does not have parameter {param}; returning full element")
                    return self.__getitem__(element)
            else:
                return self.__getitem__(element)
        else:
            warn(f"WARNING: Element {element} does not exist")
            return {}

    def getElementType(
        self,
        typ: str | list | tuple,
        param: str | list | tuple | None = None,
    ) -> dict | list | Any:
        """
        Get all elements of a hardware type, or one parameter of each.

        Parameters
        ----------
        typ: list or str or tuple
            Type, or types.
        param: str or list or tuple or None
            Parameter(s) to get; if None, each element as a dict with its ``name``.

        Returns
        -------
        dict or list or Any
            A list per type; for several ``param``, a zip of their lists.
        """
        if isinstance(typ, (list, tuple)):
            return [self.getElementType(t, param=param) for t in typ]
        if isinstance(param, (list, tuple)):
            return zip(*[self.getElementType(typ, param=p) for p in param])
        return [
            (
                {"name": element, **self.elementObjects[element].model_dump()}
                if param is None
                else getattr(self.elementObjects[element], param)
            )
            for element in list(self.elementObjects.keys())
            if isinstance(self.elementObjects[element], PhysicalBaseElement) and self.elementObjects[element].hardware_type.lower() == typ.lower()
        ]

    def setElementType(
        self,
        typ: str,
        setting: str,
        values: Any,
    ) -> None:
        """
        Set a parameter on every element of a hardware type.

        Parameters
        ----------
        typ: str
            Hardware type.
        setting: str
            Parameter to set.
        values: Any
            One value per element.

        Raises
        ------
        ValueError
            If ``values`` and the elements differ in number.
        """
        elems = self.getElementType(typ)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                setattr(self[e["name"]], setting, v)
                if self[e["name"]].hardware_type.lower() == "dipole" and setting == "angle":
                    self[e["name"]].magnetic.multipoles.K0L.normal = v

        else:
            raise ValueError

    def _default_layout(self):
        """This machine's default beam path, or ``None`` if it has no machine."""
        machine = getattr(self, "machine", None)
        if machine is None:
            return None
        return machine.lattices.get(machine.default_path)

    def _warn_if_shared_across_passes(self, element_name: str) -> None:
        """Warn when a change reaches every pass through one device; per-pass values go in layout ``overrides``."""
        layout = self._default_layout()
        if layout is None or not getattr(layout, "is_multipass", False):
            return
        passes = [
            name
            for name in layout.elements
            if split_occurrence(name)[0] == element_name
        ]
        if len(passes) > 1:
            warn(
                f"'{element_name}' is entered {len(passes)} times by beam path "
                f"'{layout.name}' ({', '.join(passes)}) and is one device, so "
                "this change applies to every pass. Use a per-pass "
                "'overrides' entry in the layout if you meant only one."
            )

    def modifyElement(
        self,
        elementName: str,
        parameter: str | list | dict,
        value: Any = None,
    ) -> None:
        """
        Modify an element (or group) parameter.

        Parameters
        ----------
        elementName: str
            Element name.
        parameter: list or str or dict
            Parameter(s); a dict maps parameters to values when ``value`` is None.
        value: Any
            Value(s) to set.
        """
        if isinstance(parameter, dict) and value is None:
            for p, v in parameter.items():
                self.modifyElement(elementName, p, v)
        elif isinstance(parameter, list) and isinstance(value, list):
            if len(parameter) != len(value):
                warn("parameter and value must be of the same length")
            for p, v in zip(parameter, value):
                self.modifyElement(elementName, p, v)
        elif elementName in self.groupObjects:
            self.groupObjects[elementName].change_Parameter(parameter, value)
        elif elementName in self.elementObjects:
            self._warn_if_shared_across_passes(elementName)
            if "." in parameter:
                obj = self.elementObjects[elementName]
                set_deep_attr(obj, parameter, value)
            else:
                setattr(self.elementObjects[elementName], parameter, value)
        elif elementName == "generator":
            setattr(self.generator, parameter, value)
        else:
            warn("incorrect parameters passed to modifyElement")

    def modifyElements(
        self,
        elementNames: str | list,
        parameter: str | list | dict,
        value: Any = None,
    ) -> None:
        """
        Modify parameters on several elements; see :meth:`modifyElement`.

        Parameters
        ----------
        elementNames: str or list
            Element name(s), or ``all``.
        parameter: list or str or dict
            Parameter(s) to modify.
        value: Any
            Value(s) to set.
        """
        if isinstance(elementNames, str):
            if elementNames.lower() == "all":
                elementNames = self.elementObjects.keys()
            else:
                elementNames = [elementNames]
        for elem in elementNames:
            self.modifyElement(elem, parameter, value)

    def modifyElementType(
        self,
        elementType: str,
        parameter: str,
        value: Any,
    ) -> None:
        """
        Set one value on every element of a hardware type.

        Parameters
        ----------
        elementType: str
            Hardware type.
        parameter: str
            Parameter to modify.
        value: Any
            Value to set.
        """
        elems = self.getElementType(elementType)
        for elementName in [e["name"] for e in elems]:
            self.modifyElement(elementName, parameter, value)

    def modifyLattice(
        self,
        latticeName: str,
        parameter: str | list | dict,
        value: Any = None,
    ) -> None:
        """
        Modify a lattice line's attributes.

        Parameters
        ----------
        latticeName: str
            Lattice name.
        parameter: str or list or dict
            Parameter(s); a dict maps parameters to values when ``value`` is None.
        value: Any
            Value(s) to set.
        """
        if isinstance(parameter, dict) and value is None:
            for p, v in parameter.items():
                self.modifyLattice(latticeName, p, v)
        elif isinstance(parameter, list) and isinstance(value, list):
            for p, v in zip(parameter, value):
                self.modifyLattice(latticeName, p, v)
        elif latticeName in self.latticeObjects:
            setattr(self.latticeObjects[latticeName], parameter, value)

    def modifyLattices(
        self,
        latticeNames: str | list,
        parameter: str | list | dict,
        value: Any = None,
    ) -> None:
        """
        Modify several lattice lines; see :meth:`modifyLattice`.

        Parameters
        ----------
        latticeNames: str or list
            Lattice name(s), or ``all``.
        parameter: str or list or dict
            Parameter(s) to modify.
        value: Any
            Value(s) to set.
        """
        if isinstance(latticeNames, str):
            if latticeNames.lower() == "all":
                latticeNames = self.latticeObjects.keys()
            else:
                latticeNames = [latticeNames]
        for latt in latticeNames:
            self.modifyLattice(latt, parameter, value)

    def add_Generator(
        self,
        default: str | None = None,
        **kwargs,
    ) -> None:
        """
        Set :attr:`generator` (and ``latticeObjects["generator"]``) from keyword arguments.

        Parameters
        ----------
        default: str or None
            Key in :attr:`generator_keywords` whose defaults override ``kwargs``.
        """
        if "code" in kwargs:
            if kwargs["code"].lower() == "gpt":
                code = GPTGenerator
            elif kwargs["code"].lower() == "astra":
                code = ASTRAGenerator
            elif kwargs["code"].lower() in ["generic", "framework", "simba"]:
                code = frameworkGenerator
            else:
                raise NotImplementedError(f"Generator {kwargs['code']} not supported; must be ASTRA or GPT")
        else:
            warn("No generator code provided; defaulting to ASTRA")
            code = ASTRAGenerator
        if default in list(self.generator_keywords.keys()):
            self.generator = code(
                executables=self.executables,
                global_parameters=self.global_parameters,
                generator_keywords=self.generator_keywords,
                **(kwargs | self.generator_keywords[default]),
            )
        else:
            self.generator = code(
                executables=self.executables,
                global_parameters=self.global_parameters,
                generator_keywords=self.generator_keywords,
                **kwargs,
            )
        self.latticeObjects["generator"] = self.generator
        self._propagate_generator()

    def _propagate_generator(self) -> None:
        """
        Hand the generator to any lattice with a ``generator`` field.

        Codes that generate and track in one run (OPAL) describe the cathode from its settings.
        """
        for name, lattice in self.latticeObjects.items():
            if name == "generator":
                continue
            if "generator" in getattr(type(lattice), "model_fields", {}):
                lattice.generator = self.generator

    def change_generator(
        self,
        generator: str,
    ) -> None:
        """
        Change the generator's code, keeping its settings.

        Parameters
        ----------
        generator: str
            New generator code.
        """
        old_kwargs = self.generator.model_dump()
        old_kwargs["code"] = generator
        if generator.lower() == "gpt":
            generator = GPTGenerator(**old_kwargs)
        elif generator.lower() in ["generic", "framework", "simba"]:
            generator = frameworkGenerator(**old_kwargs)
        else:
            if generator.lower() != "astra":
                warn(f"generator {generator} not supported; defaulting to ASTRA")
            generator = ASTRAGenerator(**old_kwargs)
        self.latticeObjects["generator"] = generator
        self.generator = generator

    def set_lattice_prefix(
        self,
        lattice: str,
        prefix: str,
    ) -> None:
        """
        Set a lattice's prefix, which determines where it looks for its starting beam.

        Parameters
        ----------
        lattice: str
            Lattice name.
        prefix: str
            Lattice prefix.
        """
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].set_prefix(prefix)
        else:
            warn(
                f"{lattice} not found in latticeObjects; valid lattices are {list(self.latticeObjects.keys())}"
            )

    def set_lattice_sample_interval(
        self,
        lattice: str,
        interval: int,
    ) -> None:
        """
        Set a lattice's :attr:`~simba.Framework_objects.frameworkLattice.sample_interval`.

        Parameters
        ----------
        lattice: str
            Lattice name.
        interval: int
            Track every ``interval``-th particle.
        """
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].sample_interval = interval
        else:
            warn(
                f"{lattice} not found in latticeObjects; valid lattices are {list(self.latticeObjects.keys())}"
            )

    def setup_remote_execution(
            self,
            lattice: str,
            code: str,
            server: str,
            ncpu: int = 1,
            ngpu: int = 1,
            exclude: str | list | tuple | None = None,
            username: str = None,
            password: str = None,
    ) -> None:
        """
        Set up a lattice line to run on a server in ``hosts.yaml``, via its ``remote_setup`` and ``executables``.

        Parameters
        ----------
        lattice: str
            Lattice line, a list of them, or ``All``.
        code: str
            Code to run; the line is switched to it with :meth:`change_Lattice_Code` if needed.
        server: str
            Server name in ``hosts.yaml``.
        ncpu: int
            Number of CPUs.
        ngpu: int
            Number of GPUs; not yet implemented.
        exclude: str | list | tuple | None, optional
            Lattice line(s) to leave local.
        username: str | None, optional
            SSH username; defaults to :attr:`username`.
        password: str | None, optional
            SSH password; defaults to :attr:`password`.

        Raises
        ------
        ValueError
            If ``server`` is unknown, ``code`` is not available on it, or there is no username or password.
        """
        if server not in list(hosts.keys()):
            raise ValueError(f"Server '{server}' not found in defined hosts.")
        if username is None:
            if self.username != "":
                username = self.username
            else:
                raise ValueError("Username must be provided for remote execution.")
        if password is None:
            if self.password != "":
                password = self.password
            else:
                raise ValueError("Password must be provided for remote execution.")
        if lattice == "All":
            [self.setup_remote_execution(lo, code, server, ncpu, ngpu, exclude, username, password) for lo in self.latticeObjects]
        elif isinstance(lattice, (tuple, list)):
            [self.setup_remote_execution(ln, code, server, ncpu, ngpu, exclude, username, password) for ln in lattice]
        else:
            if lattice != "generator" and not (
                lattice == exclude
                or (isinstance(exclude, (list, tuple)) and lattice in exclude)
            ):
                if code.lower() not in hosts[server]["codes"]:
                    raise ValueError(f"Code {code} not available on server {server}.")
                if self.latticeObjects[lattice].code.lower() != code.lower():
                    self.change_Lattice_Code(lattice, code, nowarn=True)
                remote_setup = {
                    "host": hosts[server],
                    "username": username,
                    "password": password,
                }
                executables = self.prepare_executables(location=server, ncpu=ncpu)
                self.latticeObjects[lattice].remote_setup = remote_setup
                self.latticeObjects[lattice].executables = executables
                if self.verbose:
                    print(f"Setting up remote execution for {lattice} on {server} with {code}.")

    def __getitem__(self, key: str) -> Any:
        if key in list(self.elementObjects.keys()):
            return self.elementObjects.get(key)
        elif key in list(self.latticeObjects.keys()):
            return self.latticeObjects.get(key)
        elif key in list(self.groupObjects.keys()):
            return self.groupObjects.get(key)
        else:
            try:
                return getattr(self, key)
            except Exception:
                return None

    @property
    def elements(self) -> list:
        """Names of all elements in :attr:`elementObjects`."""
        return list(self.elementObjects.keys())

    @property
    def groups(self) -> list:
        """Names of all groups in :attr:`groupObjects`."""
        return list(self.groupObjects.keys())

    @property
    def lines(self) -> list:
        """Names of all lattice lines."""
        return list(self.latticeObjects.keys())

    @property
    def lattices(self) -> list:
        """Alias of :attr:`lines`."""
        return self.lines

    @property
    def commands(self) -> list:
        """Names of all command objects."""
        return list(self.commandObjects.keys())

    def path_arc_lengths(self) -> dict:
        """Each element's arc length along the beam path, from :meth:`~laura.models.element_list.MachineLayout.arc_lengths`.

        Keys are converted from pass names (``NAME#N``) to the line's flattened names (``NAME.N``).
        """
        layout = self._default_layout()
        if layout is None:
            return {}
        try:
            return {
                flatten_occurrence(name): s for name, s in layout.arc_lengths().items()
            }
        except Exception:
            return {}

    def getSValues(self) -> list:
        """
        S values of every line (:meth:`~simba.Framework_objects.frameworkLattice.getSValues`) along the beam path.

        Each line, which starts at zero itself, is offset by its start in :meth:`path_arc_lengths`,
        or by the previous line's end if the layout cannot place it.

        Returns
        -------
        list
            S values for all elements.
        """
        offsets = self.path_arc_lengths()
        s0 = 0
        allS = []
        for lo in self.latticeObjects.values():
            try:
                start = offsets.get(flatten_occurrence(lo.start), s0)
                latticeS = [a + start for a in lo.getSValues()]
                allS = allS + latticeS
                s0 = allS[-1]
            except Exception:
                pass
        return allS

    def getSValuesElements(self) -> list:
        """
        (name, element, s) for every line; see :meth:`~simba.Framework_objects.frameworkLattice.getSNamesElems`.

        Returns
        -------
        list
            ``(name, element, s)`` tuples, offset as in :meth:`getSValues`.
        """
        offsets = self.path_arc_lengths()
        s0 = 0
        allS = []
        for lo in self.latticeObjects:
            if lo != "generator":
                latt = self.latticeObjects[lo]
                names, elems, svals = latt.getSNamesElems()
                start = offsets.get(flatten_occurrence(latt.start), s0)
                latticeS = [a + start for a in svals]
                selems = list(zip(names, elems, latticeS))
                allS = allS + selems
                s0 = latticeS[-1]
        return allS

    def getZValuesElements(self) -> list:
        """
        (name, element, z) for every line; see :meth:`~simba.Framework_objects.frameworkLattice.getZNamesElems`.

        Returns
        -------
        list
            ``(name, element, z)`` tuples, sorted by the first z value.
        """
        allZ = []
        for lo in self.latticeObjects:
            if lo != "generator":
                names, elems, zvals = self.latticeObjects[lo].getZNamesElems()
                zelems = list(zip(names, elems, zvals))
                allZ = allZ + zelems
        return sorted(allZ, key=lambda x: x[2][0])

    def _line_output_names(self, lattice_name: str) -> set:
        """Element names ``lattice_name`` writes an output beam for: screens, markers, BPMs and its end."""
        latt = self.latticeObjects.get(lattice_name)
        if latt is None:
            return set()
        names = {
            element.name
            for element in getattr(latt, "screens_and_markers_and_bpms", [])
        }
        end = getattr(latt, "end", None)
        if isinstance(end, str):
            names.add(end)
        return names

    def _mark_colliding_outputs(self, files: list) -> None:
        """Tell each line which of its outputs another line also writes, as beam files are named by element alone.

        A handoff element (one line's end, the next's start) is not marked: the downstream
        line reads it by its plain name.
        """
        seen: Dict[str, int] = {}
        per_line = {}
        starts, ends = set(), set()
        for lattice_name in files:
            if lattice_name == "generator":
                continue
            latt = self.latticeObjects[lattice_name]
            per_line[lattice_name] = self._line_output_names(lattice_name)
            for side, names in ((latt.start, starts), (latt.end, ends)):
                if isinstance(side, str):
                    names.add(side)
            for name in per_line[lattice_name]:
                seen[name] = seen.get(name, 0) + 1
        handoffs = {name for name in starts & ends if seen.get(name, 0) == 2}
        shared = {name for name, count in seen.items() if count > 1} - handoffs
        for lattice_name, names in per_line.items():
            self.latticeObjects[lattice_name].colliding_outputs = names & shared

    def track(
        self,
        files: list | None = None,
        startfile: str | None = None,
        endfile: str | None = None,
        preprocess: bool = True,
        write: bool = True,
        track: bool = True,
        postprocess: bool = True,
        save_summary: bool = True,
        frameworkDirec: bool = False,
        check_lattice: bool = True,
    ) -> Any | None:
        """
        Track each line of the machine, or of ``files``, with its code.

        Afterwards the lattice (:meth:`save_lattice`) and settings (:meth:`save_settings`) are saved.

        Parameters
        ----------
        files: list or None
            Lattice names to track; all if None.
        startfile: str or None
            First lattice to track.
        endfile: str or None
            Last lattice to track.
        preprocess: bool
            Call :meth:`~simba.Framework_objects.frameworkLattice.preProcess` on each line.
        write: bool
            Write each lattice file.
        track: bool
            Track each lattice.
        postprocess: bool
            Call :meth:`~simba.Framework_objects.frameworkLattice.postProcess` on each line.
        save_summary: bool
            Save beam and Twiss summary files.
        frameworkDirec: bool
            Return a :class:`~simba.Framework.frameworkDirectory`.
        check_lattice: bool
            Call :meth:`check_lattice` first.

        Returns
        -------
        :class:`~simba.Framework.frameworkDirectory` or None
            The run's directory if ``frameworkDirec``.
        """
        if check_lattice and not self.check_lattice():
            raise Exception("Lattice Error - check definitions")
        if not self.executables_ready:
            raise Exception("Executables not ready - check setup and paths")
        # Lattice objects are rebuilt by change_Lattice_Code after the generator
        # is created, so hand it over here rather than only at construction.
        self._propagate_generator()
        self.tracking = True
        self.progress = 0
        if files is None:
            files = (
                ["generator"] + self.lines
                if not hasattr(self, "generator")
                else self.lines
            )
        if startfile is not None and startfile in files:
            index = files.index(startfile)
            files = files[index:]
        if endfile is not None and endfile in files:
            index = files.index(endfile)
            files = files[: index + 1]
        self._mark_colliding_outputs(files)
        if self.verbose:
            pbar = tqdm(total=len(files) * 4)
        percentage_step = 100 / len(files)
        for i in range(len(files)):
            base_percentage = 100 * (i / len(files))
            lattice_name = files[i]
            self.progress = base_percentage
            if lattice_name == "generator" and hasattr(self, "generator"):
                latt = self.generator
                base_description = "Generator[" + self.generator.code + "]"
            else:
                latt = self.latticeObjects[lattice_name]
                base_description = lattice_name + "[" + latt.code + "]"
            if self.verbose:
                pbar.set_description(base_description + ":              ")  # noqa E701
            if preprocess and lattice_name != "generator":

                if self.verbose:
                    pbar.set_description(
                        base_description + ": pre-process  "
                    )  # noqa E701
                latt.preProcess()
                self.progress = base_percentage + 0.25 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
            if write:
                if self.verbose:
                    pbar.set_description(
                        base_description + ": write        "
                    )  # noqa E701
                latt.write()
                self.progress = base_percentage + 0.5 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
            if track:
                if self.verbose:
                    pbar.set_description(
                        base_description + ": track        "
                    )  # noqa E701
                latt.run()
                self.progress = base_percentage + 0.75 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
            if postprocess:
                if self.verbose:
                    pbar.set_description(
                        base_description + ": post-process "
                    )  # noqa E701
                latt.postProcess()
                self.progress = base_percentage + 1 * percentage_step
                if lattice_name != "generator":
                    for name, elem in latt.elementObjects.items():
                        if name in self.elementObjects:
                            self.elementObjects[name] = elem
            if self.verbose:
                pbar.update()  # noqa E701
        if self.verbose:
            pbar.set_description(base_description + ": Finished! ")
            pbar.close()
        self.save_lattice(directory=self.subdirectory, filename="lattice.yaml")
        self.save_settings(
            directory=self.subdirectory,
            filename="settings.def",
            elements={"filename": "lattice.yaml"},
        )
        if save_summary:
            self.save_summary_files()
        self.tracking = False
        if frameworkDirec:
            return frameworkDirectory(
                directory=self.subdirectory,
                twiss=True,
                beams=True,
                verbose=self.verbose,
            )

    def postProcess(
        self,
        files: list | None = None,
        startfile: str | None = None,
        endfile: str | None = None,
    ) -> None:
        """
        Post-process tracking files and convert them to HDF5; see :meth:`~simba.Framework_objects.frameworkLattice.postProcess`.

        Parameters
        ----------
        files: list or None
            Lattice names; all if None.
        startfile: str or None
            First lattice to process.
        endfile: str or None
            Last lattice to process.
        """
        if files is None:
            files = (
                ["generator"] + self.lines
                if not hasattr(self, "generator")
                else self.lines
            )
        if startfile is not None and startfile in files:
            index = files.index(startfile)
            files = files[index:]
        if endfile is not None and endfile in files:
            index = files.index(endfile)
            files = files[: index + 1]
        for i in range(len(files)):
            latt = files[i]
            if latt == "generator" and hasattr(self, "generator"):
                self.generator.postProcess()
            else:
                self.latticeObjects[latt].postProcess()

    def save_summary_files(self, twiss: bool = True, beams: bool = True) -> None:
        """
        Save HDF5 summaries of the Twiss and beam files in the subdirectory.

        Parameters
        ----------
        twiss: bool
            Save ``Twiss_Summary.hdf5``, via :func:`~simba.Modules.Twiss.load_directory`.
        beams: bool
            Save ``Beam_Summary.hdf5``, via :func:`~simba.Modules.Beams.save_HDF5_summary_file`.
        """
        if twiss:
            t = rtf.load_directory(self.subdirectory)
            self.stamp_twiss_turns(t)
            t.save_HDF5_twiss_file(os.path.join(self.subdirectory, "Twiss_Summary.hdf5"))
        if beams:
            rbf.save_HDF5_summary_file(
                self.subdirectory, os.path.join(self.subdirectory, "Beam_Summary.hdf5")
            )

    def stamp_twiss_turns(self, t: "rtf.twiss") -> None:
        """
        Fill the ``turn`` column (:attr:`~simba.Modules.Twiss.twiss.turn`) of a loaded twiss object.

        Rows are matched to a line by ``lattice_name``, taken from the filename.

        Parameters
        ----------
        t: :class:`~simba.Modules.Twiss.twiss`
            Modified in place.
        """
        names = np.array(t.lattice_name.val, dtype=str)
        if len(names) == 0:
            return
        turns = {
            n: o.turns
            for n, o in self.latticeObjects.items()
            if not isinstance(o, frameworkGenerator)
        }
        t.turn.val = np.array(
            [
                turns.get(n, turns.get(n.removesuffix("_twiss"), 0))
                for n in names
            ],
            dtype=int,
        )

    def pushRunSettings(self) -> None:
        """Push :attr:`runSetup` to each lattice."""
        for latticeObject in self.latticeObjects.values():
            if isinstance(latticeObject, tuple(latticeClasses)):
                latticeObject.updateRunSettings(self.runSetup)

    def setNRuns(self, nruns: int) -> None:
        """
        Set the number of runs for all lattices; see :meth:`~simba.Framework_objects.runSetup.setNRuns`.

        Parameters
        ----------
        nruns: int
            Number of runs.
        """
        self.runSetup.setNRuns(nruns)
        self.pushRunSettings()

    def setSeedValue(self, seed: int) -> None:
        """
        Set the random seed for all lattices; see :meth:`~simba.Framework_objects.runSetup.setSeedValue`.

        Parameters
        ----------
        seed: int
            Random number seed.
        """
        self.runSetup.setSeedValue(seed)
        self.pushRunSettings()

    def loadElementErrors(self, file: str) -> None:
        """
        Load an element errors file; see :meth:`~simba.Framework_objects.runSetup.loadElementErrors`.

        Parameters
        ----------
        file: str
            Errors file.
        """
        self.runSetup.loadElementErrors(file)
        self.pushRunSettings()

    def setElementScan(
        self,
        name: str,
        item: str,
        scanrange: list,
        multiplicative: bool = False,
    ) -> None:
        """
        Scan one element parameter; see :meth:`~simba.Framework_objects.runSetup.setElementScan`.

        Parameters
        ----------
        name: str
            Element name.
        item: str
            Parameter to scan.
        scanrange: list
            ``(min, max)`` of the scan.
        multiplicative: bool
            Values multiply the original rather than add to it.
        """
        self.runSetup.setElementScan(
            name=name, item=item, scanrange=scanrange, multiplicative=multiplicative
        )
        self.pushRunSettings()


class frameworkDirectory(BaseModel):
    """Load the Beam and Twiss files of a finished tracking run."""

    directory: str | None = None
    """Directory of the run."""

    twiss: bool | rtf.twiss = True
    """Load Twiss files; replaced by the loaded Twiss."""

    beams: bool | rbf.beamGroup | None = False
    """Load beam files; replaced by the loaded beams."""

    wavefronts: bool | rwf.wavefrontGroup | None = False
    """Load wavefront files; replaced by the loaded wavefronts."""

    verbose: bool = False
    """Print status updates."""

    settings: str = "settings.def"
    """Framework settings filename."""

    changes: str = "changes.yaml"
    """Lattice changes filename."""

    rest_mass: float | None = None
    """Particle rest mass; defaults to the beams' own, or the electron's."""

    framework: Framework | None = None
    """The run's :class:`~simba.Framework.Framework`; built from :attr:`settings` if not given."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        if not isinstance(self.framework, Framework):
            directory = (
                "." if self.directory is None else os.path.abspath(self.directory)
            )
            self.framework = Framework(**kwargs)
            self.framework.loadSettings(os.path.join(directory, self.settings))
        else:
            if self.directory is None:
                directory = os.path.abspath(self.framework.subdirectory)
            else:
                directory = self.directory

        if os.path.exists(os.path.join(directory, self.changes)):
            self.framework.load_changes_file(os.path.join(directory, self.changes))
        if self.beams:
            self.beams = rbf.load_HDF5_summary_file(
                os.path.join(directory, "Beam_Summary.hdf5")
            )
            if len(self.beams) < 1:
                print("No Summary File! Globbing...")
                self.beams = rbf.load_directory(directory)
            if self.rest_mass is None:
                if len(self.beams.param("particle_mass")) > 0:
                    rest_mass = self.beams.param("particle_mass")[0][0]
                else:
                    rest_mass = constants.m_e
            else:
                rest_mass = self.rest_mass
            self.twiss = rtf.twiss(rest_mass=rest_mass)
        else:
            self.beams = None
            self.twiss = rtf.twiss()
        if self.wavefronts:
            self.wavefronts = rwf.load_directory(directory)
        if self.twiss:
            self.twiss.load_directory(directory, verbose=self.verbose)

    if use_matplotlib:

        def plot(self, *args, **kwargs):
            """See :func:`~simba.Modules.plotting.plotting.plot`."""
            return groupplot.plot(self, *args, **kwargs)

        def general_plot(self, *args, **kwargs):
            """See :func:`~simba.Modules.plotting.plotting.general_plot`."""
            return groupplot.general_plot(self, *args, **kwargs)

    def __repr__(self):
        return repr(
            {"framework": self.framework, "twiss": self.twiss, "beams": self.beams}
        )

    def save_summary_files(self, twiss: bool = True, beams: bool = True):
        """
        Save summary files in the framework's subdirectory; see :meth:`Framework.save_summary_files`.

        Parameters
        ----------
        twiss: bool
            Save ``Twiss_Summary.hdf5``.
        beams: bool
            Save ``Beam_Summary.hdf5``.
        """
        self.framework.save_summary_files(twiss=twiss, beams=beams)

    def getScreen(self, screen: str) -> rbf.beam | None:
        """
        Get the beam at a screen; see :meth:`~simba.Modules.Beams.beamGroup.getScreen`.

        Parameters
        ----------
        screen: str
            Screen name.

        Returns
        -------
        :class:`~simba.Modules.Beams.beam`

        Raises
        ------
        ValueError
            If the beams have not been loaded.
        """
        if isinstance(self.beams, rbf.beamGroup):
            return self.beams.getScreen(screen)
        else:
            raise ValueError("Beam files have not been read in")

    def getScreenNames(self) -> dict:
        """
        Get the beams at all screens.

        Returns
        -------
        dict
            :class:`~simba.Modules.Beams.beam` objects by screen name.

        Raises
        ------
        ValueError
            If the beams have not been loaded.
        """
        if isinstance(self.beams, rbf.beamGroup):
            return self.beams.getScreens()
        else:
            raise ValueError("Beam files have not been read in")

    def element(self, element: str, field: str | None = None) -> Any | PhysicalBaseElement:
        """
        Get an element, or one of its fields; with no field, also print the element.

        Parameters
        ----------
        element: str
            Element name.
        field: str | None
            Field to get.

        Returns
        -------
        Any or :class:`~laura.models.element.Element`
            The field, or the whole element.
        """
        elem = self.framework.getElement(element)
        if field:
            try:
                return getattr(elem, field)
            except AttributeError:
                warn(f"{elem} does not have field {field}; returning entire element")
                return elem
        else:
            pprint(
                {
                    k.replace("object", ""): v
                    for k, v in dict(elem).items()
                    if k not in disallowed
                }
            )
            return elem


def load_directory(
    directory: str = ".", twiss: bool = True, beams: bool = False, wavefronts: bool = False, **kwargs
) -> frameworkDirectory:
    """Load a SIMBA tracking run as a :class:`frameworkDirectory`."""
    return frameworkDirectory(directory=directory, twiss=twiss, beams=beams, wavefronts=wavefronts, **kwargs)
