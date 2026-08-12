"""
SIMBA ELEGANT Module

Various objects and functions to handle ELEGANT lattices and commands. See `Elegant manual`_ for more details.

    .. _Elegant manual: https://ops.aps.anl.gov/manuals/elegant_latest/elegant.html

Classes:
    - :class:`~simba.Codes.Elegant.Elegant.ElegantLattice`: The ELEGANT lattice object, used for
    converting the :class:`~simba.Framework_objects.FrameworkObject` s defined in the
    :class:`~simba.Framework_objects.FrameworkLattice` into a string representation of
    the lattice suitable for ELEGANT input and lattice files.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantCommandFile`: Base class for defining
    commands in an ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantGlobalSettingsCommand`: Class for defining the
    &global_settings portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantRunSetupCommand`: Class for defining the
    &run_setup portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantErrorElementsCommand`: Class for defining the
    &error_elements portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantErrorElementsCommand`: Class for defining the
    &error_elements portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantScanElementsCommand`: Class for defining the
    &scan_elements portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantRunControlCommand`: Class for defining the
    &run_control portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantTwissOutputCommand`: Class for defining the
    &twiss_output portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantFloorCoordinatesCommand`: Class for defining the
    &floor_coordinates portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantMatrixOutputCommand`: Class for defining the
    &matrix_output portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantSddsBeamCommand`: Class for defining the
    &sdds_beam portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantTrackCommand`: Class for defining the
    &track portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantTrackCommand`: Class for defining the
    &track portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.ElegantOptimisation`: Class for defining the
    commands for optimization in the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.SddsFile`: Class for creating, modifying and
    saving SDDS files.
"""

import os
from copy import copy
import subprocess
import numpy as np
from warnings import warn
try:
    import sdds
except Exception:
    print("No SDDS available!")
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import ClassVar
from ...framework_objects import (
    FrameworkLattice,
    FrameworkCommand,
    elementkeywords,
    keyword_conversion_rules_elegant,
)
from ...framework_helper_functions import save_file
from ...modules import beams as rbf
from typing import Dict, List, Any
from laura.models.diagnostic import DiagnosticElement


class ElegantLattice(FrameworkLattice):
    """
    Class for defining the ELEGANT lattice object, used for
    converting the :class:`~simba.Framework_objects.FrameworkObject`s defined in the
    :class:`~simba.Framework_objects.FrameworkLattice` into a string representation of
    the lattice suitable for an ELEGANT input file.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "commandFiles": "command_files",
        "commandFilesOrder": "command_files_order",
        "createCommandFiles": "create_command_files",
        "postProcess": "post_process",
        "preProcess": "pre_process",
        "processElementErrors": "process_element_errors",
        "processElementScan": "process_element_scan",
        "processRunSettings": "process_run_settings",
        "trackBeam": "track_beam",
        "writeElements": "write_elements",
    }

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ELEGANT into the SIMBA generic 
    :class:`~simba.Modules.Beams.Beam` object and writing files"""

    code: str = "elegant"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = False
    """Flag to indicate whether negative drifts are allowed"""

    particle_definition: str | None = None
    """String representation of the initial particle distribution"""

    bunch_charge: float | None = None
    """Bunch charge"""

    q: Any = None
    """:class:`~simba.Elements.charge.charge` object"""

    track_beam: bool = True
    """Flag to indicate whether to track the beam"""

    betax: float | None = None
    """Initial beta_x for matching"""

    betay: float | None = None
    """Initial beta_y for matching"""

    alphax: float | None = None
    """Initial alpha_x for matching"""

    alphay: float | None = None
    """Initial alpha_y for matching"""

    command_files: Dict = {}
    """Dictionary of :class:`~simba.Codes.Elegant.Elegant.ElegantCommandFile`
    objects for writing to the ELEGANT input file"""

    final_screen: Any = None
    """:class:`simba.Elements.screen.screen` object at the end of the line"""

    command_files_order: List = []
    """Order in which commands are to be written in the ELEGANT input file"""

    ref_idx: int = None
    """Reference particle index"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.element_objects[self.start].name

    def write_elements(self) -> str:
        """
        Write the lattice elements defined in this object into an ELEGANT-compatible format; see
        :attr:`~simba.Framework_objects.frameworkLattice.element_objects`.

        Returns
        -------
        str
            The lattice represented as a string compatible with ELEGANT
        """
        if self.bunch_charge is not None:
            q = abs(self.bunch_charge)
        else:
            q = abs(self.global_parameters["beam"].q)
        return self.section.to_elegant(charge=q)

    def process_run_settings(self) -> tuple:
        """
        Process the runSettings object to extract the number of runs and the random number seed,
        and extract error definitions or a parameter scan definiton pertaining to this lattice section.

        Returns
        -------
        tuple
            nruns: Number of runs
            seed: Random number seedoutput
            elementErrors: Dict of errors on elements
            elementScan: Dict of elements and parameters to scan
        """
        nruns = self.run_settings.nruns
        seed = self.run_settings.seed
        elementErrors = (
            None
            if (self.run_settings.element_errors is None)
            else self.process_element_errors(self.run_settings.element_errors)
        )
        elementScan = (
            None
            if (self.run_settings.elementScan is None)
            else self.process_element_scan(self.run_settings.elementScan, nruns)
        )
        return nruns, seed, elementErrors, elementScan

    def process_element_errors(self, elementErrors: Dict) -> Dict:
        """
        Process the elementErrors dictionary to prepare it for use with the current lattice section in ELEGANT

        Parameters
        ----------
        elementErrors: Dict
            Dictionary of element names and error definitions

        Returns
        -------
        Dict
            Formatted dictionary of errors on elements
        """

        output = {}
        default_err = {
            "amplitude": 1e-6,
            "fractional": 0,
            "type": '"gaussian"',
        }

        for ele, error_defn in elementErrors.items():
            wildcard = "*" in ele

            if (ele not in self.all_elements) and (not wildcard):
                raise KeyError(
                    "Lattice element %s does not exist in the current lattice"
                    % str(ele)
                )

            element_exists = False
            if (ele in self.elements) and not wildcard:
                element_exists = True
                element_types = [self.element_objects[ele].hardware_type]
            elif wildcard:
                element_matches = [
                    x for x in self.elements if (ele.replace("*", "") in x)
                ]
                if len(element_matches) != 0:
                    element_exists = True
                    element_types = [
                        self.element_objects[x].hardware_type for x in element_matches
                    ]

            if element_exists:
                output[ele] = {}

                ele_type = str(element_types[0])
                has_expected_type = [(x == ele_type) for x in element_types]
                if not all(has_expected_type):
                    raise TypeError(
                        "All lattice elements matching a wilcarded element name must have the same type"
                    )

                for param in error_defn:
                    if param not in elementkeywords[ele_type]["keywords"]:
                        raise KeyError(
                            "Element type %s has no associated keyword %s"
                            % (str(ele_type), str(param))
                        )

                    conversions = keyword_conversion_rules_elegant[ele_type]
                    keyword = conversions[param] if (param in conversions) else param
                    output[ele][keyword] = copy(default_err)

                    for k in default_err:
                        if k in error_defn[param]:
                            output[ele][keyword][k] = error_defn[param][k]

                    if wildcard:
                        output[ele][keyword]["bind"] = 1
                        output[ele][keyword]["bind_across_names"] = 1
        return output

    def process_element_scan(self, elementScan: Dict, nsteps: int) -> Dict | None:
        """
        Process the elementScan dictionary to prepare it for use with the current lattice section in ELEGANT

        #TODO deprecated?

        Parameters
        ----------
        elementScan: Dict[name, item]
            Dictionary of elements and parameters to scan

        Returns
        -------
        Dict or None
            Dictionary of processed elements to scan if valid, else None
        """
        ele, param = elementScan["name"], elementScan["item"]

        if ele not in self.all_elements:
            raise KeyError(
                "Lattice element %s does not exist in the current lattice" % str(ele)
            )

        element_exists = ele in self.elements
        if element_exists:
            ele_type = self.element_objects[ele].hardware_type

            if param not in elementkeywords[ele_type]["keywords"]:
                raise KeyError(
                    "Element type %s has no associated parameter %s"
                    % (str(ele_type), str(param))
                )

            conversions = keyword_conversion_rules_elegant[ele_type]
            keyword = conversions[param] if (param in conversions) else param

            scan_values = np.linspace(
                elementScan["min"], elementScan["max"], int(nsteps) - 1
            )

            multiplicative = elementScan["multiplicative"]
            if multiplicative:
                if 1.0 not in list(scan_values):
                    scan_values = [1.0] + list(scan_values)
            else:
                if 0.0 not in list(scan_values):
                    scan_values = [0.0] + list(scan_values)

            scan_fname = "%s-%s.sdds" % (ele, param)
            scanSDDS = SddsFile()
            scanSDDS.add_parameter("name", [ele], type=sdds.SDDS(0).SDDS_STRING)
            scanSDDS.add_parameter("item", [keyword], type=sdds.SDDS(0).SDDS_STRING)
            scanSDDS.add_parameter("multiplicative", [int(multiplicative)])
            scanSDDS.add_parameter("nominal", [getattr(self.elements[ele], param)])
            scanSDDS.add_column("values", scan_values)
            scanSDDS.save(self.global_parameters["master_subdir"] + "/" + scan_fname)

            output = {
                "name": ele,
                "item": keyword,
                "differential": int(not multiplicative),
                "multiplicative": int(multiplicative),
                "enumeration_file": scan_fname,
                "enumeration_column": "values",
            }
            return output

        else:
            return None

    def write(self) -> None:
        """
        Write the ELEGANT lattice and command files to `master_subdir` using the functions
        :func:`~simba.Codes.Elegant.Elegant.write_elements` and
        based on the output of :func:`~simba.Codes.Elegant.Elegant.create_command_files`.
        """
        lattice_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".lte"
        )
        save_file(lattice_file, self.write_elements())
        self.files.append(lattice_file)
        # try:
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".ele"
        )
        save_file(command_file, "", "w")
        self.files.append(command_file)
        if len(self.command_files_order) > 0:
            for cfileid in self.command_files_order:
                if cfileid in self.command_files:
                    cfile = self.command_files[cfileid]
                    save_file(command_file, cfile.write_elegant(), "a")
                    self.files.append(command_file)
        else:
            warn("commandFilesOrder length is zero; run createCommandFiles first")
        # except Exception:
        #     passastrabeamfilename

    def create_command_files(self) -> None:
        """
        Create the :class:`~simba.Codes.Elegant.ElegantCommandFile` objects
        based on the run settings, lattice and beam parameters, including scans of elements,
        if defined.

        Updates :attr:`~simba.Codes.Elegant.Elegant.command_files` and
        :attr:`~simba.Codes.Elegant.Elegant.command_files_order`
        """
        if not isinstance(self.command_files, dict) or self.command_files == {}:
            # print('createCommandFiles is creating new command files!')
            # print('processRunSettings')
            nruns, seed, elementErrors, elementScan = self.process_run_settings()
            self.command_files["global_settings"] = ElegantGlobalSettingsCommand(
                # lattice=self,
                warning_limit=0
            )
            # print('run_setup')
            self.command_files["run_setup"] = ElegantRunSetupCommand(
                lattice=self.objectname + ".lte",
                p_central=np.mean(self.global_parameters["beam"].beta_gamma),
                seed=seed,
                # losses="%s.loss",
                s_start=self.start_object.physical.start.z,
                use_beamline=self.objectname,
            )

            # print('generate commands for monte carlo jitter runs')
            if elementErrors is not None:
                self.command_files["run_control"] = ElegantRunControlCommand(
                    # lattice=self,
                    n_steps=nruns,
                    n_passes=1,
                    reset_rf_for_each_step=0,
                    first_is_fiducial=1,
                )
                self.command_files["error_elements"] = ElegantErrorElementsCommand(
                    lattice=self, element_errors=elementErrors, nruns=nruns
                )
                for e in elementErrors:
                    for item in elementErrors[e]:
                        self.command_files["error_element_" + e + "_" + item] = (
                            ElegantCommandFile(
                                objectname="error_element",
                                objecttype="error_element",
                                name=e,
                                item=item,
                                allow_missing_elements=1,
                                **elementErrors[e][item],
                            )
                        )
            elif elementScan is not None:
                # print('generate commands for parameter scans without fiducialisation (i.e. jitter scans)')
                self.command_files["run_control"] = ElegantRunControlCommand(
                    # lattice=self,
                    n_steps=nruns - 1,
                    n_passes=1,
                    n_indices=1,
                    reset_rf_for_each_step=0,
                    first_is_fiducial=1,
                )
                self.command_files["scan_elements"] = ElegantScanElementsCommand(
                    # lattice=self,
                    name=elementScan["name"],
                    item=elementScan["item"],
                    enumeration_file=elementScan["enumeration_file"],
                    enumeration_column=elementScan["enumeration_column"],
                    multiplicative=int(elementScan["multiplicative"]),
                    nruns=nruns,
                )
            else:
                # print('run_control for standard runs with no jitter')
                self.command_files["run_control"] = ElegantRunControlCommand(
                    # lattice=self,
                    n_steps=1, n_passes=1
                )

            # print('twiss_output')
            self.command_files["twiss_output"] = ElegantTwissOutputCommand(
                # lattice=self,
                beam=self.global_parameters["beam"],
                beta_x=self.global_parameters["beam"].twiss.beta_x_corrected,
                beta_y=self.global_parameters["beam"].twiss.beta_y_corrected,
                alpha_x=self.global_parameters["beam"].twiss.alpha_x_corrected,
                alpha_y=self.global_parameters["beam"].twiss.alpha_y_corrected,
                # eta_x=self.global_parameters["beam"].twiss.eta_x,
                # eta_xp=self.global_parameters["beam"].twiss.eta_xp,
            )
            # print('floor_coordinates')
            self.command_files["floor_coordinates"] = ElegantFloorCoordinatesCommand(
                # lattice=self,
                X0=self.start_object.physical.start.x,
                Y0=self.start_object.physical.start.y,
                Z0=self.start_object.physical.start.z,
            )
            # print('matrix_output')
            self.command_files["matrix_output"] = ElegantMatrixOutputCommand(
                # lattice=self,
            )
            # print('sdds_beam')
            self.command_files["sdds_beam"] = ElegantSddsBeamCommand(
                lattice=self,
                input=self.objectname + "_input.sdds",
                sample_interval=self.sample_interval,
                reuse_bunch=1,
                fiducialization_bunch=0,
                center_arrival_time=0,
            )
            # print('track')
            self.command_files["track"] = ElegantTrackCommand(
                # lattice=self,
                track_beam=self.track_beam
            )
            self.command_files_order = list(
                self.command_files.keys()
            )  # ['global_settings', 'run_setup', 'error_elements', 'scan_elements', 'run_control', 'twiss', 'sdds_beam', 'track']

    def pre_process(self) -> None:
        """
        Prepare the input distribution for ELEGANT based on the `prefix` in the settings
        file for this lattice section, and create the ELEGANT command files.
        """
        super().pre_process()
        prefix = self.get_prefix()
        self.read_input_file(prefix, self.particle_definition)
        self.ref_idx = self.global_parameters["beam"].reference_particle_index
        self.global_parameters["beam"].beam.rematch_x_plane(
            **self.initial_twiss["horizontal"]
        )
        self.global_parameters["beam"].beam.rematch_y_plane(
            **self.initial_twiss["vertical"]
        )
        if self.track_beam:
            self.hdf5_to_sdds()
        self.create_command_files()

    @lox.thread(60)
    def screen_threaded_function(self, scr: DiagnosticElement, sddsindex: int, **kwargs) -> None:
        """
        Convert output from ELEGANT screen to HDF5 format

        Parameters
        ----------
        scr: PAdantic DiagnosticElement
            Screen object
        sddsindex: int
            SDDS object index
        """
        # try:
        return self.sdds_to_hdf5(
            scr,
            sddsindex,
            toffset=-1 * np.mean(self.global_parameters["beam"].particles.t),
            **kwargs,
        )
        # except Exception as e:
        #     print(f"Screen error {scr.name}, {e}")
        #     return None

    def post_process(self) -> None:
        """
        PostProcess the simulation results, i.e. gather the screens and markers
        and write their outputs to HDF5.

        :attr:`~simba.Codes.Elegant.Elegant.command_files` is also cleared
        """
        super().post_process()
        if self.track_beam:
            for i, s in enumerate(self.screens_and_markers_and_bpms):
                self.sdds_to_hdf5(
                    s,
                    toffset=-1 * np.mean(self.global_parameters["beam"].particles.t),
                    ref_index=self.ref_idx,
                )
                # self.screen_threaded_function.scatter(s, i, ref_index=self.ref_idx)
            if (
                self.final_screen is not None
                and not self.final_screen.output_filename.lower()
                in [
                    s.output_filename.lower() for s in self.screens_and_markers_and_bpms
                ]
            ):
                self.sdds_to_hdf5(
                    self.final_screen,
                    toffset=-1 * np.mean(self.global_parameters["beam"].particles.t),
                    ref_index=self.ref_idx,
                )
        #         self.screen_threaded_function.scatter(
        #             self.final_screen,
        #             len(self.screens_and_markers_and_bpms),
        #             ref_index=self.ref_idx
        #         )
        # self.screen_threaded_function.gather()
        self.command_files = {}

    def hdf5_to_sdds(self, write: bool = True) -> None:
        """
        Convert the HDF5 beam input file to an SDDS file, and create a
        :class:`~simba.Elements.charge.charge` object as the first element
        """
        sddsbeamfilename = self.objectname + "_input.sdds"
        if write:
            rbf.sdds.write_sdds_file(
                self.global_parameters["beam"],
                self.global_parameters["master_subdir"] + "/" + sddsbeamfilename,
                xyzoffset=list(self.start_object.physical.start.model_dump().values()),
            )
            self.files.append(self.global_parameters["master_subdir"] + "/" + sddsbeamfilename)

    def sdds_to_hdf5(
            self,
            screen: DiagnosticElement,
            toffset: float = 0.0,
            ref_index: int = None
    ) -> None:
        """
        Convert the SDDS beam file name to HDF5 format and write the beam file.

        Parameters
        ----------
        screen: PAdantic.models.diagnostic.DiagnosticElement
            PAdantic DiagnosticElement
        sddsindex: int
            Index for SDDS file
        toffset: float, optional
            Temporal offset
        ref_index: int, optional
            Reference particle index
        """
        beam = rbf.Beam()
        rootname = f"{self.global_parameters['master_subdir']}/{screen.name}"
        elegantbeamfilename = f"{rootname}.SDDS"
        rbf.sdds.read_sdds_beam_file(
            beam,
            elegantbeamfilename,
            xyzoffset=list(self.element_objects[screen.name].physical.start.model_dump().values()),
            ref_index=ref_index
        )
        HDF5filename = f"{rootname}.openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(beam, HDF5filename)
        if self.global_parameters["delete_tracking_files"]:
            os.remove(elegantbeamfilename)

    def run(self):
        """Run the code with input 'filename'"""
        if self.remote_setup:
            super().run_remote()
        elif not os.name == "nt":
            command = self.executables[self.code] + [self.objectname + ".ele"]
            if self.global_parameters["simcodes_location"] is None:
                my_env = {**os.environ}
            else:
                my_env = {
                    **os.environ,
                    "RPN_DEFNS": os.path.abspath(
                        self.global_parameters["simcodes_location"]
                    )
                    + "/Elegant/defns_linux.rpn",
                }
            with open(
                os.path.abspath(
                    self.global_parameters["master_subdir"]
                    + "/"
                    + self.objectname
                    + ".log"
                ),
                "w",
            ) as f:
                subprocess.call(
                    command,
                    stdout=f,
                    cwd=self.global_parameters["master_subdir"],
                    env=my_env,
                )
        else:
            code_string = " ".join(self.executables[self.code]).lower()
            command = self.executables[self.code] + [self.objectname + ".ele"]
            if "pelegant" in code_string:
                command = (
                    [command[0]]
                    + [
                        "-env",
                        "RPN_DEFNS",
                        (
                            os.path.abspath(self.global_parameters["simcodes_location"])
                            + "/Elegant/defns.rpn"
                        ).replace("/", "\\"),
                    ]
                    + command[1:]
                )
                command = [c.replace("/", "\\") for c in command]
                with open(
                    os.path.abspath(
                        self.global_parameters["master_subdir"]
                        + "/"
                        + self.objectname
                        + ".log"
                    ),
                    "w",
                ) as f:
                    subprocess.call(
                        command, stdout=f, cwd=self.global_parameters["master_subdir"]
                    )
            else:
                command = [c.replace("/", "\\") for c in command]
                with open(
                    os.path.abspath(
                        self.global_parameters["master_subdir"]
                        + "/"
                        + self.objectname
                        + ".log"
                    ),
                    "w",
                ) as f:
                    subprocess.call(
                        command,
                        stdout=f,
                        cwd=self.global_parameters["master_subdir"],
                        env={
                            "RPN_DEFNS": (
                                os.path.abspath(
                                    self.global_parameters["simcodes_location"]
                                )
                                + "/Elegant/defns.rpn"
                            ).replace("/", "\\")
                        },
                    )

    def elegant_command_file(self, *args, **kwargs):
        return ElegantCommandFile(*args, **kwargs)


class ElegantCommandFile(FrameworkCommand):
    """
    Generic class for generating elements for an ELEGANT input file
    """
    # lattice: frameworkLattice
    # """The :class:`~simba.Framework_objects.frameworkLattice` object"""
    #
    # def __init__(self, *args, **kwargs):
    #     super(ElegantCommandFile, self).__init__(*args, **kwargs)


class ElegantGlobalSettingsCommand(ElegantCommandFile):
    """
    Global settings for an ELEGANT input file; see `Elegant global settings`_

    .. _Elegant global settings: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu37.html#x45-440007.28
    """

    inhibit_fsync: int = 0
    """See this parameter in `Elegant global settings`_ for more details.    """

    mpi_io_force_file_sync: int = 0
    """See this parameter in `Elegant global settings`_ for more details."""

    mpi_io_read_buffer_size: int = 16777216
    """See this parameter in `Elegant global settings`_ for more details."""

    mpi_io_write_buffer_size: int = 16777216
    """See this parameter in `Elegant global settings`_ for more details."""

    usleep_mpi_io_kludge: int = 0
    """See this parameter in `Elegant global settings`_ for more details."""

    objectname: str = "global_settings"
    """Name of object for frameworkObject"""

    objecttype: str = "global_settings"
    """Type of object for frameworkObject"""

    # def __init__(
    #     self,
    #     *args,
    #     **kwargs,
    # ):
    #     super(elegant_global_settings_command, self).__init__(
    #         objectname="global_settings",
    #         objecttype="global_settings",
    #         *args,
    #         **kwargs,
    #     )
    #     kwargs.update(
    #         {
    #             "inhibit_fsync": self.inhibit_fsync,
    #             "mpi_io_force_file_sync": self.mpi_io_force_file_sync,
    #             "mpi_io_read_buffer_size": self.mpi_io_read_buffer_size,
    #             "mpi_io_write_buffer_size": self.mpi_io_write_buffer_size,
    #             "usleep_mpi_io_kludge": self.usleep_mpi_io_kludge,
    #         }
    #     )
    #     self.add_properties(**kwargs)


class ElegantRunSetupCommand(ElegantCommandFile):
    """
    Run setup for an ELEGANT input file; see `Elegant run setup`_

    .. _Elegant run setup: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu69.html#x77-760007.60
    """

    pcentral: float = 0.0
    """Central momentum in units of beta-gamma"""

    seed: int = 0
    """Seed for random number generators"""

    always_change_p0: int = 1
    """Match the reference momentum to the beam momentum after each element."""

    default_order: int = 3
    """The default order of transfer matrices used for elements having matrices."""

    lattice: FrameworkLattice | str = None
    """:class:`~simba.Framework_objects.frameworkLattice object"""

    centroid: str = "%s.cen"
    """File to which centroid data is to be written"""

    sigma: str = "%s.sig"
    """File to which sigma data is to be written"""

    lattice_filename: str = None
    """Name of lattice filename for ELEGANT"""

    s_start: float = 0.0
    """Starting s position"""

    objectname: str = "run_setup"
    """Name of objectname for elegant run_setup"""

    objecttype: str = "run_setup"
    """Name of objecttype for elegant run_setup"""


class ElegantErrorElementsCommand(ElegantCommandFile):
    """
    Error control for an ELEGANT input file; see `Elegant error control`_

    .. _Elegant error control: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu33.html#x41-400007.24
    """

    _DEPRECATED_METHOD_ALIASES = {
        "elementErrors": "element_errors",
    }

    element_errors: Dict = None
    """Dictionary of elements with errors"""

    nruns: int = 1
    """Number of error runs to perform"""

    lattice: FrameworkLattice = None
    """:class:`~simba.Framework_objects.frameworkLattice object"""

    no_errors_for_first_step: int = 1
    """Perform the first run without errors"""

    error_log: str = "%s.erl"
    """File to which errors are to be logged"""

    objectname: str = "error_control"
    """Name of frameworkObject objectname"""

    objecttype: str = "error_control"
    """Name of frameworkObject objecttype"""


class ElegantScanElementsCommand(ElegantCommandFile):
    """
    Error control for an ELEGANT input file; see `Elegant vary element`_

    .. _Elegant vary element: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu85.html#x93-920007.76
    """

    name: str
    """Element name to scan"""

    item: str
    """Parameter to scan"""

    enumeration_file: str
    """Name of SDDS file containing element to scan"""

    enumeration_column: str
    """Parameter to scan in enumeration_file"""

    multiplicative: int = 0
    """Whether to multiply the original value by the values in the scan range"""

    nruns: int = 1
    """Number of runs to perform"""

    index_number: int = 0
    """Scan number index"""

    lattice: FrameworkLattice = None
    """:class:`~simba.Framework_objects.frameworkLattice object"""

    objectname: str = "vary_element"
    """Name of frameworkObject objectname"""

    objecttype: str = "vary_element"
    """Name of frameworkObject objecttype"""


class ElegantRunControlCommand(ElegantCommandFile):
    """
    Run control for an ELEGANT input file; see `Elegant run control`_

    .. _Elegant run control: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu68.html#x76-750007.59
    """

    objectname: str = "run_control"
    """Name of frameworkObject objectname"""

    objecttype: str = "run_control"
    """Name of frameworkObject objecttype"""

    n_steps: int = 1
    """Number of steps"""

    n_passes: int = 1
    """Number of passes"""


class ElegantTwissOutputCommand(ElegantCommandFile):
    """
    Twiss output for an ELEGANT input file; see `Elegant twiss output`_

    .. _Elegant twiss output: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu82.html#x90-890007.73
    """

    Beam: rbf.Beam
    """Particle distribution"""

    beta_x: float | None = None
    """Initial beta_x; if `None`, take it from `beam`"""

    beta_y: float | None = None
    """Initial beta_y; if `None`, take it from `beam`"""

    alpha_x: float | None = None
    """Initial alpha_x; if `None`, take it from `beam`"""

    alpha_y: float | None = None
    """Initial alpha_y; if `None`, take it from `beam`"""

    eta_x: float | None = None
    """Initial eta_x; if `None`, take it from `beam`"""

    eta_xp: float | None = None
    """Initial eta_xp; if `None`, take it from `beam`"""

    matched: int = 0
    """Flag to indicate whether beam is matched"""

    output_at_each_step: int = 0
    """Flag to indicate whether to output twiss at each step"""

    radiation_integrals: int = 1
    """Calculate radiation integrals"""

    statistics: int = 1
    """Calculate beam statistics"""

    filename: str = "%s.twi"
    """Twiss output file"""

    objectname: str = "twiss_output"
    """Name of object"""

    objecttype: str = "twiss_output"
    """Type of object"""


class ElegantFloorCoordinatesCommand(ElegantCommandFile):
    """
    Floor coordinates for an ELEGANT input file; see `Elegant floor coordinates`_

    .. _Elegant floor coordinates: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu35.html#x43-420007.26
    """
    lattice: FrameworkLattice = None
    """:class:`~simba.Framework_objects.frameworkLattice object"""

    filename: str = "%s.flr"
    """Filename for elegant .flr file"""

    X0: float = 0.0
    """Initial horizontal floor position"""

    Y0: float = 0.0
    """Initial horizontal floor position"""

    Z0: float = 0.0
    """Initial longitudinal floor position"""

    theta0: float = 0.0
    """Initial global rotation"""

    magnet_centers: float = 0
    """Global magnet centre"""

    objectname: str = "floor_coordinates"
    """Name of object"""

    objecttype: str = "floor_coordinates"
    """Type of object"""

    @property
    def x0(self) -> float:
        return self.X0

    @property
    def y0(self) -> float:
        return self.Y0

    @property
    def z0(self) -> float:
        return self.Z0


class ElegantMatrixOutputCommand(ElegantCommandFile):
    """
    Matrix output for an ELEGANT input file; see `Elegant matrix output`_

    .. _Elegant matrix output: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu49.html#x57-560007.40
    """

    full_matrix_only: int = 0
    """A flag indicating that only the matrix of the entire accelerator is to be output."""

    SDDS_output_order: int = 2
    """Matrix output order for the SDDS file"""

    SDDS_output: str = "%s.mat"
    """File to which matrix data is to be written"""

    objectname: str = "matrix_output"
    """Name of object"""

    objecttype: str = "matrix_output"
    """Type of object"""

    @property
    def sdds_output_order(self) -> int:
        return self.SDDS_output_order

    @property
    def sdds_output(self) -> str:
        return self.SDDS_output


class ElegantSddsBeamCommand(ElegantCommandFile):
    """
    SDDS beam input for an ELEGANT input file; see `Elegant sdds beam`_

    .. _Elegant sdds beam: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu72.html#x80-790007.63
    """

    input: str = ""
    """Input filename for ELEGANT"""

    objectname: str = "sdds_beam"
    """Name of object"""

    objecttype: str = "sdds_beam"
    """Type of object"""

    sample_interval: float | int = 1
    """Fraction by which to reduce number of particles"""

    reuse_bunch: int = 1
    """Flag to indicate whether bunch is to be reused"""

    fiducialization_bunch: int = 0
    """Flag to indicate whether bunch is fiducial"""

    center_arrival_time: int = 0
    """Flag to indicate whether to centre arrival time"""


class ElegantTrackCommand(ElegantCommandFile):
    """
    Track command for an ELEGANT input file; see `Elegant track`_

    .. _Elegant track: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu83.html#x91-900007.74
    """

    _DEPRECATED_METHOD_ALIASES = {
        "trackBeam": "track_beam",
    }

    track_beam: bool = True
    """Flag to indicate whether to include the track command"""

    objectname: str = "track"
    """Name of object"""

    objecttype: str = "track"
    """Type of object"""


class ElegantOptimisation(ElegantCommandFile):
    """
    Class for generating input commands for ELEGANT optimisation.
    See `Elegant optimization variable`_ , `Elegant optimization constraint`_ ,
    and `Elegant optimization term`_

    .. _Elegant optimization variable: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu61.html#x69-680007.52
    .. _Elegant optimization constraint: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu55.html#x63-620007.46
    .. _Elegant optimization term: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu60.html#x68-670007.51
    """

    variables: Dict = {}
    """Dictionary of names and variables to be changed"""

    constraints: Dict = {}
    """Dictionary of constraints for the optimization"""

    terms: Dict = {}
    """Dictionary of terms to be optimized"""

    settings: Dict = {}
    """Dictionary of optimization settings"""

    def __init__(self, *args, **kwargs):
        super(ElegantOptimisation, self).__init__(
            *args,
            **kwargs,
        )
        for k, v in list(self.variables.items()):
            self.add_optimisation_variable(k, **v)

    def add_optimisation_variable(
            self,
            name: str,
            item: str=None,
            lower: float=None,
            upper: float=None,
            step: float=None,
            restrict_range: int=None,
    ):
        """
        Add an optimization variable and create the command

        Parameters
        ----------
        name: str
            Element name
        item: str
            Element parameter to be varied
        lower: float
            Lower limit allowed for `item`
        upper: float
            Upper limit allowed for `item`
        step: int
            Specifies grid size for optimization algorithm
        restrict_range: int
            If nonzero, the initial value is forced inside the allowed range
        """
        self.addCommand(
            name=name,
            type="optimization_variable",
            item=item,
            lower_limit=lower,
            upper_limit=upper,
            step_size=step,
            force_inside=restrict_range,
        )

    def add_optimisation_constraint(
            self,
            name: str,
            item: str=None,
            lower: float=None,
            upper: float=None
    ):
        """
        Add an optimization constraint and create the command

        Parameters
        ----------
        name: str
            Element name
        item: str
            Element parameter to be constrained
        lower: float
            Lower limit allowed for `item`
        upper: float
            Upper limit allowed for `item`
        """
        self.addCommand(
            name=name,
            type="optimization_constraint",
            quantity=item,
            lower=lower,
            upper=upper,
        )

    def add_optimisation_term(
            self,
            name: str,
            item: str=None,
            **kwargs,
    ):
        """
        Add an optimization term and create the command

        Parameters
        ----------
        name: str
            Element name
        item: str
            Element parameter to be constrained
        """
        self.addCommand(name=name, type="optimization_term", term=item, **kwargs)


class SddsFile(object):
    """simple class for writing generic column data to a new SDDS file"""

    def __init__(self):
        """initialise an SDDS instance, prepare for writing to file"""
        self.sdds = sdds.SDDS(0)

    def add_column(self, name, data, **kwargs):
        """add a column of floating point numbers to the file"""
        if not isinstance(name, str):
            raise TypeError("Column names must be string types")
        self.sdds.defineColumn(
            name,
            symbol=kwargs["symbol"] if ("symbol" in kwargs) else "",
            units=kwargs["units"] if ("units" in kwargs) else "",
            description=kwargs["description"] if ("description" in kwargs) else "",
            formatString="",
            type=self.sdds.SDDS_DOUBLE,
            fieldLength=0,
        )

        if isinstance(data, (tuple, list, np.ndarray)):
            self.sdds.setColumnValueList(name, list(data), page=1)
        else:
            raise TypeError("Column data must be a list, tuple or array-like type")

    def add_parameter(self, name, data, **kwargs):
        """add a parameter of floating point numbers to the file"""
        if not isinstance(name, str):
            raise TypeError("Parameter names must be string types")
        if "type" in kwargs:
            type = kwargs["type"]
        else:
            type = self.sdds.SDDS_DOUBLE
        self.sdds.defineParameter(
            name,
            symbol=kwargs["symbol"] if ("symbol" in kwargs) else "",
            units=kwargs["units"] if ("units" in kwargs) else "",
            description=kwargs["description"] if ("description" in kwargs) else "",
            formatString="",
            type=type,
            fixedValue="",
        )

        if isinstance(data, (tuple, list, np.ndarray)):
            self.sdds.setParameterValueList(name, list(data))
        else:
            raise TypeError("Parameter data must be a list, tuple or array-like type")

    def save(self, fname):
        """save the sdds data structure to file"""
        if not isinstance(fname, str):
            raise TypeError("SDDS file name must be a string!")
        self.sdds.save(fname)


from simba._compat import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(
    __name__,
    globals(),
    {
        "elegantLattice": "ElegantLattice",
        "elegantOptimisation": "ElegantOptimisation",
        "elegant_error_elements_command": "ElegantErrorElementsCommand",
        "elegant_floor_coordinates_command": "ElegantFloorCoordinatesCommand",
        "elegant_global_settings_command": "ElegantGlobalSettingsCommand",
        "elegant_matrix_output_command": "ElegantMatrixOutputCommand",
        "elegant_run_control_command": "ElegantRunControlCommand",
        "elegant_run_setup_command": "ElegantRunSetupCommand",
        "elegant_scan_elements_command": "ElegantScanElementsCommand",
        "elegant_sdds_beam_command": "ElegantSddsBeamCommand",
        "elegant_track_command": "ElegantTrackCommand",
        "elegant_twiss_output_command": "ElegantTwissOutputCommand",
        "sddsFile": "SddsFile",
    },
)
