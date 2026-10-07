"""
SIMBA ELEGANT Module

Various objects and functions to handle ELEGANT lattices and commands. See `Elegant manual`_ for more details.

    .. _Elegant manual: https://ops.aps.anl.gov/manuals/elegant_latest/elegant.html

Classes:
    - :class:`~simba.Codes.Elegant.Elegant.elegantLattice`: The ELEGANT lattice object, used for
      converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
      :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
      the lattice suitable for ELEGANT input and lattice files.

    - :class:`~simba.Codes.Elegant.Elegant.elegantCommandFile`: Base class for defining
      commands in an ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_global_settings_command`: Class for defining the
      &global_settings portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_run_setup_command`: Class for defining the
      &run_setup portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_error_elements_command`: Class for defining the
      &error_elements portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_error_elements_command`: Class for defining the
      &error_elements portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_scan_elements_command`: Class for defining the
      &scan_elements portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_run_control_command`: Class for defining the
      &run_control portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_twiss_output_command`: Class for defining the
      &twiss_output portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_floor_coordinates_command`: Class for defining the
      &floor_coordinates portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_matrix_output_command`: Class for defining the
      &matrix_output portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_sdds_beam_command`: Class for defining the
      &sdds_beam portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_track_command`: Class for defining the
      &track portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegant_track_command`: Class for defining the
      &track portion of the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.elegantOptimisation`: Class for defining the
      commands for optimization in the ELEGANT input file.

    - :class:`~simba.Codes.Elegant.Elegant.sddsFile`: Class for creating, modifying and
      saving SDDS files.
"""

import math
import os
import subprocess
from copy import copy
from pathlib import Path
import numpy as np
from warnings import warn
try:
    import sdds
except Exception:
    print("No SDDS available!")
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import ClassVar
from ...Framework_objects import (
    frameworkLattice,
    frameworkCommand,
    elementkeywords,
    keyword_conversion_rules_elegant,
)
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from ...Modules.SDDSFile import SDDS_Types
from typing import Dict, List, Any
from laura.models.diagnostic import DiagnosticElement
from laura.translator.converters.converter import translate_elements
from laura.translator.utils.functions import sanitize_string


def elegant_element_name(name: str) -> str:
    """The name ELEGANT knows an element by."""
    return sanitize_string(name).upper()


class elegantLattice(frameworkLattice):
    """
    Class for defining the ELEGANT lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
    the lattice suitable for an ELEGANT input file.
    """

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ELEGANT into the SIMBA generic 
    :class:`~simba.Modules.Beams.beam` object and writing files"""

    code: str = "elegant"
    """String indicating the lattice object type"""

    electrons_only: ClassVar[bool] = True
    """elegant can track other species with ``&change_particle``, but simba
    does not write it, so anything else would be tracked as electrons."""

    supports_turns: ClassVar[bool] = True
    """``run_control``'s ``n_passes``."""

    supports_radiation: ClassVar[bool] = True
    """``SYNCH_RAD`` and ``ISR`` on the bends, which LAURA writes from the
    elements' own ``sr_enable``/``isr_enable``."""

    radiates_by_default: ClassVar[bool] = True
    """Flag to state that Elegant radiates by default (based on LAURA)."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """``&find_aperture``, elegant's own search."""

    supports_frequency_map: ClassVar[bool] = True
    """``&frequency_map``, which also writes the tune *diffusion* between
    the two halves of the run."""

    supports_periodic: ClassVar[bool] = True
    """``twiss_output``'s ``matched = 1``."""

    supports_programs: ClassVar[bool] = True
    """For a ``BUMPER``/``MBUMPER``, and only for one -- see
    :meth:`program_commands`."""

    supports_ramp: ClassVar[bool] = True
    """By a ``RAMPP`` element at the head of the line; see :meth:`insert_ramp`."""

    ramp_element: ClassVar[str] = "SIMBA_RAMP"
    """Name of the ``RAMPP`` simba puts at the head of a ramped line."""

    native_rf: ClassVar[str] = "fixed"
    """Measured: an ``RFCA`` gains the same, turn by turn, ramped or not --
    on absolute time, a free-running oscillator. ``follow`` is
    ``&modulate_elements`` on its ``PHASE``; see :meth:`rf_phase_commands`."""

    otm_convention: ClassVar[str] = "x, xp, y, yp, s, delta"
    """One-turn map convention."""

    otm_longitudinal_sign: ClassVar[int] = -1
    """One-turn map longitudinal sign (i.e. R56; opposite to MAD-X, Xsuite)."""

    otm_longitudinal_scale: ClassVar[int | None] = None
    """Not a rescale. ``s`` is geometric path length, so a drift measured
    ``R56 = 0`` as opposed to ``L/(beta0.gamma0)**2``."""

    allow_negative_drifts: bool = False
    """Flag to indicate whether negative drifts are allowed"""

    particle_definition: str | None = None
    """String representation of the initial particle distribution"""

    bunch_charge: float | None = None
    """Bunch charge"""

    q: Any = None
    """:class:`~simba.Elements.charge.charge` object"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    betax: float | None = None
    """Initial beta_x for matching"""

    betay: float | None = None
    """Initial beta_y for matching"""

    alphax: float | None = None
    """Initial alpha_x for matching"""

    alphay: float | None = None
    """Initial alpha_y for matching"""

    commandFiles: Dict = {}
    """Dictionary of :class:`~simba.Codes.Elegant.Elegant.elegantCommandFile`
    objects for writing to the ELEGANT input file"""

    final_screen: Any = None
    """:class:`simba.Elements.screen.screen` object at the end of the line"""

    commandFilesOrder: List = []
    """Order in which commands are to be written in the ELEGANT input file"""

    program_elements: ClassVar[tuple] = ("bumper", "mbumper")
    """The elegant element types a device program can be written onto.
    The only ones with a ``WAVEFORM``, and so the only ones whose strength
    can change from pass to pass.
    """

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into an ELEGANT-compatible format; see
        :attr:`~simba.Framework_objects.frameworkLattice.elementObjects`.

        Returns
        -------
        str
            The lattice represented as a string compatible with ELEGANT
        """
        if self.bunch_charge is not None:
            q = abs(self.bunch_charge)
        else:
            q = abs(self.global_parameters["beam"].Q)
        return self.insert_ramp(self.section.to_elegant(charge=q))

    def write_ramp_waveform(self) -> str:
        """
        Write :meth:`ramp` as a ``RAMPP`` waveform beside the lattice.

        ``RAMPP`` reads its waveform at the **bunch's mean arrival time**;
        the shared
        :meth:`~simba.Framework_objects.frameworkLattice.ramp_clock` plus the
        input bunch's mean ``t``. It is a staircase, flat for a quarter turn
        either side of each turn's start.

        The ratio is to ``run_setup``'s ``p_central``.

        Returns
        -------
        str
            Basename of the SDDS ``WAVEFORM`` file
        """
        from ...Modules.SDDSFile import SDDSFile

        rest_energy = self.rest_energy
        p0c = self.ramp.p0c_per_pass(self.turns, rest_energy)
        p_central = self.reference_p0c
        starts = np.asarray(self.ramp_clock().times) + self.reference_t0
        times, ratios = self.pass_staircase(starts, p0c / p_central)
        basename = f"{self.objectname}_ramp.sdds"
        sdds = SDDSFile(index=1, ascii=True)
        sdds.add_columns(
            ["t", "ratio"],
            [times, ratios],
            [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE],
            ["s", ""],
            ["", ""],
        )
        path = os.path.join(self.global_parameters["master_subdir"], basename)
        sdds.write_file(path)
        self.files.append(path)
        return basename

    def rf_phase_commands(self) -> dict:
        """
        ``&modulate_elements`` commands moving each cavity's ``PHASE`` pass by
        pass, so that elegant's fixed oscillators run as :attr:`rf_mode` asks;
        see :meth:`~simba.Framework_objects.frameworkLattice.rf_phase_corrections`.

        * ``modulate_elements`` reads its table at the **bunch's** time, as
          ``RAMPP`` does;
        * a ``PHASE`` larger by the correction is the phase the reference
          sees moved by it, as for ``RFCA``'s ``PHASE`` itself.

        Returns
        -------
        dict
            Command files to insert, keyed for ``commandFiles``. Empty unless
            a correction is needed
        """
        from ...Modules.SDDSFile import SDDSFile

        corrections = self.rf_phase_corrections()
        if not corrections:
            return {}
        starts = self.pass_start_times()
        commands = {}
        for element, correction in corrections.items():
            name = sanitize_string(element)
            times, degrees = self.pass_staircase(starts, self.rf_phase_shifts(correction))
            basename = f"{name}_rf_phase.sdds"
            sdds = SDDSFile(index=1, ascii=True)
            sdds.add_columns(
                ["t", "phase"],
                [times, degrees],
                [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE],
                ["s", "deg"],
                ["", ""],
            )
            path = os.path.join(self.global_parameters["master_subdir"], basename)
            sdds.write_file(path)
            self.files.append(path)
            key = f"rf_phase_{name}"
            command = elegantCommandFile(objectname=key, objecttype="modulate_elements")
            for prop, value in (
                ("name", elegant_element_name(element)),
                ("item", "PHASE"),
                ("filename", f'"{basename}"'),
                ("time_column", '"t"'),
                ("amplitude_column", '"phase"'),
                ("differential", 1),
                ("multiplicative", 0),
                ("refresh_matrix", 1),
            ):
                command.add_property(prop, value)
            commands[key] = command
        return commands

    def insert_ramp(self, lattice: str) -> str:
        """
        Put a ``RAMPP`` at the head of the beamline, if :meth:`ramp` asks for one.

        Parameters
        ----------
        lattice: str
            The lattice file LAURA wrote

        Returns
        -------
        str
            `lattice`, with the ``RAMPP`` defined and first in the line
        """
        if not self.ramped:
            return lattice
        head = f"{self.objectname}: LINE = ("
        if head not in lattice:
            warn(
                f"Line '{self.objectname}' is ramped, but simba could not find "
                "its LINE in the elegant lattice to put the RAMPP in. The run "
                "is tracked at a fixed energy."
            )
            return lattice
        waveform = self.write_ramp_waveform()
        element = f'{self.ramp_element}: RAMPP, WAVEFORM="{waveform}=t+ratio"\n'
        return lattice.replace(head, f"{element}{head}{self.ramp_element}, ", 1)

    def processRunSettings(self) -> tuple:
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
        nruns = self.runSettings.nruns
        seed = self.runSettings.seed
        elementErrors = (
            None
            if (self.runSettings.elementErrors is None)
            else self.processElementErrors(self.runSettings.elementErrors)
        )
        elementScan = (
            None
            if (self.runSettings.elementScan is None)
            else self.processElementScan(self.runSettings.elementScan, nruns)
        )
        return nruns, seed, elementErrors, elementScan

    def processElementErrors(self, elementErrors: Dict) -> Dict:
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
            # identify element names with wildcard characters
            wildcard = "*" in ele

            # raise errors for non-wildcarded element names that don't exist in the global lattice
            if (ele not in self.allElements) and (not wildcard):
                raise KeyError(
                    "Lattice element %s does not exist in the current lattice"
                    % str(ele)
                )

            # check if the lattice element (or a wildcard match) exist in the local lattice section
            # fetch the element type (or the types of matching elements, if using a wildcard name)
            element_exists = False
            if (ele in self.elements) and not wildcard:
                element_exists = True
                element_types = [self.elementObjects[ele].hardware_type]
            elif wildcard:
                element_matches = [
                    x for x in self.elements if (ele.replace("*", "") in x)
                ]
                if len(element_matches) != 0:
                    element_exists = True
                    element_types = [
                        self.elementObjects[x].hardware_type for x in element_matches
                    ]

            # if the element exists in the local lattice, do processing
            if element_exists:
                output[ele] = {}

                # check that all matching elements have the same type
                ele_type = str(element_types[0])
                has_expected_type = [(x == ele_type) for x in element_types]
                if not all(has_expected_type):
                    raise TypeError(
                        "All lattice elements matching a wilcarded element name must have the same type"
                    )

                # check error definition associated with each of the element parameters
                # for example, an element corresponding to an RF cavity might have parameters 'amplitude' and 'phase'
                for param in error_defn:
                    # check that the current element type has this parameter
                    if param not in elementkeywords[ele_type]["keywords"]:
                        raise KeyError(
                            "Element type %s has no associated keyword %s"
                            % (str(ele_type), str(param))
                        )

                    # check for keyword conversions between simframe and elegant
                    # for example, in simframe the elegant parameter 'voltage' for RF cavities is called 'amplitude'
                    conversions = keyword_conversion_rules_elegant[ele_type]
                    keyword = conversions[param] if (param in conversions) else param
                    output[ele][keyword] = copy(default_err)

                    # fill in the define error parameters
                    for k in default_err:
                        if k in error_defn[param]:
                            output[ele][keyword][k] = error_defn[param][k]

                    # bind errors across wildcarded elements
                    if wildcard:
                        output[ele][keyword]["bind"] = 1
                        output[ele][keyword]["bind_across_names"] = 1
        return output

    def processElementScan(self, elementScan: Dict, nsteps: int) -> Dict | None:
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
        # extract the name of the beamline element, and the parameter to scan
        ele, param = elementScan["name"], elementScan["item"]

        # raise errors for element names that don't exist anywhere in the global lattice
        if ele not in self.allElements:
            raise KeyError(
                "Lattice element %s does not exist in the current lattice" % str(ele)
            )

        # check if the lattice element exists in the local lattice section and fetch the element type
        element_exists = ele in self.elements
        if element_exists:
            ele_type = self.elementObjects[ele].hardware_type

            # check that the element type has the parameter corresponding to the scan variable
            if param not in elementkeywords[ele_type]["keywords"]:
                raise KeyError(
                    "Element type %s has no associated parameter %s"
                    % (str(ele_type), str(param))
                )

            # check for keyword conversions between simframe and elegant
            conversions = keyword_conversion_rules_elegant[ele_type]
            keyword = conversions[param] if (param in conversions) else param

            # build the scan value array
            scan_values = np.linspace(
                elementScan["min"], elementScan["max"], int(nsteps) - 1
            )

            # the first scan step is always the baseline simulation, for fiducialization
            multiplicative = elementScan["multiplicative"]
            if multiplicative:
                if 1.0 not in list(scan_values):
                    scan_values = [1.0] + list(scan_values)
            else:
                if 0.0 not in list(scan_values):
                    scan_values = [0.0] + list(scan_values)

            # build the SDDS file with the scan values
            scan_fname = "%s-%s.sdds" % (ele, param)
            scanSDDS = sddsFile()
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
        :func:`~simba.Codes.Elegant.Elegant.writeElements` and
        based on the output of :func:`~simba.Codes.Elegant.Elegant.createCommandFiles`.
        """
        lattice_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".lte"
        )
        saveFile(lattice_file, self.writeElements())
        self.files.append(lattice_file)
        # try:
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".ele"
        )
        saveFile(command_file, "", "w")
        self.files.append(command_file)
        if len(self.commandFilesOrder) > 0:
            for cfileid in self.commandFilesOrder:
                if cfileid in self.commandFiles:
                    cfile = self.commandFiles[cfileid]
                    saveFile(command_file, cfile.write_Elegant(), "a")
                    self.files.append(command_file)
        else:
            warn("commandFilesOrder length is zero; run createCommandFiles first")
        # except Exception:
        #     passastrabeamfilename

    def _elegant_type(self, name: str) -> str:
        """
        The elegant element type LAURA writes `name` as, lowercased.

        Parameters
        ----------
        name: str
            Element name

        Returns
        -------
        str
            Name of ELEGANT element
        """
        element = self.elements.get(name)
        if element is None:
            return ""
        translator = translate_elements([element]).get(name)
        if translator is None:
            return ""
        return translator._convert_type_elegant(translator.hardware_type).lower()

    def write_program_waveform(self, program) -> str:
        """
        Write one program's ``(t, factor)`` table beside the lattice.

        The times are seconds from the firing pass, which is where
        elegant measures a ``WAVEFORM`` from; see
        :meth:`~simba.Modules.DeviceProgram.DeviceProgram.linear_knots`.

        Parameters
        ----------
        program: :class:`~simba.Modules.DeviceProgram.DeviceProgram`
            The program to write

        Returns
        -------
        str
            Basename of the SDDS ``WAVEFORM`` file
        """
        from ...Modules.SDDSFile import SDDSFile

        times, factors = program.factor_knots(
            self.revolution_period,
            origin_turn=program.first_turn,
            clock=self.ramp_clock(),
        )
        basename = f"{sanitize_string(program.element)}_program.sdds"
        sdds = SDDSFile(index=1, ascii=True)
        sdds.add_columns(
            ["t", "factor"],
            [times, factors],
            [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE],
            ["s", ""],
            ["", ""],
        )
        path = os.path.join(self.global_parameters["master_subdir"], basename)
        sdds.write_file(path)
        self.files.append(path)
        return basename

    def program_commands(self) -> dict:
        """
        ``&alter_elements`` commands putting each program onto its element.
        Three per program -- the peak strength, the firing pass and the
        waveform.

        ``FIRE_ON_PASS`` is **zero-based**, measured: ``FIRE_ON_PASS=2``
        first kicks on the third pass.

        Returns
        -------
        dict
            Command files to insert, keyed for ``commandFiles``. Empty if
            nothing is programmed
        """
        commands = {}
        for program in self.programs:
            etype = self._elegant_type(program.element)
            if etype not in self.program_elements:
                warn(
                    f"Line '{self.objectname}' programs '{program.element}', "
                    f"which elegant writes as a {etype or 'missing element'}. "
                    "Only a BUMPER or MBUMPER carries a WAVEFORM"
                )
                continue
            name = sanitize_string(program.element)
            waveform = self.write_program_waveform(program)
            for item, key, value in (
                ("ANGLE", "value", program.peak),
                ("FIRE_ON_PASS", "value", program.first_turn - 1),
                ("WAVEFORM", "string_value", f'"{waveform}=t+factor"'),
            ):
                key_name = f"program_{name}_{item.lower()}"
                command = elegantCommandFile(
                    objectname=key_name,
                    objecttype="alter_elements",
                    item=item,
                    **{key: value},
                )
                command.add_property("name", elegant_element_name(program.element))
                commands[key_name] = command
        return commands

    def createCommandFiles(self) -> None:
        """
        Create the :class:`~simba.Codes.Elegant.elegantCommandFile` objects
        based on the run settings, lattice and beam parameters, including scans of elements,
        if defined.

        Updates :attr:`~simba.Codes.Elegant.Elegant.commandFiles` and
        :attr:`~simba.Codes.Elegant.Elegant.commandFilesOrder`
        """
        if not isinstance(self.commandFiles, dict) or self.commandFiles == {}:
            # print('createCommandFiles is creating new command files!')
            # print('processRunSettings')
            nruns, seed, elementErrors, elementScan = self.processRunSettings()
            self.commandFiles["global_settings"] = elegant_global_settings_command(
                # lattice=self,
                warning_limit=0
            )
            # print('run_setup')
            self.commandFiles["run_setup"] = elegant_run_setup_command(
                lattice=self.objectname + ".lte",
                p_central=self.reference_p0c / self.rest_energy,
                seed=seed,
                # losses="%s.loss",
                s_start=self.entrance_s,
                use_beamline=self.objectname,
                **({"always_change_p0": 0} if self.fixed_reference else {}),
            )

            for key, command in self.program_commands().items():
                self.commandFiles[key] = command
            for key, command in self.rf_phase_commands().items():
                self.commandFiles[key] = command

            # print('generate commands for monte carlo jitter runs')
            if elementErrors is not None:
                self.commandFiles["run_control"] = elegant_run_control_command(
                    # lattice=self,
                    n_steps=nruns,
                    n_passes=self.turns,
                    reset_rf_for_each_step=0,
                    first_is_fiducial=1,
                )
                self.commandFiles["error_elements"] = elegant_error_elements_command(
                    lattice=self, elementErrors=elementErrors, nruns=nruns
                )
                for e in elementErrors:
                    for item in elementErrors[e]:
                        self.commandFiles["error_element_" + e + "_" + item] = (
                            elegantCommandFile(
                                objectname="error_element",
                                objecttype="error_element",
                                name=elegant_element_name(e),
                                item=item,
                                allow_missing_elements=1,
                                **elementErrors[e][item],
                            )
                        )
            elif elementScan is not None:
                # print('generate commands for parameter scans without fiducialisation (i.e. jitter scans)')
                self.commandFiles["run_control"] = elegant_run_control_command(
                    # lattice=self,
                    n_steps=nruns - 1,
                    n_passes=self.turns,
                    n_indices=1,
                    reset_rf_for_each_step=0,
                    first_is_fiducial=1,
                )
                self.commandFiles["scan_elements"] = elegant_scan_elements_command(
                    # lattice=self,
                    name=elegant_element_name(elementScan["name"]),
                    item=elementScan["item"],
                    enumeration_file=elementScan["enumeration_file"],
                    enumeration_column=elementScan["enumeration_column"],
                    multiplicative=int(elementScan["multiplicative"]),
                    nruns=nruns,
                )
            else:
                # print('run_control for standard runs with no jitter')
                self.commandFiles["run_control"] = elegant_run_control_command(
                    # lattice=self,
                    n_steps=1, n_passes=self.turns
                )

            # print('twiss_output')
            if not self.lsc_in_use:
                beamtwiss = self.global_parameters["beam"].twiss
                self.commandFiles["twiss_output"] = elegant_twiss_output_command(
                    # lattice=self,
                    beam=self.global_parameters["beam"],
                    matched=int(self.periodic),
                    beta_x=None if self.periodic else beamtwiss.beta_x_corrected,
                    beta_y=None if self.periodic else beamtwiss.beta_y_corrected,
                    alpha_x=None if self.periodic else beamtwiss.alpha_x_corrected,
                    alpha_y=None if self.periodic else beamtwiss.alpha_y_corrected,
                    # eta_x=self.global_parameters["beam"].twiss.eta_x,
                    # eta_xp=self.global_parameters["beam"].twiss.eta_xp,
                )
            # print('floor_coordinates')
            self.commandFiles["floor_coordinates"] = elegant_floor_coordinates_command(
                # lattice=self,
                X0=self.startObject.physical.start.x,
                Y0=self.startObject.physical.start.y,
                Z0=self.startObject.physical.start.z,
            )
            # print('matrix_output')
            if not self.lsc_in_use:
                self.commandFiles["matrix_output"] = elegant_matrix_output_command(
                    # lattice=self,
                )
            else:
                warn(
                    f"{self.objectname}: no twiss or matrix output -- elegant builds those "
                    "matrices before any beam exists, and an element doing LSC has no "
                    "matrix without a charge ('No charge defined for LSC'). Tracking is "
                    "unaffected; set lsc_enable = False on the lattice to get the optics."
                )
            # print('sdds_beam')
            self.commandFiles["sdds_beam"] = elegant_sdds_beam_command(
                lattice=self,
                input=self.objectname + "_input.sdds",
                reuse_bunch=1,
                fiducialization_bunch=0,
                center_arrival_time=0,
            )
            # print('track')
            self.commandFiles["track"] = elegant_track_command(
                # lattice=self,
                trackBeam=self.trackBeam
            )
            self.commandFilesOrder = list(
                self.commandFiles.keys()
            )  # ['global_settings', 'run_setup', 'error_elements', 'scan_elements', 'run_control', 'twiss', 'sdds_beam', 'track']

    def preProcess(self) -> None:
        """
        Prepare the input distribution for ELEGANT based on the `prefix` in the settings
        file for this lattice section, and create the ELEGANT command files.
        """
        super().preProcess()
        prefix = self.get_prefix()
        self.load_input_beam(prefix, self.particle_definition)
        if self.trackBeam:
            self.hdf5_to_sdds()
        self.createCommandFiles()

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
            toffset=-1 * np.mean(self.global_parameters["beam"].Particles.t),
            **kwargs,
        )
        # except Exception as e:
        #     print(f"Screen error {scr.name}, {e}")
        #     return None

    def _ring_study_deck(self, suffix: str, *commands) -> str | None:
        """
        Write a second, focused ``.ele`` for a ring study and run it.

        It also gets its own directory: LAURA writes each screen's
        ``WATCH`` filename literally as ``./<name>.SDDS``, so a second run
        over the same lattice silently overwrites every screen file the
        tracking run produced.

        Returns
        -------
        str | None
            Stem of the run, or None if elegant was not run.
        """
        parent = self.global_parameters["master_subdir"]
        lattice = os.path.join(parent, f"{self.objectname}.lte")
        if not os.path.isfile(lattice):
            saveFile(lattice, self.writeElements())
        stem = f"{self.objectname}_{suffix}"
        workdir = os.path.join(parent, stem)
        os.makedirs(workdir, exist_ok=True)
        setup = elegant_run_setup_command(
            lattice=os.path.join("..", f"{self.objectname}.lte"),
            p_central=self.reference_p0c / self.rest_energy,
            use_beamline=self.objectname,
            **({"always_change_p0": 0} if self.fixed_reference else {}),
        )
        twiss = elegant_twiss_output_command(
            beam=self.global_parameters["beam"], matched=1
        )
        deck = os.path.join(workdir, f"{stem}.ele")
        saveFile(deck, "", "w")
        for part in (setup, twiss, *commands):
            saveFile(deck, part.write_Elegant(), "a")
        self.files.append(deck)
        subdir = workdir
        command_line = self.executables.build_command(
            self.executables[self.code] + [f"{stem}.ele"], os.path.abspath(subdir)
        )
        try:
            subprocess.run(
                command_line, cwd=os.path.abspath(subdir), check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
            warn(f"elegant {suffix} run failed for {self.objectname}: {error}")
            return None
        return stem

    def track_reference_particle(self) -> dict:
        """
        One particle, recorded every turn, from a screen's ``WATCH`` file.
        Uses ``&bunched_beam`` with ``n_particles_per_bunch = 1``.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py`` per turn, or ``{}`` if elegant wrote
            no watch file.
        """
        screens = self.screens_and_markers_and_bpms
        if not screens:
            warn(
                f"Line '{self.objectname}' has no screen, so elegant has "
                "nowhere to record a trajectory."
            )
            return {}
        nudge = float((self.da_settings or {}).get("x_max", 1e-3)) / 100.0
        stem = self._ring_study_deck(
            "reference",
            elegant_run_control_command(n_steps=1, n_passes=self.turns),
            elegant_bunched_beam_command(
                n_particles_per_bunch=1,
                centroid_x=nudge,
                centroid_y=nudge,
            ),
            elegant_track_command(),
        )
        if stem is None:
            return {}
        path = (
            Path(self.global_parameters["master_subdir"])
            / stem
            / f"{screens[0].name}.SDDS"
        )
        if not path.is_file():
            warn(f"elegant wrote no watch file for {self.objectname}")
            return {}
        from ...Modules.SDDSFile import SDDSFile

        trajectory = {name: [] for name in ("x", "px", "y", "py")}
        columns = {"x": "x", "px": "xp", "y": "y", "py": "yp"}
        try:
            pages = rbf.sdds.count_SDDS_pages(str(path))
            for page in range(pages):
                sdds = SDDSFile(index=1)
                sdds.read_file(str(path), page=page)
                data = sdds.columns()
                for name, column in columns.items():
                    trajectory[name].append(float(np.atleast_1d(data[column].data)[0]))
        except Exception as error:
            warn(f"elegant watch file unreadable for {self.objectname}: {error}")
            return {}
        return {name: np.asarray(values) for name, values in trajectory.items()}

    def _da_bounds(self) -> dict:
        """``xmin``/``xmax``/``ymin``/``ymax``/``nx``/``ny`` from
        :meth:`da_grid`, which elegant takes as a box rather than a list."""
        xs, ys = self.da_grid()
        return {
            "xmin": -float(xs[-1]),
            "xmax": float(xs[-1]),
            "ymin": 0.0,
            "ymax": float(ys[-1]),
            "nx": int(len(xs)),
            "ny": int(len(ys)),
        }

    def run_dynamic_aperture(self) -> list:
        """
        ``&find_aperture``: elegant searches for the boundary itself.
        A boundary, not a survival grid.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` along the aperture boundary.
        """
        stem = self._ring_study_deck(
            "aperture",
            elegant_find_aperture_command(
                output=f"{self.objectname}_aperture.aper", **self._da_bounds()
            ),
        )
        if stem is None:
            return []
        path = Path(self.global_parameters["master_subdir"]) / stem / f"{stem}.aper"
        if not path.is_file():
            warn(f"elegant wrote no aperture file for {self.objectname}")
            return []
        try:
            from ...Modules.SDDSFile import SDDSFile

            sdds = SDDSFile(index=1)
            sdds.read_file(str(path))
            columns = sdds.columns()
            xs = np.atleast_1d(columns["x"].data)
            ys = np.atleast_1d(columns["y"].data)
        except Exception as error:
            warn(f"elegant aperture file unreadable for {self.objectname}: {error}")
            return []
        self.dynamic_aperture = [
            (float(x), float(y), self.turns) for x, y in zip(xs, ys)
        ]
        return self.dynamic_aperture

    def run_frequency_map(self) -> list:
        """
        ``&frequency_map``, which computes the tunes itself.
        This does not go through
        :func:`~simba.Modules.Matrices.tune_from_trajectory`, so it is an
        independent check on that function rather than a repeat of it.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y)`` per grid point.
        """
        bounds = self._da_bounds()
        stem = self._ring_study_deck(
            "fma",
            elegant_frequency_map_command(
                output=f"{self.objectname}_fma.fma", **bounds
            ),
        )
        if stem is None:
            return []
        path = Path(self.global_parameters["master_subdir"]) / stem / f"{stem}.fma"
        if not path.is_file():
            warn(f"elegant wrote no frequency map for {self.objectname}")
            return []
        try:
            from ...Modules.SDDSFile import SDDSFile

            sdds = SDDSFile(index=1)
            sdds.read_file(str(path))
            columns = sdds.columns()
            xs = np.atleast_1d(columns["x"].data)
            ys = np.atleast_1d(columns["y"].data)
            qxs = np.atleast_1d(columns["nux"].data)
            qys = np.atleast_1d(columns["nuy"].data)
            # `include_changes` writes the tune drift between the halves of
            # the run, which is the diffusion the other codes report.
            dxs = np.atleast_1d(columns["dnux"].data) if "dnux" in columns else None
            dys = np.atleast_1d(columns["dnuy"].data) if "dnuy" in columns else None
        except Exception as error:
            warn(f"elegant frequency map unreadable for {self.objectname}: {error}")
            return []
        self.frequency_map = []
        for index, (x, y, qx, qy) in enumerate(zip(xs, ys, qxs, qys)):
            if dxs is None or dys is None:
                diffusion = float("nan")
            else:
                diffusion = math.log10(
                    max(math.hypot(float(dxs[index]), float(dys[index])), 1e-16)
                )
            self.frequency_map.append(
                (float(x), float(y), float(qx), float(qy), diffusion)
            )
        return self.frequency_map

    def read_optics_summary(self) -> dict:
        """elegant writes the lot into the ``%s.twi`` parameters;
        with ``matched = 1`` they are the ring's."""
        path = Path(self.global_parameters["master_subdir"]) / f"{self.objectname}.twi"
        if not path.is_file():
            return {}
        try:
            from ...Modules.SDDSFile import SDDSFile

            sdds = SDDSFile(index=1)
            sdds.read_file(str(path))
            params = sdds.parameters()
        except Exception as error:
            warn(f"elegant twiss parameters unreadable for {self.objectname}: {error}")
            return {}
        keys = {
            "tune_x_total": "nux",
            "tune_y_total": "nuy",
            "chromaticity_x": "dnux/dp",
            "chromaticity_y": "dnuy/dp",
        }
        summary = {}
        for name, key in keys.items():
            if key in params:
                summary[name] = float(np.atleast_1d(params[key].data)[-1])
        return summary

    def read_one_turn_map(self):
        """
        The last cumulative matrix in elegant's ``%s.mat``.

        ``matrix_output`` is written on every run already.
        The matrices are cumulative from the start of the
        line, so the one covering the whole line -- the one turn -- is the
        last.

        Returns
        -------
        numpy.ndarray | None
            The 6x6 map, or None if elegant wrote no matrix file (it writes
            none when LSC is in use; see :func:`write`).
        """
        from ...Modules.Matrices import matrices

        path = Path(self.global_parameters["master_subdir"]) / f"{self.objectname}.mat"
        if not path.is_file():
            return None
        matrix = matrices()
        matrix.load(str(path), reset=True, cumulative=True)
        cumulative = matrix.cumulativeR()
        if not len(cumulative) or not len(cumulative[0]):
            return None
        return np.asarray(cumulative[0][-1], dtype=float)

    def postProcess(self) -> None:
        """
        PostProcess the simulation results, i.e. gather the screens and markers
        and write their outputs to HDF5.

        :attr:`~simba.Codes.Elegant.Elegant.commandFiles` is also cleared
        """
        super().postProcess()
        if self.trackBeam:
            for i, s in enumerate(self.screens_and_markers_and_bpms):
                self.sdds_to_hdf5(
                    s,
                    toffset=-1 * np.mean(self.global_parameters["beam"].Particles.t),
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
                    toffset=-1 * np.mean(self.global_parameters["beam"].Particles.t),
                    ref_index=self.ref_idx,
                )
        #         self.screen_threaded_function.scatter(
        #             self.final_screen,
        #             len(self.screens_and_markers_and_bpms),
        #             ref_index=self.ref_idx
        #         )
        # self.screen_threaded_function.gather()
        self.commandFiles = {}

    def hdf5_to_sdds(self, write: bool = True) -> None:
        """
        Convert the HDF5 beam input file to an SDDS file, and create a
        :class:`~simba.Elements.charge.charge` object as the first element
        """
        sddsbeamfilename = self.objectname + "_input.sdds"
        if write:
            rbf.sdds.write_SDDS_file(
                self.global_parameters["beam"],
                self.global_parameters["master_subdir"] + "/" + sddsbeamfilename,
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
        rootname = f"{self.global_parameters['master_subdir']}/{screen.name}"
        elegantbeamfilename = f"{rootname}.SDDS"
        if not os.path.isfile(elegantbeamfilename):
            if screen.hardware_type != "Beam_Position_Monitor":
                warn(f"{screen.name}: elegant wrote no beam file, nothing to convert")
            return
        xyzoffset = [0.0, 0.0, self.elementObjects[screen.name].physical.s]
        wanted = self.output_turns()
        if self.turns > 1:
            npages = rbf.sdds.count_SDDS_pages(elegantbeamfilename)
            pages = [
                (min(data_turn, npages) - 1, name_turn)
                for data_turn, name_turn in wanted
            ]
        else:
            pages = [(-1, None)]
        for page, turn in pages:
            if not self.writes_output(screen.name, turn):
                continue
            beam = rbf.beam()
            rbf.sdds.read_SDDS_beam_file(
                beam,
                elegantbeamfilename,
                page=page,
                xyzoffset=xyzoffset,
                ref_index=ref_index,
            )
            beam.turn = self.beam_turn(turn)
            HDF5filename = (
                f"{self.global_parameters['master_subdir']}/"
                f"{self.output_basename(screen.name, turn=turn)}.openpmd.hdf5"
            )
            rbf.openpmd.write_openpmd_beam_file(beam, HDF5filename)
        if self.global_parameters["delete_tracking_files"]:
            os.remove(elegantbeamfilename)

    def run(self):
        """Run the code with input 'filename'"""
        command = self.executables[self.code] + [self.objectname + ".ele"]
        workdir = os.path.abspath(self.global_parameters["master_subdir"])
        command = self.executables.build_command(command, workdir)
        logfile = os.path.join(workdir, self.objectname + ".log")
        if self.remote_setup:
            super().run_remote()
        elif not os.name == "nt":
            my_env = {**os.environ}
            if self.global_parameters["simcodes_location"] is not None:
                rpn_defns = os.path.join(
                    os.path.abspath(self.global_parameters["simcodes_location"]),
                    "Elegant",
                    "defns_linux.rpn",
                )
                if os.path.isfile(rpn_defns):
                    my_env["RPN_DEFNS"] = rpn_defns
            self.run_command(
                command,
                logfile,
                cwd=self.global_parameters["master_subdir"],
                env=my_env,
            )
        else:
            code_string = " ".join(self.executables[self.code]).lower()
            # a container command (docker/apptainer/wsl) stays POSIX-style; only a
            # native Windows executable needs its paths in native backslash form
            is_container_command = command and command[0] in ("docker", "apptainer", "wsl")
            if "pelegant" in code_string:
                if self.global_parameters["simcodes_location"] is not None:
                    rpn_defns = (
                        os.path.abspath(self.global_parameters["simcodes_location"])
                        + "/Elegant/defns.rpn"
                    )
                    if not is_container_command:
                        rpn_defns = rpn_defns.replace("/", "\\")
                    command = (
                        [command[0]]
                        + ["-env", "RPN_DEFNS", rpn_defns]
                        + command[1:]
                    )
                if not is_container_command:
                    command = [c.replace("/", "\\") for c in command]
                self.run_command(
                    command, logfile, cwd=self.global_parameters["master_subdir"]
                )
            else:
                if not is_container_command:
                    command = [c.replace("/", "\\") for c in command]
                rpn_defns = (
                    os.path.abspath(self.global_parameters["simcodes_location"])
                    + "/Elegant/defns.rpn"
                )
                if not is_container_command:
                    rpn_defns = rpn_defns.replace("/", "\\")
                self.run_command(
                    command,
                    logfile,
                    cwd=self.global_parameters["master_subdir"],
                    env={"RPN_DEFNS": rpn_defns},
                )

    def elegantCommandFile(self, *args, **kwargs):
        return elegantCommandFile(*args, **kwargs)


class elegantCommandFile(frameworkCommand):
    """
    Generic class for generating elements for an ELEGANT input file
    """
    # lattice: frameworkLattice
    # """The :class:`~simba.Framework_objects.frameworkLattice` object"""
    #
    # def __init__(self, *args, **kwargs):
    #     super(elegantCommandFile, self).__init__(*args, **kwargs)


class elegant_global_settings_command(elegantCommandFile):
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


class elegant_run_setup_command(elegantCommandFile):
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

    lattice: frameworkLattice | str = None
    """:class:`~simba.Framework_objects.frameworkLattice` object"""

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


class elegant_error_elements_command(elegantCommandFile):
    """
    Error control for an ELEGANT input file; see `Elegant error control`_

    .. _Elegant error control: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu33.html#x41-400007.24
    """

    elementErrors: Dict = None
    """Dictionary of elements with errors"""

    nruns: int = 1
    """Number of error runs to perform"""

    lattice: frameworkLattice = None
    """:class:`~simba.Framework_objects.frameworkLattice` object"""

    no_errors_for_first_step: int = 1
    """Perform the first run without errors"""

    error_log: str = "%s.erl"
    """File to which errors are to be logged"""

    objectname: str = "error_control"
    """Name of frameworkObject objectname"""

    objecttype: str = "error_control"
    """Name of frameworkObject objecttype"""


class elegant_scan_elements_command(elegantCommandFile):
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

    lattice: frameworkLattice = None
    """:class:`~simba.Framework_objects.frameworkLattice` object"""

    objectname: str = "vary_element"
    """Name of frameworkObject objectname"""

    objecttype: str = "vary_element"
    """Name of frameworkObject objecttype"""


class elegant_run_control_command(elegantCommandFile):
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


class elegant_twiss_output_command(elegantCommandFile):
    """
    Twiss output for an ELEGANT input file; see `Elegant twiss output`_

    .. _Elegant twiss output: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu82.html#x90-890007.73
    """

    beam: rbf.beam
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


class elegant_floor_coordinates_command(elegantCommandFile):
    """
    Floor coordinates for an ELEGANT input file; see `Elegant floor coordinates`_

    .. _Elegant floor coordinates: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu35.html#x43-420007.26
    """
    lattice: frameworkLattice = None
    """:class:`~simba.Framework_objects.frameworkLattice` object"""

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


class elegant_bunched_beam_command(elegantCommandFile):
    """
    Bunch generation for an ELEGANT input file; see `Elegant bunched beam`_

    .. _Elegant bunched beam: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu19.html
    """

    n_particles_per_bunch: int = 1
    """Particles generated. ``1`` is the reference-particle case."""

    one_random_bunch: int = 1
    """Reuse one bunch rather than regenerating it per step."""

    centroid_x: float | None = None
    """Launch offset in ``x``; written as ``centroid[0]``."""

    centroid_y: float | None = None
    """Launch offset in ``y``; written as ``centroid[2]``."""

    objectname: str = "bunched_beam"
    """Name of object"""

    objecttype: str = "bunched_beam"
    """Type of object"""

    def write_Elegant(self) -> str:
        """``centroid`` is an *array* keyword, so the generic writer cannot
        reach it -- ``centroid[6]`` is the declaration, and elegant wants
        the components addressed individually as ``centroid[0]`` and
        ``centroid[2]``."""
        text = super().write_Elegant()
        offsets = "".join(
            f"\t{name} = {value}\n"
            for name, value in (
                ("centroid[0]", self.centroid_x),
                ("centroid[2]", self.centroid_y),
            )
            if value is not None
        )
        return text.replace("&end\n", offsets + "&end\n") if offsets else text


class elegant_find_aperture_command(elegantCommandFile):
    """
    Dynamic-aperture search for an ELEGANT input file; see `Elegant find aperture`_

    .. _Elegant find aperture: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu40.html
    """

    output: str = "%s.aper"
    """File the aperture boundary is written to"""

    mode: str = "n-line"
    """Search mode; ``n-line`` scans along rays from the origin"""

    xmin: float = -0.01
    """Smallest horizontal amplitude searched"""

    xmax: float = 0.01
    """Largest horizontal amplitude searched"""

    ymin: float = 0.0
    """Smallest vertical amplitude searched"""

    ymax: float = 0.01
    """Largest vertical amplitude searched"""

    nx: int = 21
    """Horizontal grid points"""

    ny: int = 11
    """Vertical grid points"""

    n_lines: int = 11
    """Rays searched in ``n-line`` mode"""

    verbosity: int = 0
    """Chatter level"""

    objectname: str = "find_aperture"
    """Name of object"""

    objecttype: str = "find_aperture"
    """Type of object"""


class elegant_frequency_map_command(elegantCommandFile):
    """
    Frequency map for an ELEGANT input file; see `Elegant frequency map`_

    .. _Elegant frequency map: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu41.html
    """

    output: str = "%s.fma"
    """File the map is written to"""

    xmin: float = -0.01
    """Smallest horizontal amplitude scanned"""

    xmax: float = 0.01
    """Largest horizontal amplitude scanned"""

    ymin: float = 0.0
    """Smallest vertical amplitude scanned"""

    ymax: float = 0.01
    """Largest vertical amplitude scanned"""

    delta_min: float = 0.0
    """Smallest momentum deviation scanned"""

    delta_max: float = 0.0
    """Largest momentum deviation scanned"""

    nx: int = 21
    """Horizontal grid points"""

    ny: int = 11
    """Vertical grid points"""

    ndelta: int = 1
    """Momentum grid points"""

    include_changes: int = 1
    """Write the tune *change* between the two halves of the run -- the
    diffusion that makes a frequency map more than a tune footprint"""

    full_grid_output: int = 1
    """Write every grid point, not only the surviving ones"""

    verbosity: int = 0
    """Chatter level"""

    objectname: str = "frequency_map"
    """Name of object"""

    objecttype: str = "frequency_map"
    """Type of object"""


class elegant_matrix_output_command(elegantCommandFile):
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


class elegant_sdds_beam_command(elegantCommandFile):
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


class elegant_track_command(elegantCommandFile):
    """
    Track command for an ELEGANT input file; see `Elegant track`_

    .. _Elegant track: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu83.html#x91-900007.74
    """

    trackBeam: bool = True
    """Flag to indicate whether to include the track command"""

    objectname: str = "track"
    """Name of object"""

    objecttype: str = "track"
    """Type of object"""


class elegantOptimisation(elegantCommandFile):
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
        super(elegantOptimisation, self).__init__(
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


class sddsFile(object):
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
