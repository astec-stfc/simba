"""
Genesis 1.3 v4 backend: the lattice, and one class per ``&command`` of the input file.

SASE and HGHG are supported (EEHG is not). For HGHG, give the lattice a
:attr:`~genesisLattice.split_element` and a chicane after the first undulator; the
harmonic follows from the first undulator after the split. See the `Genesis manual`_.

.. _Genesis manual: https://github.com/svenreiche/Genesis-1.3-Version4/tree/master/manual
"""

import os
from copy import deepcopy
import numpy as np
import subprocess
from warnings import warn
from scipy.constants import speed_of_light
from random import randint

from pydantic import (
    Field,
    field_validator,
)

try:
    import sdds
except Exception:
    print("No SDDS available!")
from ...Framework_objects import (
    frameworkLattice,
    frameworkCommand,
)
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from typing import Dict, List, Literal, ClassVar
import h5py

command_files_order = [
    "setup",
    "time",
    "lattice",
    "profile_file",
    "profile_const",
    "profile_polynom",
    "profile_gauss",
    "field",
    "importdistribution",
    "importbeam",
    "importfield",
    "beam",
    "track_first",
    "alter_setup",
    "alter_beam",
    "field_2",
    "sort",
    "track",
    "write",
    "end",
]

beam_profile_properties = [
    "betax",
    "betay",
    "alphax",
    "alphay",
    "gamma",
    "delgam",
    "current",
    "ex",
    "ey"
]

class genesisLattice(frameworkLattice):
    """
    A line written as Genesis lattice (``.lat``) and input (``.in``) files.
    """

    code: str = "genesis"
    """The lattice object type."""

    allow_negative_drifts: bool = False
    """Whether negative drifts are allowed."""

    particle_definition: str | None = None
    """Name of the initial particle distribution."""

    bunch_charge: float | None = None
    """Bunch charge [C]"""

    trackBeam: bool = True
    """Whether to track the beam."""

    betax: float | None = None
    """Initial beta_x for matching"""

    betay: float | None = None
    """Initial beta_y for matching"""

    alphax: float | None = None
    """Initial alpha_x for matching"""

    alphay: float | None = None
    """Initial alpha_y for matching"""

    commandFiles: Dict = {}
    """:class:`genesisCommandFile` objects (or lists of them) for the input file, by key."""

    commandFilesOrder: List = []
    """Unused; the order is the module's ``command_files_order``."""

    element_name_converted: Dict = {}
    """Unused."""

    fundamental_wavelength: float = None
    """Fundamental wavelength [m]; by default from the beam energy and first undulator."""

    shot_noise: bool = True
    """Include shot noise in the calculation"""

    npart: int = None
    """Number of macro particles per slice; if not provided, calculate from the beam"""

    nbins: int = 16
    """Number of macro particles to be grouped into beamlets"""

    seed: int = Field(default_factory=lambda: randint(1, 10000000))
    """Random number seed"""

    match_location: float = None
    """``zmatch`` [m] for :class:`genesis_lattice_command`; unset means no matching."""

    field_power: float = 1e3
    """Initial power [W] for :class:`genesis_field_command`; see :meth:`get_field_power`."""

    dgrid: float = 1e-3
    """Grid extent [m] for :class:`genesis_field_command`"""

    ngrid: int = 251
    """Number of grid points for :class:`genesis_field_command`"""

    waist_size: float = 1e-5
    """Waist size [m] for :class:`genesis_field_command`; see :meth:`get_waist_size`."""

    beam_type: Literal["beam", "profile", "distribution"] = "beam"
    """How the beam is loaded: ``beam`` uses ``&beam`` with average values,
    ``profile`` uses ``&beam`` with profiles, ``distribution`` uses ``&importdistribution``."""

    beam_slices: int = 128
    """Number of slices for ``profile``, and that a steady-state output beam is
    replicated over for the next code."""

    slicewidth: float = 0.01
    """Fraction of the distribution length used for each slice with ``distribution``;
    see :class:`genesis_importdistribution_command`."""

    sample: int = 1
    """Simulate only every `sample`-th wavelength when time-dependent; a long bunch
    usually needs this above 1."""

    time_window: float = 99.8
    """Percentile range of the bunch the simulation window covers; see :meth:`beam_length`."""

    steady_state: bool = True
    """Run steady-state; if False, run time-dependent over :meth:`beam_length`."""

    one4one: bool = False
    """Run one-for-one; if False, use :attr:`npart` and :attr:`nbins`."""

    chicanes: str | list = None
    """Names of the chicanes, each a ``chicane`` group in the configuration file."""

    split_element: str = None
    """Element at which to split the lattice for harmonic conversion (one split only)."""

    electrons_only: ClassVar[bool] = True
    """FELs are electrons only."""


    def model_post_init(self, __context):
        super().model_post_init(__context)
        cls = self.__class__
        for f in cls.model_fields:
            if "fel" in self.file_block:
                if f in self.file_block["fel"]:
                    setattr(self, f, self.file_block["fel"][f])
            elif f in self.file_block:
                setattr(self, f, self.file_block[f])
        self.particle_definition = self.input_particle_definition

    def writeElements(self) -> str:
        """
        Genesis lattice string for this section, with chicanes from :attr:`chicanes`.

        Returns
        -------
        str
            The Genesis lattice
        """
        chicane_dict = {}
        if self.chicanes is not None:
            if isinstance(self.chicanes, str):
                self.chicanes = [self.chicanes]
            for chicane in self.chicanes:
                if chicane in self.groupObjects:
                    chicane_dict.update(
                        {
                            chicane: {
                                "start": self.groupObjects[chicane].elementObjects[0].name,
                                "end": self.groupObjects[chicane].elementObjects[-1].name,
                                "r56": self.groupObjects[chicane].r56,
                                "dipole_length": self.groupObjects[chicane].elementObjects[0].magnetic.length,
                                "drift_length": self.groupObjects[chicane].elementObjects[1].physical.start.z -
                                                self.groupObjects[chicane].elementObjects[0].physical.end.z,
                                "length": self.groupObjects[chicane].elementObjects[-1].physical.end.z -
                                          self.groupObjects[chicane].elementObjects[0].physical.start.z,
                            },
                        },
                    )
        return self.section.to_genesis(split_element=self.split_element, chicanes=chicane_dict)

    def write(self) -> None:
        """
        Write the ``.lat`` and ``.in`` files to `master_subdir`; run :meth:`preProcess` first.
        """
        lattice_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".lat"
        )
        saveFile(lattice_file, self.writeElements())
        self.files.append(lattice_file)
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(command_file, "", "w")
        if len(command_files_order) > 0:
            for cfileid in command_files_order:
                if cfileid in self.commandFiles:
                    cfile = self.commandFiles[cfileid]
                    if isinstance(cfile, genesisCommandFile):
                        saveFile(command_file, cfile.write_Genesis(), "a")
                    elif isinstance(cfile, list):
                        for cf in cfile:
                            saveFile(command_file, cf.write_Genesis(), "a")
            self.files.append(command_file)
        else:
            warn("commandFiles length is zero; run preProcess first")

    def preProcess(self) -> None:
        """
        Load the input beam and build the command files.
        """
        super().preProcess()
        prefix = self.get_prefix()
        self.load_input_beam(prefix, self.particle_definition)
        if not self.npart:
            beamlen = len(self.global_parameters["beam"].x.val)
            self.npart = beamlen
            warn(f"npart not provided; setting npart to {self.npart}")
        self.hdf5_to_genesis()
        self.write_setup_file()
        if not self.steady_state:
            self.write_time()
        if self.match_location:
            self.commandFiles["lattice"] = genesis_lattice_command(
                zmatch=self.match_location,
            )
        self.commandFiles["field"] = genesis_field_command(
            power=self.get_field_power(),
            ngrid=self.ngrid,
            dgrid=self.dgrid,
            waist_size=self.get_waist_size(),
            waist_pos=self.wigglers[0].physical.middle.z - self.startObject.physical.start.z
        )
        self.commandFiles["track"] = genesis_track_command()
        if isinstance(self.split_element, str):
            first_wiggler = None
            after_split = False
            for elem in self.elementObjects.values():
                if elem.name == self.split_element:
                    after_split = True
                elif after_split and elem.hardware_class.lower() == "wiggler":
                    first_wiggler = elem
                    break
            if first_wiggler is None:
                raise ValueError(f"No undulator found after split_element {self.split_element}")
            gamma0 = self.global_parameters["beam"].beam.centroids.mean_gamma.val
            lambda0_from_und = first_wiggler.period / (2 * gamma0 ** 2) * (1 + first_wiggler.normalized_strength ** 2)
            harmonic_number = int(round(self.fundamental_wavelength / lambda0_from_und))
            self.commandFiles["track_first"] = genesis_track_command()
            self.commandFiles["alter_setup"] = genesis_alter_setup_command(
                beamline=f"{self.objectname}_SPLIT_2",
                delz=first_wiggler.period,
                harmonic=harmonic_number,
            )
            self.commandFiles["field_2"] = genesis_field_command(
                power=0,
                ngrid=self.ngrid,
                dgrid=self.dgrid,
                accumulate=True,
            )
        self.commandFiles["write"] = genesis_write_command(
            field=f"{self.end}_FIELD",
            beam=f"{self.end}_BEAM",
        )

    def postProcess(self) -> None:
        """
        Write the final beam to openPMD, rename monitor outputs, shift output ``z`` to
        lattice coordinates and clear :attr:`commandFiles`.
        """
        super().postProcess()
        beam = rbf.beam()
        rootname = f"{self.global_parameters['master_subdir']}/{self.end}"
        genesisbeamfilename = f"{rootname}_BEAM.par.h5"
        expand = {}
        if self.steady_state:
            expand = {
                "steady_state": True,
                "bunch_length": self.beam_length(),
                "n_slices": self.beam_slices,
            }
        rbf.genesis.read_genesis_beam_file(beam, genesisbeamfilename, **expand)
        HDF5filename = (
            f"{self.global_parameters['master_subdir']}/"
            f"{self.output_basename(self.end)}.openpmd.hdf5"
        )
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            HDF5filename,
            pos=list(self.endObject.physical.start.model_dump().values()),
        )
        self.commandFiles = {}
        outfields = sorted(
            [
                e for e in os.listdir(self.global_parameters["master_subdir"]) if ".fld.h5" in e and self.end not in e
            ]
        )
        outbeams = sorted(
            [
                e for e in os.listdir(self.global_parameters["master_subdir"]) if ".par.h5" in e and self.end not in e
            ]
        )
        subd = self.global_parameters["master_subdir"]
        for outf, photon in zip(outfields, self.getElementType("photon_monitor")):
            os.rename(f"{subd}/{outf}", f"{subd}/{photon.name}.fld.h5")
        for outb, scr in zip(outbeams, self.getElementType("screen")):
            os.rename(f"{subd}/{outb}", f"{subd}/{scr.name}.par.h5")
        with h5py.File(f"{self.global_parameters['master_subdir']}/{self.objectname}.out.h5", "r+") as f:
            dset = f["/Lattice/z"]
            dset[:] = dset[:] + self.startObject.physical.start.z
        try:
            with h5py.File(f"{self.global_parameters['master_subdir']}/{self.objectname}.Run2.out.h5", "r+") as f:
                dset = f["/Lattice/z"]
                dset[:] = dset[:] + self.elementObjects[self.split_element].physical.start.z
        except FileNotFoundError:
            pass

    def write_setup_file(self) -> None:
        gamma0 = self.global_parameters["beam"].beam.centroids.mean_gamma.val
        first_wiggler = self.wigglers[0]
        if not self.fundamental_wavelength:
            self.fundamental_wavelength = first_wiggler.period / (2 * gamma0**2)
            self.fundamental_wavelength *= (1 + first_wiggler.normalized_strength**2)
        else:
            lambda0_from_und = first_wiggler.period / (2 * gamma0**2) * (1 + first_wiggler.normalized_strength**2)
            if not np.isclose([self.fundamental_wavelength], [lambda0_from_und]):
                warn("First undulator strength is not close to fundamental_wavelength")
        delz = first_wiggler.period
        if isinstance(self.split_element, str):
            if self.split_element in self.elements:
                beamline = f"{self.objectname}_SPLIT_1"
            else:
                raise ValueError(f"split_element {self.split_element} not found in elements")
        else:
            beamline = self.objectname
        self.commandFiles["setup"] = genesis_setup_command(
            rootname=self.objectname,
            lattice=self.objectname + ".lat",
            beamline=beamline,
            one4one=self.one4one,
            lambda0=self.fundamental_wavelength,
            gamma0=gamma0,
            delz=delz,
            shotnoise=self.shot_noise,
            nbins=self.nbins,
            npart=self.npart,
            seed=self.seed,
        )

    def beam_length(self) -> float:
        """Length [m] of the input bunch over the :attr:`time_window` percentile range.

        Percentiles stop a few far-tail particles (e.g. after an arc) setting the
        window, which the slice count and run time scale with.
        """
        t = np.asarray(self.global_parameters["beam"].t)
        edges = [50 - self.time_window / 2, 50 + self.time_window / 2]
        return float(np.ptp(np.percentile(t, edges))) * speed_of_light

    def write_time(self) -> None:
        self.commandFiles["time"] = genesis_time_command(
            slen=self.beam_length(),
            time=True,
            sample=self.sample,
        )

    def hdf5_to_genesis(self) -> None:
        """
        Set up the beam command(s) for :attr:`beam_type`, writing a Genesis beam file if needed.
        """
        hdf5outname = f'{self.global_parameters["master_subdir"]}/{self.start}.hdf5'
        genesisbeamfilename = hdf5outname.replace("hdf5", "genesis.hdf5")
        if self.beam_type == "profile":
            rbf.genesis.write_genesis_beam_file(
                self.global_parameters["beam"],
                genesisbeamfilename,
                n_slice = int(self.beam_slices),
            )
            props = {}
            self.commandFiles["profile_file"] = []
            for b in beam_profile_properties:
                self.commandFiles["profile_file"].append(
                    genesis_profile_file_command(
                        label=f"{b}_profile",
                        xdata=f"{self.start}.genesis.hdf5/s",
                        ydata=f"{self.start}.genesis.hdf5/{b}",
                    )
                )
                props.update({b: f"@{b}_profile"})
            self.commandFiles["beam"] = genesis_beam_command(**props)
            self.files.append(f"{self.global_parameters['master_subdir']}/{self.start}.genesis.hdf5")
        elif self.beam_type == "beam":
            beam_properties = self.get_average_beam_properties()
            self.commandFiles["beam"] = genesis_beam_command(**beam_properties)
        elif self.beam_type == "distribution":
            if self.steady_state:
                raise ValueError("beam_type 'distribution' needs steady_state = False")
            rbf.genesis.write_genesis_beam_distribution(
                self.global_parameters["beam"],
                genesisbeamfilename,
                pos=[-v for v in self.startObject.physical.start.model_dump().values()],
            )
            self.commandFiles["importdistribution"] = genesis_importdistribution_command(
                file=f"{self.start}.genesis.hdf5",
                charge=abs(float(self.global_parameters["beam"].total_charge)),
                slicewidth=self.slicewidth,
                gamma0=float(self.global_parameters["beam"].beam.centroids.mean_gamma.val),
                settimewindow=True,
            )
            self.files.append(genesisbeamfilename)
        else:
            raise ValueError(f"beam_type {self.beam_type} not understood")

    def get_average_beam_properties(self) -> Dict:
        beam = deepcopy(self.global_parameters["beam"])
        if isinstance(float(beam.E0_eV.val), float):
            E0_eV = float(beam.E0_eV.val)
        else:
            E0_eV = float(beam.E0_eV.val[0])
        slmom = beam.slice.slice_momentum_spread.val
        lenslmom = len(slmom)
        delgam = float(np.mean(slmom[int(lenslmom/2 - 3): int(lenslmom/2 + 3)])) / E0_eV
        return {
            "betax": float(beam.twiss.beta_x.val),
            "betay": float(beam.twiss.beta_y.val),
            "alphax": float(beam.twiss.alpha_x.val),
            "alphay": float(beam.twiss.alpha_y.val),
            "gamma": float(beam.centroids.mean_gamma.val),
            "delgam": delgam,
            "current": float(beam.slice.peak_current.val),
            "ex": float(beam.emittance.normalized_horizontal_emittance.val),
            "ey": float(beam.emittance.normalized_vertical_emittance.val),
        }

    def get_field_power(self) -> float | str:
        """
        Initial field power: a laser profile label if the first undulator has a laser,
        else :attr:`field_power`.

        Returns
        -------
        float | str
            :attr:`field_power` or the ``@<name>_laser_profile`` label
        """
        first_wiggler = self.wigglers[0]
        if first_wiggler.laser:
            self.commandFiles["profile_gauss"] = genesis_profile_gauss_command(
                label=first_wiggler.name + "_laser_profile",
                c0=first_wiggler.laser.max_power,
                s0=first_wiggler.laser.initial_position,
                sig=first_wiggler.laser.pulse_duration_rms * speed_of_light,
            )
            return f"@{first_wiggler.name}_laser_profile"
        else:
            return self.field_power

    def get_waist_size(self) -> float:
        """
        Waist size [m] of the first undulator's laser, else :attr:`waist_size`.
        """
        first_wiggler = self.wigglers[0]
        if first_wiggler.laser:
            return first_wiggler.laser.waist
        else:
            return self.waist_size

    def run(self) -> None:
        """
        Run Genesis in `master_subdir`, logging to ``<name>.log``; uses :meth:`run_remote`
        if :attr:`remote_setup` is set.
        """
        if self.remote_setup:
            self.run_remote()
        else:
            command = self.executables[self.code] + [self.name + ".in"]
            workdir = os.path.abspath(self.global_parameters["master_subdir"])
            command = self.executables.build_command(command, workdir)
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

class genesisCommandFile(frameworkCommand):
    """
    Base class for Genesis input-file namelists.
    """

class genesis_setup_command(genesisCommandFile):
    """
    ``&setup`` namelist.
    """

    objectname: str = "setup"
    """Name of the namelist object."""

    objecttype: str = "setup"
    """Type of the namelist object."""

    rootname: str
    """Prefix for all output files unless overridden by ``&write``."""

    lattice: str
    """Lattice filename, relative to the input file."""

    beamline: str
    """Beamline name, defined in the lattice file."""

    gamma0: float
    """Reference energy in units of the electron rest mass."""

    lambda0: float
    """Reference wavelength [m]; also sets the sample spacing in time-dependent runs."""

    delz: float
    """Preferred integration step [m]."""

    seed: int = Field(default_factory=lambda: randint(1, 10000000))
    """Random seed for shot noise and lattice errors."""

    npart: int
    """Macroparticles per slice; must be a multiple of :attr:`nbins`. Ignored if :attr:`one4one`."""

    nbins: int
    """Macroparticles per beamlet for shot noise. Ignored if :attr:`one4one`."""

    one4one: bool = False
    """Resolve every electron; needed for sorting/slicing but can use a lot of memory."""

    shotnoise: bool = True
    """Add shot noise per slice; best off for steady-state or scans."""

    beam_global_stat: bool = False
    """Write whole-bunch beam statistics to ``Beam/Global``."""

    field_global_stat: bool = False
    """Write whole-pulse field statistics, as :attr:`beam_global_stat`."""

    exclude_spatial_output: bool = False
    """Omit transverse position/size datasets from the output."""

    exclude_fft_output: bool = False
    """Omit field divergence/pointing (skips the FFT)."""

    exclude_intensity_output: bool = False
    """Omit near/far-field intensity and phase; spectra then cannot be computed."""

    exclude_energy_output: bool = False
    """Omit mean energy and energy spread datasets."""

    exclude_aux_output: bool = False
    """Omit auxiliary datasets (currently the long-range longitudinal field)."""

    exclude_current_output: bool = True
    """Write the current profile only once; set False when it can change
    (one-for-one with sorting, e.g. ESASE/HGHG)."""

    exclude_field_dump: bool = False
    """Skip the field dump to ``.fld.h5``."""

class genesis_alter_setup_command(genesisCommandFile):
    """
    ``&alter_setup`` namelist.
    """

    objectname: str = "alter_setup"
    """Name of the namelist object."""

    objecttype: str = "alter_setup"
    """Type of the namelist object."""

    rootname: str | None = None
    """New output-file prefix; see :class:`genesis_write_command`."""

    beamline: str
    """Beamline to switch to, defined in the lattice file."""

    delz: float
    """Preferred integration step [m]; Genesis may adjust it."""

    harmonic: int = 1
    """Up-convert to this harmonic: scales wavelength, sample rate and phases, and
    keeps only the matching harmonic field as the new fundamental."""

    subharmonic: int = 1
    """Down-convert by this factor; the fundamental becomes a harmonic, so a new
    fundamental field must be defined before tracking."""

    resample: bool = False
    """With :attr:`~genesis_setup_command.one4one`, resample slices to the new
    wavelength on (sub)harmonic conversion."""

    disable: bool = False
    """Disable non-matching radiation harmonic."""


class genesis_lattice_command(genesisCommandFile):
    """
    ``&lattice`` namelist.
    """

    objectname: str = "lattice"
    """Name of the namelist object."""

    objecttype: str = "lattice"
    """Type of the namelist object."""

    zmatch: float = 0.0
    """If non-zero, position [m] at which to compute periodic matched optics, used as
    the default for the following beam."""

    element: str = ""
    """Element type to alter (first 4 letters suffice; MARKER unsupported)."""

    field: str = ""
    """Element attribute to alter, as named in the lattice file."""

    value: float | str = 0.0
    """New value, or a sequence reference."""

    instance: int = 0
    """Which occurrence to alter; 0 means all."""

    add: bool = True
    """Add to the existing value rather than overwrite it."""


class genesis_time_command(genesisCommandFile):
    """
    ``&time`` namelist; its presence makes the run time-dependent.
    """

    objectname: str = "time"
    """Name of the namelist object."""

    objecttype: str = "time"
    """Type of the namelist object."""

    s0: float = 0.0
    """Start of the time window [m]."""

    slen: float = 0.0
    """Length of the time window [m]; may be enlarged to fit the MPI size."""

    sample: int = 1
    """Sample rate in units of ``lambda0``; slices = slen / lambda0 / sample."""

    time: bool = True
    """Time-dependent run; False gives a scan (no slippage)."""


class genesis_profile_const_command(genesisCommandFile):
    """
    ``&profile_const`` namelist.
    """

    objectname: str = "profile_const"
    """Name of the namelist object."""

    objecttype: str = "profile_const"
    """Type of the namelist object."""

    label: str
    """Profile name, referenced as ``@label``."""

    c0: float
    """Constant value."""


class genesis_profile_gauss_command(genesisCommandFile):
    """
    ``&profile_gauss`` namelist.
    """

    objectname: str = "profile_gauss"
    """Name of the namelist object."""

    objecttype: str = "profile_gauss"
    """Type of the namelist object."""

    label: str
    """Profile name, referenced as ``@label``."""

    c0: float
    """Peak value."""

    s0: float
    """Centre of the Gaussian [m]."""

    sig: float
    """RMS width of the Gaussian [m]."""

class genesis_profile_step_command(genesisCommandFile):
    """
    ``&profile_step`` namelist.
    """

    objectname: str = "profile_step"
    """Name of the namelist object."""

    objecttype: str = "profile_step"
    """Type of the namelist object."""

    label: str
    """Profile name, referenced as ``@label``."""

    c0: float
    """Value inside the step."""

    s_start: float
    """Start of the step [m]."""

    s_end: float
    """End of the step [m]."""

class genesis_profile_polynom_command(genesisCommandFile):
    """
    ``&profile_polynom`` namelist.
    """

    objectname: str = "profile_polynom"
    """Name of the namelist object."""

    objecttype: str = "profile_polynom"
    """Type of the namelist object."""

    label: str
    """Profile name, referenced as ``@label``."""

    c0: float
    """Constant term."""

    c1: float = 0.0
    """Term proportional to s."""

    c2: float = 0.0
    """Term proportional to s^2."""

    c3: float = 0.0
    """Term proportional to s^3."""

    c4: float = 0.0
    """Term proportional to s^4."""

class genesis_profile_file_command(genesisCommandFile):
    """
    ``&profile_file`` namelist (look-up table from HDF5).
    """

    objectname: str = "profile_file"
    """Name of the namelist object."""

    objecttype: str = "profile_file"
    """Type of the namelist object."""

    label: str
    """Profile name, referenced as ``@label``."""

    xdata: str
    """HDF5 dataset of s-positions, as ``filename/group/.../dataset``."""

    ydata: str
    """HDF5 dataset of values, as :attr:`xdata`."""

    isTime: bool = False
    """:attr:`xdata` is time, multiplied by c to give position."""

    reverse: bool = False
    """Reverse the table order (time and position can differ in sign)."""

    autoassign: bool = False
    """Use the HDF5 file from :attr:`xdata`."""


class genesis_sequence_const_command(genesisCommandFile):
    """
    ``&sequence_const`` namelist.
    """

    objectname: str = "sequence_const"
    """Name of the namelist object."""

    objecttype: str = "sequence_const"
    """Type of the namelist object."""

    label: str
    """Sequence name, referenced in the lattice."""

    c0: float
    """Constant value."""

class genesis_sequence_polynom_command(genesisCommandFile):
    """
    ``&sequence_polynom`` namelist.
    """

    objectname: str = "sequence_polynom"
    """Name of the namelist object."""

    objecttype: str = "sequence_polynom"
    """Type of the namelist object."""

    label: str
    """Sequence name, referenced in the lattice."""

    c0: float
    """Constant term."""

    c1: float = 0.0
    """Term proportional to s."""

    c2: float = 0.0
    """Term proportional to s^2."""

    c3: float = 0.0
    """Term proportional to s^3."""

    c4: float = 0.0
    """Term proportional to s^4."""

class genesis_sequence_power_command(genesisCommandFile):
    """
    ``&sequence_power`` namelist.
    """

    objectname: str = "sequence_power"
    """Name of the namelist object."""

    objecttype: str = "sequence_power"
    """Type of the namelist object."""

    label: str
    """Sequence name, referenced in the lattice."""

    c0: float
    """Constant term."""

    dc: float
    """Scale of the power series added to :attr:`c0`."""

    alpha: float
    """Power of the series."""

    n0: int = 1
    """Index at which the power growth starts."""

    c4: float = 0.0
    """Not a Genesis ``&sequence_power`` parameter."""

class genesis_sequence_random_command(genesisCommandFile):
    """
    ``&sequence_random`` namelist.
    """

    objectname: str = "sequence_random"
    """Name of the namelist object."""

    objecttype: str = "sequence_random"
    """Type of the namelist object."""

    label: str
    """Sequence name, referenced in the lattice."""

    c0: float = 0.0
    """Mean value."""

    dc: float = 0.0
    """RMS (normal) or half-range (uniform) of the error."""

    seed: int = 100
    """Random seed."""

    normal: bool = True
    """Gaussian distribution; if False, uniform."""


class genesis_beam_command(genesisCommandFile):
    """
    ``&beam`` namelist; any value may be a ``@profile`` label.
    """

    objectname: str = "beam"
    """Name of the namelist object."""

    objecttype: str = "beam"
    """Type of the namelist object."""

    gamma: float | str
    """Mean energy in units of the electron rest mass."""

    delgam: float | str
    """RMS energy spread in units of the electron rest mass."""

    current: float | str
    """Current [A]."""

    ex: float | str
    """Normalised horizontal emittance [m-rad]."""

    ey: float | str
    """Normalised vertical emittance [m-rad]."""

    betax: float | str
    """Initial horizontal beta [m]; defaults to the matched value if
    :attr:`genesis_lattice_command.zmatch` was used."""

    betay: float | str
    """Initial vertical beta [m]; see :attr:`betax`."""

    alphax: float | str
    """Initial horizontal alpha; see :attr:`betax`."""

    alphay: float | str
    """Initial vertical alpha; see :attr:`betax`."""

    xcenter: float | str = 0.0
    """Horizontal centroid [m]."""

    ycenter: float | str = 0.0
    """Vertical centroid [m]."""

    pxcenter: float | str = 0.0
    """Horizontal centroid momentum [γβx]."""

    pycenter: float | str = 0.0
    """Vertical centroid momentum [γβy]."""

    bunch: float | str = 0.0
    """Initial bunching."""

    bunchphase: float | str = 0.0
    """Initial bunching phase."""

    emod: float | str = 0.0
    """Energy modulation at the reference wavelength, in units of the electron rest mass."""

    emodphase: float | str = 0.0
    """Energy modulation phase."""


class genesis_alter_beam_command(genesisCommandFile):
    """
    ``&alter_beam`` namelist.
    """

    objectname: str = "alter_beam"
    """Name of the namelist object."""

    objecttype: str = "alter_beam"
    """Type of the namelist object."""

    dgamma: float | str = 0.0
    """Sinusoidal energy modulation amplitude, in units of the electron rest mass."""

    phase: float | str = 0.0
    """Energy modulation phase [rad]."""

    # lambda (a Python keyword): wavelength in m of the external energy modulation;
    # pass it as an extra field, e.g. ``**{"lambda": 1e-6}``, so write_Genesis finds it.

    r56: float = 0
    """Chicane R56 [m]."""


class genesis_field_command(genesisCommandFile):
    """
    ``&field`` namelist (Gauss-Hermite mode).
    """

    objectname: str = "field"
    """Name of the namelist object."""

    objecttype: str = "field"
    """Type of the namelist object."""

    # lambda (a Python keyword): central frequency of the radiation mode, defaulting to
    # genesis_setup_command.lambda0; pass it as an extra field, e.g. ``**{"lambda": 1e-9}``.

    power: float | str = 0.0
    """Radiation power [W]."""

    phase: float | str = 0.0
    """Radiation phase [rad]; a linear profile shifts the wavelength."""

    waist_pos: float | str = 0.0
    """Focus position relative to the undulator entrance [m]; negative is upstream."""

    waist_size: float | str | None = None
    """Waist size w0 (Siegman's definition) [m]."""

    xcenter: float = 0.0
    """Horizontal centre [m]."""

    ycenter: float = 0.0
    """Vertical centre [m]."""

    xangle: float = 0.0
    """Horizontal injection angle [rad]."""

    yangle: float = 0.0
    """Vertical injection angle [rad]."""

    dgrid: float = 0.001
    """Half-width of the grid [m]."""

    ngrid: int = Field(default=151, gt=1)
    """Grid points per dimension; must be odd so a point sits on axis."""

    harm: int = 1
    """Harmonic of the reference wavelength."""

    nx: int = 0
    """Horizontal mode number."""

    ny: int = 0
    """Vertical mode number."""

    accumulate: bool = True
    """Add to an existing field rather than overwrite it."""

    @field_validator("ngrid")
    def check_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("ngrid must be odd")
        return v


class genesis_importdistribution_command(genesisCommandFile):
    """
    ``&importdistribution`` namelist.
    """

    objectname: str = "importdistribution"
    """Name of the namelist object."""

    objecttype: str = "importdistribution"
    """Type of the namelist object."""

    file: str
    """Distribution filename."""

    charge: float
    """Total charge [C]."""

    slicewidth: float
    """Fraction of the distribution length used to reconstruct each slice."""

    center: bool = False
    """Recentre position, momentum and energy to the values below."""

    gamma0: float
    """New mean energy with :attr:`center`, in units of the electron rest mass."""

    x0: float = 0.0
    """New horizontal centre with :attr:`center` [m]."""

    y0: float = 0.0
    """New vertical centre with :attr:`center` [m]."""

    px0: float = 0.0
    """New horizontal mean momentum with :attr:`center` [γβx]."""

    py0: float = 0.0
    """New vertical mean momentum with :attr:`center` [γβy]."""

    match: bool = False
    """Match the distribution to the optics below."""

    betax: float = 15.0
    """New horizontal beta with :attr:`match` [m]."""

    betay: float = 15.0
    """New vertical beta with :attr:`match` [m]."""

    alphax: float = 0.0
    """New horizontal alpha with :attr:`match`."""

    alphay: float = 0.0
    """New vertical alpha with :attr:`match`."""

    eval_start: float = 0.0
    """Evaluation start."""

    eval_end: float = 1.0
    """Evaluation end."""

    settimewindow: bool = False
    """Set time window."""


class genesis_importbeam_command(genesisCommandFile):
    """
    ``&importbeam`` namelist.
    """

    objectname: str = "importbeam"
    """Name of the namelist object."""

    objecttype: str = "importbeam"
    """Type of the namelist object."""

    file: str
    """Genesis 1.3 slice-wise particle dump (HDF5)."""

    time: bool = True
    """If False and no time window is set, run as a scan (no slippage)."""


class genesis_importfield_command(genesisCommandFile):
    """
    ``&importfield`` namelist.
    """

    objectname: str = "importfield"
    """Name of the namelist object."""

    objecttype: str = "importfield"
    """Type of the namelist object."""

    file: str
    """Genesis 1.3 field dump (HDF5)."""

    harmonic: int = 1
    """Harmonic the field is imported as."""

    time: bool = True
    """If False and no time window is set, run as a scan (no slippage)."""

    attenuation: float = 1.0
    """Scale factor applied to the imported field."""

    offset: float = 0.0
    """Time-frame offset of the field; should be a multiple of the dump's slice length."""


class genesis_importtransformation_command(genesisCommandFile):
    """
    ``&importtransformation`` namelist.
    """

    objectname: str = "importtransformation"
    """Name of the namelist object."""

    objecttype: str = "importtransformation"
    """Type of the namelist object."""

    file: str
    """HDF5 file holding the vector and matrix."""

    vector: str
    """Dataset of the vector, shape (6) or (n,6)."""

    matrix: str
    """Dataset of the matrix, shape (6,6) or (n,6,6)."""

    slen: float = 0.0
    """Spacing [m] between sample points for interpolation; 0 applies the first entry globally."""


class genesis_efield_command(genesisCommandFile):
    """
    ``&efield`` namelist (space charge).
    """

    objectname: str = "efield"
    """Name of the namelist object."""

    objecttype: str = "efield"
    """Type of the namelist object."""

    longrange: bool = False
    """Compute the long-range space-charge field."""

    rmax: float = 0.0
    """Radial grid size [m]; enlarged automatically if the beam outgrows it."""

    nz: int = 0.0
    """Longitudinal Fourier components of the short-range field; keep compatible with the beamlet size."""

    nphi: int = 0.0
    """Azimuthal modes of the short-range field."""

    ngrid: int = 100
    """Radial grid points of the short-range field."""


class genesis_sponrad_command(genesisCommandFile):
    """
    ``&sponrad`` namelist (spontaneous radiation).
    """

    objectname: str = "sponrad"
    """Name of the namelist object."""

    objecttype: str = "sponrad"
    """Type of the namelist object."""

    seed: int = 1234
    """Random seed for quantum fluctuations."""

    doLoss: bool = False
    """Apply spontaneous-radiation energy loss."""

    doSpread: bool = False
    """Apply quantum-fluctuation energy spread growth."""


class genesis_wake_command(genesisCommandFile):
    """
    ``&wake`` namelist.
    """

    objectname: str = "wake"
    """Name of the namelist object."""

    objecttype: str = "wake"
    """Type of the namelist object."""

    loss: float | str = 0.0
    """Global loss [eV/m], independent of the current profile."""

    radius: float = 0.0025
    """Pipe radius, or half-gap for parallel plates [m]."""

    roundpipe: bool = True
    """Round pipe; if False, parallel plates."""

    conductivity: float = 0.0
    """Wall conductivity for resistive-wall wakes."""

    relaxation: float = 0.0
    """Wall relaxation distance (mean free path) for resistive-wall wakes."""

    material: Literal["CU", "AL", ""] = ""
    """Set conductivity and relaxation for copper or aluminium, overriding both."""

    gap: float = 0.0
    """Longitudinal gap length [mm] for geometric wakes."""

    lgap: float = 1.0
    """Length [m] over which one gap applies (its period)."""

    hrough: float = 0.0
    """Roughness corrugation amplitude [m]."""

    lrough: float = 0.0
    """Roughness corrugation period [m]."""

    transient: bool = False
    """Model wake catch-up from :attr:`ztrans` (slower; updates every step); False is steady-state."""

    ztrans: float = 0.0
    """Position of the wake source relative to the undulator start [m]."""

    output: str = ""
    """Root of the ``.wake.h5`` file for single-particle wakes."""


class genesis_sort_command(genesisCommandFile):
    """
    ``&sort`` namelist.
    """

    objectname: str = "sort"
    """Name of the namelist object."""

    objecttype: str = "sort"
    """Type of the namelist object."""

class genesis_write_command(genesisCommandFile):
    """
    ``&write`` namelist.
    """

    objectname: str = "write"
    """Name of the namelist object."""

    objecttype: str = "write"
    """Type of the namelist object."""

    field: str = ""
    """Root of the ``.fld.h5`` field dump (all harmonics, suffixed ``.hNNN``)."""

    beam: str = ""
    """Root of the ``.par.h5`` particle dump."""


class genesis_track_command(genesisCommandFile):
    """
    ``&track`` namelist.
    """

    objectname: str = "track"
    """Name of the namelist object."""

    objecttype: str = "track"
    """Type of the namelist object."""

    zstop: float = 1e9
    """Stop tracking here [m] if shorter than the lattice."""

    output_step: int = 1
    """Integration steps between output samples."""

    field_dump_step: int = 0
    """Integration steps between field dumps (files can be large when time-dependent)."""

    beam_dump_step: int = 0
    """Integration steps between particle dumps (files can be large when time-dependent)."""

    sort_step: int = 0
    """Integration steps between particle sorts (one-for-one only)."""

    s0: float = None
    """Override the :class:`genesis_time_command` window start."""

    slen: float = None
    """Override the :class:`genesis_time_command` window length."""

    field_dump_at_undexit: bool = False
    """Dump the field at each undulator exit."""

    bunchharm: int = Field(default=1, ge=1)
    """Bunching harmonic for output."""
