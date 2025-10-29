"""
SIMBA Genesis Module

Various objects and functions to handle Genesis lattices and commands. See `Genesis manual`_ for more details.

    .. _Genesis manual: https://github.com/svenreiche/Genesis-1.3-Version4/tree/master/manual

Classes:
    - :class:`~simba.Codes.Genesis.Genesis.genesisLattice`: The Genesis lattice object, used for
    creating a string representation of the lattice suitable for Genesis input and lattice files.

    - :class:`~simba.Codes.Genesis.Genesis.genesisCommandFile`: Base class for defining
    commands in a Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_setup_command`: Class for defining the
    &setup portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_alter_setup_command`: Class for defining the
    &alter_setup portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_lattice_command`: Class for defining the
    &lattice portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_time_command`: Class for defining the
    &time portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_profile_const_command`: Class for defining the
    &profile_const portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_profile_gauss_command`: Class for defining the
    &profile_gauss portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_profile_step_command`: Class for defining the
    &profile_step portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_profile_polynom_command`: Class for defining the
    &profile_polynom portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_profile_file_command`: Class for defining the
    &profile_file portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_sequence_const_command`: Class for defining the
    &sequence_const portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_sequence_polynom_command`: Class for defining the
    &sequence_polynom portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_sequence_power_command`: Class for defining the
    &sequence_power portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_sequence_random_command`: Class for defining the
    &sequence_random portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_beam_command`: Class for defining the
    &beam portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_alter_beam_command`: Class for defining the
    &alter_beam portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_field_command`: Class for defining the
    &field portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_importdistribution_command`: Class for defining the
    &importdistribution portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_importbeam_command`: Class for defining the
    &importbeam portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_importfield_command`: Class for defining the
    &importfield portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_importtransformation_command`: Class for defining the
    &importtransformation portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_efield_command`: Class for defining the
    &efield portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_sponrad_command`: Class for defining the
    &sponrad portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_wake_command`: Class for defining the
    &wake portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_sort_command`: Class for defining the
    &sort portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_write_command`: Class for defining the
    &write portion of the Genesis input file.

    - :class:`~simba.Codes.Genesis.Genesis.genesis_track_command`: Class for defining the
    &track portion of the Genesis input file.
"""

import os
import time
from copy import copy, deepcopy
import subprocess
import numpy as np
from random import randint
from warnings import warn
from scipy.constants import speed_of_light

from pydantic import (
    computed_field,
    Field,
    field_validator,
)

from ...Elements.fel_modulator import fel_modulator
from ...Elements.wiggler import wiggler

try:
    import sdds
except Exception:
    print("No SDDS available!")
from typing import ClassVar
from ...Framework_objects import (
    frameworkLattice,
    frameworkCommand,
)
from ...FrameworkHelperFunctions import saveFile, expand_substitution
from ...Modules import Beams as rbf
from typing import Dict, List, Literal

command_files_order = [
    "setup",
    "time",
    "lattice",
    "profile_file",
    "profile_const",
    "profile_polynom",
    "profile_gauus",
    "field",
    "importdistribution",
    "importbeam",
    "importfield",
    "beam",
    "alter_setup",
    "alter_beam",
    "sort",
    "track",
    "write",
    "end",
]

class genesisLattice(frameworkLattice):
    """
    Class for defining the Genesis lattice object, used for
    creating a string representation of
    the lattice suitable for a Genesis input file.
    """

    code: str = "genesis"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = False
    """Flag to indicate whether negative drifts are allowed"""

    particle_definition: str | None = None
    """String representation of the initial particle distribution"""

    bunch_charge: float | None = None
    """Bunch charge"""

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
    """Dictionary of :class:`~simba.Codes.Genesis.Genesis.elegantCommandFile`
    objects for writing to the Genesis input file"""

    commandFilesOrder: List = []
    """Order in which commands are to be written in the Genesis input file"""

    element_name_converted: Dict = {}
    """Genesis elements may need their names to be converted"""

    fundamental_wavelength: float = None
    """Fundamental wavelength of the beamline; if not provided, it is calculated from the beam
    energy and the strength of the first undulator"""

    shot_noise: bool = True
    """Include shot noise in the calculation"""

    npart: int = None
    """Number of macro particles per slice; if not provided, calculate from the beam"""

    nbins: int = 4
    """Number of macro particles to be grouped into beamlets"""

    seed: int = randint(1,10000000)
    """Random number seed"""

    match: float = None
    """If True, use `zmatch` in :class:`~simba.Codes.Genesis.Genesis.genesis_lattice_command`
    and set the location to the middle of the first undulator"""

    field_power: float = 1e3
    """Initial power for :class:`~simba.Codes.Genesis.Genesis.genesis_field_command`"""

    dgrid: float = 1e-4
    """Grid size for :class:`~simba.Codes.Genesis.Genesis.genesis_field_command`"""

    ngrid: int = 251
    """Number of grids for :class:`~simba.Codes.Genesis.Genesis.genesis_field_command`"""

    waist_size: float = 1e-5
    """Waist size for :class:`~simba.Codes.Genesis.Genesis.genesis_field_command`"""

    beam_type: Literal["beam", "profile", "distribution"] = "beam"
    """Method for loading in electron beam: `beam`=use `&beam` with numerical values; 
    `profile`=use `&beam` with profile labels; `distribution`: use `&importdistribution`"""

    beam_slices: int = 128
    """Number of beam slices for `profile`"""

    steady_state: bool = True
    """If `True`, run in steady-state mode; if not, set `time=true` and set up simulation window
    based on beam length"""

    one4one: bool = False
    """If `True`, run in one-for-one mode; if not, use :attr:`~npart` and :attr:`~nbins`"""


    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.elementObjects[self.start].objectname

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into a Genesis-compatible format; see
        :attr:`~simba.Framework_objects.frameworkLattice.elementObjects`.

        Returns
        -------
        str
            The lattice represented as a string compatible with Genesis
        """
        elements = self.createDrifts()
        text = ""
        for i, element in enumerate(list(elements.values())):
            if not element.subelement:
                text += element.write_Genesis()
        text += self.objectname + ": Line={"
        for e, element in list(elements.items()):
            if not element.subelement:
                text += e + ", "
        fulltext = text[:-2] + " };\n"
        return fulltext

    def write(self) -> None:
        """
        Write the Genesis lattice and command files to `master_subdir` using the functions
        :func:`~simba.Codes.Genesis.Genesis.writeElements` and
        based on the output of :func:`~simba.Codes.Genesis.Genesis.createCommandFiles`.
        """
        lattice_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".lat"
        )
        saveFile(lattice_file, self.writeElements())
        # try:
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(command_file, "", "w")
        if len(command_files_order) > 0:
            for cfileid in command_files_order:
                if cfileid in self.commandFiles:
                    cfile = self.commandFiles[cfileid]
                    saveFile(command_file, cfile.write_Genesis(), "a")
        else:
            warn("commandFilesOrder length is zero; run createCommandFiles first")

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.hdf5_to_genesis(prefix)
        self.write_setup_file()
        if self.steady_state:
            self.write_time()
        if self.match:
            self.commandFiles["lattice"] = genesis_lattice_command(
                zmatch=self.match,
            )
        self.commandFiles["field"] = genesis_field_command(
            power=self.field_power,
            ngrid=self.ngrid,
            dgrid=self.dgrid,
            waist_size=self.waist_size,
        )
        self.commandFiles["track"] = genesis_track_command()
        self.commandFiles["write"] = genesis_write_command(
            field=self.elementObjects[self.end].objectname,
            beam=self.elementObjects[self.end].objectname,
        )

    def write_setup_file(self) -> None:
        gamma0 = self.global_parameters["beam"].beam.centroids.mean_gamma.val
        first_wiggler = self.wigglers_and_modulators[0]
        if not self.fundamental_wavelength:
            self.fundamental_wavelength = first_wiggler.period_length / (2 * gamma0**2)
            self.fundamental_wavelength *= (1 + first_wiggler.k**2 / 2)
        else:
            lambda0_from_und = first_wiggler.period_length / (2 * gamma0**2) * (1 + first_wiggler.k**2 / 2)
            if not np.isclose([self.fundamental_wavelength], [lambda0_from_und]):
                warn("First undulator strength is not close to fundamental_wavelength")
        if not self.npart:
            beamlen = len(self.global_parameters["beam"].x.val)
            parts_per_lambda = int(np.std(self.global_parameters["beam"].z.val) / self.fundamental_wavelength)
            self.npart = 1 << (parts_per_lambda - 1).bit_length()
            warn(f"npart not provided; setting npart to {self.npart}")
        delz = first_wiggler.period_length
        self.commandFiles["setup"] = genesis_setup_command(
            rootname=self.objectname,
            lattice=self.objectname + ".lat",
            outputdir=self.global_parameters["master_subdir"],
            beamline=self.objectname,
            one4one=self.one4one,
            lambda0=self.fundamental_wavelength,
            gamma0=gamma0,
            delz=delz,
            shotnoise=self.shot_noise,
            nbins=self.nbins,
            npart=self.npart,
        )

    def write_time(self) -> None:
        self.global_parameters["beam"].beam.slice.bin_time()
        tbins = self.global_parameters["beam"].beam.slice._t_Bins.val
        print(tbins)
        slen = (tbins[-1] - tbins[0]) * speed_of_light
        print(slen)
        self.commandFiles["time"] = genesis_time_command(
            slen=slen,
            time=True,
            sample=1
        )

    def hdf5_to_genesis(self, prefix: str="", write: bool=True) -> None:
        """
        Convert the initial HDF5 particle distribution to Genesis format.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        HDF5filename = prefix + self.particle_definition + ".hdf5"
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        else:
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        rbf.hdf5.read_HDF5_beam_file(
            self.global_parameters["beam"],
            os.path.abspath(filepath),
        )
        hdf5outname = f'{self.global_parameters["master_subdir"]}/{self.elementObjects[self.start].objectname}.hdf5'
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            hdf5outname,
        )
        genesisbeamfilename = hdf5outname.replace("hdf5", "genesis.hdf5")
        if self.beam_type == "profile":
            rbf.genesis.write_genesis_beam_profiles(
                self.global_parameters["beam"],
                genesisbeamfilename,
                self.beam_slices,
            )
            beam_profile_properties = self.get_beam_profile_properties()
            props = {}
            self.commandFiles["profile_file"] = []
            for b in beam_profile_properties:
                self.commandFiles["profile_file"].append(
                    genesis_profile_file_command(
                        label=f"{b}_profile",
                        xdata="z",
                        ydata="b",
                    )
                )
                props.update({b: f"{b}_profile"})
            self.commandFiles["beam"] = genesis_beam_command(**props)
        elif self.beam_type == "beam":
            beam_properties = self.get_average_beam_properties()
            self.commandFiles["beam"] = genesis_beam_command(**beam_properties)
        else:
            raise ValueError(f"beam_type {self.beam_type} not understood")

    def get_average_beam_properties(self) -> Dict:
        beam = deepcopy(self.global_parameters["beam"])
        ddd =  {
            "betax": float(beam.twiss.beta_x.val),
            "betay": float(beam.twiss.beta_y.val),
            "alphax": float(beam.twiss.alpha_x.val),
            "alphay": float(beam.twiss.alpha_y.val),
            "gamma0": float(beam.centroids.mean_gamma.val),
            "delgam": float(beam.sigmas.sigma_cp_eV.val) / float(beam.E0_eV.val[0]),
            "current": float(beam.slice.peak_current.val),
            "xcenter": float(beam.centroids.mean_x.val),
            "ycenter": float(beam.centroids.mean_y.val),
            "pxcenter": float(beam.centroids.mean_cpx.val) / float(beam.E0_eV.val[0]),
            "pycenter": float(beam.centroids.mean_cpy.val) / float(beam.E0_eV.val[0]),
            "ex": float(beam.emittance.normalized_horizontal_emittance.val),
            "ey": float(beam.emittance.normalized_vertical_emittance.val),
        }
        return ddd

    def get_beam_profile_properties(self) -> List:
        return [
            "betax"
            "betay"
            "alphax"
            "alphay"
            "gamma0"
            "delgam"
            "current"
            "xcenter"
            "ycenter"
            "pxcenter"
            "pycenter"
        ]

        # ocebeamfilename = hdf5outname.replace("hdf5", "ocelot.npz")
        # self.pin = rbf.beam.write_ocelot_beam_file(
        #     self.global_parameters["beam"], ocebeamfilename, write=write
        # )

class genesisCommandFile(frameworkCommand):
    """
    Generic class for generating elements for a Genesis input file
    """

class genesis_setup_command(genesisCommandFile):
    """
    Class for defining the &setup portion of the Genesis input file.
    """

    objectname: str = "setup"
    """Name of object for frameworkObject"""

    objecttype: str = "setup"
    """Type of object for frameworkObject"""

    rootname: str
    """The basic string, with which all output files will start, 
    unless the output filename is directly overwritten (see write namelist)"""

    outputdir: str
    """Output directory name."""

    lattice: str
    """The name of the file which contains the undulator lattice description. 
    This can also include some relative paths if the lattice file is not in the same directory as the input file."""

    beamline: str
    """The name of the beamline, which has to be defined within the lattice file."""

    gamma0: float
    """The reference energy in units of the electron rest mass."""

    lambda0: float
    """The reference wavelength in meter, which is used as the wavelength in 
    steady-state simulation or for defining the sample distance in time-dependent runs. 
    It also acts as the default value when field distributions are generated."""

    delz: float
    """Preferred integration stepsize in meter."""

    seed: int = 123456789
    """Seed to initialize the random number generator, 
    which is used for shot noise calculation and undulator lattice errors"""

    npart: int
    """Number of macro particles per slice. Note that the number must be a multiple of the used 
    bins nbins otherwise Genesis will exit with an error. 
    If one-for-one simulations are used, this parameter has no meaning."""

    nbins: int
    """Number of macro particles, which are grouped into beamlets for generating the correct shot noise. 
    For one-for-one simulations this parameter has no meaning"""

    one4one: bool = False
    """Flag to enable or disable resolving each electron in the simulation. 
    This is mandatory for certain features, such as sorting or slicing of particle distributions. 
    If set to true other parameters such as :attr:`~npart` and attr:`~nbins` are obsolete 
    and do not need to be defined. 
    It is recommended to estimate the number of electrons, which are generated in the simulations, 
    because this can easily required memory beyond what is available on the computer."""

    shotnoise: bool = True
    """Flag to enable the calculation of shotnoise per each slice during generation 
    of the electron distribution. It is recommended to set the value to false for 
    steady-state or scan simulations."""

    beam_global_stat: bool = False
    """Flag to enable extra output of beam parameters of the entire bunch, 
    such as energy, energy spread etc. 
    The data are placed in the HDF group ”Global” within the group ”Beam” of the output file"""

    field_global_stat: bool = False
    """Flag for the field output, similar to attr:`~beam_global_stat`."""

    exclude_spatial_output: bool = False
    """Flag to suppress the datasets in the output file for the x- and y-position and size 
    (both Beam and Field) and px- and py-position (Beam only). 
    This might be useful to reduce the file size of the output file, 
    if these datasets are not needed for the post-processing"""

    exclude_fft_output: bool = False
    """Flag to suppress the datasets in the output file for the field divergence and pointing. 
    Since it also disable the FFT calculation of the 2D wavefronts it speeds up the 
    execution time slightly. If the code has been compiled without the support of the 
    FFTW library this parameter has no effect."""

    exclude_intensity_output: bool = False
    """Flag to suppress the datasets for the near and farfield intensity and phase 
    for the radiation field. If excluded the output file size becomes smaller 
    but no post-processing calculation of the spectra is possible."""

    exclude_energy_output: bool = False
    """Flag to suppress the datasets in the output file for the mean 
    energy and energy spread of the electron beam."""

    exclude_aux_output: bool = False
    """Flag to suppress the auxiliary datasets in the output file. 
    In the moment it is the long-range longitudinal electric field as seen by the electrons."""

    exclude_current_output: bool = True
    """Flag to reduce the size of the current dataset for the electron beam. 
    Under most circumstances the current profile is constant and only the initial current 
    profile is written out. However, simulation with one-4-one set to true and sorting 
    events the current profile might change. Example are ESASE/HGHG schemes. 
    By setting the flag to false the current profile is written out at each output step 
    similar to radiation power and bunching profile."""

    exclude_twiss_output: bool = True
    """Flag to reduce the size of the twiss (emittance, beta and alpha values) dataset 
    for the electron beam. Under most circumstances the twiss parameters are constant 
    and only the initial values are written out. However, simulation with :attr:`~one4one` 
    set to True and sorting events the twiss parameters might change. Example are 
    ESASE/HGHG schemes. By setting the flag to false the twiss values written out 
    at each output step similar to radiation power and bunching profile."""

    exclude_field_dump: bool = False
    """Exclude the field dump to .fld.h5."""

    write_meta_file: bool = False
    """Write a metadata file."""

    semaphore_file_name: str = ""
    """Providing a file name for the semaphore file always switches on writing the 
    "done" semaphore file, overriding 'write_semaphore_file' flag. 
    This allows to switch on semaphore functionality just by specifying corresponding 
    command line argument -- no modification of G4 input file needed."""

    write_semaphore_file: bool = False
    """Write a semaphore file when the simulation has completed."""

    write_semaphore_file_done: bool = False
    """Alias for write_semaphore_file. 
    This takes precedence over :attr:`~write_semaphore_file` if both are specified."""

    write_semaphore_file_started: bool = False
    """Write a semaphore file at startup, after the setup block is parsed."""

class genesis_alter_setup_command(genesisCommandFile):
    """
    Class for defining the &alter_setup portion of the Genesis input file.
    """

    objectname: str = "alter_setup"
    """Name of object for frameworkObject"""

    objecttype: str = "alter_setup"
    """Type of object for frameworkObject"""

    rootname: str
    """The basic string, with which all output files will start, 
    unless the output filename is directly overwritten (see 
    :class:`~simba.Codes.Genesis.Genesis.genesis_write_command`)"""

    beamline: str
    """The name of the beamline, which has to be defined within the lattice file. 
    This way another beamline can be selected in the case the simulation has multiple stages"""

    delz: float
    """Preferred integration stepsize in meter. Note that this is not a strict value because 
    Genesis tries to optimized the stepsize according to the elements it can resolve."""

    harmonic: int = 1
    """If the value is not 1 than a harmonic conversion is done. This has several consequences. 
    The reference wavelength in setup is divided by the harmonic number, the sample rate in time 
    is multiplied by the harmonic number, the ponderomotive phases of all macro particles are 
    scaled with the harmonic number, all radiation fields, which are not identical to the harmonic 
    numbers are deleted, while an existing harmonic field is changed to be at the fundamental wavelength"""

    subharmonic: int = 1
    """If the value is not 1 than a down conversion is done. It is similar to the action of harmonics 
    but in the opposite directions. For the radiation field all field definitions are deleted except 
    for the fundamental, which is converted to a harmonic. In this case the fundamental field needs 
    to be defined before another tracking is called."""

    resample: bool = False
    """If this is set to true and only if one-for-one simulations are used the harmonic and subharmonic 
    conversion can re-sample to the new wavelength. In the case of up-conversion the slices are 
    split and the total number of slices increases. Same with the radiation field. An previously 
    existing harmonic field, which is now becoming the fundamental, is interpolated between the 
    existing sample points (still needs to be implemented). If a new field is generated it has 
    automatically the new number of slices. If also prevents that the sample rate is changed by 
    remaining unchanged."""

    disable: bool = False
    """Disable non-matching radiation harmonic."""


class genesis_lattice_command(genesisCommandFile):
    """
    Class for defining the &lattice portion of the Genesis input file.
    """

    objectname: str = "lattice"
    """Name of object for frameworkObject"""

    objecttype: str = "lattice"
    """Type of object for frameworkObject"""

    zmatch: float = 0.0
    """If the position within the undulator in meter is non-zero than Genesis tries to 
    calculate the matched optics function for a periodic solution. In the case that it 
    cannot find a solution than it will report it. Found solution will also be the default 
    values for a succeeding beam generation, so that no explicit optical functions need to 
    be defined any longer. If the lattice is highly non-periodic it is recommended to 
    find the matching condition with an external program such as MAdX."""

    element: str = ""
    """Name of the element type, which will be changed, e.g. Undulator if undulator modules 
    are altered. Only the first 4 letters need to be defined. If there is no match, e.g. due 
    to a type, nothing will be changed. It acts rather as a filter than a mandatory element. 
    Elements of the type MARKER are not supported."""

    field: str = ""
    """Attribute name for a given element. The names are the same as in the definition of the 
    lattice file. The field acts as a filter again. With non-matching events nothing will be changed."""

    value: float | str = 0.0
    """The new value. If a reference to a sequence is used, values can be different depending 
    on how many elements are changed. For a double the value would be the same for all elements affected."""

    instance: int = 0
    """The instances of affected elements. If a positive value is given, than only that element is 
    changed, where its occurence matches the number. E.g. for a value of 3 only the third element 
    is selected. For a value of 0 all elements are changed. The ability to change more than 
    one but less than all is currently not supported."""

    add: bool = True
    """If true, the changes are added to the existing value; if false, the old values are overwritten."""


class genesis_time_command(genesisCommandFile):
    """
    Class for defining the &time portion of the Genesis input file.
    """

    objectname: str = "time"
    """Name of object for frameworkObject"""

    objecttype: str = "time"
    """Type of object for frameworkObject"""

    s0: float = 0.0
    """Starting point of the time-window in meters."""

    slen: float = 0.0
    """Length of the time window in meters. 
    Note that for parallel jobs this might be adjusted towards larger values."""

    sample: int = 1
    """Sample rate in units of the reference wavelength from the 
    :class:`~simba.Codes.Genesis.Genesis.genesis_setup_command` namelist, 
    so that the number of slices is given by SLEN / LAMBDA0 / SAMPLE after SLEN 
    has been adjusted to fit the MPI size."""

    time: bool = True
    """Flag to indicate time-dependent run. Note that time-dependent simulations are 
    enabled already by using this namelist. This flag has the functionality to differentiate 
    between time-dependent run and scans, which disable the slippage in the tracking. 
    To restrict the simulation to steady-state the time namelist has to be omitted from the input deck."""


class genesis_profile_const_command(genesisCommandFile):
    """
    Class for defining the &profile_const portion of the Genesis input file.
    """

    objectname: str = "profile_const"
    """Name of object for frameworkObject"""

    objecttype: str = "profile_const"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""


class genesis_profile_gauss_command(genesisCommandFile):
    """
    Class for defining the &profile_gauss portion of the Genesis input file.
    """

    objectname: str = "profile_gauss"
    """Name of object for frameworkObject"""

    objecttype: str = "profile_gauss"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""

    s0: float
    """Center point of the Gaussian distribution"""

    sig: float
    """Standard deviation of the Gaussian distribution"""

class genesis_profile_step_command(genesisCommandFile):
    """
    Class for defining the &profile_step portion of the Genesis input file.
    """

    objectname: str = "profile_step"
    """Name of object for frameworkObject"""

    objecttype: str = "profile_step"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""

    s_start: float
    """Starting point of the step function"""

    s_end: float
    """End point of the step function"""

class genesis_profile_polynom_command(genesisCommandFile):
    """
    Class for defining the &profile_polynom portion of the Genesis input file.
    """

    objectname: str = "profile_polynom"
    """Name of object for frameworkObject"""

    objecttype: str = "profile_polynom"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""

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
    Class for defining the &profile_file portion of the Genesis input file.
    """

    objectname: str = "profile_file"
    """Name of object for frameworkObject"""

    objecttype: str = "profile_file"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    xdata: str
    """Points to a dataset in an HDF5 file to define the s-position for the look-up table. 
    The format is filename/group1/.../groupn/datasetname, where the naming of groups is 
    not required if the dataset is at root level of the HDF file"""

    ydata: str
    """Same as :attr:`~xdata` but for the function values of the look-up table."""

    isTime: bool = False
    """If true the s-position is a time variable and therefore multiplied 
    with the speed of light c to get the position in meters."""

    reverse: bool = False
    """If true the order in the look-up table is reverse. 
    This is sometimes needed because time and spatial coordinates differ sometimes by a minus sign."""

    autoassign: bool = False
    """Use the HDF5 file from :attr:`~xdata`."""


class genesis_sequence_const_command(genesisCommandFile):
    """
    Class for defining the &sequence_const portion of the Genesis input file.
    """

    objectname: str = "sequence_const"
    """Name of object for frameworkObject"""

    objecttype: str = "sequence_const"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""

class genesis_sequence_polynom_command(genesisCommandFile):
    """
    Class for defining the &sequence_polynom portion of the Genesis input file.
    """

    objectname: str = "sequence_polynom"
    """Name of object for frameworkObject"""

    objecttype: str = "sequence_polynom"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""

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
    Class for defining the &sequence_power portion of the Genesis input file.
    """

    objectname: str = "sequence_power"
    """Name of object for frameworkObject"""

    objecttype: str = "sequence_power"
    """Type of object for frameworkObject"""

    label: str
    """Name of the profile, which is used to refer to it in later calls of namelists"""

    c0: float
    """Constant value to be used."""

    dc: float
    """Term scaling the growing power series before added to the constant term."""

    alpha: float
    """Power of the series."""

    n0: int = 1
    """Starting index of power growth. Otherwise the sequence uses only the constant term"""

    c4: float = 0.0
    """Term proportional to s^4."""

class genesis_sequence_random_command(genesisCommandFile):
    """
    Class for defining the &sequence_random portion of the Genesis input file.
    """

    objectname: str = "sequence_random"
    """Name of object for frameworkObject"""

    objecttype: str = "sequence_random"
    """Type of object for frameworkObject"""

    label: str
    """Name of the sequence, which is used to refer to it in the lattice"""

    c0: float = 0.0
    """Mean value"""

    dc: float = 0.0
    """Amplitude of the error, either the standard division for normal 
    distribution or the min and max value for uniform distribution."""

    seed: int = 100
    """Seed for the random number generator"""

    normal: bool = True
    """Flag for Gaussian distribution. If False a uniform distribution is used."""


class genesis_beam_command(genesisCommandFile):
    """
    Class for defining the &beam portion of the Genesis input file.
    """

    objectname: str = "beam"
    """Name of object for frameworkObject"""

    objecttype: str = "beam"
    """Type of object for frameworkObject"""

    gamma0: float | str
    """Mean energy in units of the electron rest mass."""

    delgam: float | str
    """RMS energy spread in units of the electron rest mass."""

    current: float | str
    """Current in Amperes."""

    ex: float | str
    """Normalized emittance in x in units of m-rad"""

    ey: float | str
    """Normalized emittance in y in units of m-rad"""

    betax: float | str
    """Initial beta-function in x in meters. If the matched command has been invoked 
    before the default values are set to the results;
    see :attr:`~simba.Codes.Genesis.Genesis.genesis_lattice_command.zmatch`."""

    betay: float | str
    """Initial beta-function in y in meters; see :attr:`~betax`"""

    alphax: float | str
    """Initial alpha-function in x; see :attr:`~betax`"""

    alphay: float | str
    """Initial alpha-function in y; see :attr:`~betax`"""

    xcenter: float | str = 0.0
    """Initial centroid position in x in meter."""

    ycenter: float | str = 0.0
    """Initial centroid position in y in meter."""

    pxcenter: float | str = 0.0
    """Initial centroid momentum in x in units of γβx."""

    pycenter: float | str = 0.0
    """Initial centroid momentum in y in units γβy."""

    bunch: float | str = 0.0
    """Initial bunching value"""

    bunchphase: float | str = 0.0
    """Initial phase of the bunching"""

    emod: float | str = 0.0
    """Initial energy modulation in units of the electron rest mass. 
    This modulation is on the scale of the reference wavelength"""

    emodphase: float | str = 0.0
    """Initial phase of the energy modulation"""


class genesis_alter_beam_command(genesisCommandFile):
    """
    Class for defining the &alter_beam portion of the Genesis input file.
    """

    objectname: str = "alter_beam"
    """Name of object for frameworkObject"""

    objecttype: str = "alter_beam"
    """Type of object for frameworkObject"""

    dgamma: float | str = 0.0
    """Amplitude of the sinusoidal modulation in units of the electron rest mass"""

    phase: float | str = 0.0
    """Phase of the energy modulation in units of radians."""

    _lambda: float
    """Wavelength in m of the external energy modulation"""

    r56: float = 0
    """R56 element of the magnetic chicane in m"""

    @computed_field(alias="lambda")
    @property
    def lambda_(self) -> float:
        return self._lambda


class genesis_field_command(genesisCommandFile):
    """
    Class for defining the &field portion of the Genesis input file.
    """

    objectname: str = "field"
    """Name of object for frameworkObject"""

    objecttype: str = "field"
    """Type of object for frameworkObject"""

    _lambda: float
    """Central frequency of the radiation mode. 
    The default value is the reference wavelength from 
    :attr:`~simba.Codes.Genesis.Genesis.genesis_setup_command.lambda0`."""

    power: float | str = 0.0
    """Radiation power in Watts"""

    phase: float | str = 0.0
    """Radiation phase in rads. Note that a linear profile results in a shift in the 
    radiation wavelength, which is also the method if for the variable lambda a different 
    value than the reference wavelength is used. In case of conflicts the profile 
    for the phase definition has priority."""

    waist_pos: float | str = 0.0
    """Position where the focal point is located relative to the undulator entrance. 
    Negative values place it before, resulting in a diverging radiation field."""

    waist_size: float | str
    """Waist size according to the definition of w 0 according to Siegman’s ’Laser’ handbook"""

    xcenter: float = 0.0
    """Center position in x in meter of the Gauss-Hermite mode"""

    ycenter: float = 0.0
    """Center position in y in meter of the Gauss-Hermite mode"""

    xangle: float = 0.0
    """Injection angle in x in rad of the Gauss-Hermite mode"""

    yangle: float = 0.0
    """Injection angle in y in rad of the Gauss-Hermite mode"""

    dgrid: float = 0.001
    """Grid extension from the center to one edge. The whole grid is twice as 
    large with 0 as the center position"""

    ngrid: int = Field(default=151, gt=1)
    """Number of grid points in one dimension. This value should be odd to 
    enforce a grid point directly on axis. Otherwise the convergence in the simulations could be worse."""

    harm: int = 1
    """Harmonic number of the radiation field with respect to the reference wavelength."""

    nx: int = 0
    """Mode number in x of the Gauss-Hermite mode"""

    ny: int = 0
    """Mode number in y of the Gauss-Hermite mode"""

    accumulate: bool = True
    """If True the generated field is added to an existing field instead of overwriting it."""

    @field_validator("ngrid")
    def check_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("ngrid must be odd")
        return v


class genesis_importdistribution_command(genesisCommandFile):
    """
    Class for defining the &importdistribution portion of the Genesis input file.
    """

    objectname: str = "importdistribution"
    """Name of object for frameworkObject"""

    objecttype: str = "importdistribution"
    """Type of object for frameworkObject"""

    file: str
    """The file name of the distribution, including possible relative directories."""

    charge: float
    """Total charge of the distribution to calculate the current and individual charge per macro particle."""

    slicewidth: float
    """The fraction in length of the distribution which is used for reconstruction. 
    E.g if the length is 10 μm and slice width 0.02 then the reconstruction at the 
    positions s = 4 μm is using those particles in the distribution, 
    which are located in the slice from 3.9 μm to 4.1 μm."""

    center: bool = False
    """If True the particle distribution is recentered in transverse position, momenta and energy."""

    gamma0: float
    """If :attr:`~center` is enabled, new center in energy in units of electron rest mass."""

    x0: float = 0.0
    """If :attr:`~center` is enabled, new center in x in meter."""

    y0: float = 0.0
    """If :attr:`~center` is enabled, new center in y in meter."""

    px0: float = 0.0
    """If :attr:`~center` is enabled, new mean momentum in x in γβx."""

    py0: float = 0.0
    """If :attr:`~center` is enabled, new mean momentum in y in γβy."""

    match: bool = False
    """If True, the particle distribution is matched to new optical function values."""

    betax: float = 15.0
    """If matching is enabled, new beta function in x in meters."""

    betay: float = 15.0
    """If matching is enabled, new beta function in y in meters."""

    alphax: float = 0.0
    """If matching is enabled, new alpha function in x."""

    alphay: float = 0.0
    """If matching is enabled, new alpha function in y."""

    eval_start: float = 0.0
    """Evaluation start."""

    eval_end: float = 1.0
    """Evaluation end."""

    settimewindow: bool = False
    """Set time window."""


class genesis_importbeam_command(genesisCommandFile):
    """
    Class for defining the &importbeam portion of the Genesis input file.
    """

    objectname: str = "importbeam"
    """Name of object for frameworkObject"""

    objecttype: str = "importbeam"
    """Type of object for frameworkObject"""

    file: str
    """File name of a hdf5 complient datafile to contain the slice-wise particle distribution. 
    It has to follow the internal Genesis 1.3 syntax."""

    time: bool = True
    """If the time window hasn’t been defined it allows to run Genesis with the imported distribution 
    in scan mode, when set to false. This would disable all slippage and long-range 
    collective effects in the simulation"""


class genesis_importfield_command(genesisCommandFile):
    """
    Class for defining the &importfield portion of the Genesis input file.
    """

    objectname: str = "importfield"
    """Name of object for frameworkObject"""

    objecttype: str = "importfield"
    """Type of object for frameworkObject"""

    file: str
    """File name of a hdf5 compliant datafile to contain the slice-wise particle distribution. 
    It has to follow the internal Genesis 1.3 syntax."""

    harmonic: int = 1
    """Defines the harmonic for the given Genesis run."""

    time: bool = True
    """If the time window hasn’t been defined it allows to run Genesis with the 
    imported distribution in scan mode, when set to false. This would disable 
    all slippage and long-range collective effects in the simulation"""

    attenuation: float = 1.0
    """Apply an on-the-flight scaling factor to the field to be imported, 
    without the need of modifying the original field file."""

    offset: float = 0.0
    """Additional offset of the field with respect to the time frame. 
    It should be an integer multiple of the slice length as defined in the field dump file"""


class genesis_importtransformation_command(genesisCommandFile):
    """
    Class for defining the &importtransformation portion of the Genesis input file.
    """

    objectname: str = "importtransformation"
    """Name of object for frameworkObject"""

    objecttype: str = "importtransformation"
    """Type of object for frameworkObject"""

    file: str
    """File name of a hdf5 compliant datafile to contain the vector and matrix informations"""

    vector: str
    """Name of the dataset which contains the vector information. The shape must be either (6) or (n,6)"""

    matrix: str
    """Name of the dataset which contains the matrix information. The shape must be either (6,6) or (n,6,6)"""

    slen: float = 0.0
    """The length in meters between adjacent sample points (n>1), needed for the interpolation. 
    If the value is zero only a global transformation is applied using the first entry."""


class genesis_efield_command(genesisCommandFile):
    """
    Class for defining the &efield portion of the Genesis input file.
    """

    objectname: str = "efield"
    """Name of object for frameworkObject"""

    objecttype: str = "efield"
    """Type of object for frameworkObject"""

    longrange: bool = False
    """Flag to enable the calculation of the long range space charge field."""

    rmax: float = 0.0
    """Size of radial grid in meters. If the beam size gets larger than the grid the size is 
    automatically adjusted to the maximum radius of the electrons with an additional 50% extension. 
    When the mesh size is adjusted a message will be printed on screen."""

    nz: int = 0.0
    """Number of longitudinal Fourier component of the short range space charge field. 
    Note that this should be not in conflict with the beamlet size."""

    nphi: int = 0.0
    """Number of azimuthal modes in the calculation of the short range space charge field."""

    ngrid: int = 100
    """Number of grid points of the radial grid for the short range space charge field."""


class genesis_sponrad_command(genesisCommandFile):
    """
    Class for defining the &sponrad portion of the Genesis input file.
    """

    objectname: str = "sponrad"
    """Name of object for frameworkObject"""

    objecttype: str = "sponrad"
    """Type of object for frameworkObject"""

    seed: int = 1234
    """Seed for random number generator to model the quantum fluctuation of hard photons."""

    doLoss: bool = False
    """If True, electrons will lose energy due to the emission of spontaneous radiation within the undulator"""

    doSpread: bool = False
    """If True, the energy spread will increase due to the fluctuation in the emission 
    of hard photons of the spontaneous radiation."""


class genesis_wake_command(genesisCommandFile):
    """
    Class for defining the &wake portion of the Genesis input file.
    """

    objectname: str = "wake"
    """Name of object for frameworkObject"""

    objecttype: str = "wake"
    """Type of object for frameworkObject"""

    loss: float | str = 0.0
    """Loss in eV/m . This is a global loss function (in particular if a profile is defined). 
    Its function values V(s) remains unchanged even if the current profile changes"""

    radius: float = 0.0025
    """Radius of the aperture if it is a round chamber or half the distance in the case of two parallel plates."""

    roundpipe: bool = True
    """Flag to indicate the shape of the transverse cross-section of the aperture. 
    If True, a round aperture is assumed, otherwise the model has two parallel plates."""

    conductivity: float = 0.0
    """Conductivity of the vacuum material for the resistive wall wakefield function"""

    relaxation: float = 0.0
    """Relaxation distance (aka the mean free path of the electron in the vacuum material) 
    for the resistive wall wakefields"""

    material: Literal["CU", "AL", ""] = ""
    """String literal to define conductivity and relaxation distance for either copper or aluminum 
    by using the two character label ’CU’ or ’AL’ respectively. 
    This overwrites also any explicit definition of the conductivity and relaxation value."""

    gap: float = 0.0
    """Length in mm of a longitudinal gap in the aperture, exciting geometric wakes."""

    lgap: float = 1.0
    """Effective length over which a single gap is applied. E.g. if there is a periodicity of 
    4.5 m at which there is always the same gap in the aperture for the geometric wakes,
    then this value should be put to 4.5 m."""

    hrough: float = 0.0
    """Amplitude in meters of a sinusoidal corrugation, modeling the effect of surface roughness wakes."""

    lrough: float = 0.0
    """Period length in meters of the sinusoidal corrugation of the surface roughness model."""

    transient: bool = False
    """If True, Genesis includes the catch-up length of the origin of the wakefield 
    to the particle effects. E.g. particles do not see immediately the wake from those closer 
    ahead of them than those further away. The catch-up distance is the distance in the 
    undulator added to the starting position :attr:`~ztrans`. If set to false the steady-state model is 
    used, effectively setting :attr:`~ztrans` to infinity. Enabling transient calculation will update 
    the wakefield at each integration step, which can slow down the calculations."""

    ztrans: float = 0.0
    """Reference location of the first source of the wake fields. 
    A positive value means that the condition for wakes (e.g. a small aperture in the vacuum chamber) 
    has already started and there has been already some length to establish the wakes. 
    For a value of zero the source is right at the undulator start, while a negative value prevents 
    any wake, till the interation position has passed that point."""

    output: str = ""
    """Root of the filename, where the single particle wakes are written. 
    The root is extended by .wake.h5 to form the filename."""


class genesis_sort_command(genesisCommandFile):
    """
    Class for defining the &sort portion of the Genesis input file.
    """

    objectname: str = "sort"
    """Name of object for frameworkObject"""

    objecttype: str = "sort"
    """Type of object for frameworkObject"""

class genesis_write_command(genesisCommandFile):
    """
    Class for defining the &write portion of the Genesis input file.
    """

    objectname: str = "write"
    """Name of object for frameworkObject"""

    objecttype: str = "write"
    """Type of object for frameworkObject"""

    field: str = ""
    """If a filename is defined, Genesis writes out the field distribution of all harmonics. 
    The harmonics are indicated by the suffix ’.hxxx.’ where xxx is the harmonic number. 
    The filename gets the extension.fld.h5 automatically"""

    beam: str = ""
    """If a filename is defined, Genesis writes out the particle distribution. 
    The filename gets the extension.par.h5 automatically"""

    stride: int = 1
    """For values larger than 1 the amount of particles written to the file is reduced 
    by only writing each stride-th particle to the dump file."""


class genesis_track_command(genesisCommandFile):
    """
    Class for defining the &track portion of the Genesis input file.
    """

    objectname: str = "track"
    """Name of object for frameworkObject"""

    objecttype: str = "track"
    """Type of object for frameworkObject"""

    zstop: float = 1e9
    """If zstop is shorter than the lattice length the tracking stops at the specified position."""

    output_step: int = 1
    """Defines the number of integration steps before the particle and field distribution is analyzed for output."""

    field_dump_step: int = 0
    """Defines the number of integration steps before a field dump is written. 
    Be careful because for time-dependent simulation it can generate many large output files."""

    beam_dump_step: int = 0
    """Defines the number of integration steps before a particle dump is written. 
    Be careful because for time-dependent simulation it can generate many large output files."""

    sort_step: int = 0
    """Defines the number of steps of integration before the particle distribution is sorted. 
    Works only for one-4-one simulations."""

    s0: float = None
    """Option to override the default time window start from 
    :class:`~simba.Codes.Genesis.Genesis.genesis_time_command`."""

    slen: float = None
    """Option to override the default time window length from 
    :class:`~simba.Codes.Genesis.Genesis.genesis_time_command`."""

    field_dump_at_undexit: bool = False
    """Field dumps at the exit of the undulator (one dump for each undulator in the expanded lattice)."""

    bunchharm: int = Field(default=1, gt=1)
    """Bunching harmonic output setting. Must be >= 1."""

    exclusive_harmonics: bool = False
    """If set to true than only the requested bunching harmonic is included in output. 
    Otherwise all harmonic sup and including the specified harmonics are included."""
