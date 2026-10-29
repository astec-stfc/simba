"""
Particle beams (:class:`beam`) and groups of beams (:class:`beamGroup`), with readers and
writers for each supported code's distribution format.
"""
import os
from pydantic import (
    BaseModel,
    ConfigDict,
)
from typing import Dict, Any, List
import numpy as np
from warnings import warn
import re
import copy
import glob
import h5py
from ..units import UnitValue
from .. import constants
from .Particles import Particles
from . import astra
from . import sdds
from . import gdf
from . import hdf5
from . import mad8
from . import madx
from . import openpmd
from . import xsuite
from . import opal
from . import genesis

try:
    from . import plot

    use_matplotlib = True
except ImportError as e:
    print("Import error - plotting disabled. Missing package:", e)
    use_matplotlib = False

from .Particles.emittance import emittance as emittanceobject
from .Particles.twiss import twiss as twissobject
from .Particles.slice import slice as sliceobject
from .Particles.sigmas import sigmas as sigmasobject
from .Particles.centroids import centroids as centroidsobject
from .Particles.kde import kde as kdeobject

try:
    from .Particles.mve import MVE as MVEobject

    imported_mve = True
except ImportError:
    imported_mve = False

SPECIES = {
    "electron": (constants.m_e, -1),
    "positron": (constants.m_e, 1),
    "proton": (constants.m_p, 1),
    "antiproton": (constants.m_p, -1),
}
"""Mass [kg] and charge sign of each species :meth:`beam.set_species` accepts."""

SPECIES_ALIASES = {f"{name}s": name for name in SPECIES} | {"hydrogen": "proton"}


def get_properties(obj):
    props = [f for f in dir(obj) if type(getattr(obj, f)) is property and f != "__fields_set__"]
    if hasattr(obj, "model_fields"):
        props += list(obj.model_fields.keys())
    return props


parameters = {
    "data": get_properties(Particles),
    "emittance": get_properties(emittanceobject),
    "twiss": get_properties(twissobject),
    "slice": get_properties(sliceobject),
    "sigmas": get_properties(sigmasobject),
    "centroids": get_properties(centroidsobject),
    "kde": get_properties(kdeobject),
    "mve": get_properties(MVEobject) if imported_mve else [],
}


class particlesGroup(BaseModel):
    """The same analysis object (e.g. ``emittance``) from each beam in a :class:`beamGroup`."""
    particles: List = None
    """:class:`~simba.Modules.Beams.Particles.Particles` or analysis objects, one per beam"""


class statsGroup:
    """Applies a numpy reduction (e.g. ``np.mean``) to a parameter of each beam in a :class:`beamGroup`."""

    def __init__(self, beam, function):
        self._beam = beam
        self._func = function

    def __getattr__(self, key):
        var = self._beam.__getitem__(key)
        return UnitValue([self._func(v) for v in var], units="m")


class beamGroup(BaseModel):
    """
    A set of :class:`beam` objects, e.g. from :func:`load_directory`; analysis properties
    return a :class:`particlesGroup`.
    """

    sddsindex: int = 0
    """Index for SDDS files"""

    beams: Dict = {}
    """:class:`beam` objects keyed by filename"""

    def __repr__(self):
        return repr(list(self.beams.keys()))

    def __len__(self):
        return len(list(self.beams.keys()))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.beams.values())[key]
        elif isinstance(key, slice):
            return beamGroup(beams=list(self.beams.items())[key])
        elif hasattr(np, key):
            return statsGroup(self, getattr(np, key))
        for p in parameters:
            if key in parameters[p]:
                return getattr(self, key)
        else:
            return getattr(self, key)

    def __init__(self, filenames=[], beams=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sddsindex = 0
        self.beams = {}
        self._parameters = parameters
        for k, v in beams:
            self.beams[k] = v
        if isinstance(filenames, str):
            filenames = [filenames]
        for f in filenames:
            self.add(f)

    @property
    def data(self):
        return particlesGroup(particles=[b._beam for b in self.beams.values()])

    @property
    def sigmas(self):
        return particlesGroup(particles=[b._beam.sigmas for b in self.beams.values()])

    @property
    def centroids(self):
        return particlesGroup(particles=[b._beam.centroids for b in self.beams.values()])

    @property
    def twiss(self):
        return particlesGroup(particles=[b._beam.twiss for b in self.beams.values()])

    @property
    def slice(self):
        return particlesGroup(particles=[b._beam.slice for b in self.beams.values()])

    @property
    def emittance(self):
        return particlesGroup(particles=[b._beam.emittance for b in self.beams.values()])

    @property
    def kde(self):
        return particlesGroup(particles=[b._beam.kde for b in self.beams.values()])

    @property
    def mve(self):
        return particlesGroup(particles=[b._beam.mve for b in self.beams.values()])

    def sort(self, key="z", function="mean", *args, **kwargs):
        if isinstance(function, str) and hasattr(np, function):
            func = getattr(np, function)
        else:
            func = function
        self.beams = dict(
            sorted(
                self.beams.items(), key=lambda item: func(getattr(item[1], key)), *args, **kwargs
            )
        )
        return self

    def add(self, filename):
        if isinstance(filename, str):
            filename = [filename]
        for file in filename:
            if os.path.isdir(file):
                self.add(glob.glob(os.path.join(file, "*.hdf5")))
            elif os.path.isfile(file):
                file = file.replace("\\", "/")
                try:
                    self.beams[file] = beam(filename=file)
                except Exception:
                    if file in self.beams:
                        del self.beams[file]

    def param(self, param):
        return [getattr(b._beam, param) for b in self.beams.values()]

    def getScreen(self, screen):
        for b in self.beams:
            if screen == os.path.splitext(os.path.basename(b))[0]:
                return self.beams[b]
        return None

    def getScreens(self):
        return {os.path.splitext(os.path.basename(b))[0]: b for b in self.beams}


class beam(BaseModel):
    """
    A particle distribution (:class:`~simba.Modules.Beams.Particles.Particles`, via
    :attr:`Particles`) plus its analysis objects and per-code read/write methods.
    """
    q_over_c: UnitValue = UnitValue(constants.elementary_charge / constants.speed_of_light, "C/c")
    """Elementary charge divided by speed of light"""

    speed_of_light: UnitValue = UnitValue(constants.speed_of_light, "m/s")
    """Speed of light"""

    filename: str | None = None
    """Beam distribution file; loaded on instantiation if given"""

    sddsindex: int = 0
    """Index for SDDS files"""

    code: str | None = None
    """Code from which the beam distribution was generated"""

    turn: int | None = None
    """Turn of a multi-turn run this distribution was recorded on (1-based; 1 for single-pass lines)"""

    reference_particle: np.ndarray | None  = None
    """Reference particle for ASTRA-type distributions"""

    reference_particle_index: int | None  = None
    """Reference particle index for ASTRA-type distributions"""

    longitudinal_reference: np.ndarray | str | None = None
    """Longitudinal reference position for ASTRA-type distributions"""

    starting_position: list | np.ndarray = [0, 0, 0]
    """Beam starting position [x,y,z]"""

    theta: float = 0
    """Horizontal angle of beam distribution"""

    offset: list | np.ndarray = [0, 0, 0]
    """Beam offset from nominal axis [x,y,z]"""

    particle_mass: np.ndarray | None = None
    """Particle mass in kg"""

    species: str = "electron"

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    reference_particle_coords: list = [
        "x",
        "y",
        "z",
        "cpx",
        "cpy",
        "cpz",
        "t",
        "charge",
        "status",
    ]

    def __init__(self, filename=None, step=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._beam = Particles()
        self._parameters = parameters
        self.filename = filename
        self.code = None
        if self.filename is not None:
            self.read_beam_file(self.filename, step=step)
            self.set_species(self.species)

    def model_dump(self, *args, **kwargs) -> Dict:
        full_dump = super().model_dump(*args, **kwargs)
        full_dump.update({"Particles": self._beam.model_dump()})
        return full_dump

    def set_species(self, value):
        name = SPECIES_ALIASES.get(value, value)
        if name not in SPECIES:
            raise ValueError(
                "Species must be one of: electron(s), positron(s), proton(s), antiproton(s), hydrogen"
            )
        mass, sign = SPECIES[name]
        self.species = name
        self.set_particle_mass(mass)
        self._beam.particle_charge = UnitValue(np.full(len(self.x), sign * constants.elementary_charge), units="C")
        # unary minus, not sign * ...: UnitValue.__rmul__ drops the units
        self._beam.charge = abs(self._beam.charge) if sign > 0 else -abs(self._beam.charge)
        self._beam.total_charge = abs(self._beam.total_charge) if sign > 0 else -abs(self._beam.total_charge)
        self._beam.particle_rest_energy_eV = UnitValue(mass * constants.speed_of_light**2 / constants.elementary_charge, units="eV/c")

    @property
    def E0_eV(self) -> float:
        """Particle rest mass energy in eV, falling back to the electron's if the mass is unknown."""
        if self._beam.particle_rest_energy_eV is not None:
            return self._beam.particle_rest_energy_eV
        elif self._beam.particle_rest_energy is not None:
            return np.mean(self._beam.particle_rest_energy) / constants.elementary_charge
        else:
            particle_mass = UnitValue(constants.m_e, "kg")
            E0 = UnitValue(particle_mass * constants.speed_of_light ** 2, "J")
            return UnitValue(E0 / constants.elementary_charge, "eV/c")

    @property
    def beam(self) -> Particles:
        """The particle distribution."""
        return self._beam

    @property
    def fullbeam(self):
        return self.beam.fullbeam

    @fullbeam.setter
    def fullbeam(self, beam):
        self.beam.fullbeam = beam

    @property
    def status(self):
        return np.full(len(self.x), 1)

    @property
    def Particles(self) -> Particles:
        """The particle distribution."""
        return self._beam

    @property
    def data(self) -> Particles:
        """The particle distribution."""
        return self._beam

    @property
    def sigmas(self) -> sigmasobject:
        """Beam sigmas (:class:`~simba.Modules.Beams.Particles.sigmas.sigmas`)."""
        return self._beam.sigmas

    @property
    def centroids(self) -> centroidsobject:
        """Beam centroids (:class:`~simba.Modules.Beams.Particles.centroids.centroids`)."""
        return self._beam.centroids

    @property
    def twiss(self) -> twissobject:
        """Twiss parameters (:class:`~simba.Modules.Beams.Particles.twiss.twiss`)."""
        return self._beam.twiss

    @property
    def slice(self) -> sliceobject:
        """Slice properties (:class:`~simba.Modules.Beams.Particles.slice.slice`)."""
        return self._beam.slice

    @property
    def emittance(self) -> emittanceobject:
        """Emittances (:class:`~simba.Modules.Beams.Particles.emittance.emittance`)."""
        return self._beam.emittance

    @property
    def kde(self) -> kdeobject:
        """Kernel density estimator (:class:`~simba.Modules.Beams.Particles.kde.kde`)."""
        return self._beam.kde

    @property
    def mve(self) -> Any:
        """Minimum volume ellipse (:class:`~simba.Modules.Beams.Particles.mve.MVE`)."""
        return self._beam.mve

    def rms(self, x, axis: int=None) -> float | np.ndarray   :
        """
        RMS of an array (about zero, not the mean).

        Parameters
        ----------
        x: np.ndarray
            Input array
        axis: int, optional
            Axis along which to reduce

        Returns
        -------
        float or np.ndarray
            RMS of ``x``
        """
        return np.sqrt(np.mean(x**2, axis=axis))

    def __len__(self):
        return len(self._beam.x)

    def __setitem__(self, key, value):
        for p in parameters:
            if key in parameters[p]:
                return setattr(getattr(self, p), key, value)
        if hasattr(self, "_beam") and hasattr(self._beam, key):
            return setattr(self._beam, key, value)
        else:
            try:
                return setattr(self, key, value)
            except KeyError:
                raise AttributeError(key)

    def __getattr__(self, key):
        for p in parameters:
            if key in parameters[p]:
                return getattr(getattr(self, p), key)
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        # write particle data where __getattr__ reads it, not into the pydantic extras
        own = key in type(self).model_fields or isinstance(getattr(type(self), key, None), property)
        if not own and key in parameters["data"]:
            try:
                return setattr(self._beam, key, value)
            except AttributeError:
                return warn(f"beam.{key} is derived from the particle data and cannot be set; ignoring")
        super().__setattr__(key, value)

    def __repr__(self):
        return repr(
            {
                "filename": self.filename,
                "code": self.code,
            }
        )

    def set_particle_mass(self, mass: float=constants.m_e) -> None:
        """
        Set every particle's mass.

        Parameters
        ----------
        mass: float
            Particle mass in kg
        """
        self.particle_mass = UnitValue(np.full(len(self.x), mass), units="kg")
        self._beam.particle_mass = UnitValue(np.full(len(self.x), mass), units="kg")

    def set_mass_and_charge(self, mass, charge, n: int | None = None) -> None:
        """
        Set the per-particle mass and charge, and the rest energies that follow from the mass.

        Parameters
        ----------
        mass: float | np.ndarray
            Particle mass in kg; a scalar applies to every particle
        charge: float | np.ndarray
            Particle charge in C; a scalar applies to every particle
        n: int, optional
            Number of particles; defaults to ``len(self.x)``
        """
        n = len(self.x) if n is None else n
        self._beam.particle_mass = UnitValue(np.full(n, mass), units="kg")
        self._beam.particle_rest_energy = UnitValue(
            self._beam.particle_mass * constants.speed_of_light**2, units="J"
        )
        self._beam.particle_rest_energy_eV = UnitValue(
            self._beam.particle_rest_energy / constants.elementary_charge, units="eV/c"
        )
        self._beam.particle_charge = UnitValue(np.full(n, charge), units="C")

    def set_momenta(self, cpx, cpy, cpz) -> None:
        """
        Set px, py and pz from momenta in eV/c.

        Parameters
        ----------
        cpx, cpy, cpz: np.ndarray
            Momentum components in eV/c
        """
        self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
        self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
        self._beam.pz = UnitValue(cpz * self.q_over_c, units="kg*m/s")

    def set_z_from_t(self, z0: float, ref_index: int | None = None) -> None:
        """
        Set z from t, about the reference particle's t if there is one, else about mean(t),
        and set :attr:`reference_particle` to match.

        Parameters
        ----------
        z0: float
            z at the reference time, in m
        ref_index: int, optional
            Index of the reference particle
        """
        if ref_index is None:
            tref = np.mean(self._beam.t)
        else:
            self.reference_particle_index = int(ref_index)
            tref = self._beam.t[self.reference_particle_index]
        self._beam.z = UnitValue(
            z0 + (-1 * self._beam.Bz * constants.speed_of_light) * (self._beam.t - tref),
            units="m",
        )
        self.reference_particle = None if ref_index is None else [
            getattr(self._beam, coord)[self.reference_particle_index]
            for coord in self.reference_particle_coords
        ]

    def normalise_to_ref_particle(self, array, index=0, subtractmean=False) -> np.ndarray:
        """
        Add the reference particle (element 0, as in ASTRA files) to the other elements.

        Parameters
        ----------
        array: np.ndarray
            Values with elements 1: relative to element 0
        index: int
            Unused
        subtractmean: bool
            If true, then subtract the reference particle from the result

        Returns
        -------
        np.ndarray
            A copy of ``array`` with absolute values
        """
        array = copy.copy(array)
        array[1:] = array[0] + array[1:]
        if subtractmean:
            array = array - array[0]
        return array

    def reset_dicts(self) -> None:
        """Replace the particle distribution with an empty one."""
        self._beam = Particles()

    def read_HDF5_beam_file(self, *args, **kwargs):
        """Load an HDF5 beam file; see :func:`~simba.Modules.Beams.hdf5.read_HDF5_beam_file`."""
        hdf5.read_HDF5_beam_file(self, *args, **kwargs)

    def read_SDDS_beam_file(self, *args, **kwargs):
        """Load an SDDS beam file; see :func:`~simba.Modules.Beams.sdds.read_SDDS_beam_file`."""
        sdds.read_SDDS_beam_file(self, *args, **kwargs)

    def read_gdf_beam_file(self, *args, **kwargs):
        """Load a GDF beam file; see :func:`~simba.Modules.Beams.gdf.read_gdf_beam_file`."""
        gdf.read_gdf_beam_file(self, *args, **kwargs)

    def read_astra_beam_file(self, *args, **kwargs):
        """Load an ASTRA beam file; see :func:`~simba.Modules.Beams.astra.read_astra_beam_file`."""
        astra.read_astra_beam_file(self, *args, **kwargs)

    def read_xsuite_beam_file(self, *args, **kwargs):
        """Load an Xsuite beam file; see :func:`~simba.Modules.Beams.xsuite.read_xsuite_beam_file`."""
        xsuite.read_xsuite_beam_file(self, *args, **kwargs)

    def read_ocelot_beam_file(self, *args, **kwargs):
        """Load an OCELOT beam file; see :func:`~simba.Modules.Beams.ocelot.read_ocelot_beam_file`."""
        from . import ocelot

        ocelot.read_ocelot_beam_file(self, *args, **kwargs)

    def read_opal_beam_file(self, *args, **kwargs):
        """Load an OPAL beam file; see :func:`~simba.Modules.Beams.opal.read_opal_beam_file`."""
        opal.read_opal_beam_file(self, *args, **kwargs)

    def write_openpmd_beam_file(self, *args, **kwargs):
        """Write an openPMD beam file; see :func:`~simba.Modules.Beams.openpmd.write_openpmd_beam_file`."""
        openpmd.write_openpmd_beam_file(self, *args, **kwargs)

    def write_HDF5_beam_file(self, *args, **kwargs):
        """Write an HDF5 beam file; see :func:`~simba.Modules.Beams.hdf5.write_HDF5_beam_file`."""
        hdf5.write_HDF5_beam_file(self, *args, **kwargs)

    def write_SDDS_beam_file(self, *args, **kwargs):
        """Write an SDDS beam file; see :func:`~simba.Modules.Beams.sdds.write_SDDS_file`."""
        sdds.write_SDDS_file(self, *args, **kwargs)

    def write_gdf_beam_file(self, *args, **kwargs):
        """Write a GDF beam file; see :func:`~simba.Modules.Beams.gdf.write_gdf_beam_file`."""
        gdf.write_gdf_beam_file(self, *args, **kwargs)

    def write_astra_beam_file(self, *args, **kwargs):
        """Write an ASTRA beam file; see :func:`~simba.Modules.Beams.astra.write_astra_beam_file`."""
        astra.write_astra_beam_file(self, *args, **kwargs)

    def write_xsuite_beam_file(self, *args, **kwargs):
        """Write an Xsuite beam file; see :func:`~simba.Modules.Beams.xsuite.write_xsuite_beam_file`."""
        return xsuite.write_xsuite_beam_file(self, *args, **kwargs)

    def write_ocelot_beam_file(self, *args, **kwargs):
        """Write an OCELOT beam file; see :func:`~simba.Modules.Beams.ocelot.write_ocelot_beam_file`."""
        from . import ocelot

        return ocelot.write_ocelot_beam_file(self, *args, **kwargs)

    def write_opal_beam_file(self, *args, **kwargs):
        """Write an OPAL beam file; see :func:`~simba.Modules.Beams.opal.write_opal_beam_file`."""
        opal.write_opal_beam_file(self, *args, **kwargs)

    def write_cheetah_beam_file(self, *args, **kwargs):
        from . import cheetah
        return cheetah.write_cheetah_beam_file(self, *args, **kwargs)

    def write_mad8_beam_file(self, *args, **kwargs):
        """Write a MAD8 beam file; see :func:`~simba.Modules.Beams.mad8.write_mad8_beam_file`."""
        mad8.write_mad8_beam_file(self, *args, **kwargs)

    def beam_to_madx_coords(self, *args, **kwargs):
        """MAD-X canonical coordinates (X, PX, Y, PY, T, PT); see :func:`~simba.Modules.Beams.madx.beam_to_madx_coords`."""
        return madx.beam_to_madx_coords(self, *args, **kwargs)

    def madx_coords_to_beam(self, *args, **kwargs):
        """New beam from MAD-X coordinates, with this beam's mass/charge/species; see :func:`~simba.Modules.Beams.madx.madx_coords_to_beam`."""
        return madx.madx_coords_to_beam(self, *args, **kwargs)

    def read_beam_file(self, filename, run_extension="001", step=0, turn=None):
        """
        Load a beam distribution file, picking the reader from the file extension.

        Parameters
        ----------
        filename: str
            File to load
        run_extension: str
            ASTRA run extension
        step: int, optional
            Step number in an OPAL output file
        turn: int, optional
            Turn of a multi-turn openPMD file (default: the last);
            see :func:`~simba.Modules.Beams.openpmd.openpmd_turns`
        """
        pre, ext = os.path.splitext(os.path.basename(filename))
        if ext.lower()[:4] == ".hdf":
            # cheetah writes plain openPMD, it just doesn't say so in the name
            if "openpmd" in pre.lower() or "cheetah" in pre.lower():
                openpmd.read_openpmd_beam_file(self, filename, turn=turn)
            else:
                hdf5.read_HDF5_beam_file(self, filename)
        elif ext.lower() == ".sdds":
            sdds.read_SDDS_beam_file(self, filename)
        elif ext.lower() == ".gdf":
            gdf.read_gdf_beam_file(self, filename)
        elif (ext.lower() == ".npz") and (".ocelot" in filename):
            from . import ocelot

            ocelot.read_ocelot_beam_file(self, filename)
        elif ext.lower() == ".astra":
            astra.read_astra_beam_file(self, filename)
        elif (ext.lower() == ".h5") and ("opal" in pre):
            opal.read_opal_beam_file(self, filename, step=step)
        elif ext.lower() == ".json":
            xsuite.read_xsuite_beam_file(self, filename)
        elif re.match(r".*.\d\d\d\d." + run_extension, filename):
            astra.read_astra_beam_file(self, filename)
        else:
            try:
                with open(filename) as f:
                    firstline = f.readline()
                    if "SDDS" in firstline:
                        sdds.read_SDDS_beam_file(self, filename)
            except UnicodeDecodeError:
                if gdf.rgf.is_gdf_file(filename):
                    gdf.read_gdf_beam_file(self, filename)
                else:
                    warn("Could not load file")

    if use_matplotlib:

        def plot(self, **kwargs):
            return plot.plot(self, **kwargs)

        def slice_plot(self, *args, **kwargs):
            return plot.slice_plot(self, *args, **kwargs)

        def plotScreenImage(self, **kwargs):
            return plot.plotScreenImage(self, **kwargs)

    def resample(self, npart, **kwargs) -> beam:
        """
        Resample the beam to ``npart`` particles with the :attr:`kde`.

        Parameters
        ----------
        npart: int
            Number of particles in the new beam

        Returns
        -------
        :class:`~simba.Modules.Beams.beam`
            The resampled beam
        """
        postbeam = self.kde.resample(npart, **kwargs)
        newbeam = beam()
        E0 = UnitValue(self.beam.particle_mass[0] * constants.speed_of_light ** 2, "J")
        newbeam.Particles.particle_rest_energy_eV = UnitValue(E0 / constants.elementary_charge, "eV/c")
        newbeam.Particles.x = UnitValue(postbeam[0], "m")
        newbeam.Particles.y = UnitValue(postbeam[1], "m")
        newbeam.Particles.z = UnitValue(postbeam[2], "m")
        newbeam.Particles.px = UnitValue(postbeam[3], "kg*m/s")
        newbeam.Particles.py = UnitValue(postbeam[4], "kg*m/s")
        newbeam.Particles.pz = UnitValue(postbeam[5], "kg*m/s")
        newbeam.Particles.total_charge = self.total_charge
        single_charge = newbeam.Particles.total_charge / (len(newbeam.x))
        newbeam.Particles.charge = UnitValue(np.full(len(newbeam.Particles.x), single_charge), "C")
        newbeam.Particles.nmacro = UnitValue(np.full(len(newbeam.Particles.x), 1), "")
        newbeam.Particles.t = UnitValue(newbeam.Particles.z / (-1 * newbeam.Particles.Bz * constants.speed_of_light), "s")
        newbeam.code = "KDE"
        newbeam.longitudinal_reference = "z"

        return newbeam

    def rotate_beamXZ(self, theta, preOffset=[0, 0, 0], postOffset=[0, 0, 0]):
        hdf5.rotate_beamXZ(self, theta, preOffset=preOffset, postOffset=postOffset)

    def unrotate_beamXZ(self):
        hdf5.unrotate_beamXZ(self)


def load_directory(directory=".", types={"SIMBA": ".hdf5"}, verbose=False) -> beamGroup:
    """
    Load every beam file in a directory into a :class:`beamGroup`.

    Parameters
    ----------
    directory: str
        Directory to search
    types: Dict
        File suffix to glob for, keyed by code name
    verbose: bool
        If true, print progress

    Returns
    -------
    :class:`~simba.Modules.Beams.beamGroup`
        The loaded beams, sorted by mean z
    """
    bg = beamGroup()
    if verbose:
        print("Directory:", directory)
    for code, string in types.items():
        beam_files = glob.glob(directory + "/*" + string)
        if verbose:
            print(code, [os.path.basename(t) for t in beam_files])
        bg.add(beam_files)
        bg.sort()
    return bg


def load_file(filename, *args, **kwargs) -> beam:
    """
    Load a beam distribution file into a new :class:`beam`.

    Parameters
    ----------
    filename: str
        File to load

    Returns
    -------
    :class:`~simba.Modules.Beams.beam`
        The loaded beam
    """
    b = beam()
    b.read_beam_file(filename)
    return b


def save_HDF5_summary_file(directory: str = ".", filename: str = "./Beam_Summary.hdf5", screens: list = None, files: list = None) -> None:
    if screens is not None:
        files = []
        for scr in screens:
            if os.path.isfile(bf := os.path.join(directory, scr + '.openpmd.hdf5')):
                files.append(bf)
            else:
                pass
    if files is None:
        beam_files = glob.glob(directory + "/*openpmd.hdf5")
        files = []
        for bf in beam_files:
            with h5py.File(bf, "r") as f:
                if openpmd.is_openpmd_beam_file(f):
                    files.append(bf)
    hdf5.write_HDF5_summary_file(filename, files)


def load_HDF5_summary_file(filename):
    dir = os.path.dirname(filename)
    bg = beamGroup()
    with h5py.File(filename, "r") as f:
        for screen in list(f.keys()):
            bg.add(os.path.join(dir, screen + ".hdf5"))
    bg.sort()
    return bg
