"""
SIMBA FBPIC Module

Various objects and functions to handle Wake-T lattices and commands. See `FBPIC github`_ for more details.

    .. _FBPIC github: https://github.com/fbpic/fbpic

Classes:
    - :class:`~simba.Codes.FBPIC.FBPIC.fbpicLattice`: The FBPIC lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an FBPIC lattice object,
    and for tracking through it.

"""

from ...Framework_objects import frameworkLattice
from laura.models.element import Plasma
from warnings import warn
from typing import Dict, Literal, List, Any
from copy import deepcopy
import os
from yaml import safe_load
import numpy as np
from scipy.constants import c

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/fbpic_defaults.yaml",
    "r",
) as infile:
    fbpicglobal = safe_load(infile)


def all_subclasses(cls):
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses


class fbpicLattice(frameworkLattice):
    """
        Class for defining the FBPIC lattice object, used for
        converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
        :class:`~simba.Framework_objects.frameworkLattice` into an FBPIC lattice object,
        and for tracking through it.
        """

    code: str = "fbpic"
    """String indicating the lattice object type"""

    beamline: List = []
    """List of elements in the beamline"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    allow_negative_drifts: bool = True
    """Allow drifts to be of negative length (could be necessary for plasma injection)"""

    fbpicglobal: Dict = {}
    """Global settings for FBPIC simulations; defaults in `fbpic_defaults.yaml`"""

    particle_definition: str = None
    """Name of the initial object in the lattice"""

    diag_period: int = 50
    """Period of the diagnostics in number of timesteps"""

    save_checkpoints: bool = False
    """Whether to write checkpoint files"""

    checkpoint_period: int = 100
    """Period for writing the checkpoints"""

    use_restart: bool = False
    """Whether to restart from a previous checkpoint"""

    track_bunch: bool = True
    """Whether to track and write particle ids"""

    use_cuda: bool = False
    """Whether to use CUDA for GPU acceleration"""

    n_order: int = -1
    """Use -1 for infinite order (advised for single-GPU/single-CPU simulation).
    Use a positive number (and multiple of 2) for a finite-order stencil
    (required for multi-GPU/multi-CPU with MPI)
    """

    boost: Any | bool = True
    """Boosted frame converter; set to True if you want this set up during pre-processing"""

    gamma_boost: float = 10.0
    """Boosted frame -- Lorentz factor"""

    number_of_modes: int = 2
    """Number of modes for `FBPIC Simulation class`_ for more details.

    .. _FBPIC github: https://github.com/fbpic/fbpic/blob/dev/fbpic/main.py"""

    simulation: Any | None = None
    """`FBPIC Simulation class`_"""

    boundaries: Dict = {'z': 'open', 'r': 'reflective'}
    """Boundaries for `FBPIC Simulation class`_"""

    v_comoving: float = None
    """Co-moving velocity for boosted simulations"""

    particle_shape: Literal["cubic", "linear"] = "cubic"
    """Set the particle shape for the charge/current deposition."""

    v_window: float = c
    """Moving window """

    interaction_time: float = None
    """Interaction time (seconds) (to calculate number of PIC iterations)"""

    interaction_length: float = None
    "The interaction length of the simulation, in the lab frame (meters)"

    diags: List = []
    """Diagnostics for :attr:`~simulation`"""

    n_step: int = 0
    """Number of simulation steps"""

    n_boosted_diag: int = 16
    """Number of discrete diagnostic snapshots, for the diagnostics in the
    boosted frame"""

    n_lab_diag: int = 11
    """Number of discrete diagnostic snapshots, for the diagnostics in the
    lab frame"""

    zstep: float = 0
    """Distance from max to min longitudinal position of plasma"""

    write_period: int = 50
    """Period of writing the cached, backtransformed lab frame diagnostics to disk"""

    plasma_electrons: Any | None = None
    """Plasma electrons for when in boosted-frame mode; should be FBPIC `Particles` object"""

    min_longitudinal_position: float = 0
    """Min position of the box along z (meters) -- same as plasma"""

    max_longitudinal_position: float = 0
    """Max position of the box along z (meters) -- same as plasma"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "FBPICsettings" in list(self.settings["global"].keys()):
            for k, v in self.settings["global"]["FBPICsettings"].items():
                if isinstance(v, Dict):
                    for k1, v1 in v.items():
                        getattr(self, k).update({k1: v1})
                else:
                    setattr(self, k, v)
        if (
                "input" in self.file_block
                and "particle_definition" in self.file_block["input"]
        ):
            if (
                    self.file_block["input"]["particle_definition"]
                    == "initial_distribution"
            ):
                self.particle_definition = "laser"
            else:
                self.particle_definition = self.file_block["input"][
                    "particle_definition"
                ]
        else:
            self.particle_definition = self.elementObjects[self.start].name


    def write(self) -> None:
        """
        Create the beamline object via :func:`~writeElements`;
        not that FBPIC appears not to support the writing of a lattice to a file.
        """
        self.writeElements()

    def writeElements(self) -> None:
        """
        Create FBPIC objects for all the elements in the lattice and set the
        :attr:`~simba.Codes.FBPIC.FBPIC.fbpicLattice.beamline`.
        """
        self.beamline = self.section.to_fbpic()
        self.configure_simulation(self.beamline)

    def configure_boost(self, plas: Plasma, omega0: float) -> None:
        """
        Add a `fbpic.lpa_utils.boosted_frame.BoostConverter` based on :attr:`~gamma_boost`.
        Sets :attr:`~boost`.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma
            Plasma object
        omega0: float
            Laser angular frequency
        """
        from fbpic.lpa_utils.boosted_frame import BoostConverter
        self.boost = BoostConverter(self.gamma_boost)
        self.v_comoving = - c * np.sqrt(1. - 1. / self.boost.gamma0 ** 2)
        self.v_window = c*(1 - 0.5*plas.density/plas.plasma.critical_density(omega0))

    def configure_simulation(self, beamline: list) -> None:
        """
        Determine the parameters of the plasma and laser elements in order
        to configure the :attr:`~simulation` object correctly.

        Parameters
        ----------
        beamline: list
            The :class:`~SimulationFramework.Framework_objects.frameworkElement` objects
            defining the line
        """
        from fbpic.lpa_utils.boosted_frame import BoostConverter
        from fbpic.lpa_utils.laser import add_laser_pulse
        from fbpic.main import Simulation
        omega0 = 2*np.pi*c/(800e-9)
        n_longitudinal = 0
        n_radial = 0
        max_radial_position = 0
        lasers = None
        plasmas = None
        for element in beamline:
            if element.hardware_type.lower() == "plasma":
                plasmas = element
                n_longitudinal = element.n_longitudinal
                n_radial = element.n_radial
                max_radial_position = element.r_max
                self.max_longitudinal_position = element.max_longitudinal_position
                self.min_longitudinal_position = element.min_longitudinal_position
                self.zstep = (self.max_longitudinal_position - self.min_longitudinal_position)
                time_step = self.zstep / n_longitudinal / c
            if element.hardware_type.lower() == "laser":
                lasers = element.to_fbpic()
                omega0 = element.angular_frequency()

        if plasmas is None:
            raise ValueError(f"No plasmas found in {self.name}; aborting")

        self.interaction_length = plasmas.length  # increase to simulate longer distance!
        # Interaction time (seconds) (to calculate number of PIC iterations)
        self.interaction_time = (self.interaction_length + self.zstep) / self.v_window

        if self.boost is True:
            if self.gamma_boost > 0:
                self.configure_boost(plas=plasmas, omega0=omega0)
                fac11 = max_radial_position
                fac12 = (2 * self.boost.gamma0 * n_radial)
                time_step = min(fac11 / fac12 / c, self.zstep / n_longitudinal / c)
                self.interaction_length = plasmas.length  # the plasma length
                # Interaction time, in the boosted frame (seconds)
                self.interaction_time = self.boost.interaction_time(self.interaction_length, self.zstep, self.v_window)
            else:
                warn("gamma_boost not set and boost is True; cannot configure BoostConverter")

        from numba import cuda
        use_cuda = True if cuda.is_available() else False

        self.simulation = Simulation(
            n_longitudinal,
            self.max_longitudinal_position,
            n_radial,
            max_radial_position,
            self.number_of_modes,
            time_step,
            zmin=self.min_longitudinal_position,
            n_order=self.n_order,
            use_cuda=use_cuda,
            boundaries=self.boundaries,
            v_comoving=self.v_comoving,
            particle_shape=self.particle_shape,
        )
        self.pin = self.prepare_bunch()
        gamma_boost = None
        if isinstance(self.boost, BoostConverter):
            self.plasma_electrons = self.simulation.add_new_species(
                **self.add_plasma_species(plasmas, typ="electron"),
            )
            self.simulation.add_new_species(
                **self.add_plasma_species(plasmas, typ="hydrogen"),
            )

        if self.track_bunch:
            self.pin.track(self.simulation.comm)

        if lasers is not None:
            z0_antenna = lasers.initial_position if self.method == "antenna" else None
            add_laser_pulse(
                self.simulation,
                lasers,
                gamma_boost=gamma_boost,
                method=lasers.method,
                z0_antenna=z0_antenna,
            )

        if isinstance(self.boost, BoostConverter):
            v_window_boosted, = self.boost.velocity([self.v_window])
            self.simulation.set_moving_window(v=v_window_boosted)
        else:
            self.simulation.set_moving_window(v=self.v_window)

        self.simulation.diags.extend(self.prepare_diagnostics())

        self.n_step = int(self.interaction_time / self.simulation.dt)

    def add_plasma_species(
            self,
            plas: Plasma,
            typ: Literal["electron", "hydrogen", "positron"],
    ) -> Dict:
        """
        Define a new plasma species to add to :attr:`~simulation`.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            SimFrame `plasma` object
        typ: Literal["electron", "ion"]
            Name of plasma species

        Returns
        -------
        Dict
            Dictionary containing plasma parameters
        """
        dens_func = plas._density_profile if plas.density_profile else None
        boost_positions = True if self.boost else False
        q = plas.plasma.charge(typ)
        m = plas.plasma.mass(typ)

        plas_dict = {
            "q": q,
            "m": m,
            "n": plas.density,
            "dens_func": dens_func,
            "p_zmin": plas.min_longitudinal_position,
            "p_zmax": plas.max_longitudinal_position,
            "p_rmax": plas.r_max,
            "p_nr": plas.particles_per_radial_cell,
            "p_nz": plas.particles_per_longitudinal_cell,
            "p_nt": plas.particles_per_angular_cell,
            "boost_positions_in_dens_func": boost_positions,
        }
        return plas_dict

    def prepare_diagnostics(self) -> List:
        """
        Prepare the Diagnostic objects for :attr:`~simulation`

        Returns
        -------
        List
            List of diagnostic objects
        """
        from fbpic.lpa_utils.boosted_frame import BoostConverter
        from fbpic.openpmd_diag import (
            FieldDiagnostic,
            ParticleDiagnostic,
            BackTransformedFieldDiagnostic,
            BackTransformedParticleDiagnostic,
            set_periodic_checkpoint,
            restart_from_checkpoint,
        )
        if not isinstance(self.boost, BoostConverter):
            field_diag = FieldDiagnostic(
                self.diag_period,
                self.simulation.fld,
                comm=self.simulation.comm,
                # write_dir = self.global_parameters["master_subdir"],
            )
            part_diag = ParticleDiagnostic(
                self.diag_period,
                species={"electrons": deepcopy(self.pin)},
                select={"uz": [1., None]},
                comm=self.simulation.comm,
                # write_dir=self.global_parameters["master_subdir"],
            )
            return [field_diag, part_diag]
        else:
            dt_lab_diag_period = (self.interaction_length + self.zstep) / self.v_window / (self.n_lab_diag - 1)
            # Time interval between diagnostic snapshots *in the boosted frame*
            dt_boosted_diag_period = self.interaction_time / (self.n_boosted_diag - 1)
            fld = FieldDiagnostic(
                dt_period=dt_boosted_diag_period,
                fldobject=self.simulation.fld,
                comm=self.simulation.comm,
                write_dir=self.global_parameters["master_subdir"] + "/diags/",
            )
            part = ParticleDiagnostic(
                dt_period=dt_boosted_diag_period,
                species={"electrons": self.plasma_electrons, "bunch": self.pin},
                comm=self.simulation.comm,
                write_dir=self.global_parameters["master_subdir"] + "/diags/",
            )
            # Diagnostics in the lab frame (back-transformed)
            # btfld = BackTransformedFieldDiagnostic(
            #     self.min_longitudinal_position,
            #     self.max_longitudinal_position,
            #     self.v_window,
            #     dt_lab_diag_period,
            #     self.n_lab_diag,
            #     self.boost.gamma0,
            #     fieldtypes=['rho', 'E', 'B'],
            #     period=self.write_period,
            #     fldobject=self.simulation.fld,
            #     comm=self.simulation.comm,
            #     write_dir=self.global_parameters["master_subdir"] + "/lab_diags/",
            # )
            btpart = BackTransformedParticleDiagnostic(
                self.min_longitudinal_position,
                self.max_longitudinal_position,
                self.v_window,
                dt_lab_diag_period,
                self.n_lab_diag,
                self.boost.gamma0,
                self.write_period,
                self.simulation.fld,
                select={'uz': [0., None]},
                species={'bunch': self.pin},
                comm=self.simulation.comm,
                write_dir=self.global_parameters["master_subdir"] + "/lab_diags/",
            )
            return [fld, part, btpart]#, btfld]

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        super().preProcess()
        # os.chdir(self.global_parameters["master_subdir"])

    def hdf5_to_particles(self, prefix="", write=True) -> "Particles":
        """
        Convert the initial HDF5 particle distribution to FBPIC format and set
        :attr:`~pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        from ...Modules.Beams.fbpic import beam_to_particles
        prefix = self.get_prefix()
        self.read_input_file(prefix, self.particle_definition)
        self.global_parameters["beam"].beam.rematchXPlane(**self.initial_twiss["horizontal"])
        self.global_parameters["beam"].beam.rematchYPlane(**self.initial_twiss["vertical"])
        return beam_to_particles(
            self.global_parameters["beam"],
            simulation=self.simulation,
            boost=self.boost,
            zstart=float(self.global_parameters["beam"].beam.centroids.mean_z.val),
        )

    def prepare_bunch(self) -> "Particles":
        """
        Once :attr:`~simulation` has been prepared and the elements created, the
        initial bunch distribution :attr:`~pin` can be read in.
        See :func:`~hdf5_to_particles`.
        """
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        return self.hdf5_to_particles(prefix)

    def run(self) -> None:
        """
        Run the code, and set :attr:`~bunch_list`
        """
        self.simulation.step(self.n_step)
