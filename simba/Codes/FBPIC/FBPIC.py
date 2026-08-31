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
from ...Modules import Beams as rbf
from laura.models.element import Plasma
from warnings import warn
from typing import Dict, Literal, List, Any
import glob
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

    use_cuda: bool = True
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

    bunch_z_position: float = 0
    """Position in the FBPIC box, in metres, at which the incoming bunch centroid
    is placed. The default of 0 puts it at the box origin. It has to be settable
    because the bunch's position relative to the driver is what phases it in the
    wake, and because a boosted-frame run needs the laser at z = 0 (see
    :func:`~configure_boost`), which leaves the bunch nowhere else to go."""

    zstart: float = 0
    """Lattice z-position that FBPIC's z = 0 corresponds to."""

    write_period: int = 50
    """Period of writing the cached, backtransformed lab frame diagnostics to disk"""

    plasma_electrons: Any | None = None
    """Plasma electrons for when in boosted-frame mode; should be FBPIC `Particles` object"""

    min_longitudinal_position: float = 0
    """Min position of the simulation box along z (meters)"""

    max_longitudinal_position: float = 0
    """Max position of the simulation box along z (meters)"""

    include_ions: bool = False
    """Whether to add a (mobile) hydrogen ion species alongside the plasma electrons"""

    laser_injection_method: Literal["direct", "antenna"] = "direct"
    """How the laser pulse is introduced; see `fbpic.lpa_utils.laser.add_laser_pulse`"""

    pin: Any | None = None
    """FBPIC `Particles` object holding the injected bunch"""

    bunch_list: List[Any] | None = None
    """Output distributions produced by tracking"""

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
        if not self.include_ions:
            warn(
                "Running in a boosted frame without an ion species. In the boosted "
                "frame the ions stream backwards at -beta0*c and carry a current "
                "that is not negligible, unlike in the lab frame, so FBPIC's "
                "documentation requires them. Set include_ions=True."
            )
        # `direct` injection stamps the analytic vacuum profile onto the mesh at
        # boosted t' = 0, which corresponds to a lab time that grows with the
        # laser's lab-frame start position. The pulse therefore arrives having
        # apparently propagated ~2*gamma^2*z0 in vacuum, and a Gaussian that far
        # past its waist is weaker by 1/sqrt(1 + (drift/z_R)^2). The failure is
        # silent -- a weak driver just makes a weak wake -- so check it here.
        laser = getattr(plas, "laser", None)
        if laser is not None and self.laser_injection_method == "direct":
            z_rayleigh = np.pi * laser.waist ** 2 / laser.wavelength
            drift = self.boost.gamma0 * self.boost.beta0 * (
                self.boost.gamma0 * (1 + self.boost.beta0)
            ) * laser.initial_position
            attenuation = 1 / np.sqrt(1 + (drift / z_rayleigh) ** 2)
            if attenuation < 0.95:
                warn(
                    f"The laser starts at z0 = {laser.initial_position * 1e6:.1f} um, "
                    f"so at gamma_boost = {self.gamma_boost} `direct` injection puts it "
                    f"on the mesh as though it had propagated {drift * 1e6:.0f} um "
                    f"({drift / z_rayleigh:.1f} Rayleigh lengths) in vacuum, cutting a0 "
                    f"to {attenuation:.1%} of its intended value. Move the laser to "
                    "z0 ~ 0 (shift the whole lattice if need be), widen the waist, or "
                    "lower gamma_boost; 2*gamma_boost^2*z0 has to stay well inside the "
                    "Rayleigh length."
                )
        # gamma_boost above roughly omega0/omega_p leaves the wake under-resolved;
        # FBPIC's documentation asks for gamma_boost < gamma_wake / 2.
        gamma_wake = omega0 / plas.plasma.plasma_frequency()
        if self.gamma_boost >= 0.5 * gamma_wake:
            warn(
                f"gamma_boost = {self.gamma_boost} is not below gamma_wake / 2 = "
                f"{0.5 * gamma_wake:.1f} for this plasma density; the boosted-frame "
                "result is unlikely to be converged."
            )

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
        laser_z0 = None
        plasmas = None
        for element in beamline:
            if element.hardware_type.lower() == "plasma":
                plasmas = element
                n_longitudinal = element.simulation.n_longitudinal
                n_radial = element.simulation.n_radial
                max_radial_position = element.simulation.r_max
                self.max_longitudinal_position = element.simulation.max_longitudinal_position
                self.min_longitudinal_position = element.simulation.min_longitudinal_position
                self.zstep = (self.max_longitudinal_position - self.min_longitudinal_position)
                time_step = self.zstep / n_longitudinal / c
                if lasers is None and element.laser is not None:
                    lasers = element.laser_to_fbpic()
                    omega0 = element.laser.angular_frequency
                    laser_z0 = element.laser.initial_position
            if element.hardware_type.lower() == "laser":
                lasers = element.to_fbpic()
                omega0 = element.laser.angular_frequency
                laser_z0 = element.laser.initial_position

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
        use_cuda = True if cuda.is_available() and self.use_cuda else False

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
            gamma_boost=(
                self.boost.gamma0 if isinstance(self.boost, BoostConverter) else None
            ),
        )
        self.plasma_electrons = self.simulation.add_new_species(
            **self.add_plasma_species(plasmas, typ="electron"),
        )
        if self.include_ions:
            self.simulation.add_new_species(
                **self.add_plasma_species(plasmas, typ="hydrogen"),
            )

        self.pin = self.prepare_bunch()

        if self.track_bunch:
            self.pin.track(self.simulation.comm)

        if lasers is not None:
            gamma_boost = (
                self.boost.gamma0 if isinstance(self.boost, BoostConverter) else None
            )
            # The antenna needs the laser's starting position. Take it from the
            # LAURA element rather than the FBPIC profile object, which does not
            # carry a `z0` attribute (it lives on the longitudinal sub-profile).
            z0_antenna = (
                laser_z0 if self.laser_injection_method == "antenna" else None
            )
            add_laser_pulse(
                self.simulation,
                lasers,
                gamma_boost=gamma_boost,
                method=self.laser_injection_method,
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
        from fbpic.lpa_utils.boosted_frame import BoostConverter

        dens_func = (
            plas._relative_density_profile if plas.plasma.density_profile else None
        )
        boost_positions = isinstance(self.boost, BoostConverter)
        q = plas.plasma.charge(typ)
        m = plas.plasma.mass(typ)

        plas_dict = {
            "q": q,
            "m": m,
            "n": plas.plasma.density,
            "dens_func": dens_func,
            "p_zmin": plas.simulation.p_zmin,
            "p_zmax": plas.simulation.p_zmax,
            "p_rmax": plas.simulation.p_rmax,
            "p_nr": plas.simulation.particles_per_radial_cell,
            "p_nz": plas.simulation.particles_per_longitudinal_cell,
            "p_nt": plas.simulation.particles_per_angular_cell,
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
                write_dir=self.global_parameters["master_subdir"] + "/diags/",
            )
            species = {"electrons": self.plasma_electrons}
            if self.pin is not None:
                species["bunch"] = self.pin
            part_diag = ParticleDiagnostic(
                self.diag_period,
                species=species,
                select={"uz": [1., None]},
                comm=self.simulation.comm,
                write_dir=self.global_parameters["master_subdir"] + "/diags/",
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
        prefix = prefix if prefix else self.get_prefix()
        self.read_input_file(prefix, self.particle_definition)
        self.global_parameters["beam"].beam.rematchXPlane(**self.initial_twiss["horizontal"])
        self.global_parameters["beam"].beam.rematchYPlane(**self.initial_twiss["vertical"])
        # Shifting by the centroid alone would land the bunch at z = 0; subtracting
        # the requested position as well puts it where it was asked for, and keeps
        # `zstart` the offset that postProcess adds back to recover lattice
        # coordinates.
        self.zstart = (
            float(self.global_parameters["beam"].beam.centroids.mean_z.val)
            - self.bunch_z_position
        )
        return beam_to_particles(
            self.global_parameters["beam"],
            simulation=self.simulation,
            boost=self.boost,
            zstart=self.zstart,
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
        self.bunch_list = [self.pin]

    def final_lab_diagnostic(self) -> str | None:
        """
        Path to the last back-transformed lab-frame snapshot, if there is one.

        Returns
        -------
        str or None
            The snapshot with the highest iteration number, or None if the
            diagnostic wrote nothing
        """
        pattern = os.path.join(
            self.global_parameters["master_subdir"], "lab_diags", "hdf5", "*.h5"
        )
        snapshots = sorted(glob.glob(pattern))
        return snapshots[-1] if snapshots else None

    def postProcess(self) -> None:
        """
        Convert the tracked bunch back to a `beam` object and write it to
        `master_subdir` as openPMD, so the next line in the lattice can pick it up.

        In a boosted-frame run the live species arrays are in the *boosted* frame,
        so the lab-frame distribution is taken from the last snapshot written by
        the `BackTransformedParticleDiagnostic` instead.
        """
        from fbpic.lpa_utils.boosted_frame import BoostConverter
        from ...Modules.Beams.fbpic import particles_to_beam, read_fbpic_beam_file

        super().postProcess()
        if self.pin is None:
            return
        outbeamname = f'{self.global_parameters["master_subdir"]}/{self.end}.openpmd.hdf5'
        if isinstance(self.boost, BoostConverter):
            snapshot = self.final_lab_diagnostic()
            if snapshot is None:
                warn(
                    "No back-transformed lab-frame diagnostics were written, so the "
                    "output distribution of boosted-frame line "
                    f"{self.objectname} cannot be produced in the lab frame."
                )
                return
            read_fbpic_beam_file(
                self.global_parameters["beam"], snapshot, z_offset=self.zstart
            )
            captured = len(self.global_parameters["beam"].beam.x)
            if captured < self.pin.Ntot:
                warn(
                    f"The final lab-frame snapshot captured {captured} of "
                    f"{self.pin.Ntot} macroparticles. A particle is only recorded "
                    "once it crosses the snapshot plane, so the shortfall is "
                    "particles that had not crossed it when the run ended; step "
                    "a little past the last snapshot to capture them."
                )
        else:
            particles_to_beam(
                self.global_parameters["beam"],
                self.pin,
                zpos=self.zstart,
            )
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            outbeamname,
        )
