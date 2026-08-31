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
import numpy as np
from scipy.constants import c


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

    particle_definition: str = None
    """Name of the initial object in the lattice"""

    diag_period: int = 50
    """Period of the diagnostics in number of timesteps"""

    save_checkpoints: bool = False
    """Whether to write checkpoint files, to ``checkpoints/`` under the run
    directory. See :func:`~configure_checkpoints`."""

    checkpoint_period: int = 100
    """Period for writing the checkpoints, in number of timesteps"""

    use_restart: bool = False
    """Whether to restart from the last checkpoint in ``checkpoints/``. Warns
    and starts from the beginning if there is no such directory."""

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
    """Number of modes for FBPIC `Simulation` class; see `FBPIC github`_ for more details.

    .. _FBPIC github: https://github.com/fbpic/fbpic/blob/dev/fbpic/main.py"""

    boundaries: Dict = {'z': 'open', 'r': 'reflective'}
    """Boundaries for `FBPIC Simulation class`_"""

    particle_shape: Literal["cubic", "linear"] = "cubic"
    """Set the particle shape for the charge/current deposition."""

    use_galilean: bool = False
    """Whether to use the Galilean scheme, in which the spectral solver follows
    :attr:`~v_comoving` rather than staying in the frame of the grid.

    Only has an effect when :attr:`~v_comoving` is set, which
    :func:`~configure_boost` does for a boosted run."""

    n_boosted_diag: int = 16
    """Number of discrete diagnostic snapshots, for the diagnostics in the
    boosted frame"""

    n_lab_diag: int = 11
    """Number of discrete diagnostic snapshots, for the diagnostics in the
    lab frame"""

    bunch_z_position: float = 0
    """Position in the FBPIC box, in metres, at which the incoming bunch centroid
    is placed; 0 puts it at the box origin."""

    write_period: int = 50
    """Period of writing the cached, backtransformed lab frame diagnostics to disk"""

    lab_field_diagnostics: bool = False
    """Whether a boosted-frame run also writes back-transformed *fields* to
    ``lab_diags/``, alongside the back-transformed particles."""

    lab_fieldtypes: List[str] = ["E", "B", "rho"]
    """Field types written by the back-transformed lab-frame field diagnostic.
    ``rho`` and ``J`` are only available if the corresponding arrays are being
    deposited; see :func:`~prepare_diagnostics`."""

    include_ions: bool = False
    """Whether to add a (mobile) hydrogen ion species alongside the plasma electrons"""

    laser_injection_method: Literal["direct", "antenna"] = "direct"
    """How the laser pulse is introduced; see `fbpic.lpa_utils.laser.add_laser_pulse`"""

    pin: Any | None = None
    """FBPIC `Particles` object holding the injected bunch"""

    bunch_list: List[Any] | None = None
    """Output distributions produced by tracking"""

    _simulation: Any | None = None
    _plasma_electrons: Any | None = None
    _ionizable_species: Any | None = None
    _v_window: float = c
    _v_comoving: float | None = None
    _zstart: float = 0.0
    _n_step: int | None = None

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

    @property
    def simulation(self) -> Any | None:
        """`FBPIC Simulation class`_ object, once :func:`~write` has built it"""
        return self._simulation

    @property
    def plasma_electrons(self) -> Any | None:
        """FBPIC `Particles` object holding the plasma electrons"""
        return self._plasma_electrons

    @property
    def ionizable_species(self) -> Any | None:
        """FBPIC `Particles` object holding the ionizable atoms, if the plasma
        element defines any; see :func:`~add_ionizable_species`"""
        return self._ionizable_species

    @property
    def v_comoving(self) -> float | None:
        """Co-moving velocity for boosted simulations, set by
        :func:`~configure_boost`"""
        return self._v_comoving

    @property
    def v_window(self) -> float:
        """Velocity of the moving window. The group velocity of the driver in
        the plasma for a boosted run (:func:`~configure_boost`), and `c`
        otherwise."""
        return self._v_window

    @property
    def zstart(self) -> float:
        """Lattice z-position that FBPIC's z = 0 corresponds to."""
        return self._zstart

    @property
    def plasma(self) -> Any | None:
        """
        The plasma element of :attr:`~beamline`, which most of the geometry
        below is read from.

        Returns
        -------
        The LAURA plasma translator object, or None before :func:`~write` has
        run or if the line holds no plasma
        """
        for element in self.beamline:
            if element.hardware_type.lower() == "plasma":
                return element
        return None

    @property
    def boosted(self) -> bool:
        """Whether :attr:`~boost` has been resolved into a `BoostConverter`,
        i.e. whether this line is being run in a boosted frame"""
        from fbpic.lpa_utils.boosted_frame import BoostConverter
        return isinstance(self.boost, BoostConverter)

    @property
    def min_longitudinal_position(self) -> float:
        """Min position of the simulation box along z (meters)"""
        return self.plasma.simulation.min_longitudinal_position if self.plasma else 0.0

    @property
    def max_longitudinal_position(self) -> float:
        """Max position of the simulation box along z (meters)"""
        return self.plasma.simulation.max_longitudinal_position if self.plasma else 0.0

    @property
    def zstep(self) -> float:
        """Length of the simulation box along z (meters)"""
        return self.max_longitudinal_position - self.min_longitudinal_position

    @property
    def interaction_length(self) -> float:
        """The interaction length of the simulation, in the lab frame (meters);
        the length of the plasma element"""
        return self.plasma.length if self.plasma else 0.0

    @property
    def interaction_time(self) -> float:
        """
        Time to track for, in the frame being simulated (seconds).

        The window has to traverse the plasma *and* its own length before the
        far end of the box has seen the whole stage, hence the extra
        :attr:`~zstep`. In a boosted frame the equivalent time is shorter by
        roughly ``2*gamma_boost^2``; FBPIC's `BoostConverter` works it out.
        """
        if self.boosted:
            return self.boost.interaction_time(
                self.interaction_length, self.zstep, self.v_window
            )
        return (self.interaction_length + self.zstep) / self.v_window

    @property
    def n_step(self) -> int:
        """
        Number of PIC steps left to take.

        Derived from :attr:`~interaction_time` and the timestep FBPIC settled
        on, less whatever a restart has already done
        (:func:`~configure_checkpoints`) -- so a resumed run still ends where an
        uninterrupted one would have. Can be set manually.
        """
        if self._n_step is not None:
            return self._n_step
        if self.simulation is None:
            return 0
        total = int(self.interaction_time / self.simulation.dt)
        return max(total - self.simulation.iteration, 0)

    @n_step.setter
    def n_step(self, value: int) -> None:
        self._n_step = int(value)

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
        Sets :attr:`~boost`, :attr:`~v_comoving` and :attr:`~v_window`.

        A Lorentz-boosted frame is what makes a metre-scale plasma stage
        affordable, but it should be checked to ensure the simulation is sensible.

        :func:`~boost_limits` bounds :attr:`~gamma_boost` from above, and warns
        if the requested value is past the smallest bound.

        :func:`~check_boost_geometry` warns about the lattice itself: missing
        ions, a laser that starts far enough from ``z = 0`` that boosting
        defocuses it before the run begins, and a density profile that stops
        short of where the boosted particle injector will ask for it.

        Boosted simulations should also be checked against the lab frame for
        consistency; this check only attempts to ensure that the boosted results
        are sensible.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            Plasma object, carrying the laser element if there is one
        omega0: float
            Laser angular frequency
        """
        from fbpic.lpa_utils.boosted_frame import BoostConverter
        self.boost = BoostConverter(self.gamma_boost)
        self._v_comoving = - c * np.sqrt(1. - 1. / self.boost.gamma0 ** 2)
        self._v_window = c*(1 - 0.5*plas.density/plas.plasma.critical_density(omega0))
        limits = self.boost_limits(plas, omega0)
        binding, ceiling = min(limits.items(), key=lambda item: item[1])
        if self.gamma_boost > ceiling:
            detail = ", ".join(f"{name} {value:.1f}" for name, value in limits.items())
            warn(
                f"gamma_boost = {self.gamma_boost} is past the {binding} limit of "
                f"{ceiling:.1f} for this lattice (limits: {detail}). Try "
                f"gamma_boost = {self.suggest_gamma_boost(plas, omega0):.0f}, and "
                "check it against a lab-frame run before trusting the result."
            )
        self.check_boost_geometry(plas, omega0)

    def boost_limits(
        self, plas: Plasma, omega0: float, a0_tolerance: float = 0.95
    ) -> Dict[str, float]:
        """
        Largest :attr:`~gamma_boost` that each boosted-frame criterion permits.

        Three things bound the boost, and only the smallest of them matters:

        ``resolution``
            FBPIC's documentation asks for ``gamma_boost < gamma_wake / 2``,
            where ``gamma_wake = omega0 / omega_p``. Above that the plasma wave
            is under-resolved in the boosted frame.
        ``efficiency``
            ``gamma_boost**2 < L_interact / l_window``. Past this the boost
            stops paying for itself, because the numerical-Cherenkov condition
            ``c*dt_lab < dr_lab / (2*gamma_boost)`` shrinks the timestep faster
            than the boost shortens the run; the step count reaches a minimum
            and then climbs again.
        ``laser_defocus``
            Only for ``direct`` injection, and only when the laser starts away
            from ``z = 0``. See :func:`~check_boost_geometry`.

        Treat the smallest limit as a ceiling to stay below rather than as a
        recommendation.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            Plasma object, carrying the laser element if there is one
        omega0: float
            Laser angular frequency
        a0_tolerance: float
            Smallest fraction of the intended ``a0`` the laser may be delivered
            with before the ``laser_defocus`` limit binds

        Returns
        -------
        Dict[str, float]
            Criterion name against the largest gamma_boost it permits, with
            `np.inf` for a criterion that does not apply to this lattice
        """
        limits = {
            "resolution": 0.5 * omega0 / plas.plasma.plasma_frequency(),
            "efficiency": (
                np.sqrt(self.interaction_length / self.zstep)
                if self.interaction_length and self.zstep
                else np.inf
            ),
            "laser_defocus": np.inf,
        }
        laser = getattr(plas, "laser", None)
        if (
            laser is not None
            and self.laser_injection_method == "direct"
            and laser.initial_position
        ):
            z_rayleigh = np.pi * laser.waist ** 2 / laser.wavelength
            max_drift = z_rayleigh * np.sqrt(1 / a0_tolerance ** 2 - 1)
            limits["laser_defocus"] = np.sqrt(
                max_drift / (2 * abs(laser.initial_position))
            )
        return limits

    def suggest_gamma_boost(
        self, plas: Plasma, omega0: float, safety: float = 0.7
    ) -> float:
        """
        A :attr:`~gamma_boost` to start from for this lattice.

        This is a starting point and not an optimum;
        confirm any boosted result against a lab-frame one.
        Running at the ceiling is not good enough, which is why ``safety``
        exists.

        A return value of 1 means the lattice as it stands should not be
        boosted at all -- usually because the stage is too short relative to the
        simulation box for a boost to pay for itself, or because the laser sits
        too far from ``z = 0``.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            Plasma object, carrying the laser element if there is one
        omega0: float
            Laser angular frequency
        safety: float
            Fraction of the binding limit to suggest staying under

        Returns
        -------
        float
            Suggested gamma_boost, never less than 1
        """
        ceiling = min(self.boost_limits(plas, omega0).values())
        return max(1.0, float(np.floor(safety * ceiling)))

    def check_boost_geometry(self, plas: Plasma, omega0: float) -> None:
        """
        Warn about lattice geometry that a boosted frame handles badly.

        All three failures below produce a run that completes normally and
        gives the wrong answer, which is why they are checked rather than left
        to be noticed.

        *Missing ions.* In the lab frame the ions barely move over the
        interaction time and can be left out; in the boosted frame they stream
        backwards at ``-beta0*c`` and carry a current that is not negligible.

        *A defocused driver.* ``direct`` injection stamps the analytic vacuum
        profile onto the mesh at boosted time ``t' = 0``, which corresponds to a
        lab time at which the pulse has already propagated
        roughly ``2*gamma**2*z0``. A Gaussian that far past its waist is weaker by
        ``1/sqrt(1 + (drift/z_rayleigh)**2)``, so the driver arrives with
        the wrong amplitude.

        *A density profile that stops too soon.* The boosted particle injector
        queries ``dens_func`` at lab positions well beyond the interaction
        length, and a profile that has ended by then injects no plasma at all
        where the laser is.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            Plasma object, carrying the laser element if there is one
        omega0: float
            Laser angular frequency
        """
        if not self.include_ions:
            warn(
                "Running in a boosted frame without an ion species. In the boosted "
                "frame the ions stream backwards at -beta0*c and carry a current "
                "that is not negligible, unlike in the lab frame, so FBPIC's "
                "documentation requires them. Set include_ions=True."
            )

        laser = getattr(plas, "laser", None)
        if (
            laser is not None
            and self.laser_injection_method == "direct"
            and laser.initial_position
        ):
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

        if plas.plasma.density_profile and self.interaction_length:
            reach = self.gamma_boost ** 2 * self.interaction_length
            if float(plas._relative_density_profile(reach)) < 0.01:
                warn(
                    f"The density profile is essentially zero at z = "
                    f"{reach * 1e3:.2f} mm, which is gamma_boost^2 * L_interact "
                    f"({self.gamma_boost}^2 * {self.interaction_length * 1e3:.2f} mm). "
                    "The boosted particle injector asks for the density well beyond "
                    "L_interact, and where the profile has ended it injects no plasma "
                    "at all -- including where the laser is. Extend the profile "
                    "(plateau and p_zmax) to cover that reach; it is analytic, so "
                    "over-providing costs nothing."
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
        from fbpic.lpa_utils.laser import add_laser_pulse
        from fbpic.main import Simulation
        omega0 = 2*np.pi*c/(800e-9)
        lasers = None
        laser_z0 = None
        plasmas = self.plasma
        if plasmas is None:
            raise ValueError(f"No plasmas found in {self.name}; aborting")
        if sum(e.hardware_type.lower() == "plasma" for e in beamline) > 1:
            warn(
                f"{self.name} holds more than one plasma element, but an FBPIC "
                "`Simulation` describes a single box and a single plasma. Only "
                f"{plasmas.name} is simulated; split the others into "
                "separate lattice sections."
            )

        n_longitudinal = plasmas.simulation.n_longitudinal
        n_radial = plasmas.simulation.n_radial
        max_radial_position = plasmas.simulation.r_max
        time_step = self.zstep / n_longitudinal / c
        if plasmas.laser is not None:
            lasers = plasmas.laser_to_fbpic()
            omega0 = plasmas.laser.angular_frequency
            laser_z0 = plasmas.laser.initial_position
        for element in beamline:
            if element.hardware_type.lower() == "laser":
                lasers = element.to_fbpic()
                omega0 = element.laser.angular_frequency
                laser_z0 = element.laser.initial_position

        if self.boost is True:
            if self.gamma_boost > 0:
                self.configure_boost(plas=plasmas, omega0=omega0)
                fac11 = max_radial_position
                fac12 = (2 * self.boost.gamma0 * n_radial)
                time_step = min(fac11 / fac12 / c, self.zstep / n_longitudinal / c)
            else:
                warn("gamma_boost not set and boost is True; cannot configure BoostConverter")

        from numba import cuda
        use_cuda = True if cuda.is_available() and self.use_cuda else False

        self._simulation = Simulation(
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
            use_galilean=self.use_galilean,
            particle_shape=self.particle_shape,
            gamma_boost=self.boost.gamma0 if self.boosted else None,
        )
        self._plasma_electrons = self.simulation.add_new_species(
            **self.add_plasma_species(plasmas, typ="electron"),
        )
        if self.include_ions:
            self.simulation.add_new_species(
                **self.add_plasma_species(plasmas, typ="hydrogen"),
            )
        self.add_ionizable_species(plasmas)

        self.pin = self.prepare_bunch()

        if self.track_bunch:
            self.pin.track(self.simulation.comm)

        if lasers is not None:
            z0_antenna = (
                laser_z0 if self.laser_injection_method == "antenna" else None
            )
            add_laser_pulse(
                self.simulation,
                lasers,
                gamma_boost=self.boost.gamma0 if self.boosted else None,
                method=self.laser_injection_method,
                z0_antenna=z0_antenna,
            )

        self.configure_checkpoints()

        if self.boosted:
            v_window_boosted, = self.boost.velocity([self.v_window])
            self.simulation.set_moving_window(v=v_window_boosted)
        else:
            self.simulation.set_moving_window(v=self.v_window)

        self.simulation.diags.extend(self.prepare_diagnostics())

    def configure_checkpoints(self) -> None:
        """
        Set up checkpointing and, if asked, restart from the last checkpoint.

        Restarting reads back the iteration FBPIC last wrote, which
        :attr:`~n_step` subtracts so that the run still ends where an
        uninterrupted one would have.
        """
        from fbpic.openpmd_diag import (
            set_periodic_checkpoint,
            restart_from_checkpoint,
        )

        checkpoint_dir = (
            self.global_parameters["master_subdir"] + "/checkpoints/"
        )
        if self.use_restart:
            if not os.path.isdir(os.path.join(checkpoint_dir, "proc0", "hdf5")):
                warn(
                    f"use_restart is set for line {self.objectname} but there "
                    f"are no checkpoints under {checkpoint_dir}; starting from "
                    "the beginning instead."
                )
            else:
                restart_from_checkpoint(
                    self.simulation, checkpoint_dir=checkpoint_dir
                )
        if self.save_checkpoints:
            set_periodic_checkpoint(
                self.simulation,
                self.checkpoint_period,
                checkpoint_dir=checkpoint_dir,
            )

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
        typ: Literal["electron", "hydrogen", "positron"]
            Name of plasma species

        Returns
        -------
        Dict
            Dictionary containing plasma parameters
        """
        u_th = plas.plasma.thermal_momentum(typ)
        plas_dict = {
            "q": plas.plasma.charge(typ),
            "m": plas.plasma.mass(typ),
            "n": plas.plasma.density or None,
            "ux_th": u_th,
            "uy_th": u_th,
            "uz_th": u_th,
            **self.plasma_extent(plas),
        }
        return plas_dict

    def plasma_extent(self, plas: Plasma) -> Dict:
        """
        The arguments of `add_new_species` that say where a plasma species is
        laid down and how finely it is sampled.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            SimFrame `plasma` object

        Returns
        -------
        Dict
            Dictionary of `add_new_species` keyword arguments
        """
        return {
            "dens_func": (
                plas._relative_density_profile
                if plas.plasma.density_profile
                else None
            ),
            "p_zmin": plas.simulation.p_zmin,
            "p_zmax": plas.simulation.p_zmax,
            "p_rmin": plas.simulation.p_rmin,
            "p_rmax": plas.simulation.p_rmax,
            "p_nr": plas.simulation.particles_per_radial_cell,
            "p_nz": plas.simulation.particles_per_longitudinal_cell,
            "p_nt": plas.simulation.particles_per_angular_cell,
            "boost_positions_in_dens_func": self.boosted,
        }

    def add_ionizable_species(self, plas: Plasma) -> Any | None:
        """
        Add the ionizable atoms the plasma element asks for, if any, and set
        :attr:`~ionizable_species`.

        This is what makes ionization injection available.
        FBPIC frees electrons from them with the ADK model as the driver field
        passes, and the freed electrons join :attr:`~plasma_electrons`. The
        plasma proper is then the pre-ionized background the wake forms in.

        ``ionization_initial_level`` is worth setting rather than leaving at
        zero. Nitrogen's first five electrons come off far ahead of the wake
        and only add noise and cost; starting at level 5 spends the
        macroparticles on the two K-shell electrons that are actually injected.

        Parameters
        ----------
        plas: :class:`~laura.models.element.Plasma`
            SimFrame `plasma` object

        Returns
        -------
        The FBPIC `Particles` object holding the atoms, or None if the element
        defines no ionizable species

        Raises
        ------
        ValueError
            If ``ionizable`` is set without ``ionization_element``
        """
        if not plas.plasma.ionizable:
            return None
        element = plas.plasma.ionization_element
        if not element:
            raise ValueError(
                f"{plas.name} sets ionizable but no ionization_element. Give "
                "the atomic symbol of the gas to ionize, e.g. 'N' or 'He'."
            )
        import periodictable
        from scipy.constants import physical_constants

        try:
            atom = getattr(periodictable, element)
        except AttributeError:
            raise ValueError(
                f"{plas.name} asks to ionize '{element}', which is not an "
                "atomic symbol. Use e.g. 'N', not 'Nitrogen'."
            )
        mass = atom.mass * physical_constants["atomic mass constant"][0]
        level_start = plas.plasma.ionization_initial_level
        u_th = plas.plasma.thermal_momentum(mass=mass)

        self._ionizable_species = self.simulation.add_new_species(
            q=level_start * abs(plas.plasma.charge("hydrogen")),
            m=mass,
            n=plas.plasma.ionization_density or plas.plasma.density or None,
            ux_th=u_th,
            uy_th=u_th,
            uz_th=u_th,
            **self.plasma_extent(plas),
        )
        self._ionizable_species.make_ionizable(
            element,
            target_species=self.plasma_electrons,
            level_start=level_start,
            level_max=plas.plasma.ionization_max_level,
        )
        return self._ionizable_species

    def prepare_diagnostics(self) -> List:
        """
        Prepare the Diagnostic objects for :attr:`~simulation`

        Returns
        -------
        List
            List of diagnostic objects
        """
        from fbpic.openpmd_diag import (
            FieldDiagnostic,
            ParticleDiagnostic,
            BackTransformedFieldDiagnostic,
            BackTransformedParticleDiagnostic,
        )
        if not self.boosted:
            field_diag = FieldDiagnostic(
                self.diag_period,
                self.simulation.fld,
                comm=self.simulation.comm,
                write_dir=self.global_parameters["master_subdir"] + "/diags/",
            )
            species = {"electrons": self.plasma_electrons}
            if self.ionizable_species is not None:
                species["atoms"] = self.ionizable_species
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
            diags = [fld, part, btpart]
            if self.lab_field_diagnostics:
                diags.append(
                    BackTransformedFieldDiagnostic(
                        self.min_longitudinal_position,
                        self.max_longitudinal_position,
                        self.v_window,
                        dt_lab_diag_period,
                        self.n_lab_diag,
                        self.boost.gamma0,
                        fieldtypes=self.lab_fieldtypes,
                        period=self.write_period,
                        fldobject=self.simulation.fld,
                        comm=self.simulation.comm,
                        write_dir=self.global_parameters["master_subdir"]
                        + "/lab_diags/",
                    )
                )
            return diags

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
        self._zstart = (
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
        from ...Modules.Beams.fbpic import particles_to_beam, read_fbpic_beam_file

        super().postProcess()
        if self.pin is None:
            return
        outbeamname = f'{self.global_parameters["master_subdir"]}/{self.end}.openpmd.hdf5'
        if self.boosted:
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
            if captured == 0:
                warn(
                    "The final lab-frame snapshot captured none of the "
                    f"{self.pin.Ntot} macroparticles, so boosted-frame line "
                    f"{self.objectname} has no output distribution to write. A "
                    "particle is only recorded once it crosses the snapshot "
                    "plane; this normally means the run ended before the first "
                    "snapshot, so check n_step against interaction_time."
                )
                return
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
