"""
SIMBA Xsuite module: builds and tracks an Xsuite line. See `Xsuite github`_.

    .. _Xsuite github: https://github.com/xsuite
"""
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

from ... import exceptions
from ...Framework_objects import frameworkLattice, getGrids
from ...Modules import Beams as rbf
from ...Modules.constants import speed_of_light
from copy import deepcopy
import numpy as np
import json

from typing import Dict, List, Any, ClassVar, Literal
from warnings import warn


def _select_turn(data: Dict, turn: int) -> Dict:
    """One turn's rows out of a ``ParticlesMonitor``'s flattened dump.

    Anything not the same length as ``at_turn`` (scalars, metadata) is passed
    through untouched.
    """
    at_turn = np.asarray(data.get("at_turn", []))
    if not at_turn.size:
        return data
    mask = at_turn == turn
    selected = {}
    for key, value in data.items():
        array = np.asarray(value) if hasattr(value, "__len__") else None
        if array is not None and array.ndim == 1 and array.size == at_turn.size:
            selected[key] = array[mask]
        else:
            selected[key] = value
    return selected


class xsuiteLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` built and tracked as an Xsuite line."""

    code: str = "xsuite"
    """String indicating the lattice object type"""

    supports_turns: ClassVar[bool] = True
    """``line.track(num_turns=...)``."""

    native_time: ClassVar[tuple[str, str]] = ("zeta", "m")
    """``zeta = s - beta0 c t``, measured from the reference."""

    supports_radiation: ClassVar[bool] = True
    """``line.configure_radiation(model=...)``."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """A grid of single particles tracked for `turns`; `state` says who was
    lost and `at_turn` says when."""

    supports_frequency_map: ClassVar[bool] = True
    """The same grid with a `ParticlesMonitor`, then the shared tune
    extractor."""

    supports_periodic: ClassVar[bool] = True
    """``line.twiss()`` with no initial conditions."""

    supports_nsuperperiods: ClassVar[bool] = True
    """As N passes per turn: Xtrack has no notion of a sector."""

    supports_programs: ClassVar[bool] = True
    """Natively: attributes bound to a ``FunctionPieceWiseLinear`` of ``t_turn_s``."""

    supports_ramp: ClassVar[bool] = True
    """Natively, by ``xt.EnergyProgram``; see :meth:`bind_ramp`."""

    native_rf: ClassVar[str] = "synchronous"
    """Cavities are re-phased to the reference every pass; moved by :meth:`bind_rf_phases`."""

    rf_phase_sign: ClassVar[float] = 1.0
    """``lag`` (degrees) moved by this times a phase the reference sees."""

    otm_convention: ClassVar[str] = "x, px, y, py, zeta, delta"
    """One-turn-map convention"""

    otm_longitudinal_sign: ClassVar[int] = 1
    """One-turn-map longitudinal sign (R56)"""

    otm_longitudinal_scale: ClassVar[int | None] = 0
    """Canonical by definition: this is the convention the others convert to."""

    trackBeam: bool = True
    """Track the beam; if False, run single-particle and generate output beams from Gaussians."""

    names: List = None
    """Names of elements in the lattice"""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    final_screen: Any = None
    """Final screen object"""

    env: Any = None
    """Xsuite environment object"""

    line: Any = None
    """Xsuite line object"""

    pin: Any | None = None
    """Xsuite input particle distribution"""

    pout: Any | None = None
    """Xsuite output particle distribution"""

    tws: Any | None = None
    """Xsuite Twiss Table output"""

    reference_at_start: Any | None = None
    """The line's reference particle before tracking: a ramp moves it"""

    end_monitor: Any | None = None
    """The beam at the end of each pass but the last of a multi-pass run; the
    last is :attr:`pout`"""

    _closed_orbit: Any = None
    """The run's ``find_closed_orbit()``, found once by whichever of
    :meth:`read_one_turn_map` and :meth:`read_closed_orbit` asks first"""

    matrices: Dict | None = None
    """Dictionary of R-matrices produced by tracking"""

    context: Any | None = None
    """Xsuite context object"""

    beam_data: Dict = {}
    """Data containing beam statistics at each element"""

    grids: getGrids = None
    """Class for calculating the required number of space charge grids"""

    pic_solver: Literal["FFTSolver3D", "FFTSolver2p5D", "FFTSolver2p5DAveraged"] = "FFTSolver3D"
    """xfields' Poisson solver for space charge."""

    space_charge_step: float = 0.1
    """Metres between space-charge kicks, as Ocelot's ``unit_step``"""

    space_charge_sigmas: ClassVar[tuple] = (8.0, 8.0, 5.0)
    """Half-width of each space-charge grid in x, y and zeta, in rms beam
    sizes beyond the beam's centre"""

    space_charge_modes_off: ClassVar[tuple] = ("", "false", "none", "0", "off")
    """``charge: space_charge_mode`` values that mean no space charge"""

    program_attributes: ClassVar[dict] = {
        "Magnet": ("knl[0]", "ksl[0]"),
        "Multipole": ("knl[0]", "ksl[0]"),
        "ACDipole": ("volt", "volt"),
    }
    """``(horizontal, vertical)`` attribute an unqualified program sets, per
    Xtrack class."""

    program_signs: ClassVar[dict] = {
        "knl[0]": -1.0,
        "ksl[0]": 1.0,
    }
    """What a lattice deflection angle becomes in that attribute."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        import xobjects as xo
        self.particle_definition = self.input_particle_definition
        if has_cupy:
            self.context = xo.ContextCupy()
        else:
            self.context = xo.ContextCpu()
        self.grids = getGrids()

    @classmethod
    def native_time_scale(cls, beta0: float) -> float:
        """``zeta = -beta0 c (t - reference_time)``."""
        return -beta0 * speed_of_light


    def writeElements(self) -> None:
        """Build :attr:`line` from the section, install monitors and set :attr:`names`."""
        import xtrack as xt
        self.env = xt.Environment()
        particle_ref = xt.Particles(
            p0c=[self.reference_p0c],
            mass0=self.rest_energy,
            q0=self.reference_charge,
            zeta=0.0,
        )
        beam_length = len(self.global_parameters["beam"].x.val)
        self.line = self.section.to_xsuite(
            beam_length,
            env=self.env,
            particle_ref=particle_ref,
            save=True,
            turns=self.turns,
        )
        self.install_monitors(beam_length)
        self.names = self.line.element_names

    def install_monitors(self, num_particles: int) -> None:
        """
        Add a ``ParticlesMonitor`` at every screen, marker and BPM, recording every pass.

        Parameters
        ----------
        num_particles: int
            Particles in the beam
        """
        import xtrack as xt
        for elem in self.screens_and_markers_and_bpms:
            if elem.name in self.line.element_dict:
                self.line.element_dict[elem.name] = xt.ParticlesMonitor(
                    num_particles=num_particles,
                    start_at_turn=0,
                    stop_at_turn=self.turns * self.passes_per_turn,
                )

    @property
    def space_charge_mode(self) -> str:
        """``charge: space_charge_mode``, lower-cased; ``""`` if not given."""
        charge = self.file_block.get("charge") or {}
        return str(charge.get("space_charge_mode", "")).lower()

    @property
    def space_charge(self) -> bool:
        """
        Whether the beam is tracked with space charge: ``3d``, as Ocelot reads it::

            files:
              LINE:
                code: xsuite
                charge: {space_charge_mode: 3d}

        Any other mode but off is warned about in :meth:`preProcess`.
        """
        return self.space_charge_mode == "3d" and self.trackBeam

    @property
    def space_charge_resize(self) -> bool:
        """
        ``charge: space_charge_resize``: re-size every space-charge grid from a
        pass *with* space charge, before tracking::

            charge: {space_charge_mode: 3d, space_charge_resize: true}

        The grids are otherwise sized from a pass without it
        (:class:`~simba.exceptions.SpaceChargeOffGridWarning`). Off by default.
        """
        charge = self.file_block.get("charge") or {}
        value = charge.get("space_charge_resize", False)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "on", "1")
        return bool(value)

    @property
    def single_particle_line(self):
        """:attr:`line` without collective elements, for single-particle tracking (orbit, DA, FMA)."""
        if self.line.iscollective:
            return self.line._get_non_collective_line()
        return self.line

    def install_space_charge(self) -> None:
        """
        Put a space-charge kick every :attr:`space_charge_step` metres of :attr:`line`.

        Grids are sized from one pass of a beam sample without space charge
        (again with it, if :attr:`space_charge_resize`).
        """
        import xtrack as xt

        length = self.line.get_length()
        count = max(1, int(np.ceil(length / self.space_charge_step - 1e-9)))
        step = length / count
        names = [f"simba_space_charge_{i}" for i in range(count)]
        env = self.line.env
        for name in names:
            env.new(name, xt.Marker)
        self.line.insert(
            [env.place(name, at=(i + 0.5) * step) for i, name in enumerate(names)]
        )
        self.names = self.line.element_names
        half_widths = self.space_charge_half_widths(set(names))
        self.line.build_tracker(_context=self.context, compile=False)
        buffer = self.line._buffer
        self.line.discard_tracker()
        self._install_space_charge_grids(half_widths, step, buffer)
        if self.space_charge_resize:
            half_widths = self.space_charge_half_widths(
                set(names), with_space_charge=True
            )
            self._install_space_charge_grids(half_widths, step, buffer)

    def _install_space_charge_grids(
        self, half_widths: dict, step: float, buffer: Any
    ) -> None:
        """
        Put a ``SpaceCharge3D`` at every kick in ``half_widths``, replacing whatever is there.

        Parameters
        ----------
        half_widths: dict
            ``{kick: (x, y, zeta)}``, from :meth:`space_charge_half_widths`
        step: float
            Length each kick stands for, in m
        buffer: xobjects buffer
            The line's buffer
        """
        import xfields as xf
        import xobjects as xo

        n = self.grids.getGridSizes(len(self.pin.x))
        hz = max((width[2] for width in half_widths.values()), default=0.0)
        ladder = 1.25
        buckets = {}
        for name, (hx, hy, _) in half_widths.items():
            # widths rounded up a rung of the ladder, so neighbours share a grid
            key = tuple(
                int(np.ceil(np.log(max(h, 1e-9)) / np.log(ladder))) for h in (hx, hy)
            )
            buckets.setdefault(key, []).append(name)
        for (kx, ky), members in buckets.items():
            hx, hy = ladder**kx, ladder**ky
            pic = xf.SpaceCharge3D(
                _buffer=buffer,
                length=step,
                update_on_track=True,
                apply_z_kick=self.pic_solver == "FFTSolver3D",
                x_range=(-hx, hx),
                y_range=(-hy, hy),
                z_range=(-max(hz, 1e-9), max(hz, 1e-9)),
                nx=n,
                ny=n,
                nz=n,
                solver=self.pic_solver,
                gamma0=float(self.line.particle_ref.gamma0[0]),
                fftplan=self._space_charge_fftplan(n) if isinstance(self.context, xo.ContextCpu) else None,
            )
            # the copies go in the same buffer, which cannot move once used
            pic._buffer.grow(10 * 1024**2)
            for name in members:
                self.line.element_dict[name] = pic.copy(_buffer=buffer)

    def _space_charge_fftplan(self, n: int):
        """A numpy FFT plan for :attr:`pic_solver` on a grid of ``n`` cells a side."""
        from xobjects.context_cpu import FFTCpu

        shape, axes = {
            "FFTSolver3D": ((2 * n, 2 * n, 2 * n), (0, 1, 2)),
            "FFTSolver2p5D": ((2 * n, 2 * n, n), (0, 1)),
            "FFTSolver2p5DAveraged": ((2 * n, 2 * n), (0, 1)),
        }[self.pic_solver]
        return FFTCpu(np.zeros(shape, dtype=complex, order="F"), axes=axes, threads=0)

    def space_charge_half_widths(
        self, kicks: set, sample: int = 10000, with_space_charge: bool = False
    ) -> dict:
        """
        Beam half-widths at each space-charge kick, from one pass of a sample of :attr:`pin`.

        Parameters
        ----------
        kicks: set
            Names of the kicks
        sample: int
            Most particles to walk
        with_space_charge: bool
            Apply the installed kicks, with weights scaled up to the whole beam's charge

        Returns
        -------
        dict
            ``{kick: (x, y, zeta)}``: :attr:`space_charge_sigmas` rms sizes
            beyond the beam's centre, from zero. Kicks no live particle reaches are absent.
        """
        import xtrack as xt

        stride = max(1, int(np.ceil(len(self.pin.x) / sample)))
        keep = np.zeros(len(self.pin.x), dtype=bool)
        keep[::stride] = True
        particles = self.pin.filter(keep)
        if with_space_charge:
            particles.weight *= float(np.sum(self.pin.weight)) / float(
                np.sum(particles.weight)
            )
        recentre = self.turns * self.passes_per_turn == 1
        self.line.build_tracker(_context=self.context)
        widths = {}
        try:
            for element, name in zip(self.line.elements, self.line.element_names):
                if recentre:
                    # as run() does, one pass at a time
                    particles.zeta -= np.mean(particles.zeta)
                if name in kicks:
                    alive = np.asarray(particles.state) > 0
                    if alive.any():
                        widths[name] = tuple(
                            abs(np.mean(values)) + sigmas * np.std(values)
                            for values, sigmas in zip(
                                (np.asarray(getattr(particles, coord))[alive]
                                 for coord in ("x", "y", "zeta")),
                                self.space_charge_sigmas,
                            )
                        )
                    if with_space_charge:
                        self._track_element(element, particles)
                elif not isinstance(element, xt.ParticlesMonitor):
                    element.track(particles, increment_at_element=True)
        finally:
            self.line.discard_tracker()
        return widths

    def check_space_charge_grids(self, particles, tolerance: float = 1e-3) -> None:
        """
        Warn if more than ``tolerance`` of the beam leaving the line is outside
        the last space-charge grid.

        Parameters
        ----------
        particles: xtrack.Particles
            The beam at the end of the line
        tolerance: float
            Fraction of the beam allowed off the grid
        """
        import xfields as xf

        kicks = [e for e in self.line.elements if isinstance(e, xf.SpaceCharge3D)]
        alive = np.asarray(particles.state) > 0
        if not kicks or not alive.any():
            return
        fieldmap = kicks[-1].fieldmap
        outside = np.zeros(int(alive.sum()), dtype=bool)
        for coord, grid in (("x", fieldmap.x_grid), ("y", fieldmap.y_grid), ("zeta", fieldmap.z_grid)):
            values = np.asarray(getattr(particles, coord))[alive]
            outside |= (values < grid[0]) | (values > grid[-1])
        fraction = float(np.mean(outside))
        if fraction > tolerance:
            warn(exceptions.SpaceChargeOffGridWarning(self.objectname, fraction))

    def write(self) -> None:
        """Build the line via :meth:`writeElements`."""
        self.writeElements()

    def preProcess(self) -> None:
        """Load the input beam, convert it to Xsuite and check the space-charge mode."""
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.load_input_beam(prefix, self.particle_definition)
        self.hdf5_to_json(prefix)
        mode = self.space_charge_mode
        if mode not in self.space_charge_modes_off + ("3d",):
            warn(exceptions.SpaceChargeModeWarning(self.objectname, "Xsuite", mode, "'3d'"))

    def hdf5_to_json(self, prefix: str = "", write: bool = True) -> None:
        """
        Convert the input beam to Xsuite and set :attr:`pin`.

        Parameters
        ----------
        prefix: str
            Unused
        write: bool
            Also save it as ``<particle_definition>.xsuite.json``
        """
        xsuitebeamfilename = self.global_parameters["master_subdir"] + "/" + self.particle_definition + ".xsuite.json"
        self.pin = rbf.beam.write_xsuite_beam_file(
            self.global_parameters["beam"],
            xsuitebeamfilename,
            write=write,
            s_start=self.entrance_s,
            p0c=self.reference_p0c,
            t0=self.reference_t0,
        )

    def insert_reference_energy_increases(self) -> None:
        """Insert an ``xtrack.ReferenceEnergyIncrease`` ahead of every cavity in :attr:`line`."""
        import xtrack as xt
        from xtrack import Cavity, ReferenceEnergyIncrease

        if any(isinstance(e, ReferenceEnergyIncrease) for e in self.line.elements):
            return
        new_elements, new_names = [], []
        mass0 = float(self.line.particle_ref.mass0)
        p0c = float(self.line.particle_ref.p0c[0])
        for el, name in zip(self.line.elements, self.line.element_names):
            if isinstance(el, Cavity):
                on_crest_phase = el.phase + el.lag * np.pi / 180
                energy = np.hypot(p0c, mass0) + el.voltage * np.sin(on_crest_phase)
                p0c_after = np.sqrt(energy**2 - mass0**2)
                new_elements.append(
                    xt.ReferenceEnergyIncrease(Delta_p0c=p0c_after - p0c)
                )
                p0c = p0c_after
                new_names.append(f"{name}_p0c")
            new_elements.append(el)
            new_names.append(name)
        particle_ref = self.line.particle_ref
        self.line = xt.Line(elements=new_elements, element_names=new_names)
        self.line.particle_ref = particle_ref
        self.names = self.line.element_names

    @property
    def revolution_period(self) -> float:
        """
        Seconds per turn: ``passes_per_turn * line_length / (beta0 * clight)``.

        ``t_turn_s`` advances by one *pass* per Xsuite turn, hence the superperiod factor.
        """
        from ...Modules.constants import speed_of_light

        beta0 = float(np.atleast_1d(self.line.particle_ref.beta0)[0])
        pass_time = self.line.get_length() / (beta0 * speed_of_light)
        return self.passes_per_turn * pass_time

    def bind_programs(self) -> None:
        """Bind each programmed element's attribute to a function of ``t_turn_s``, once before tracking."""
        import xtrack as xt

        if not self.programs:
            return
        period = self.revolution_period
        clock = self.ramp_clock(self.line.get_length())
        bound = False
        for program in self.programs:
            element = self.line.element_dict.get(program.element)
            attribute = self.program_attribute(
                program, None if element is None else type(element).__name__
            )
            if attribute is None:
                continue
            sign = self.program_signs.get(attribute, 1.0) if not program.parameter else 1.0
            times, values = program.time_knots(period, clock=clock)
            name = f"{program.element}_simba_program"
            self.line.functions[name] = xt.FunctionPieceWiseLinear(
                x=times, y=[sign * value for value in values]
            )
            target = self.line.element_refs[program.element]
            *path, last = attribute.replace("]", "").split("[")
            for piece in path:
                target = target[int(piece)] if piece.isdigit() else getattr(target, piece)
            expression = self.line.functions[name](self.line.vars["t_turn_s"])
            if last.isdigit():
                target[int(last)] = expression
            else:
                setattr(target, last, expression)
            bound = True
        if bound:
            self.line.enable_time_dependent_vars = True

    def bind_rf_phases(self) -> None:
        """
        Bind each cavity's ``lag`` to a per-pass staircase in ``t_turn_s``.

        Xsuite re-phases cavities to the reference every pass; the staircase makes them run
        as :attr:`rf_mode` asks (see
        :meth:`~simba.Framework_objects.frameworkLattice.rf_phase_corrections`).
        """
        import xtrack as xt

        corrections = self.rf_phase_corrections()
        if not corrections:
            return
        passes = self.turns * self.passes_per_turn
        clock = self.ramp_clock(self.line.get_length())
        if clock is not None:
            starts = np.asarray(clock.times[:passes], dtype=float)
        else:
            starts = np.arange(passes) * self.revolution_period / self.passes_per_turn
        for name, correction in corrections.items():
            if name not in self.line.element_dict:
                warn(
                    f"Line '{self.objectname}' moves the RF phase of '{name}', "
                    "which is not in the Xsuite line. It runs as given."
                )
                continue
            lag = float(self.line.element_dict[name].lag)
            times, lags = self.pass_staircase(
                starts, lag + self.rf_phase_shifts(correction)
            )
            function = f"{name}_simba_rf_phase"
            self.line.functions[function] = xt.FunctionPieceWiseLinear(
                x=times, y=lags
            )
            self.line.element_refs[name].lag = self.line.functions[function](
                self.line.vars["t_turn_s"]
            )
        self.line.enable_time_dependent_vars = True

    def bind_ramp(self) -> None:
        """
        Give the line an ``xt.EnergyProgram`` holding :attr:`ramp`, one knot per pass.

        Knot times come from :meth:`~simba.Framework_objects.frameworkLattice.ramp_clock`.
        """
        import xtrack as xt

        if not self.ramped:
            return
        clock = self.ramp_clock(self.line.get_length())
        self.line.energy_program = xt.EnergyProgram(
            t_s=np.asarray(clock.times),
            p0c=self.ramp.p0c_per_pass(
                self.turns, self.rest_energy, self.passes_per_turn
            ),
        )

    def run(self) -> None:
        """Track the beam and set :attr:`tws` and :attr:`pout`."""
        if not self.fixed_reference:
            self.insert_reference_energy_increases()
        self.bind_ramp()
        if self.space_charge:
            self.install_space_charge()
        self.line.build_tracker(_context=self.context)
        if self.radiation not in (None, "off"):
            self.line.configure_radiation(model=self.radiation)
        self.line.freeze_energy(state=False, force=True)
        self.line.config["FREEZE_VAR_zeta"] = False
        self.bind_programs()
        self.bind_rf_phases()
        if self.line.energy_program is not None:
            self.line.enable_time_dependent_vars = True
        self.reference_at_start = self.line.particle_ref.copy()
        self._closed_orbit = None
        pin = deepcopy(self.pin)

        passes = self.turns * self.passes_per_turn
        if passes > 1:
            import xtrack as xt
            self.end_monitor = xt.ParticlesMonitor(
                num_particles=len(pin.x), start_at_turn=0, stop_at_turn=passes
            )
            self.line.track(pin, num_turns=passes, turn_by_turn_monitor=self.end_monitor)
            self.pout = pin
            self.check_space_charge_grids(pin)
            self.collect_beam_data(deepcopy(pin))
            self.tws = self._twiss()
            return

        for el, name in zip(self.line.elements, self.line.element_names):
            pin.zeta -= np.mean(pin.zeta)  # Center zeta
            self._track_element(el, pin)  # Track in-place
            self.beam_data.update({name: self.bunch_statistics(pin)})
        self.beam_data.update({"_end_point": self.bunch_statistics(pin)})
        self.pout = pin
        self.check_space_charge_grids(pin)
        self.tws = self._twiss()

    @staticmethod
    def _track_element(element, particles) -> None:
        """Track ``particles`` through one element, in place. A collective
        element (space charge) takes no ``increment_at_element``."""
        if getattr(element, "iscollective", False):
            element.track(particles)
        else:
            element.track(particles, increment_at_element=True)

    def bunch_statistics(self, particles) -> dict:
        """Tracked (not ``line.twiss()``) bunch statistics at one point, for the twiss file.

        Parameters
        ----------
        particles: xtrack.Particles
            Distribution at one point in the line

        Returns
        -------
        dict
            One row of :attr:`beam_data`
        """
        # lost particles keep the coordinates where they were lost
        if np.any(np.asarray(particles.state) <= 0):
            particles = particles.filter(particles.state > 0)
        return {
            'mean_x': np.mean(particles.x),
            'mean_y': np.mean(particles.y),
            'sigma_x': np.std(particles.x),
            'sigma_px': np.std(particles.px),
            'sigma_y': np.std(particles.y),
            'sigma_py': np.std(particles.py),
            'sigma_zeta': np.std(particles.zeta),
            'sigma_delta': np.std(particles.delta),
            'momentum': np.mean(particles.energy) - particles.mass0,
            'emit_xn': np.mean(
                self.compute_norm_emit(particles.x, particles.px, particles)
            ),
            'emit_yn': np.mean(
                self.compute_norm_emit(particles.y, particles.py, particles)
            ),
            'emit_xn_corrected': np.mean(
                self.compute_norm_emit_corrected(
                    particles.x, particles.px, particles
                )
            ),
            'emit_yn_corrected': np.mean(
                self.compute_norm_emit_corrected(
                    particles.y, particles.py, particles
                )
            ),
        }

    def collect_beam_data(self, particles) -> None:
        """Fill :attr:`beam_data` by walking ``particles`` down the line.

        Parameters
        ----------
        particles: xtrack.Particles
            A copy of the beam entering the pass
        """
        stats = None
        for el, name in zip(self.line.elements, self.line.element_names):
            self._track_element(el, particles)
            stats = self.bunch_statistics(particles)
            self.beam_data.update({name: stats})
        if stats is not None:
            self.beam_data.update({"_end_point": stats})

    def _twiss(self):
        """
        Twiss the line as at turn 1: periodic if :attr:`periodic`, else from the beam's twiss.

        Returns
        -------
        xtrack.TwissTable
        """
        dependent = self.line.enable_time_dependent_vars
        if not dependent:
            return self._twiss_now()
        self.line.enable_time_dependent_vars = False
        self.line.vars["t_turn_s"] = 0.0
        if self.reference_at_start is not None:
            self.line.particle_ref = self.reference_at_start.copy()
        try:
            return self._twiss_now()
        finally:
            self.line.enable_time_dependent_vars = dependent

    def _twiss_now(self):
        """:meth:`_twiss`, on the line as it stands."""
        kwargs = {
            "compute_R_element_by_element": False,
            "method": "6d",
            "freeze_energy": False,
        }
        if self.periodic:
            if not self.rf_voltage:
                kwargs.update(method="4d")
                kwargs.pop("freeze_energy")
            return self.line.twiss(**kwargs)
        return self.line.twiss(
            betx=self.global_parameters["beam"].twiss.beta_x.val,
            alfx=self.global_parameters["beam"].twiss.alpha_x.val,
            bety=self.global_parameters["beam"].twiss.beta_y.val,
            alfy=self.global_parameters["beam"].twiss.alpha_y.val,
            **kwargs,
        )

    def track_reference_particle(self) -> dict:
        """
        Track one particle launched just off the closed orbit, sampled after each turn.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py``, each of length :attr:`turns`.
        """
        import xtrack as xt

        orbit = self.read_closed_orbit()
        if orbit is None:
            orbit = np.zeros(6)
        nudge = float((self.da_settings or {}).get("x_max", 1e-3)) / 100.0
        particles = self.line.build_particles(
            x=np.array([orbit[0] + nudge]),
            px=np.array([orbit[1]]),
            y=np.array([orbit[2] + nudge]),
            py=np.array([orbit[3]]),
        )
        passes = self.turns * self.passes_per_turn
        monitor = xt.ParticlesMonitor(
            _context=self.context,
            start_at_turn=0,
            stop_at_turn=passes,
            num_particles=1,
        )
        self.single_particle_line.track(
            particles, num_turns=passes, turn_by_turn_monitor=monitor
        )
        stride = self.passes_per_turn
        return {
            name: np.append(
                np.asarray(getattr(monitor, name))[0][stride::stride],
                float(np.atleast_1d(getattr(particles, name))[0]),
            )
            for name in ("x", "px", "y", "py")
        }

    def _da_particles(self, points=None):
        """One particle per ``(x, y)`` start; the :meth:`da_grid` by default."""
        if points is None:
            xs, ys = self.da_grid()
            points = [(x, y) for y in ys for x in xs]
        grid_x, grid_y = (np.array(v, dtype=float) for v in zip(*points))
        return grid_x, grid_y, self.line.build_particles(x=grid_x, y=grid_y)

    def run_dynamic_aperture(self) -> list:
        """
        Track one particle per point of :meth:`da_rays` in a single call and record survival.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per start.
        """
        grid_x, grid_y, particles = self._da_particles(self.da_rays())
        self.single_particle_line.track(
            particles, num_turns=self.turns * self.passes_per_turn
        )
        order = np.argsort(particles.particle_id)
        turns = np.asarray(particles.at_turn)[order] // self.passes_per_turn
        self.dynamic_aperture = [
            (float(x), float(y), int(turn))
            for x, y, turn in zip(grid_x, grid_y, turns)
        ]
        return self.dynamic_aperture

    def run_frequency_map(self) -> list:
        """
        Tune footprint over the aperture grid, from turn-by-turn monitor data.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y, diffusion)`` per surviving grid point.
        """
        import xtrack as xt

        grid_x, grid_y, particles = self._da_particles()
        passes = self.turns * self.passes_per_turn
        monitor = xt.ParticlesMonitor(
            _context=self.context,
            start_at_turn=0,
            stop_at_turn=passes,
            num_particles=len(grid_x),
        )
        self.single_particle_line.track(
            particles, num_turns=passes, turn_by_turn_monitor=monitor
        )
        stride = self.passes_per_turn
        coords = [np.asarray(getattr(monitor, k))[:, ::stride] for k in ("x", "px", "y", "py")]
        # tracking moves lost particles to the end; the monitor is by id
        state = np.asarray(particles.state)[np.argsort(particles.particle_id)]
        return self._footprint(
            (grid_x[i], grid_y[i], *(c[i] for c in coords))
            for i in range(len(grid_x))
            if int(state[i]) > 0
        )

    def find_closed_orbit(self):
        """``line.find_closed_orbit()``, once a run."""
        if self._closed_orbit is None:
            self._closed_orbit = self.line.find_closed_orbit()
        return self._closed_orbit

    def read_closed_orbit(self):
        """:meth:`find_closed_orbit`, already called for the one-turn map."""
        try:
            orbit = self.find_closed_orbit()
        except Exception as error:
            warn(f"Xsuite found no closed orbit for {self.objectname}: {error}")
            return None
        return np.array(
            [
                float(np.atleast_1d(getattr(orbit, name))[0])
                for name in ("x", "px", "y", "py", "zeta", "delta")
            ]
        )

    def read_optics_summary(self) -> dict:
        """Xsuite reports all four off the periodic twiss directly."""
        if self.tws is None:
            return {}
        keys = {
            "tune_x_total": "qx",
            "tune_y_total": "qy",
            "chromaticity_x": "dqx",
            "chromaticity_y": "dqy",
        }
        summary = {}
        for name, attribute in keys.items():
            value = getattr(self.tws, attribute, None)
            if value is not None:
                summary[name] = float(value)
        return summary

    def read_one_turn_map(self):
        """
        ``line.compute_R_matrix()``, which finite-differences about the
        closed orbit.

        Returns
        -------
        numpy.ndarray | None
            The 6x6 map, or None if Xsuite could not find a closed orbit.
        """
        try:
            closed_orbit = self.find_closed_orbit()
            result = self.line.compute_R_matrix(particle_on_co=closed_orbit)
            return np.asarray(result["R_matrix"], dtype=float)
        except Exception as error:
            warn(
                f"Line '{self.objectname}': Xsuite could not compute a "
                f"one-turn matrix ({error}). An unstable or unclosed lattice "
                "has no one-turn map."
            )
            return None

    def compute_norm_emit(self, coord, mom, particles):
        cov = np.cov(coord, mom)
        emit = np.sqrt(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2)
        gamma = particles.energy / particles.mass0
        beta = particles.beta0 * (1 + particles.delta.mean()) / (1 + particles.delta.mean() * particles.beta0 ** 2)
        return gamma * beta * emit

    def compute_norm_emit_corrected(self, coord, mom, particles):
        """
        Normalised emittance with the linear correlation to ``delta`` removed.

        Parameters
        ----------
        coord: np.ndarray
            x or y
        mom: np.ndarray
            px or py
        particles: xtrack.Particles
            Source of ``delta`` and the normalisation

        Returns
        -------
        float
        """
        delta = np.asarray(particles.delta)
        var_delta = np.var(delta)
        if var_delta > 0:
            coord = np.asarray(coord) - (np.cov(coord, delta)[0, 1] / var_delta) * delta
            mom = np.asarray(mom) - (np.cov(mom, delta)[0, 1] / var_delta) * delta
        return self.compute_norm_emit(coord, mom, particles)

    def postProcess(self) -> None:
        """
        Write beams at every monitor and the line end, for each of :meth:`output_turns`, and the twiss CSV.
        """
        super().postProcess()
        svals = {
            name: value + self.entrance_s
            for name, value in self.getSValues(as_dict=True).items()
        }
        ends = self.end_monitor.data.to_dict() if self.end_monitor is not None else None
        for data_turn, name_turn in self.output_turns():
            turn = self.beam_turn(data_turn)
            if turn == self.turns or ends is None:
                payload = self.pout.to_dict()
            else:
                # the end of turn k is the start of the pass after it
                payload = _select_turn(ends, turn * self.passes_per_turn)
            self._write_xsuite_beam(
                payload, self.end, name_turn, turn,
                self.endObject.physical.middle.z, svals[self.end],
            )
        for elem in self.screens_and_markers_and_bpms:
            # the end's is written above
            if elem.name == self.end or elem.name not in self.line.element_dict:
                continue
            if not self.writes_output(elem.name):
                continue
            data = self.line[elem.name].data.to_dict()
            for data_turn, name_turn in self.output_turns():
                turn = self.beam_turn(data_turn)
                self._write_xsuite_beam(
                    _select_turn(data, turn * self.passes_per_turn - 1),
                    elem.name, name_turn, turn,
                    elem.physical.middle.z, svals[elem.name],
                )
        self._write_twiss_csv()

    def _write_xsuite_beam(
        self, payload: dict, name: str, name_turn: int | None, turn: int,
        zstart: float, s: float,
    ) -> None:
        """
        Write one beam via :meth:`write_beam_file`; the last turn's also as ``.xsuite.json``.

        Parameters
        ----------
        payload: dict
            ``Particles.to_dict()``, or one turn of a monitor's
        name: str
            Element the beam is at
        name_turn: int | None
            Turn as :meth:`output_turns` names it
        turn: int
            Turn the beam is from
        zstart: float
            z of the element
        s: float
            s of the element, in the machine
        """
        import xobjects as xo
        if turn == self.turns:
            stem = self.output_basename(name)
            fname = f"{self.global_parameters['master_subdir']}/{stem}.xsuite.json"
            with open(fname, "w") as fid:
                json.dump(payload, fid, cls=xo.JEncoder)
        t_reference = None
        if self.uses_reference_clock:
            t_reference = self.reference_time(s - self.entrance_s, self.last_pass(turn))
        beam = deepcopy(self.global_parameters["beam"])
        beam.read_xsuite_beam_file(
            payload, zstart=zstart, s=s, ref_index=self.ref_idx, t_reference=t_reference
        )
        beam.turn = turn
        self.write_beam_file(beam, name, name_turn)

    def _write_twiss_csv(self) -> None:
        """The twiss table, with the tracked beam statistics beside it."""
        df = self.tws.to_pandas()
        # Anchor s to the lattice entrance (not the incoming beam's accumulated s),
        # as every code does.
        df["s"] += self.entrance_s
        svals = np.array(self.getSValues(at_entrance=False)) + df["s"][0]
        zvals = [a[-1] for a in self.getZValues()]
        df["z"] = np.interp(df["s"], svals, zvals)
        if self.beam_data:
            for k in next(iter(self.beam_data.values())):
                df[k] = [x[k] for x in self.beam_data.values()]
        df.to_csv(f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.csv')
