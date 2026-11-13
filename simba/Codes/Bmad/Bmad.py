"""
SIMBA Bmad module: writes Bmad lattices and tracks them with PyTao. See `Bmad manual`_
and `Tao manual`_.

    .. _Bmad manual: https://www.classe.cornell.edu/bmad/manual.html

    .. _Tao manual: https://www.classe.cornell.edu/bmad/tao.html
"""

import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar
from warnings import warn

import numpy as np
from laura.models.simulation import TwissMatchSimulationElement
from laura.translator.converters.converter import translate_elements
from laura.translator.utils.functions import sanitize_string

from ... import exceptions
from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules import constants
from ...Modules.Twiss.bmad import save_bmad_twiss_hdf

LATTICE_TWISS = {
    "s": "ele.s",
    "e_tot": "ele.e_tot",
    "p0c": "ele.p0c",
    "design_beta_x": "ele.a.beta",
    "design_alpha_x": "ele.a.alpha",
    "design_gamma_x": "ele.a.gamma",
    "design_beta_y": "ele.b.beta",
    "design_alpha_y": "ele.b.alpha",
    "design_gamma_y": "ele.b.gamma",
    "design_eta_x": "ele.x.eta",
    "design_etap_x": "ele.x.etap",
    "design_eta_y": "ele.y.eta",
    "design_etap_y": "ele.y.etap",
    "mu_x": "ele.a.phi",
    "mu_y": "ele.b.phi",
}
"""Lattice functions to extract from Tao, keyed by their name in the twiss file"""

BEAM_TWISS = {
    "beam_charge": "charge_live",
    "beam_n_particle": "n_particle_live",
    "beam_t": "centroid_t",
    "beam_p0c": "centroid_p0c",
    "beam_x": "centroid_vec_1",
    "beam_y": "centroid_vec_3",
    "beam_delta": "centroid_vec_6",
    "beam_sigma_x": "twiss_sigma_x",
    "beam_sigma_xp": "twiss_sigma_p_x",
    "beam_sigma_y": "twiss_sigma_y",
    "beam_sigma_yp": "twiss_sigma_p_y",
    "beam_sigma_z": "twiss_sigma_z",
    "beam_sigma_delta": "twiss_sigma_p_z",
    "beam_sigma_t": "sigma_t",
    "beam_emit_x": "twiss_emit_x",
    "beam_emit_y": "twiss_emit_y",
    "beam_emit_z": "twiss_emit_z",
    "beam_norm_emit_x": "twiss_norm_emit_x",
    "beam_norm_emit_y": "twiss_norm_emit_y",
    "beam_norm_emit_z": "twiss_norm_emit_z",
    "beam_norm_emit_a": "twiss_norm_emit_a",
    "beam_norm_emit_b": "twiss_norm_emit_b",
    "beam_beta_x": "twiss_beta_x",
    "beam_alpha_x": "twiss_alpha_x",
    "beam_gamma_x": "twiss_gamma_x",
    "beam_beta_y": "twiss_beta_y",
    "beam_alpha_y": "twiss_alpha_y",
    "beam_gamma_y": "twiss_gamma_y",
    "beam_beta_a": "twiss_beta_a",
    "beam_alpha_a": "twiss_alpha_a",
    "beam_beta_b": "twiss_beta_b",
    "beam_alpha_b": "twiss_alpha_b",
    "beam_beta_z": "twiss_beta_z",
    "beam_alpha_z": "twiss_alpha_z",
    "beam_gamma_z": "twiss_gamma_z",
    "beam_eta_x": "twiss_eta_x",
    "beam_etap_x": "twiss_etap_x",
    "beam_eta_y": "twiss_eta_y",
    "beam_etap_y": "twiss_etap_y",
}
"""Bunch parameters to extract from Tao, keyed by their name in the twiss file"""


class bmadLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` written as Bmad and tracked with PyTao."""

    code: str = "bmad"
    """String indicating the lattice type"""

    supports_csr: ClassVar[bool] = True
    supports_lsc: ClassVar[bool] = True

    radiates_by_default: ClassVar[bool] = True
    """Bmad lattices from LAURA radiate by default."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """Survival along :meth:`da_rays` through ``long_term_tracking``; see :meth:`run_dynamic_aperture`."""

    supports_frequency_map: ClassVar[bool] = True
    """The grid through ``long_term_tracking``; see :meth:`_track_grid_turn_by_turn`."""

    supports_radiation: ClassVar[bool] = True
    """``bmad_com[radiation_damping_on]``/``[radiation_fluctuations_on]``, written by LAURA."""

    supports_periodic: ClassVar[bool] = True
    """``parameter[geometry] = closed``, written by LAURA; Twiss from the one-turn map."""

    supports_programs: ClassVar[bool] = True
    """As ``ramper`` elements in ``long_term_tracking`` (:meth:`_rampers`), by ``set
    element`` in Tao (:meth:`apply_programs`)."""

    supports_ramp: ClassVar[bool] = True
    """A ``ramper`` on every element's ``p0c``; see :meth:`_rampers`."""

    supports_turns: ClassVar[bool] = True
    """Bmad's ``long_term_tracking`` tracks every turn but the last, Tao the last, for
    the screens; see :meth:`_run_long_term_tracking`."""

    native_rf: ClassVar[str] = "fixed"
    """Turns are tracked in absolute time (``bmad_com%absolute_time_tracking``), so each
    cavity is a free-running oscillator."""

    rf_phase_per_radian: ClassVar[float] = 1 / (2 * np.pi)
    """``phi0`` is in turns; see :meth:`cavity_phase`."""

    rf_phase_sign: ClassVar[float] = -1.0
    """Measured against elegant's ``follow`` ramp: an rfcavity's phase goes as ``2 pi phi0 - w t``."""

    tracks_multiple_species: ClassVar[bool] = True
    """Each Bmad particle has its own ``species``; only those with the tracked one's mass,
    though (see :meth:`check_other_species`)."""

    otm_convention: ClassVar[str] = "x, px, y, py, z, pz"
    """One-turn map convention."""

    otm_longitudinal_sign: ClassVar[int] = 1
    """One-turn map longitudinal sign (i.e. R56)"""

    otm_longitudinal_scale: ClassVar[int | None] = 0
    """Canonical by definition: this is the convention the others convert to."""

    particle_definition: str | None = None
    """Initial particle distribution as a string"""

    input_beam_file: str | None = None
    """Input beam file name"""

    lattice_file: str | None = None
    """Lattice file name"""

    tao_init_file: str | None = None
    """Tao initialization file name"""

    tao: Any | None = None
    """PyTao instance"""

    collective_step: float = 0.01
    """Bmad ``space_charge_com[ds_track_step]`` [m] used when CSR or space charge
    is on and the lattice does not set one."""

    space_charge_n_bin: int | None = None
    """Number of space-charge bins"""

    _SAVED_AT_LIMIT: int = 200
    """Tao's character limit on ``beam_saved_at``"""

    _ALIVE: int = 1
    """Bmad/openPMD status flag for a live particle"""

    _GRID_CHARGE: ClassVar[float] = 1e-15
    """Charge given to each frequency-map grid particle."""

    _POSITION_FILE_CMD: ClassVar[str] = "set beam_init position_file = {path}"
    """Tao command pointing at a particle file"""

    _STALE_CLOCK_KEYS: ClassVar[frozenset] = frozenset({
        "quadrupole", "sbend", "rbend", "sol_quad", "solenoid", "sextupole", "octupole",
        "thick_multipole", "hkicker", "vkicker", "kicker", "ac_kicker", "elseparator",
        "monitor", "instrument", "pipe", "rcollimator", "ecollimator",
    })
    """Element types Bmad's ``bmad_standard`` tracking times by their ``delta_ref_time``
    (``track1_bmad``); drifts, cavities and wigglers time each particle afresh.
    See :meth:`_clock_slips`."""

    _clock: tuple | None = None
    """This run's :meth:`_clock_slips`, if ramped."""

    _ramper_lines: str = ""
    """This run's :meth:`_rampers`, made before :meth:`_set_last_turn` moves Tao on to turn N."""

    _PHASE_SPACE: ClassVar[tuple] = ("x", "px", "y", "py", "z", "pz")
    """Bmad's six phase-space coordinates"""

    libtao: str | None = None
    """Location of libtao.so"""

    chromaticity_delta: float = 1e-4
    """Momentum step for the chromaticity finite difference; Tao's ``delta_e_chrom`` default."""

    program_attributes: ClassVar[dict] = {
        "hkicker": ("kick", "kick"),
        "vkicker": ("kick", "kick"),
        "kicker": ("hkick", "vkick"),
        "ac_kicker": ("bl_hkick", "bl_vkick"),
    }
    """``(horizontal, vertical)`` Bmad attribute an unqualified program
    sets, per element type."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition
        if self.libtao is None:
            self.libtao = self.executables["tao"][0]

    def preProcess(self) -> None:
        """Load the input beam and write it as a Bmad beam file."""
        space_charge_n_bin = self.csr_bins or self.lsc_bins
        super().preProcess()
        self.space_charge_n_bin = space_charge_n_bin
        self.load_input_beam(self.get_prefix(), self.particle_definition)
        beam = self.global_parameters["beam"]
        self._input_species = beam.particle_species if beam.is_mixed else None
        if self._input_species is not None and self.turns > 1:
            raise NotImplementedError(
                f"Line '{self.objectname}': Bmad cannot track a beam of several "
                "species for more than one turn"
            )
        self._warn_collective_species()
        self.input_beam_file = str(
            Path(self.global_parameters["master_subdir"])
            / f"{self.objectname}.bmad.beam"
        )
        self._write_bmad_beam_file()

    def _warn_collective_species(self) -> None:
        """
        Warn if a beam of several species meets Bmad's CSR or space charge.

        Bmad bins the bunch by each particle's ``charge`` (a positive macroparticle
        weight) and takes one species for the whole bunch, the first particle's
        (``csr_and_space_charge_mod``: ``csr%species = bunch%particle(1)%species``),
        so positrons add to an electron bunch's charge density rather than cancel it.
        """
        if getattr(self, "_input_species", None) is None:
            return
        space_charge = self.space_charge_mode == "3d" or self.lsc_enable
        effects = [name for name, on in (("CSR", self.csr_enable), ("space charge", space_charge)) if on]
        if effects:
            warn(exceptions.CollectiveSpeciesWarning(
                self.objectname, "Bmad", " and ".join(effects),
                "bins every particle by the size of its charge, as the first particle's species",
            ))

    def check_other_species(self) -> None:
        """
        As :meth:`~simba.Framework_objects.frameworkLattice.check_other_species`, but
        refuse charged species of another mass than the tracked one.

        Bmad scales a magnet's strength for such a particle by its charge-to-mass ratio
        relative to the reference (``rel_tracking_charge_to_mass``), but takes its
        velocity from its momentum, so no input gets both its steering and its timing
        right.

        Raises
        ------
        :class:`~simba.exceptions.MixedMassError`
        """
        super().check_other_species()
        tracked = self.global_parameters["beam"].species
        mass = rbf.SPECIES[tracked][0]
        others = [
            name for name in (getattr(self, "_carried_species", None) or {})
            if rbf.SPECIES[name][1] != 0 and rbf.SPECIES[name][0] != mass
        ]
        if others:
            raise exceptions.MixedMassError(
                self.objectname, "Bmad", tracked, others,
                "Bmad steers a particle of another mass by its charge-to-mass ratio "
                "but times it by its momentum, so cannot get both right.",
            )

    def _reference_value(self, values, fallback: float = None) -> float:
        """The reference particle's value, if the beam has one; else
        ``fallback`` (the incoming beam's mean, before sampling), else the
        mean of ``values``."""
        values = np.asarray(values)
        if not values.size:
            raise ValueError("Cannot create a Bmad lattice without beam particles")
        if self.ref_idx is not None and 0 <= self.ref_idx < len(values):
            return float(values[self.ref_idx])
        if fallback is not None:
            return float(fallback)
        return float(np.mean(values))

    def _reference_p0c(self) -> float:
        """The :attr:`ramp`'s first turn, else a ring's :attr:`design_p0c`, else the
        reference particle's ``cp`` in eV, else :attr:`reference_p0c`."""
        if self.ramped:
            return self.ramp_p0c(1)
        if self.design_p0c is not None:
            return self.design_p0c
        beam = self.global_parameters["beam"]
        return self._reference_value(beam.cp.val, self.reference_p0c)

    def _write_bmad_beam_file(self) -> None:
        """Write the beam to :attr:`input_beam_file` in Bmad's ASCII format."""
        beam = self.global_parameters["beam"]
        p0c = self._reference_p0c()
        z = beam.z.val - self._reference_value(beam.z.val, self.reference_z0)
        particles = np.column_stack(
            (
                beam.x.val,
                beam.cpx.val / p0c,
                beam.y.val,
                beam.cpy.val / p0c,
                z,
                beam.cp.val / p0c - 1,
                np.abs(beam.charge.val),
                beam.t.val,
            )
        )
        species = getattr(self, "_input_species", None)
        if species is None:
            np.savetxt(
                self.input_beam_file,
                particles,
                header=(
                    f"# species = {beam.species}\n"
                    "# state = alive\n"
                    f"# p0c = {p0c}\n"
                    f"# charge_tot = {abs(float(beam.Q.val))}\n"
                    "#! x px y py z pz charge time"
                ),
                comments="",
            )
            return
        # several species: one per particle in a column (Bmad's names are simba's).
        # charge_tot is the sum of the macroparticle sizes, not the net charge
        self._warn_near_axis(particles, species, beam.species)
        with open(self.input_beam_file, "w") as f:
            f.write(
                "# state = alive\n"
                f"# p0c = {p0c}\n"
                f"# charge_tot = {float(np.sum(particles[:, 6]))}\n"
                "#! x px y py z pz charge time species\n"
            )
            for row, name in zip(particles, species):
                f.write(" ".join(f"{v:.18e}" for v in row) + f" {name}\n")

    def _warn_near_axis(self, particles, species, tracked: str) -> None:
        """
        Warn of particles of another species that Bmad would track as the reference one.

        Bmad's bend switches to the reference particle's linear map near the zero orbit
        (``track_a_bend``: ``|x g|, |px|, |py|, |pz| < 1e-9``) without checking the
        particle is of the reference species, so such a particle is bent as the reference.
        """
        near = np.all(np.abs(particles[:, [1, 3, 5]]) < 1e-9, axis=1) & (np.asarray(species) != tracked)
        if np.any(near):
            warn(exceptions.NearAxisSpeciesWarning(
                self.objectname, int(np.count_nonzero(near)), tracked,
                "Bmad's bends use the reference particle's map",
            ))

    def _reference_energy(self) -> float:
        """
        Reference energy: the :attr:`ramp`'s first turn's, else :attr:`reference_energy` if :attr:`design_p0c` is set, else the reference particle's.

        Returns
        -------
        float
        """
        if self.ramped:
            return float(np.hypot(self._reference_p0c(), self.rest_energy))
        if self.design_p0c is not None:
            return self.reference_energy
        return self._reference_value(
            self.global_parameters["beam"].energy.val, self.reference_energy
        )

    def _bmad_initial_twiss(self) -> TwissMatchSimulationElement | None:
        """
        Initial Twiss from the incoming beam.

        Returns
        -------
        TwissMatchSimulationElement | None
            None for a periodic line.
        """
        if self.periodic:
            return None
        twiss = self.global_parameters["beam"].twiss
        return TwissMatchSimulationElement(
            beta_x=float(twiss.beta_x.val),
            alpha_x=float(twiss.alpha_x.val),
            beta_y=float(twiss.beta_y.val),
            alpha_y=float(twiss.alpha_y.val),
        )

    def _write_position_file(self, path: str, rows, states=None) -> None:
        """
        Write particles in the ASCII form ``beam_init%position_file`` reads.

        ``time`` is written as zero rather than the tracked time: Bmad takes
        the particle's time at the start element from ``z``.

        Parameters
        ----------
        path: str
            File to write
        rows: array
            ``n_particles x 6`` of ``x px y py z pz`` in Bmad coordinates (as
            ``Tao.bunch1`` gives, not ``bunch_data``)
        states: array | None
            Tao's integer particle states; None writes every particle alive
        """
        rows = np.asarray(rows, dtype=float)
        alive = (
            np.ones(len(rows), dtype=bool)
            if states is None
            else np.asarray(states) == self._ALIVE
        )
        body = "\n".join(
            " ".join(f"{value:.15e}" for value in row)
            + f" {self._GRID_CHARGE:.15e} 0.0 "
            + ("Alive" if is_alive else "Lost")
            for row, is_alive in zip(rows, alive)
        )
        with open(path, "w") as handle:
            handle.write(
                f"# species = {self.global_parameters['beam'].species}\n"
                "# state = alive\n"
                f"# p0c = {self._reference_p0c()}\n"
                f"# charge_tot = {self._GRID_CHARGE * len(rows)}\n"
                "#! x px y py z pz charge time state\n"
                f"{body}\n"
            )

    def _write_grid_beam_file(self, path: str, points: list | None = None) -> list:
        """
        Write one particle per start, offset from the closed orbit.

        Parameters
        ----------
        path: str
            File to write
        points: list | None
            ``(x, y)`` starts; the frequency-map grid (:meth:`_grid_points`) if None

        Returns
        -------
        list
            The ``(x, y)`` the particles were placed at, in file order.
        """
        orbit = self.read_closed_orbit()
        if orbit is None:
            orbit = np.zeros(6)
        points = self._grid_points() if points is None else points
        self._write_position_file(
            path,
            [
                [orbit[0] + x, orbit[1], orbit[2] + y, orbit[3], 0.0, 0.0]
                for x, y in points
            ],
        )
        return points

    def run_frequency_map(self) -> list:
        """
        Tune footprint from :meth:`_track_grid_turn_by_turn`.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y, diffusion)`` per surviving grid point.
        """
        points = self._grid_points()
        tracks = self._track_grid_turn_by_turn()
        if not tracks:
            return []
        return self._footprint(
            (x, y, tracks["x"][i], tracks["px"][i], tracks["y"][i], tracks["py"][i])
            for i, (x, y) in enumerate(points)
            # a lost particle keeps its coordinates where it was lost
            if i < len(tracks["x"]) and np.all(tracks["state"][i] == self._ALIVE)
        )

    def _grid_points(self) -> list:
        """``(x, y)`` of the scan grid, in the order the beam file holds."""
        xs, ys = self.da_grid()
        return [(float(x), float(y)) for y in ys for x in xs]

    def _track_grid_turn_by_turn(self) -> dict:
        """
        Track the grid with ``long_term_tracking``, as the beam is: same programs, ramp and RF.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py``/``state``, each ``n_particles x turns``. Empty
            if it could not be tracked, or every particle was lost.
        """
        if self.tao is None:
            warn(
                f"Line '{self.objectname}': Tao must be run before the grid "
                "can be tracked."
            )
            return {}
        grid_file = Path(self.lattice_file).with_suffix(".fma.beam")
        try:
            self._write_grid_beam_file(str(grid_file))
            # ASCII: Bmad's own px/py, which the openPMD file leaves as momenta
            files = self._run_long_term_tracking(
                f"{self.objectname}.fma.turns", grid_file.name, self.turns, 1
            )
        except Exception as error:
            warn(f"long_term_tracking failed on the grid of {self.objectname}: {error}")
            return {}
        bunches = [self._read_ascii_beam(files[turn]) for turn in range(1, self.turns + 1)]
        return {name: np.column_stack([b[name] for b in bunches]) for name in bunches[0]}

    def _read_ascii_beam(self, path: Path) -> dict:
        """
        ``x``/``px``/``y``/``py``/``state`` from a beam file in Bmad's ASCII form, one entry per particle.

        ``state`` is :attr:`_ALIVE` for a live particle, else 0.
        """
        lines = path.read_text().splitlines()
        columns = next(line for line in lines if line.startswith("#!")).split()[1:]
        rows = [line.split() for line in lines if line.strip() and not line.startswith("#")]
        beam = {
            name: np.array([float(row[columns.index(name)]) for row in rows])
            for name in ("x", "px", "y", "py")
        }
        state = columns.index("state")
        beam["state"] = np.array([self._ALIVE if row[state] == "Alive" else 0 for row in rows])
        return beam

    def apply_programs(self, turn: int) -> None:
        """
        ``set element`` each programmed element's attribute for `turn` in Tao.

        Parameters
        ----------
        turn: int
            Turn number, 1-based
        """
        if self.tao is None:
            return
        for program, name, attribute, before in self._programs():
            self.tao.cmd(
                f"set element {name} {attribute} = {program.value_at(turn, before)}",
                raises=False,
            )

    def _programs(self):
        """
        Each program Bmad can set, with the value Tao's design gives before its first knot.

        Yields
        ------
        tuple
            ``(program, Bmad name, attribute, design value or None)``
        """
        for program in self.programs:
            etype = (
                self._bmad_type(program.element)
                if program.element in self.elements else None
            )
            attribute = self.program_attribute(program, etype)
            if attribute is None:
                continue
            name = sanitize_string(program.element)
            before = None
            if program.first_turn > 1:
                design = self.tao.ele_gen_attribs(name, which="design", raises=False)
                before = (design or {}).get(attribute.upper())
                if before is None:
                    warn(
                        f"Line '{self.objectname}': Tao has no design {attribute} for "
                        f"'{program.element}', so the turns before {program.first_turn} "
                        "take the first knot's value."
                    )
            yield program, name, attribute, before

    def _phi0_sign(self, name: str) -> float:
        """+1 for an ``rfcavity``, whose phase goes as ``2 pi phi0 - w t``; -1 for an ``lcavity``'s ``2 pi phi0 + w t``."""
        key = self.tao.ele_head(sanitize_string(name))["key"]
        return -1.0 if key.lower() == "lcavity" else 1.0

    def cavity_phase(self, name: str) -> float | None:
        """A cavity's ``phi0`` from Tao, signed by :meth:`_phi0_sign` so that every cavity moves alike."""
        if self.tao is None:
            return None
        attribs = self.tao.ele_gen_attribs(sanitize_string(name), raises=False) or {}
        if "PHI0" not in attribs:
            return None
        return self._phi0_sign(name) * float(attribs["PHI0"])

    def set_cavity_phase(self, name: str, phase: float) -> None:
        """Set a cavity's ``phi0`` in Tao; the inverse of :meth:`cavity_phase`."""
        phi0 = self._phi0_sign(name) * phase
        self.tao.cmd(f"set element {sanitize_string(name)} phi0 = {float(phi0)!r}")

    def _bmad_type(self, name: str) -> str:
        """
        The Bmad element type LAURA writes `name` as, lowercased.

        Parameters
        ----------
        name: str
            Element name

        Returns
        -------
        str
            ``""`` if unknown
        """
        element = self.elements.get(name)
        if element is None:
            return ""
        translator = translate_elements([element]).get(name)
        if translator is None:
            return ""
        return translator._convert_type_bmad(translator.hardware_type).lower()

    def track_reference_particle(self) -> dict:
        """Track one particle turn by turn, as a one-point frequency-map grid."""
        settings = dict(self.da_settings or {})
        nudge = float(settings.get("x_max", 1e-3)) / 100.0
        original = self.file_block.get("tracking")
        tracking = dict(original or {})
        tracking["dynamic_aperture"] = {
            "nx": 1, "ny": 1, "x_max": nudge, "y_max": nudge,
        }
        self.file_block["tracking"] = tracking
        try:
            trajectory = self._track_grid_turn_by_turn()
        finally:
            if original is None:
                self.file_block.pop("tracking", None)
            else:
                self.file_block["tracking"] = original
        if not trajectory:
            return {}
        return {name: values[0] for name, values in trajectory.items()}

    def run_dynamic_aperture(self) -> list:
        """
        Track one particle per point of :meth:`da_rays` through ``long_term_tracking`` and record survival.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per start.
        """
        if self.tao is None:
            warn(
                f"Line '{self.objectname}': Tao must be run before a "
                "dynamic-aperture scan."
            )
            return []
        starts = self.da_rays()
        beam = Path(self.lattice_file).with_suffix(".da.beam")
        try:
            self._write_grid_beam_file(str(beam), starts)
            files = self._run_long_term_tracking(
                f"{self.objectname}.da.turns", beam.name, self.turns, 1, complete=False
            )
        except Exception as error:
            warn(f"long_term_tracking failed on the rays of {self.objectname}: {error}")
            return []
        survived = sum(
            self._read_ascii_beam(files[turn])["state"] == self._ALIVE
            for turn in range(1, self.turns + 1) if turn in files
        )
        self.dynamic_aperture = [
            (float(x), float(y), int(turns)) for (x, y), turns in zip(starts, survived)
        ]
        return self.dynamic_aperture

    def write(self) -> None:
        """Write the Bmad lattice and Tao init file to `master_subdir`."""
        if self.native_multipass:
            lattice = self._native_multipass_lattice()
        else:
            section = self.section
            section.reference_energy = self._reference_energy()
            lattice = section.to_bmad(
                particle=self.global_parameters["beam"].species,
                space_charge_n_bin=self.space_charge_n_bin,
                initial_twiss=self._bmad_initial_twiss(),
            )
        lattice += self._collective_methods(lattice)
        lattice = f"beginning[ref_time] = {float(self.reference_t0)!r}\n" + lattice
        path = Path(self.global_parameters["master_subdir"]) / f"{self.objectname}.bmad"
        path.write_text(lattice, encoding="utf-8")
        self.lattice_file = str(path)
        tao_path = path.with_suffix(".tao.init")
        tao_path.write_text(
            f'&tao_design_lattice\n  design_lattice(1)%file = "{path.name}"\n/\n'
            '&tao_params\n  global%lat_sigma_calc_uses_emit_from = "beam"\n/\n'
            f'&tao_beam_init\n  beam_init%position_file = "{self.objectname}.bmad.beam"\n'
            f'  beam_saved_at = "{self._saved_at(lattice)}"\n/\n',
            encoding="utf-8",
        )
        self.tao_init_file = str(tao_path)

    def _write_ltt_init(self, output: Path, beam: str, turns: int, every: int) -> Path:
        """
        A ``long_term_tracking`` namelist beside `output`, and its lattice: the Bmad lattice and :attr:`_ramper_lines`.

        Parameters
        ----------
        output: Path
            Beam file it writes, numbered by turn: openPMD if ``.h5``, else Bmad's ASCII
        beam: str
            Position file to start from
        turns: int
            Turns to track
        every: int
            Write the beam every this many turns

        Returns
        -------
        Path
            The namelist
        """
        lattice = Path(self.lattice_file)
        rampers = self._ramper_lines
        ltt_lattice = lattice.with_suffix(".ltt.bmad")
        ltt_lattice.write_text(f"call, file = {lattice.name}\n{rampers}", encoding="utf-8")
        init = output.with_suffix(".init")
        init.write_text(
            "&params\n"
            f"  ltt%lat_file = '{ltt_lattice.name}'\n"
            + ("  ltt%ramping_on = T\n" if rampers else "")
            + "  ltt%simulation_mode = 'BEAM'\n"
            "  ltt%tracking_method = 'BMAD'\n"
            f"  ltt%n_turns = {turns}\n"
            f"  ltt%beam_output_file = '{output.name}'\n"
            f"  ltt%beam_output_every_n_turns = {every}\n"
            "  ltt%only_live_particles_out = F\n"
            "  ltt%add_closed_orbit_to_init_position = F\n"
            "  ltt%random_seed = 1\n"
            "  ltt%print_info_messages = F\n"
            "  bmad_com%absolute_time_tracking = T\n"
            # lost beyond 1 m where there is no aperture, as Xsuite and ImpactX
            "  bmad_com%max_aperture_limit = 1.0\n"
            f"  beam_init%position_file = '{beam}'\n"
            "/\n",
            encoding="utf-8",
        )
        return init

    def _collective_methods(self, lattice: str) -> str:
        """
        Bmad attribute settings that switch on the collective effects asked for.

        ``bmad_com[csr_and_space_charge_on]`` alone computes nothing: each
        element also needs a ``csr_method`` / ``space_charge_method``. Set them
        on every element by wildcard: :attr:`csr_enable` gives Bmad's 1D CSR, and
        ``charge: space_charge_mode: 3d`` its 3D space charge. Else :attr:`lsc_enable`
        gives ``slice`` space charge.

        Parameters
        ----------
        lattice: str
            The lattice text, to check the global switch is on

        Returns
        -------
        str
            Lines to append to the lattice
        """
        lines = []
        if self.csr_enable:
            lines.append("*[csr_method] = 1_dim")
        if self.space_charge_mode == "3d":
            lines.append("*[space_charge_method] = fft_3d")
        elif self.lsc_enable:
            lines.append("*[space_charge_method] = slice")
        if lines and "bmad_com[csr_and_space_charge_on] = T" not in lattice:
            lines.append("bmad_com[csr_and_space_charge_on] = T")
        if lines and "space_charge_com[ds_track_step]" not in lattice:
            lines.append(f"space_charge_com[ds_track_step] = {self.collective_step}")
        return "".join(f"{line}\n" for line in lines)

    @property
    def native_multipass(self) -> bool:
        """
        Write a multipass path as Bmad multipass (``multipass: native`` on the
        line): shared hardware defined once, a lord with one slave per pass,
        rather than one copy per pass. The line has to run the whole path::

            files:
              ERL:
                code: bmad
                multipass: native
        """
        return str(self.file_block.get("multipass", "")).lower() == "native"

    def _native_multipass_lattice(self) -> str:
        """The whole beam path from LAURA's native Bmad multipass export."""
        from laura.translator.converters.layout import MachineLayoutTranslator

        layout = self.machine.lattices.get(self.machine.default_path)
        if layout is None or not layout.is_multipass:
            raise ValueError(f"Line '{self.objectname}' asks for native multipass, "
                             "but its beam path is not multipass")
        names = layout.elements
        if (self.start, self.end) != (names[0], names[-1]):
            raise ValueError(
                f"Line '{self.objectname}' asks for native multipass, which writes the "
                f"whole beam path: it has to run from {names[0]} to {names[-1]}, "
                f"not {self.start} to {self.end}"
            )
        return MachineLayoutTranslator.from_layout(layout, multipass=True).to_bmad_multipass(
            particle=self.global_parameters["beam"].species,
            space_charge_n_bin=self.space_charge_n_bin,
            initial_twiss=self._bmad_initial_twiss(),
            reference_energy=self._reference_energy(),
        )

    def _tao_name(self, name: str) -> str:
        """
        Tao's name for one of this line's elements: one pass of a multipass
        element (``NAME.N`` in the line) is the slave ``NAME\\N`` when written
        natively.
        """
        if self.native_multipass:
            base, _, number = name.rpartition(".")
            if base and number.isdigit() and f"{base}#{number}" in self.machine.lattices[
                self.machine.default_path
            ].elements:
                return f"{sanitize_string(base)}\\{number}"
        return sanitize_string(name)

    @property
    def _saved_elements(self) -> list[str]:
        return list(
            dict.fromkeys(
                self._tao_name(element.name)
                for element in self.screens_and_markers_and_bpms
            )
        )

    def _saved_at(self, lattice: str) -> str:
        """
        Build the Tao ``beam_saved_at`` list covering every output element.

        Parameters
        ----------
        lattice: str
            Bmad lattice text, to look up each element's class

        Returns
        -------
        str
            Comma-separated element classes, or ``*`` if over :attr:`_SAVED_AT_LIMIT`.
        """
        classes = {}
        for line in lattice.splitlines():
            name, _, remainder = line.partition(":")
            element_class = remainder.strip().partition(",")[0].strip().lower()
            if element_class:
                classes[name.strip()] = element_class
        saved_at = ", ".join(
            dict.fromkeys(
                [
                    f"{classes[name]}::*"
                    for name in self._saved_elements
                    if name in classes
                ]
                + ["END"]
            )
        )
        return saved_at if len(saved_at) <= self._SAVED_AT_LIMIT else "*"

    def run(self) -> None:
        """Track the beam through PyTao; turns before the last through :meth:`_run_long_term_tracking`."""
        if (
            self.lattice_file is None
            or self.input_beam_file is None
            or self.tao_init_file is None
        ):
            raise RuntimeError(
                "Bmad lattice and input beam must be written before tracking"
            )
        from pytao import Tao

        with contextlib.chdir(Path(self.lattice_file).parent):
            self.tao = Tao(
                init_file=Path(self.tao_init_file).name,
                lattice_file=Path(self.lattice_file).name,
                beam_init_position_file=Path(self.input_beam_file).name,
                so_lib=self.libtao,
                noplot=True,
            )
            for name in self._saved_elements:
                location = self._exit_location(name)
                if location != name:
                    self.tao.cmd(f"set beam add_saved_at = {location}")
            self._clock = self._clock_slips() if self.ramped else None
            self._ramper_lines = ""
            if self.turns > 1:
                self.begin_rf_phases()
                self._ramper_lines = self._rampers()
                turns = self.turns - 1
                self._run_long_term_tracking(
                    f"{self.objectname}.ltt.h5", Path(self.input_beam_file).name,
                    turns, 1 if self.write_turns else turns,
                )
                self._set_last_turn()
            self.tao.track_beam("BEGINNING", "END", use_progress_bar=False)
            if self.turns > 1:
                # the optics read after want it off
                self.tao.cmd("set bmad_com absolute_time_tracking = F")

    def _set_last_turn(self) -> None:
        """
        Set Tao up for turn N as :meth:`_rampers` would, on the beam ``long_term_tracking`` left after turn N-1.

        Tao, unlike ``long_term_tracking``, re-times the lattice for a new ``p0c``, so
        each cavity's phase is moved back by the shift in its ``ref_time_start``, and
        on by however far ahead :meth:`_clock_slips` left the beam's clock.
        """
        turn = self.turns
        index = self.pass_index(turn)
        self.apply_programs(turn)
        self.apply_rf_phases(index)
        if self.ramped:
            cavities = {name: sanitize_string(name) for name in self.live_cavities()}
            before = {
                name: float(self.tao.ele_gen_attribs(tao_name)["REF_TIME_START"])
                for name, tao_name in cavities.items()
            }
            self.tao.cmd(f"set element beginning p0c = {float(self.pass_p0c()[index])!r}")
            for name, tao_name in cavities.items():
                attribs = self.tao.ele_gen_attribs(tao_name)
                late = self._clock[0][index] - (float(attribs["REF_TIME_START"]) - before[name])
                self.set_cavity_phase(
                    name, self.cavity_phase(name) + float(attribs["RF_FREQUENCY"]) * late
                )
        self.tao.cmd(
            self._POSITION_FILE_CMD.format(path=self._ltt_beam_files()[turn - 1].name)
        )
        # carry on long_term_tracking's clock
        self.tao.cmd("set bmad_com absolute_time_tracking = T")

    def _rampers(self) -> str:
        """
        ``ramper`` elements on absolute time, so ``long_term_tracking`` moves the ramp, programs and RF phases turn by turn.

        It sets each element from its rampers at the bunch's mean time mid-element, so
        programs and RF phases are :meth:`pass_staircase` tables around the reference's
        arrival there, on its own clock (:meth:`_clock_ahead`). The ramp is one table for
        all elements, held from pass start to pass start.

        Returns
        -------
        str
            Bmad lattice lines; empty if nothing varies.
        """
        passes = self.turns * self.passes_per_turn
        s_in = self.getSValues(as_dict=True, at_entrance=True)
        s_out = self.getSValues(as_dict=True)

        def arrivals(element):
            s = 0.5 * (s_in[element] + s_out[element])
            return np.array([
                self.reference_time(s, j) + self._clock_ahead(element, j)
                for j in range(passes)
            ])

        tables = []
        if self.ramped:
            starts = self.pass_start_times()
            periods = np.diff(np.append(starts, 2 * starts[-1] - starts[-2]))
            # steps 1% of a pass early, so a bunch up to that early or late still
            # reads its own pass; a zero-length element right at the end reads the next pass's
            edges = starts - periods / 100 + self._clock[0][:passes]
            times = np.column_stack((edges, edges + periods * (1 - 1e-6))).ravel()
            tables.append(("*[p0c]", times, np.repeat(self.pass_p0c(), 2)))
        for program, name, attribute, before in self._programs():
            values = [program.value_at(turn, before) for turn in range(1, self.turns + 1)]
            tables.append(
                (f"{name}[{attribute}]", *self.pass_staircase(arrivals(program.element), values))
            )
        corrections = self._rf_corrections or {}
        cavities = self.live_cavities() if self.ramped else corrections
        for element in cavities:
            phase = self.cavity_phase(element)
            if phase is None:
                continue
            phase = phase + self.rf_phase_shifts(corrections.get(element, np.zeros(passes)))
            if self.ramped:
                ahead = [self._clock_ahead(element, j) for j in range(passes)]
                phase = phase + float(self.elements[element].cavity.frequency) * np.array(ahead)
            phi0 = self._phi0_sign(element) * phase
            tables.append(
                (f"{sanitize_string(element)}[phi0]", *self.pass_staircase(arrivals(element), phi0))
            )
        return "".join(
            self._ramper(f"simba_ramper{index}", *table)
            for index, table in enumerate(tables, start=1)
        )

    def _clock_slips(self) -> tuple:
        """
        How far ``long_term_tracking``'s clock runs ahead of the reference under a ramp.

        It ramps ``p0c`` but not ``delta_ref_time``, by which Bmad times the elements in
        :attr:`_STALE_CLOCK_KEYS`, so on pass ``j`` each puts the clock
        ``delta_ref_time (1 - beta_0 / beta_j)`` ahead, turn on turn.
        Read from Tao before it is re-timed.

        Returns
        -------
        tuple
            ``(ahead, slip, within)``: seconds ahead at the start of each pass, and after
            the last; ``1 - beta_0 / beta_j`` per pass; and by Tao name, the stale
            ``delta_ref_time`` up to the element's middle.
        """
        names = self.tao.lat_list("*", "ele.name")
        keys = self.tao.lat_list("*", "ele.key")
        ref_time = np.asarray(self.tao.lat_list("*", "ele.ref_time"), dtype=float)
        stale = np.diff(ref_time, prepend=ref_time[0]) * np.array(
            [key.lower() in self._STALE_CLOCK_KEYS for key in keys]
        )
        within = {}
        for name, middle in zip(names, np.cumsum(stale) - stale / 2):
            within.setdefault(name, float(middle))
        beta = self.pass_beta0()
        slip = 1 - beta[0] / beta
        ahead = np.concatenate(([0.0], np.cumsum(np.sum(stale) * slip)))
        return ahead, slip, within

    def _clock_ahead(self, element: str, index: int) -> float:
        """Seconds ``long_term_tracking``'s clock is ahead mid-`element` on pass `index`; see :meth:`_clock_slips`."""
        if self._clock is None:
            return 0.0
        ahead, slip, within = self._clock
        return float(ahead[index] + slip[index] * within.get(sanitize_string(element), 0.0))

    @staticmethod
    def _ramper(name: str, target: str, times, values) -> str:
        """
        One linear ``ramper`` of `target` against ``time``.

        The parser evaluates every ramper at time 0, and errs off the end of a table, so
        each is held a second beyond its ends and time 0.
        """
        times = [min(times[0], 0.0) - 1.0, *times, max(times[-1], 0.0) + 1.0]
        values = [values[0], *values, values[-1]]

        def table(numbers):
            return ",\n    ".join(
                ", ".join(repr(float(x)) for x in numbers[i:i + 8])
                for i in range(0, len(numbers), 8)
            )

        return (
            f"{name}: ramper = {{{target}: {{{table(values)}}}}}, var = {{time}},\n"
            f"  x_knot = {{{table(times)}}}, interpolation = linear\n"
        )

    def _run_long_term_tracking(
        self, output: str, beam: str, turns: int, every: int, complete: bool = True
    ) -> dict:
        """
        Track `turns` turns of `beam` with Bmad's ``long_term_tracking``, from the namelist :meth:`_write_ltt_init` writes.
        The beam comes out at the line's start every `every` turns; see :meth:`_ltt_beam_files`.

        Parameters
        ----------
        output: str
            Beam file name, beside the lattice; see :meth:`_write_ltt_init`
        beam: str
            Position file to start from, beside the lattice
        turns: int
        every: int
        complete: bool
            Raise if it stopped early, as it does once every particle is lost

        Returns
        -------
        dict[int, Path]
            The beam files, by turn

        Raises
        ------
        RuntimeError
            If it exits non-zero, or if `complete` and it writes no beam for the last turn.
        """
        workdir = Path(self.lattice_file).parent
        for stale in self._ltt_beam_files(output).values():
            stale.unlink()
        init = self._write_ltt_init(workdir / output, beam, turns, every)
        command = self.executables["long_term_tracking"] + [init.name]
        logfile = init.with_suffix(".log")
        self.run_command(
            self.executables.build_command(command, str(workdir.resolve())),
            str(logfile),
            cwd=workdir,
        )
        files = self._ltt_beam_files(output)
        if complete and turns not in files:
            tail = "".join(logfile.read_text().splitlines(True)[-20:]).strip()
            raise RuntimeError(
                f"long_term_tracking wrote no beam for turn {turns} of "
                f"{self.objectname}.\nLast lines of {logfile}:\n{tail}"
            )
        return files

    def _ltt_beam_files(self, output: str | None = None) -> dict:
        """
        ``long_term_tracking``'s beam files, by turn: ``<name>.ltt-NN.h5``, zero-padded to its turn count.

        Parameters
        ----------
        output: str | None
            The ``ltt%beam_output_file`` they were numbered from; the tracking run's if None

        Returns
        -------
        dict[int, Path]
        """
        output = Path(output or f"{self.objectname}.ltt.h5")
        workdir = Path(self.lattice_file).parent
        stem, suffix = f"{output.stem}-", output.suffix
        return {
            int(path.name[len(stem):-len(suffix)]): path
            for path in workdir.glob(f"{stem}*{suffix}")
            if path.name[len(stem):-len(suffix)].isdigit()
        }

    def _exit_location(self, element: str) -> str:
        """
        Where Tao holds the beam leaving ``element``.

        The element itself, unless it was superimposed (on a quadrupole, say),
        when it is a super-lord that is not tracked through: then its last
        super-slave, as a ``branch>>index`` location.

        Parameters
        ----------
        element: str
            Element name in the Bmad lattice

        Returns
        -------
        str
            ``element``, or the location of its last slave
        """
        try:
            rows = self.tao.ele_lord_slave(element)
        except Exception:
            return element
        # super-slaves only: a multipass slave also lists its lord's other passes
        slaves = [
            row["location_name"]
            for row in rows
            if row.get("type") == "Slave" and row.get("status", "Super_Slave") == "Super_Slave"
        ]
        if not slaves:
            return element
        return max(slaves, key=lambda location: int(location.split(">>")[-1]))

    def _particles_at(self, element: str, zstart: float = 0, late: float = 0.0) -> tuple:
        """
        Live particles at an element; Bmad keeps lost ones, frozen where they died.

        Parameters
        ----------
        element: str
            Element name
        zstart: float
            Position of the element along the machine
        late: float
            Seconds to take off each particle's ``t``; see :meth:`_clock_slips`

        Returns
        -------
        tuple[ParticleGroup, int | None, numpy.ndarray | None]
            The particles, the reference particle's index (None if it was lost), and
            for a beam of several species, each live particle's species (else None).
        """
        data = self.tao.bunch_data(self._exit_location(element))
        species = getattr(self, "_input_species", None)
        if species is not None:
            data["species"] = self.global_parameters["beam"].species
        return self._live_particles(data, zstart, species, late)

    def _live_particles(self, data: dict, zstart: float, species=None, late: float = 0.0) -> tuple:
        """
        As :meth:`_particles_at`, from a bunch's ``data`` (Tao's ``bunch_data``, or a ``ParticleGroup``'s).

        Parameters
        ----------
        data: dict
            ParticleGroup data, lost particles included
        zstart: float
            Position of the element along the machine
        species: array | None
            Each particle's species, for a beam of several
        late: float
            Seconds to take off each particle's ``t``
        """
        if late:
            data = {**data, "t": np.asarray(data["t"]) - late}
        alive = np.asarray(data["status"]) == self._ALIVE
        if species is not None:
            species = np.asarray(species)[alive]
        ref_idx = (
            int(np.count_nonzero(alive[: self.ref_idx]))
            if self.ref_idx is not None
            and 0 <= self.ref_idx < len(alive)
            and alive[self.ref_idx]
            else None
        )
        particles = rbf.openpmd.ParticleGroup(
            data={
                key: value[alive] if np.shape(value) == alive.shape else value
                for key, value in data.items()
            }
        )
        reference_time = (
            particles.t[ref_idx] if ref_idx is not None else np.mean(particles.t)
        )
        particles.z = zstart + (
            -particles.beta_z
            * constants.speed_of_light
            * (particles.t - reference_time)
        )
        return particles, ref_idx, species

    def _twiss_data(self) -> dict:
        """
        Twiss and bunch parameters along the lattice from Tao.

        Returns
        -------
        dict
            For :func:`~simba.Modules.Twiss.bmad.save_bmad_twiss_hdf`.
        """
        indices = self.tao.lat_list("*", "ele.ix_ele", flags="-array_out -track_only")
        twiss = {
            name: self.tao.lat_list("*", command, flags="-array_out -track_only")
            for name, command in LATTICE_TWISS.items()
        }
        twiss["element_name"] = np.asarray(
            self.tao.lat_list("*", "ele.name", flags="-track_only")
        )
        bunch_params = [self.tao.bunch_params(int(index)) for index in indices]
        twiss.update(
            {
                name: np.array([params[key] for params in bunch_params])
                for name, key in BEAM_TWISS.items()
            }
        )
        twiss["s"] = twiss["s"] + self.entrance_s
        s_values = np.array(self.getSValues(at_entrance=False)) + self.entrance_s
        z_values = [z[-1] for z in self.getZValues()]
        twiss["z"] = np.interp(twiss["s"], s_values, z_values)
        return twiss

    def _tao_tunes(self) -> dict:
        """
        Tunes from the phase advance at the end of the lattice; unlike a one-turn map, keeps the integer part.
        """
        tunes = {}
        for name, attribute in (("x", "ele.a.phi"), ("y", "ele.b.phi")):
            out = self.tao.cmd(f"pipe lat_list 1@0>>END|model {attribute}")
            if out and str(out[0]).strip():
                tunes[name] = float(str(out[0]).strip()) / (2 * np.pi)
        return tunes

    def read_closed_orbit(self):
        """Tao's ``orbit.vec.N`` at the start of the lattice."""
        if self.tao is None:
            return None
        try:
            with contextlib.chdir(Path(self.lattice_file).parent):
                values = [
                    float(
                        str(
                            self.tao.cmd(
                                f"pipe lat_list 1@0>>BEGINNING|model orbit.vec.{i}"
                            )[0]
                        ).strip()
                    )
                    for i in range(1, 7)
                ]
        except Exception as error:
            warn(f"Tao closed orbit unavailable for {self.objectname}: {error}")
            return None
        return np.array(values)

    def read_optics_summary(self) -> dict:
        """
        Tune and chromaticity from Tao.

        Chromaticity is the tune difference at ``pz`` = +/- :attr:`chromaticity_delta`, with RF off.
        """
        if self.tao is None:
            return {}
        summary = {}
        try:
            with contextlib.chdir(Path(self.lattice_file).parent):
                for plane, tune in self._tao_tunes().items():
                    summary[f"tune_{plane}_total"] = tune
                delta = self.chromaticity_delta
                rf_on = next(
                    (
                        line.split(";")[2]
                        for line in self.tao.cmd("pipe global")
                        if line.startswith("rf_on;")
                    ),
                    "T",
                )
                try:
                    self.tao.cmd("set global rf_on = F")
                    self.tao.cmd(f"set particle_start pz = {delta}")
                    plus = self._tao_tunes()
                    self.tao.cmd(f"set particle_start pz = {-delta}")
                    minus = self._tao_tunes()
                finally:
                    self.tao.cmd("set particle_start pz = 0")
                    self.tao.cmd(f"set global rf_on = {rf_on}")
                for plane in ("x", "y"):
                    if plane in plus and plane in minus:
                        summary[f"chromaticity_{plane}"] = (
                            plus[plane] - minus[plane]
                        ) / (2 * delta)
        except Exception as error:
            warn(f"Tao optics summary unavailable for {self.objectname}: {error}")
        return summary

    def read_one_turn_map(self):
        """Tao's ``matrix`` from ``BEGINNING`` to itself: the one-turn map on a closed lattice.

        Returns
        -------
        numpy.ndarray | None
            The 6x6 ``mat6``, or None if Tao gave nothing back.
        """
        if self.tao is None:
            return None
        with contextlib.chdir(Path(self.lattice_file).parent):
            result = self.tao.matrix("BEGINNING", "BEGINNING")
        mat6 = (result or {}).get("mat6")
        return None if mat6 is None else np.asarray(mat6, dtype=float)

    def postProcess(self) -> None:
        """Write the beams at each output element and the twiss HDF5 to `master_subdir`."""
        super().postProcess()
        if self.tao is None:
            raise RuntimeError("Bmad tracking must finish before post-processing")
        source_beam = self.global_parameters["beam"]
        s_values = {
            name: value + self.entrance_s
            for name, value in self.getSValues(as_dict=True).items()
        }
        outputs = {
            element.name: (self._tao_name(element.name), element.physical.start.z)
            for element in self.screens_and_markers_and_bpms
        }
        end = self.end_name
        outputs[end] = ("END", self.endObject.physical.end.z)
        if self.bundles_turns:
            # the line's end on the turns before the last, where long_term_tracking left them
            files = self._ltt_beam_files()
            for turn in range(1, self.turns):
                beam = deepcopy(source_beam)
                particles, ref_idx, _ = self._live_particles(
                    rbf.openpmd.ParticleGroup(str(files[turn])).data, outputs[end][1],
                    late=self._clock[0][turn] if self._clock is not None else 0.0,
                )
                rbf.openpmd.read_particle_group(
                    beam, particles, s=s_values[end], reference_particle_index=ref_idx
                )
                beam.turn = turn
                self.write_beam_file(beam, end, turn)
        _, last_turn = self.output_turns()[-1]
        late = self._clock[0][self.turns - 1] if self._clock is not None else 0.0
        final_beam = None
        for output_name, (tao_element, zstart) in outputs.items():
            if not self.writes_output(output_name):
                continue
            beam = deepcopy(source_beam)
            particles, ref_idx, species = self._particles_at(tao_element, zstart, late)
            rbf.openpmd.read_particle_group(
                beam,
                particles,
                s=s_values[output_name],
                reference_particle_index=ref_idx,
            )
            if species is not None:
                # the writer splits the species back out
                beam.set_particle_species(species, primary=source_beam.species)
            beam.turn = self.beam_turn(last_turn)
            self.write_beam_file(beam, output_name, last_turn)
            if output_name == end:
                final_beam = beam
        save_bmad_twiss_hdf(
            filename=str(
                Path(self.global_parameters["master_subdir"])
                / f"{self.objectname}_twiss.bmad.hdf5"
            ),
            twiss=self._twiss_data(),
        )
        self.global_parameters["beam"] = final_beam
        if self.turns > 1 and self.global_parameters["delete_tracking_files"]:
            for path in self._ltt_beam_files().values():
                path.unlink()
