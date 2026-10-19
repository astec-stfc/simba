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

    radiates_by_default: ClassVar[bool] = True
    """Bmad lattices from LAURA radiate by default."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """Tao's own scan; see :meth:`_tao_dynamic_aperture_namelist`."""

    supports_frequency_map: ClassVar[bool] = True
    """By tracking the grid a turn at a time; see :meth:`_track_grid_turn_by_turn`."""

    supports_radiation: ClassVar[bool] = True
    """``bmad_com[radiation_damping_on]``/``[radiation_fluctuations_on]``, written by LAURA."""

    supports_periodic: ClassVar[bool] = True
    """``parameter[geometry] = closed``, written by LAURA; Twiss from the one-turn map."""

    supports_programs: ClassVar[bool] = True
    """By ``set element`` inside the Tao turn loops; see :meth:`apply_programs`."""

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
        self.input_beam_file = str(
            Path(self.global_parameters["master_subdir"])
            / f"{self.objectname}.bmad.beam"
        )
        self._write_bmad_beam_file()

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
        """A ring's :attr:`design_p0c`, else the reference particle's ``cp``
        in eV, else :attr:`reference_p0c`."""
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

    def _reference_energy(self) -> float:
        """
        Reference energy: :attr:`reference_energy` if :attr:`design_p0c` is set, else the reference particle's.

        Returns
        -------
        float
        """
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

    def _tao_dynamic_aperture_namelist(self) -> str:
        """
        The ``&tao_dynamic_aperture`` block for the Tao init file, written on every run.

        Tao searches ``n_angle`` rays, so the result is a boundary, not a survival grid.
        """
        xs, ys = self.da_grid()
        return (
            "\n&tao_dynamic_aperture\n"
            "  ix_universe = 1\n"
            "  pz = 0.0\n"
            "  da_param%min_angle = 0.0\n"
            f"  da_param%max_angle = {np.pi}\n"
            f"  da_param%n_angle = {max(3, int(self.da_settings.get('n_angle', 9)))}\n"
            f"  da_param%n_turn = {self.turns}\n"
            f"  da_param%x_init = {float(xs[-1])}\n"
            f"  da_param%y_init = {float(ys[-1])}\n"
            "/\n"
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

    def _write_grid_beam_file(self, path: str) -> list:
        """
        Write one particle per frequency-map grid point, offset from the closed orbit.

        Parameters
        ----------
        path: str
            File to write

        Returns
        -------
        list
            The ``(x, y)`` the particles were placed at, in file order.
        """
        orbit = self.read_closed_orbit()
        if orbit is None:
            orbit = np.zeros(6)
        xs, ys = self.da_grid()
        points = [(float(x), float(y)) for y in ys for x in xs]
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
        Track the grid one turn at a time, feeding the bunch at ``END`` back in as the position file.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py``/``state``, each ``n_particles x turns``. Empty
            if Tao could not be driven.
        """
        if self.tao is None:
            warn(
                f"Line '{self.objectname}': Tao must be run before the grid "
                "can be tracked."
            )
            return {}
        history = {name: [] for name in ("x", "px", "y", "py", "state")}
        try:
            with contextlib.chdir(Path(self.lattice_file).parent):
                grid_file = str(Path(self.lattice_file).with_suffix(".fma.beam"))
                self._write_grid_beam_file(grid_file)
                self.tao.cmd(self._POSITION_FILE_CMD.format(path=grid_file))
                self.tao.cmd("set beam add_saved_at = END")
                self.tao.cmd("set global track_type = beam", raises=False)
                for turn in range(1, self.turns + 1):
                    self.apply_programs(turn)
                    self.tao.track_beam("BEGINNING", "END", use_progress_bar=False)
                    bunch = {
                        name: np.asarray(
                            self.tao.bunch1(
                                "END", coordinate=name, which="model", ix_bunch=1
                            )
                        )
                        for name in self._PHASE_SPACE + ("state",)
                    }
                    for name in history:
                        history[name].append(bunch[name])
                    self._write_position_file(
                        grid_file,
                        np.column_stack([bunch[name] for name in self._PHASE_SPACE]),
                        states=bunch["state"],
                    )
                    self.tao.cmd(self._POSITION_FILE_CMD.format(path=grid_file))
        except Exception as error:
            warn(f"Tao turn-by-turn tracking failed for {self.objectname}: {error}")
            return {}
        return {name: np.asarray(values).T for name, values in history.items()}

    def apply_programs(self, turn: int) -> None:
        """
        ``set element`` each programmed element's attribute for `turn`, inside the Tao turn loop.

        Parameters
        ----------
        turn: int
            Turn number, 1-based
        """
        if self.tao is None:
            return
        for program in self.programs:
            etype = (
                self._bmad_type(program.element)
                if program.element in self.elements else None
            )
            attribute = self.program_attribute(program, etype)
            if attribute is None:
                continue
            self.tao.cmd(
                f"set element {sanitize_string(program.element)} {attribute} = "
                f"{program.value_at(turn)}",
                raises=False,
            )

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
        Tao's dynamic-aperture scan; needs lattice apertures.

        Returns
        -------
        list
            ``(x, y, turns)`` along the aperture boundary; the turn count is
            :attr:`turns` for every point.
        """
        if self.tao is None:
            warn(
                f"Line '{self.objectname}': Tao must be run before a "
                "dynamic-aperture scan can be read."
            )
            return []
        try:
            with contextlib.chdir(Path(self.lattice_file).parent):
                self.tao.cmd("set universe 1 dynamic_aperture_calc on")
                rows = self.tao.cmd("pipe da_aperture")
        except Exception as error:
            warn(f"Tao dynamic aperture failed for {self.objectname}: {error}")
            return []
        aperture = []
        for row in rows:
            fields = str(row).split(";")
            if len(fields) < 4:
                continue
            try:
                aperture.append(
                    (float(fields[2]), float(fields[3]), self.turns)
                )
            except ValueError:
                continue
        if not aperture:
            warn(
                f"Line '{self.objectname}': Tao returned no aperture points. "
                "Does the lattice have apertures?"
            )
        self.dynamic_aperture = aperture
        return aperture

    def write(self) -> None:
        """Write the Bmad lattice and Tao init file to `master_subdir`."""
        section = self.section
        section.reference_energy = self._reference_energy()
        lattice = section.to_bmad(
            particle=self.global_parameters["beam"].species,
            space_charge_n_bin=self.space_charge_n_bin,
            initial_twiss=self._bmad_initial_twiss(),
        )
        path = Path(self.global_parameters["master_subdir"]) / f"{self.objectname}.bmad"
        path.write_text(lattice, encoding="utf-8")
        self.lattice_file = str(path)
        tao_path = path.with_suffix(".tao.init")
        tao_path.write_text(
            f'&tao_beam_init\n  beam_saved_at = "{self._saved_at(lattice)}"\n/\n'
            + self._tao_dynamic_aperture_namelist(),
            encoding="utf-8",
        )
        self.tao_init_file = str(tao_path)

    @property
    def _saved_elements(self) -> list[str]:
        return list(
            dict.fromkeys(
                sanitize_string(element.name)
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
        """Track the beam through PyTao."""
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
            self.tao.track_beam("BEGINNING", "END", use_progress_bar=False)

    def _particles_at(self, element: str, zstart: float = 0) -> tuple:
        """
        Live particles at an element; Bmad keeps lost ones, frozen where they died.

        Parameters
        ----------
        element: str
            Element name
        zstart: float
            Position of the element along the machine

        Returns
        -------
        tuple[ParticleGroup, int | None]
            The particles, and the reference particle's index (None if it was lost).
        """
        data = self.tao.bunch_data(element)
        alive = np.asarray(data["status"]) == self._ALIVE
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
        return particles, ref_idx

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
            element.name: (sanitize_string(element.name), element.physical.start.z)
            for element in self.screens_and_markers_and_bpms
        }
        outputs[self.end] = ("END", self.endObject.physical.end.z)
        final_beam = None
        for output_name, (tao_element, zstart) in outputs.items():
            if not self.writes_output(output_name):
                continue
            beam = deepcopy(source_beam)
            particles, ref_idx = self._particles_at(tao_element, zstart=zstart)
            rbf.openpmd.read_particle_group(
                beam,
                particles,
                s=s_values[output_name],
                reference_particle_index=ref_idx,
            )
            # one pass, whatever `turns` says
            beam.turn = 1
            rbf.openpmd.write_openpmd_beam_file(
                beam,
                str(
                    Path(self.global_parameters["master_subdir"])
                    / f"{self.output_basename(output_name)}.openpmd.hdf5"
                ),
            )
            if output_name == self.end:
                final_beam = beam
        save_bmad_twiss_hdf(
            filename=str(
                Path(self.global_parameters["master_subdir"])
                / f"{self.objectname}_twiss.bmad.hdf5"
            ),
            twiss=self._twiss_data(),
        )
        self.global_parameters["beam"] = final_beam
