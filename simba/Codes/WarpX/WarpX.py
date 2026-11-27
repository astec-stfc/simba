"""
SIMBA WarpX module: one WarpX (RZ) run per line, of its plasma stage or plasma lens,
in the lab frame with a moving window. See the `WarpX docs`_.

The stage's physics comes from LAURA's
:meth:`~laura.translator.converters.plasma.PlasmaTranslator.to_warpx`; the incoming
beam, run control and anything in :attr:`~warpxLattice.inputs` are added here.

    .. _WarpX docs: https://warpx.readthedocs.io
"""

import glob
import os
import shutil
from types import SimpleNamespace
from typing import Dict, List
from warnings import warn

import h5py
import numpy as np
from scipy.constants import c

from laura.translator.utils.hipace import hipace_inputs

from ... import exceptions
from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules.Beams.fbpic import _openpmd_record, particles_to_beam
from ...Modules.Beams.hipace import write_hipace_beam_file
from ...Modules.units import UnitValue


class warpxLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` run as one WarpX simulation."""

    code: str = "warpx"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    allow_negative_drifts: bool = True
    """Allow drifts to be of negative length (could be necessary for plasma injection)"""

    particle_definition: str | None = None
    """Name of the input particle distribution. ``input: {particle_definition: null}``
    in the settings file means there is no incoming beam (a laser-driven stage)."""

    bunch_z_position: float = 0
    """Position in the box of a plasma stage, in metres, at which
    :attr:`~simba.Framework_objects.frameworkLattice.plasma_box_origin` -- the
    incoming reference particle, or else the centroid -- is placed. Other lines
    centre their box on the beam."""

    include_ions: bool = False
    """Model the background ions as mobile protons"""

    antenna_offset: float | None = None
    """How far ahead of the laser centre the antenna sits, in metres; ``3*c*tau``
    if None, as :attr:`~simba.Codes.FBPIC.FBPIC.fbpicLattice.antenna_offset`"""

    injected_uz_min: float = 2.0
    """Electrons freed from the ionizable species join the output beam only if
    their ``uz`` (gamma*beta) exceeds this at the end of the run."""

    single_precision: bool | None = None
    """Whether WarpX was built with single-precision particles, which the beam
    file must match; None reads it from the executable's name (``.PSP.``)."""

    stage_inputs: Dict = {
        "boundary.field_lo": ["none", "pec"],
        "boundary.field_hi": ["pec", "pec"],
        "algo.particle_shape": 3,
        "warpx.cfl": 1.0,
        "warpx.use_filter": 1,
        "warpx.filter_npass_each_dir": [0, 1],  # RZ FDTD cannot filter radially
    }
    """Numerics of a line with a plasma stage or lens (RZ, electromagnetic). 
    FBPIC's solver is ``{"algo.maxwell_solver": "psatd", "boundary.field_lo": "none damped",
    "boundary.field_hi": "none damped"}``. ``warpx.n_rz_azimuthal_modes`` is 2 with a
    laser (its polarization is m = 1) and 1 without, where WarpX can start the beam
    from its own field (``initialize_self_fields`` is single-mode in RZ)."""

    gamma_boost: float | None = None
    """Run a plasma stage in a frame boosted by this Lorentz factor along z (None or 1: the
    lab frame). The plasma, laser and box stay in the lab frame, which WarpX transforms;
    the solver is PSATD with a Galilean shift (:attr:`~boost_inputs`), and the output
    :attr:`~n_out` + 1 back-transformed lab-frame snapshots. The background ions are
    always included (:attr:`~include_ions`): an electron-only plasma streaming through the
    boosted box carries an unneutralised charge and current, whose fields swamped the wake.
    Steps fall roughly as ``(1 + beta) gamma^2`` for a laser stage; the wake's phase
    velocity bounds the boost.

    Validated on PLASMA's LWFA (1.2 cm, lambda/20) against Wake-T: at gamma 3 the witness
    matches (energy -0.5 %, eps_n +13 %); at gamma 6 the wake does but the witness does not
    (eps_n x4.5, bunch lengthened). A beam-driven stage (PWFA) is not validated."""

    boost_inputs: Dict = {
        "algo.maxwell_solver": "psatd",
        "psatd.use_default_v_galilean": 1,
        "psatd.current_correction": 1,
        "warpx.grid_type": "collocated",
        "boundary.field_lo": ["none", "damped"],
        "boundary.field_hi": ["none", "damped"],
        "warpx.cfl": 0.99,
        "warpx.use_filter": 1,
        "warpx.filter_npass_each_dir": [1, 4],
        "warpx.use_filter_compensation": 1,
    }
    """Numerics of a boosted plasma stage, from WarpX's RZ boosted laser-wakefield example
    (``inputs_test_rz_laser_acceleration_boosted_opmd``); over :attr:`~stage_inputs`."""

    inputs: Dict = {}
    """WarpX inputs applied last, over everything else, e.g. ``{"warpx.verbose": 1}``;
    ``WARPXsettings: {inputs: {...}}`` adds to them"""

    n_cell: List[int] = [32, 32, 64]
    """Grid of a line without a plasma stage (3D), in x, y, z"""

    box_sigmas: List[float] = [10.0, 10.0, 10.0]
    """Half-size of the box of a line without a plasma stage, in incoming-beam rms
    sizes, about the beam centre. Particles leaving the box are lost, so a beam
    whose size grows along the line needs a larger one."""

    cells_per_step: float = 1.0
    """Cells the beam moves per time step in a line without a plasma stage. WarpX's
    moving window may shift by no more than its guard cells (a few) per step, so the
    step follows the cell size."""

    opposing_p0c: float | None = None
    """Momentum of the opposing bunch at a BeamBeam [eV/c]; the incoming beam's if None"""

    opposing_rest_energy: float | None = None
    """Rest energy of the opposing particles at a BeamBeam [eV]; the incoming beam's if None"""

    opposing_macroparticles: int = 10000
    """Macroparticles in the opposing bunch at a BeamBeam"""

    collision_sigmas: float = 6.0
    """Transverse half-size of a beam-beam box, in the larger of the two bunches'
    rms sizes. QED (beamstrahlung, pairs) is not switched on; add it through :attr:`~inputs`."""

    _drift_after: float | None = None
    _fel_box: dict | None = None
    _apertures: list = []

    fel: bool = True
    """Run a line with a wiggler as a boosted-frame FEL (radiation included);
    otherwise the undulator is only a field in a transport line"""

    cells_per_wavelength: int = 16
    """Cells per FEL resonant wavelength along z"""

    fel_fields: List[str] = ["Ex", "Ey", "Bx", "By"]
    """Fields in the FEL's back-transformed snapshots (the radiation)"""

    fel_space_charge: bool = True
    """Keep the FEL beam's space charge; if False, inject it twice at half weight, as
    electrons and as positrons, so that its self-fields cancel while the radiation adds.
    The beam written out is the electrons at full weight."""

    auto_phase: bool = True
    """Phase the line's cavities by tracking the reference particle along the axis
    (:func:`~laura.translator.utils.fields.onaxis.autophase`);
    otherwise each cavity's stored ``crest`` is taken as is."""

    n_out: int | None = None
    """Number of evenly spaced particle dumps along the line, in ``<name>_diags/``;
    the first and last steps are always written. None takes the plasma stage's
    ``simulation.n_out``, else 1 (the first and last only)."""

    _zstart: float = 0.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.inputs = dict(self.inputs)
        for k, v in (self.settings["global"].get("WARPXsettings") or {}).items():
            if isinstance(v, Dict):
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)
        stated = self.file_block.get("input") or {}
        if "particle_definition" in stated and stated["particle_definition"] is None:
            self.particle_definition = None
        else:
            self.particle_definition = self.input_particle_definition

    @property
    def subdir(self) -> str:
        return self.global_parameters["master_subdir"]

    @property
    def zstart(self) -> float:
        """Lattice z of the box's z = 0 at t = 0; WarpX z is lab z less this."""
        return self._zstart

    def preProcess(self) -> None:
        """Read the input beam and fix :attr:`zstart`."""
        super().preProcess()
        if self.particle_definition is not None:
            prefix = self.get_prefix()
            prefix = prefix if self.trackBeam else prefix + self.particle_definition
            self.load_input_beam(prefix, self.particle_definition)
            self._zstart = self.plasma_box_origin - self.bunch_z_position
        else:
            self._zstart = float(self.startObject.physical.start.z)

    @property
    def executable(self) -> str:
        """``warpx`` (the RZ build) for a line with a plasma stage or lens, else ``warpx_3d``;
        set by :func:`~write`."""
        return self._executable

    _executable: str = "warpx"

    def _overlaid(self) -> bool:
        """Whether the beam goes in as electrons and positrons at half weight (see :attr:`~fel_space_charge`)."""
        return self._fel_box is not None and not self.fel_space_charge

    def _particle_dtype(self) -> type:
        single = self.single_precision
        if single is None:
            exe = self.executables[self.executable][-1]
            single = ".PSP." in os.path.basename(os.path.realpath(exe))
        return np.float32 if single else np.float64

    def write(self) -> None:
        """
        Write ``<name>.in`` and, with an incoming beam, ``<name>_beam.h5`` to the
        master subdirectory. With a beam the run lasts until the reference
        particle reaches the end of the line.

        A line with a plasma stage or lens runs in RZ in its box, with
        :attr:`~stage_inputs`. Any other line runs in 3D, in a box of
        :attr:`~box_sigmas` about the beam moving at its speed, with WarpX's
        relativistic electrostatic solver when ``space_charge_mode`` is ``3d``
        and no field solve otherwise.

        A line with a :class:`~laura.models.element.BeamBeam` is a strong-strong
        collision instead; see :func:`~_collision`.

        Raises
        ------
        ValueError
            For a line without a beam or plasma stage, or with the beam past its end
        """
        for diags in ("diags", "aperture_diags"):  # postProcess reads the last dump: none may be left from a run before
            shutil.rmtree(os.path.join(self.subdir, f"{self.objectname}_{diags}"), ignore_errors=True)
        collision = [e for e in self.elementObjects.values() if e.hardware_type == "BeamBeam"]
        if collision:
            return self._finish(*self._collision(collision), staged=False)
        p0c = self.reference_p0c if self.particle_definition is not None else 0.0
        brho = self.rigidity(p0c) if p0c else 0.0
        staged = any(e.hardware_type.lower() in ("plasma", "plasmalens") for e in self.elementObjects.values())
        if not staged:  # the box follows the beam, so the lattice can start at the line's start
            self._zstart = float(self.startObject.physical.start.z)
        beam = self.particle_definition is not None
        origin = self.plasma_box_origin if beam else 0.0  # taken before any drift of the beam
        wigglers = [e for e in self.elementObjects.values() if e.hardware_type == "Wiggler"]
        fel = beam and not staged and self.fel and wigglers
        if fel:  # WarpX z = 0 at the box's front, where back-transformed snapshot times start
            self._fel_box = self._fel_frame(p0c, wigglers)
            origin -= self._fel_box["shift"]
            self._zstart = self._fel_box["top"]
        def export() -> dict:  # with a beam, stop_time is when the reference particle reaches the end of the line
            return self.section.to_warpx(
                include_ions=self.include_ions or (self.gamma_boost or 1) > 1,  # see gamma_boost
                zstart=self._zstart, antenna_offset=self.antenna_offset,
                Brho=abs(brho), charge_sign=-1 if brho < 0 else 1, directory=self.subdir,
                z_reference=origin - self._zstart if beam else None,
                rest_energy=self.rest_energy, auto_phase=self.auto_phase,
            )

        params = export()
        staged = "geometry.dims" in params  # a stage wholly behind the beam leaves none
        boosted = staged and (self.gamma_boost or 1) > 1
        front = float(params["geometry.prob_hi"][1]) if boosted else 0.0
        if front:
            self._zstart += front
            params = export()
            if not beam:  # the run lasts until the box's old z = 0, now `front` behind the new one, leaves the stage
                params["stop_time"] += front / c  # (with a beam, the reference particle moves with the box)
            for key in ("geometry.prob_lo", "geometry.prob_hi"):
                params[key] = [params[key][0], float(params[key][1]) - front]
            names = params.get("lasers.names", [])
            for name in [names] if isinstance(names, str) else names:
                x, y, z = params[f"{name}.position"]
                params[f"{name}.position"] = [x, y, float(z) - front]
        self._apertures = params.pop("_apertures", [])
        self._executable = "warpx" if staged else "warpx_3d"
        species = list(params.get("particles.species_names", []))
        if staged:
            modes = 2 if "lasers.names" in params else 1
            params = self.stage_inputs | {"warpx.n_rz_azimuthal_modes": modes} | params
            if boosted:
                params = self._boost(params)
        elif not beam:
            raise ValueError(f"{self.objectname}: a WarpX line without a plasma stage needs a beam.")
        if beam:
            if fel and not staged:
                params = self._fel(params, wigglers)
            elif not staged:  # the window keeps up with the reference particle on average
                end = float(self.endObject.physical.end.z)
                travel = end - origin
                params |= self._transport(travel / (params["stop_time"] * c), travel, params["stop_time"])
                params |= self._aperture_diagnostic(params["warpx.const_dt"], params["stop_time"])
            if self._apertures and (staged or fel):
                warn(f"{self.objectname}: the apertures {[a['name'] for a in self._apertures]} are drifts in a "
                     "WarpX plasma stage or FEL line (they are applied in transport lines only).")
                self._apertures = []
            beam_file = f"{self.objectname}_beam.h5"
            overlay = self._overlaid()
            write_hipace_beam_file(
                self.global_parameters["beam"], os.path.join(self.subdir, beam_file),
                self._zstart, dtype=self._particle_dtype(), weight=0.5 if overlay else 1.0,
            )
            species.append("beam")
            params |= {"beam.injection_style": "external_file", "beam.injection_file": beam_file}
            if overlay:  # the same particles as positrons: no net charge or current, the radiation adds
                species.append("positrons")
                params |= {"positrons.injection_style": "external_file", "positrons.injection_file": beam_file,
                           "positrons.charge": "q_e", "particles.rigid_injected_species": ["beam", "positrons"],
                           "positrons.zinject_plane": params["beam.zinject_plane"], "positrons.rigid_advance": 0}
            bounds = [(params | self.inputs).get(k, []) for k in ("boundary.field_lo", "boundary.field_hi")]
            open_ends = staged and {"damped", "pml", "absorbing_silver_mueller"} & {
                str(b).lower() for v in bounds for b in (v.split() if isinstance(v, str) else v)}
            if boosted:
                warn(f"{self.objectname}: a beam in a boosted WarpX stage is not validated: WarpX cannot start "
                     f"it from its own field with {sorted(open_ends) or 'these'} boundaries. A witness "
                     "behind a laser held its quality at gamma_boost 3, not at 6 (see gamma_boost); "
                     "run beam-driven stages in the lab frame.")
            elif staged and open_ends:
                warn(f"{self.objectname}: WarpX cannot start the beam from its own field with {sorted(open_ends)} "
                     "boundaries (its electrostatic solve refuses them); the field forms in a transient, so start "
                     "the beam clear of the plasma.")
            elif staged:  # electromagnetic: start from the beam's own field, as FBPIC
                modes = int((params | self.inputs)["warpx.n_rz_azimuthal_modes"])
                if modes == 1:
                    params["beam.initialize_self_fields"] = 1
                else:
                    warn(f"{self.objectname}: WarpX starts a beam from its own field only with one "
                         f"azimuthal mode in RZ; with {modes} the beam starts without it, and its "
                         "field forms in a transient at the start.")
        self._finish(params, species, staged)

    def _finish(self, params: dict, species: list, staged: bool) -> None:
        """Add the species list, diagnostics, overrides and time step; write ``<name>.in``."""
        params["particles.species_names"] = species
        out = [s for s in ("beam", "injected") if s in species]
        params = {  # a line's own diagnostics (an FEL's back-transformed ones) win
            "amr.max_level": 0,  # mandatory; no mesh refinement
            "diagnostics.diags_names": "diag",
            "diag.diag_type": "Full",
            "diag.format": "openpmd",
            "diag.openpmd_backend": "h5",
            "diag.file_prefix": f"{self.objectname}_diags",
            "diag.fields_to_plot": "none",
            # WarpX takes "species = none" for a species named none and aborts (a laser-only stage)
            **({"diag.species": out} if out else {"diag.write_species": 0}),
        } | params | self.inputs
        if staged and "warpx.const_dt" not in params and "warpx.gamma_boost" not in params:
            params["warpx.const_dt"] = self._time_step(params)  # boosted: WarpX's, in the boosted frame
        if "amr.blocking_factor" not in params:
            # AMReX needs every grid dimension divisible by it (default 8)
            cells = [int(n) for n in params["amr.n_cell"]]
            params["amr.blocking_factor"] = max(b for b in (1, 2, 4, 8) if all(n % b == 0 for n in cells))
        if params["diag.diag_type"] == "Full":
            n_out = self.n_out if self.n_out is not None else self._stage_n_out(staged)
            steps = float(params["stop_time"]) / float(params["warpx.const_dt"])
            # Step 0 and every steps/n_out; the last always (dump_last_timestep).
            params.setdefault("diag.intervals", max(1, int(steps // n_out)) if n_out > 1 else 10**9)
        with open(os.path.join(self.subdir, f"{self.objectname}.in"), "w") as f:
            f.write(hipace_inputs(params))

    def _boost(self, params: dict) -> dict:
        """
        A plasma stage in the frame boosted by :attr:`~gamma_boost`: :attr:`~boost_inputs`, and
        :attr:`~n_out` + 1 back-transformed lab-frame snapshots (the stage's ``n_out`` if None),
        the last at the lab ``stop_time``, which sets the run length.
        """
        stop = params.pop("stop_time")
        snapshots = max(2, (self.n_out if self.n_out is not None else self._stage_n_out(True)) + 1)
        return params | self.boost_inputs | {
            "warpx.gamma_boost": self.gamma_boost,
            "warpx.boost_direction": "z",
            "warpx.compute_max_step_from_btd": 1,
            "diag.diag_type": "BackTransformed",
            "diag.num_snapshots_lab": snapshots,
            "diag.dt_snapshots_lab": stop / (snapshots - 1),
            "diag.buffer_size": 64,
        }

    def _fel_frame(self, p0c: float, wigglers: list) -> dict:
        """
        The boosted frame and box of an FEL line, in lattice z: the ponderomotive
        ``gamma* = gamma / sqrt(1 + K^2/2)`` (``1 + K^2`` helical), the resonant
        wavelength, and a box holding the bunch plus the radiation's slippage over
        every undulator period. Its front (``top``) becomes WarpX's z = 0: WarpX
        times the back-transformed snapshots from ``gamma beta z_max / c``.
        """
        from laura.translator.converters.magnet import undulator_peak_field
        from laura.translator.utils.fields.onaxis import K_PER_TESLA_METRE

        def k(w):
            return K_PER_TESLA_METRE * undulator_peak_field(w.magnetic) * float(w.magnetic.period)

        first = min(wigglers, key=lambda w: w.physical.start.z)
        if len({round(k(w), 6) for w in wigglers}) > 1:
            warn(f"{self.objectname}: the undulators' K differ; the boosted frame is that of {first.name}.")
        helical, period = bool(first.magnetic.helical), float(first.magnetic.period)
        factor = 1 + k(first) ** 2 * (1 if helical else 0.5)
        gamma = np.hypot(p0c, self.rest_energy) / self.rest_energy
        wavelength = period * factor / (2 * gamma**2)
        beam = self.global_parameters["beam"]
        margin = 0.1 * float(np.ptp(beam.z.val))
        # Rigid injection needs the whole beam before the first undulator
        shift = max(float(np.max(beam.z.val)) - (float(first.physical.start.z) - margin), 0.0)
        if shift:
            _drift(beam, -shift)
        z = beam.z.val
        periods = sum(round(float(w.physical.length) / float(w.magnetic.period)) for w in wigglers)
        swing = 2 * k(first) / (gamma * 2 * np.pi / period)
        gamma_star = gamma / np.sqrt(factor)
        stretch = 1 + (gamma**2 - gamma_star**2) / (gamma**2 + gamma_star**2)
        top = float(z.max()) + periods * wavelength + margin
        return {"first": first, "gamma_star": gamma_star, "wavelength": wavelength,
                "bottom": top - (top - float(z.min()) + margin) * stretch, "top": top,
                "swing": [swing, swing if helical else 0.0], "shift": shift}

    def _fel(self, params: dict, wigglers: list) -> dict:
        """
        An FEL line in the frame and box of :func:`~_fel_frame`: the Yee solver,
        the beam rigidly injected up to the first undulator, and :attr:`~n_out` + 1
        back-transformed lab-frame snapshots, the last as the reference particle
        reaches the end of the line, at :attr:`~cells_per_wavelength` cells per
        resonant wavelength.

        Returns
        -------
        dict
            The inputs, without ``stop_time``: the snapshots set the run length
        """
        frame = self._fel_box
        first, gamma_star, wavelength = frame["first"], frame["gamma_star"], frame["wavelength"]
        entrance = float(first.physical.start.z) - self._zstart
        early = [e.name for e in self.elementObjects.values()
                 if e.hardware_type not in ("Drift", "Marker", "Wiggler") and float(e.physical.length or 0)
                 and float(e.physical.end.z) - self._zstart <= entrance]
        if early:
            warn(f"{self.objectname}: the beam is rigidly injected up to {first.name}, so {early} "
                 "before it act on nothing; put them in a line of their own.")
        beam = self.global_parameters["beam"]
        lo, hi = frame["bottom"] - self._zstart, frame["top"] - self._zstart  # hi = 0
        nz = 8 * int(np.ceil((hi - lo) / (wavelength / self.cells_per_wavelength) / 8))
        if nz > 200_000:
            warn(f"{self.objectname}: the FEL box needs {nz} cells along z to resolve the "
                 f"{wavelength * 1e9:.3g} nm resonant wavelength; expect a very long run.")
        centre = [np.mean(beam.x.val), np.mean(beam.y.val)]
        # the bunch, plus its swing either way in the undulator (planar: x only), a fifth to spare
        half = [s * max(float(np.std(v)), 1e-6) + 1.2 * w
                for s, v, w in zip(self.box_sigmas, (beam.x.val, beam.y.val), frame["swing"])]
        snapshots = max(2, (self.n_out or 1) + 1)
        stop = params["stop_time"]  # lab time for the reference to reach the end of the line
        params = {key: value for key, value in params.items() if key != "stop_time"}
        return params | {
            "geometry.dims": 3,
            "amr.n_cell": [self.n_cell[0], self.n_cell[1], nz],
            "geometry.prob_lo": [centre[0] - half[0], centre[1] - half[1], lo],
            "geometry.prob_hi": [centre[0] + half[0], centre[1] + half[1], hi],
            "warpx.gamma_boost": gamma_star,
            "warpx.boost_direction": "z",
            "warpx.do_moving_window": 1,
            "warpx.moving_window_dir": "z",
            "warpx.moving_window_v": np.sqrt(1 - 1 / gamma_star**2),
            "algo.maxwell_solver": "yee",
            "algo.particle_shape": 2,
            "algo.particle_pusher": "vay",
            "warpx.cfl": 0.99,
            "boundary.field_lo": ["absorbing_silver_mueller"] * 3,
            "boundary.field_hi": ["absorbing_silver_mueller"] * 3,
            "boundary.particle_lo": ["absorbing"] * 3,
            "boundary.particle_hi": ["absorbing"] * 3,
            "particles.rigid_injected_species": "beam",
            "beam.zinject_plane": entrance,
            "beam.rigid_advance": 0,
            "warpx.compute_max_step_from_btd": 1,
            "diag.diag_type": "BackTransformed",
            "diag.num_snapshots_lab": snapshots,
            "diag.dt_snapshots_lab": stop / (snapshots - 1),
            "diag.buffer_size": 64,
            "diag.fields_to_plot": self.fel_fields,
        }

    def _collision(self, elements: list) -> tuple[dict, list]:
        """
        A strong-strong collision at a :class:`~laura.models.element.BeamBeam`, as in
        WarpX's beam-beam examples and Formenti et al. (arXiv:2610.17557).
        The incoming beam is drifted to ``d = 4 max(sigma_z)`` before the IP, the opposing
        bunch (:meth:`~laura.translator.converters.beam_beam.BeamBeamTranslator.to_warpx`)
        starts ``d`` after it, and the run lasts ``2d / c``; :func:`~postProcess` then
        drifts the beam on to the end of the line.

        Returns
        -------
        tuple[dict, list]
            The inputs and the species

        Raises
        ------
        ValueError
            Without a beam, or with more than the one element in the line
        """
        from laura.translator.converters.converter import translate_elements

        others = [e.name for e in self.elementObjects.values()
                  if e.hardware_type not in ("BeamBeam", "Drift", "Marker") and float(e.physical.length or 0)]
        if self.particle_definition is None or len(elements) > 1 or others:
            raise ValueError(
                f"{self.objectname}: a WarpX beam-beam line holds one BeamBeam (and drifts) and an "
                f"incoming beam; it has {[e.name for e in elements] + others}."
            )
        (element,) = elements
        beam = self.global_parameters["beam"]
        sigma = [float(np.std(beam.x.val)), float(np.std(beam.y.val)), float(np.std(beam.z.val))]
        sim = element.simulation
        d = 4 * max(sigma[2], sim.width or 0.0)
        ip, end = float(element.physical.middle.z), float(self.endObject.physical.end.z)
        _drift(beam, ip - d - self.plasma_box_origin)
        self._zstart, self._drift_after = ip, end - ip - d
        self._executable = "warpx_3d"
        translator = translate_elements([element])[element.name]
        params = translator.to_warpx(
            self.opposing_p0c or self.reference_p0c, self.opposing_rest_energy or self.rest_energy,
            z_centre=d, sigma_z=sigma[2], macroparticles=self.opposing_macroparticles,
        )
        k = self.collision_sigmas
        half = [k * max(sigma[0], sim.horizontal_sigma) + abs(sim.horizontal_offset),
                k * max(sigma[1], sim.vertical_sigma) + abs(sim.vertical_offset), 2 * d]
        beam_file = f"{self.objectname}_beam.h5"
        write_hipace_beam_file(beam, os.path.join(self.subdir, beam_file), self._zstart,
                               dtype=self._particle_dtype())
        params |= {
            "geometry.dims": 3,
            "amr.n_cell": self.n_cell,
            "geometry.prob_lo": [-h for h in half],
            "geometry.prob_hi": half,
            "boundary.field_lo": ["open"] * 3,
            "boundary.field_hi": ["open"] * 3,
            "boundary.particle_lo": ["Absorbing"] * 3,
            "boundary.particle_hi": ["Absorbing"] * 3,
            "warpx.do_electrostatic": "relativistic",
            "warpx.poisson_solver": "fft",
            "warpx.use_2d_slices_fft_solver": 1,
            "warpx.grid_type": "collocated",
            "algo.particle_pusher": "vay",
            "algo.particle_shape": 3,
            "stop_time": 2 * d / c,
            "warpx.const_dt": 4 * d / (c * self.n_cell[2]),  # a cell per step
            "beam.injection_style": "external_file",
            "beam.injection_file": beam_file,
            "beam.initialize_self_fields": 1,
            "opposing.initialize_self_fields": 1,
        }
        return params, ["beam", "opposing"]

    def _stage_n_out(self, staged: bool) -> int:
        """``simulation.n_out`` of the line's plasma stage or lens, else 1."""
        for element in self.elementObjects.values():
            if staged and element.hardware_type.lower() in ("plasma", "plasmalens"):
                return int(element.simulation.n_out or 1)
        return 1

    @staticmethod
    def _time_step(params: dict) -> float:
        """
        The time step WarpX's ``ComputeDt`` would take for this box and solver,
        made explicit so the dumps can be spaced: ``cfl * min(dr, dz) / c`` for
        PSATD, the multi-mode Courant limit of ``CylindricalYeeAlgorithm`` for Yee.
        """
        nr, nz = params["amr.n_cell"]
        (r0, z0), (r1, z1) = params["geometry.prob_lo"], params["geometry.prob_hi"]
        dr, dz = (r1 - r0) / nr, (z1 - z0) / nz
        cfl = float(params.get("warpx.cfl", 1.0))
        if str(params.get("algo.maxwell_solver", "yee")).lower() == "psatd":
            return cfl * min(dr, dz) / c
        modes = int(params.get("warpx.n_rz_azimuthal_modes", 1))
        coeffs = (0.2105, 1.0, 3.5234, 8.5104, 15.5059, 24.5037)
        alpha = coeffs[modes - 1] if modes < 7 else (modes - 1) ** 2 - 0.4
        return cfl / (c * np.sqrt((1 + alpha) / dr**2 + 1 / dz**2))

    def _transport(self, beta: float, length: float, stop_time: float) -> dict:
        """
        3D box, moving window, time step and field solve of a line ``length`` long without a
        plasma stage. The step divides ``stop_time`` evenly: WarpX finishes its last whole step,
        so a remainder would overshoot the end of the line by up to a step.
        """
        beam = self.global_parameters["beam"]
        centre = [np.mean(beam.x.val), np.mean(beam.y.val), np.mean(beam.z.val) - self._zstart]
        # floored so a zero-size (single-particle) beam still has a box
        half = [k * max(float(np.std(v)), 1e-6) for k, v in
                zip(self.box_sigmas, (beam.x.val, beam.y.val, beam.z.val))]
        step = self.cells_per_step * 2 * half[2] / self.n_cell[2]
        if length / step > 1e5:
            warn(f"{self.objectname}: {length / step:.3g} steps of {step:.3g} m (a cell) to cross "
                 f"{length:.3g} m; WarpX's moving window cannot take longer steps.")
        params = {
            "geometry.dims": 3,
            "amr.n_cell": self.n_cell,
            "geometry.prob_lo": [m - h for m, h in zip(centre, half)],
            "geometry.prob_hi": [m + h for m, h in zip(centre, half)],
            "warpx.do_moving_window": 1,
            "warpx.moving_window_dir": "z",
            "warpx.moving_window_v": beta,
            "warpx.const_dt": stop_time / np.ceil(stop_time * beta * c / step),
            "algo.particle_shape": 3,
        }
        mode = self.space_charge_mode
        if mode == "3d":
            # A collocated grid, so that E and v x B (which cancel to 1/gamma^2) are gathered at the
            # same points: on the staggered grid a 1 GeV beam blew up ~1e4 times too fast.
            return params | {"warpx.do_electrostatic": "relativistic", "warpx.poisson_solver": "fft",
                             "boundary.field_lo": ["open"] * 3, "boundary.field_hi": ["open"] * 3,
                             "warpx.grid_type": "collocated", "algo.particle_pusher": "vay"}
        if mode:
            warn(exceptions.SpaceChargeModeWarning(self.objectname, "WarpX", mode, "'3d'"))
        return params | {"algo.maxwell_solver": "none"}

    def _aperture_diagnostic(self, dt: float, stop_time: float) -> dict:
        """
        A second diagnostic, ``<name>_aperture_diags/``, dumping the beam at every step in which
        a particle may cross an end of an aperture (:attr:`_apertures`, from LAURA): each
        particle's step is estimated from its start and speed, two steps to spare either side.
        :func:`_aperture_losses` reads them.
        """
        if not self._apertures:
            return {}
        beam = self.global_parameters["beam"]
        z = np.asarray(beam.z.val) - self._zstart
        cp = np.asarray(beam.cp.val)
        beta = cp / np.hypot(cp, self.rest_energy)
        last = int(np.ceil(stop_time / dt))
        windows = []
        for plane in sorted({p for a in self._apertures for p in a["z"]}):
            t = (plane - z) / (beta * c)
            lo, hi = int(np.floor(t.min() / dt)) - 2, int(np.ceil(t.max() / dt)) + 2
            if hi >= 0 and lo <= last:
                windows.append(f"{max(lo, 0)}:{min(hi, last)}:1")
        return {
            "diagnostics.diags_names": ["diag", "aperture"],
            "aperture.diag_type": "Full",
            "aperture.format": "openpmd",
            "aperture.openpmd_backend": "h5",
            "aperture.file_prefix": f"{self.objectname}_aperture_diags",
            "aperture.fields_to_plot": "none",
            "aperture.species": "beam",
            "aperture.intervals": ",".join(windows),
        }

    def _aperture_losses(self) -> tuple[set, set]:
        """
        The ids of the beam particles outside an aperture where they cross one of its ends,
        each interpolated in z between the two dumps on either side (a straight track through
        a convex aperture is outside it somewhere only if it is at an end), and the ids seen
        crossing every end.
        """
        snaps = []
        for path in glob.glob(os.path.join(self.subdir, f"{self.objectname}_aperture_diags", "*.h5")):
            with h5py.File(path) as f:
                for it in f["data"].values():
                    if "beam" in it.get("particles", {}):
                        snaps.append((float(it.attrs["time"]), _species(it["particles"]["beam"])))
        snaps = [s for _, s in sorted(snaps, key=lambda p: p[0])]
        lost, everywhere = set(), None
        for aperture in self._apertures:
            (x0, y0), (a, b) = aperture["centre"], aperture["half"]
            for plane in aperture["z"]:
                checked = set()
                for s0, s1 in zip(snaps, snaps[1:]):
                    ids, i0, i1 = np.intersect1d(s0.id, s1.id, return_indices=True)
                    z0, z1 = s0.z[i0], s1.z[i1]
                    cross = (z0 < plane) & (z1 >= plane)
                    f = (plane - z0[cross]) / (z1[cross] - z0[cross])
                    x = s0.x[i0][cross] + f * (s1.x[i1][cross] - s0.x[i0][cross]) - x0
                    y = s0.y[i0][cross] + f * (s1.y[i1][cross] - s0.y[i0][cross]) - y0
                    if aperture["shape"] == "elliptical":
                        out = (x / a) ** 2 + (y / b) ** 2 >= 1
                    else:
                        out = ((abs(x) >= a) if a else False) | ((abs(y) >= b) if b else False)
                    lost.update(ids[cross][out].tolist())
                    checked.update(ids[cross].tolist())
                everywhere = checked if everywhere is None else everywhere & checked
        return lost, everywhere or set()

    def run(self) -> None:
        """Run WarpX on ``<name>.in`` in the master subdirectory."""
        workdir = os.path.abspath(self.subdir)
        command = self.executables.build_command(
            self.executables[self.executable] + [f"{self.objectname}.in"], workdir
        )
        self.run_command(command, os.path.join(workdir, f"{self.objectname}.log"), cwd=workdir)

    def final_diagnostic(self) -> str | None:
        """The last openPMD dump WarpX wrote, or None."""
        dumps = sorted(glob.glob(os.path.join(self.subdir, f"{self.objectname}_diags", "*.h5")))
        return dumps[-1] if dumps else None

    def postProcess(self) -> None:
        """
        Read the beam, and the trapped injected electrons, from the last dump and
        write them as openPMD for the next line.
        """
        super().postProcess()
        dump = self.final_diagnostic()
        if dump is None:
            raise RuntimeError(f"{self.objectname}: WarpX wrote no diagnostics.")
        with h5py.File(dump) as f:
            particles = next(iter(f["data"].values())).get("particles", {})
            found = [
                (_species(particles[name]), cut)
                for name, cut in (("beam", None), ("injected", self.injected_uz_min))
                if name in particles
            ]
            has_beam = "beam" in particles
        if not found:
            return  # a laser-only stage that injected nothing
        if has_beam and self._overlaid():  # the electrons went in at half weight
            found[0][0].w = found[0][0].w * 2
        beam = self.global_parameters["beam"]
        if self.particle_definition is not None:
            n_in = len(beam.x.val)  # still the incoming beam
            n_out = len(found[0][0].x) if has_beam else 0
            if n_out < n_in:
                warn(f"{self.objectname}: {n_in - n_out} of {n_in} beam particles left the WarpX box "
                     "and were absorbed; enlarge box_sigmas (collision_sigmas at a BeamBeam) if "
                     "that is not the physics.")
        if has_beam and self._apertures:
            out = found[0][0]
            lost, checked = self._aperture_losses()
            keep = ~np.isin(out.id, list(lost))
            unchecked = int(np.sum(~np.isin(out.id, list(checked | lost))))
            if unchecked:
                warn(f"{self.objectname}: {unchecked} beam particles crossed an aperture end between dumps "
                     "and were not checked against it (their speed was misjudged: a cavity?).")
            if lost and self.space_charge_mode == "3d":
                warn(f"{self.objectname}: {len(lost)} particles stopped at the apertures were removed after the "
                     "run; their space charge still acted on the rest downstream of them.")
            n = len(out.x)
            found[0] = (SimpleNamespace(**{k: v[keep] if isinstance(v, np.ndarray) and len(v) == n else v
                                           for k, v in vars(out).items()}), found[0][1])
        particles_to_beam(beam, [s for s, _ in found], zpos=self._zstart, uz_min=[u for _, u in found])
        if self._drift_after is not None:  # from d past a BeamBeam to the end of the line
            _drift(beam, self._drift_after)
        beam.code = "warpx"
        rbf.openpmd.write_openpmd_beam_file(
            beam, os.path.join(self.subdir, f"{self.output_basename(self.end)}.openpmd.hdf5")
        )


def _drift(beam, length: float) -> None:
    """Move every particle of ``beam`` ``length`` [m] along z in a straight line (negative: back)."""
    b = beam._beam
    xp, yp = b.px.val / b.pz.val, b.py.val / b.pz.val
    b.x = UnitValue(b.x.val + xp * length, "m")
    b.y = UnitValue(b.y.val + yp * length, "m")
    b.z = UnitValue(b.z.val + length, "m")


def _species(group: h5py.Group) -> SimpleNamespace:
    """One species of a WarpX openPMD dump in the shape ``particles_to_beam`` reads."""
    z = group["position/z"]  # a constant record (attribute ``shape``) when the species is empty
    n = int(z.shape[0] if isinstance(z, h5py.Dataset) else z.attrs["shape"][0])
    record = {k: _openpmd_record(group, k, n) for k in (
        "position/x", "position/y", "position/z", "momentum/x", "momentum/y", "momentum/z",
        "weighting", "charge", "mass")}
    if "positionOffset" in group:  # WarpX writes one
        for axis in "xyz":
            record[f"position/{axis}"] = record[f"position/{axis}"] + _openpmd_record(
                group, f"positionOffset/{axis}", n)
    mc = record["mass"] * c
    return SimpleNamespace(
        x=record["position/x"], y=record["position/y"], z=record["position/z"],
        ux=record["momentum/x"] / mc, uy=record["momentum/y"] / mc, uz=record["momentum/z"] / mc,
        w=record["weighting"], q=float(record["charge"][0]) if n else 0.0,
        m=float(record["mass"][0]) if n else 1.0,
        id=np.asarray(group["id"], dtype=np.uint64) if isinstance(group.get("id"), h5py.Dataset) else np.arange(n),
    )
