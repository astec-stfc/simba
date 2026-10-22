"""
`pyAT`_ (Accelerator Toolbox) backend.

.. _pyAT: https://github.com/atcollab/at
"""

from copy import deepcopy
from typing import Any, ClassVar, List
from warnings import filterwarnings, warn

import numpy as np
from laura.translator.utils.at import cavity_phase, time_lag

from ...Framework_objects import frameworkLattice
from ...Modules.Beams import at as rbf_at
from ...Modules.constants import speed_of_light
from ...Modules.EnergyRamp import wrap_phase

AT_ORDER = [0, 1, 2, 3, 5, 4]
"""pyAT's ``(x, px, y, py, delta, ct)`` in :attr:`atLattice.otm_convention`
order, and back."""


class atLattice(frameworkLattice):
    """A line converted to a pyAT ``Lattice`` and tracked through it."""

    code: str = "at"
    """The lattice object type."""

    supports_turns: ClassVar[bool] = True
    """By ``lattice_track``, a pass at a time."""

    native_time: ClassVar[tuple[str, str]] = ("ct", "m")
    """``ct``, measured from the reference: positive behind it."""

    supports_frequency_map: ClassVar[bool] = True
    """The aperture grid, recorded every pass, then the shared tune extractor."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """``lattice_track(losses=True)``: its loss map says when."""

    supports_nsuperperiods: ClassVar[bool] = True
    """As N passes per turn."""

    supports_radiation: ClassVar[bool] = True
    """``Lattice.enable_6d``; ``quantum`` adds a ``gen_quantdiff_elem``."""

    supports_periodic: ClassVar[bool] = True
    """``get_optics`` with no initial conditions."""

    native_rf: ClassVar[str] = "synchronous"
    """Once :meth:`rf_phase_corrections` undoes AT's own slip; moved through ``TimeLag``."""

    rf_phase_sign: ClassVar[float] = -1.0
    """LAURA phase (degrees) moved by this times a phase the reference sees; measured against elegant."""

    supports_programs: ClassVar[bool] = True
    """Set between turns by :meth:`apply_programs`."""

    program_attributes: ClassVar[dict] = {
        name: ("KickAngle[0]", "KickAngle[1]")
        for name in ("Corrector", "Dipole", "Quadrupole", "Sextupole", "Multipole", "ThinMultipole")
    }
    """``(horizontal, vertical)`` attribute an unqualified program sets, per
    pyAT class; ``KickAngle`` has a deflection angle's sign."""

    otm_convention: ClassVar[str] = "x, px, y, py, ct, delta"
    """One-turn map convention, after :data:`AT_ORDER`."""

    otm_longitudinal_sign: ClassVar[int] = -1
    """``ct`` runs the other way to ``zeta``."""

    otm_longitudinal_scale: ClassVar[int | None] = 0
    """AT takes ``beta = 1``, where its ``ct`` is Xsuite's ``-zeta``."""

    native_turns: bool | None = None
    """Default for ``tracking: native_turns``; see :attr:`use_native_turns`."""

    particle_definition: str = None
    """Name of the initial particle distribution."""

    ring: Any = None
    """The pyAT ``Lattice``."""

    tracked: Any = None
    """:attr:`ring`, with any quantum diffusion appended; what the beam goes through."""

    pin: Any = None
    """Initial 6xN pyAT coordinates."""

    pout: Any = None
    """Final 6xN pyAT coordinates."""

    tws: Any = None
    """``get_optics`` at every element and the end: ``(ld0, rd, ld)``."""

    beam_data: List = []
    """Tracked bunch statistics at every element's entrance and the end, on the last pass."""

    _optics: Any = None
    """The run's periodic ``get_optics`` (``()`` if there is none), solved once
    by :meth:`periodic_optics`."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition

    @classmethod
    def native_time_scale(cls, beta0: float) -> float:
        """``ct = beta0 c (t - reference_time)``."""
        return beta0 * speed_of_light

    def writeElements(self) -> None:
        """
        Build :attr:`ring` and :attr:`tracked`, radiating as :attr:`radiation` asks.
        """
        import at

        particle = at.Particle(
            "simba", rest_energy=self.rest_energy, charge=self.reference_charge
        )
        filterwarnings("ignore", "AT tracking still assumes beta", at.AtWarning)
        self.ring = self.section.to_at(energy=self.reference_energy, particle=particle)
        self.tracked = self.ring
        self._optics = None
        gamma0 = self.reference_energy / self.rest_energy
        if gamma0 < 100:
            warn(
                f"Line '{self.objectname}': AT's pass methods take beta = 1, so "
                f"its time of flight, and any synchrotron motion, is off by "
                f"1 / gamma0**2 = {gamma0**-2:.1e}."
            )
        if self.radiation in ("mean", "quantum"):
            self.ring.enable_6d()
        if self.radiation == "quantum":
            try:
                diffusion = at.gen_quantdiff_elem(self.ring)
            except Exception as error:
                warn(
                    f"Line '{self.objectname}': AT found no closed orbit to "
                    f"diffuse about ({error}), so radiation is 'mean'."
                )
            else:
                self.tracked = self.ring.copy()
                self.tracked.append(diffusion)

    def write(self) -> None:
        """
        Build the lattice via :meth:`writeElements`.
        """
        self.writeElements()

    def preProcess(self) -> None:
        """
        Load the input beam and convert it to :attr:`pin`.
        """
        super().preProcess()
        self.load_input_beam(self.get_prefix(), self.particle_definition)
        self.pin = rbf_at.write_at_beam(
            self.global_parameters["beam"], p0c=self.reference_p0c, t0=self.reference_t0
        )

    def run(self) -> None:
        """
        Track every pass, writing beams at the screens, markers, BPMs and end,
        then set :attr:`tws`.
        """
        import at

        names = [e.FamName for e in self.ring]
        s_in = self.getSValues(as_dict=True, at_entrance=True)
        points = [
            (names.index(e.name), e.name, e.physical.start.z, s_in[e.name])
            for e in self.screens_and_markers_and_bpms
            if e.name in names and e.name != self.end and self.writes_output(e.name)
        ]
        points.sort()
        points.append((
            len(self.tracked), self.end, self.endObject.physical.end.z,
            self.getSValues(as_dict=True)[self.end],
        ))
        refpts = [point[0] for point in points]
        last = self.turns * self.passes_per_turn - 1
        self.pout = self.pin.copy(order="F")
        self._optics = None

        def track_pass(turn, pass_index, name_turn, record):
            # AT's turn count runs on, for its AC dipoles; see rf_phase_corrections
            if pass_index == last:
                self.collect_beam_data(self.pout, pass_index)
            rout, *_ = at.lattice_track(
                self.tracked, self.pout, refpts=refpts if record else [], in_place=True,
                turn=pass_index,
            )
            for k, (_, name, z, s) in enumerate(points if record else []):
                self._write_beam(rout[:, :, k, 0], name, name_turn, turn, z, self.entrance_s + s)

        if self.use_native_turns:
            self.run_native(track_pass)
        else:
            self.run_turns(track_pass)
        self.tws = self._twiss()

    @property
    def use_native_turns(self) -> bool:
        """
        Whether :meth:`run_native` tracks, rather than :meth:`run_turns`.

        ``native_turns`` in the ``tracking`` block, else :attr:`native_turns`:
        unset, only with no live cavity; true, with one too, as plain AT would,
        (see :meth:`run_native`) and ``rf`` is not imposed;
        false, never. Never with programs.
        """
        tracking = self.file_block.get("tracking") or {}
        native = tracking.get("native_turns", self.native_turns)
        if self.programs or native is False:
            return False
        return bool(native) or not self.live_cavities()

    def rf_phase_corrections(self) -> dict:
        """
        :meth:`~simba.Framework_objects.frameworkLattice.rf_phase_corrections`,
        less AT's own slip.

        ``RFCavityPass`` moves a cavity by ``2 pi f T0`` a turn (mod ``2 pi``),
        with ``T0`` the lattice length over ``c``.
        """
        corrections = super().rf_phase_corrections()
        passes = np.arange(self.turns * self.passes_per_turn)
        t0 = sum(element.Length for element in self.tracked) / speed_of_light
        for name in self.live_cavities():
            for element in self._cavities(name)[:1]:
                slip = 2 * np.pi * element.Frequency * t0 * passes
                corrections[name] = wrap_phase(corrections.get(name, 0.0) - slip)
        return corrections

    def run_native(self, track_pass) -> None:
        """
        :meth:`run_turns` with nothing to change between passes: one ``nturns``
        call up to each recorded pass, which ``track_pass`` tracks.

        Each ``lattice_track`` call costs about 0.25 ms. AT's turn count runs
        on across the calls, and slips a live
        cavity by ``2 pi h (1 / beta - 1)`` a turn.
        """
        import at

        done = 0
        for turn, name_turn in self.output_turns():
            index = self.last_pass(self.beam_turn(turn))
            if index > done:
                at.lattice_track(
                    self.tracked, self.pout, nturns=index - done, refpts=[], in_place=True,
                    turn=done,
                )
            track_pass(self.beam_turn(turn), index, name_turn, True)
            done = index + 1

    def _write_beam(
        self, r: np.ndarray, name: str, name_turn: int | None, turn: int,
        zstart: float, s: float,
    ) -> None:
        """
        Write one beam via :meth:`write_beam_file`.

        Parameters
        ----------
        r: np.ndarray
            6xN pyAT coordinates
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
        t_reference = None
        if self.uses_reference_clock:
            t_reference = self.reference_time(s - self.entrance_s, self.last_pass(turn))
        beam = deepcopy(self.global_parameters["beam"])
        rbf_at.read_at_beam(
            beam, r, self.reference_p0c, zstart=zstart, s=s,
            ref_index=self.ref_idx, t_reference=t_reference,
        )
        beam.turn = turn
        self.write_beam_file(beam, name, name_turn)

    def collect_beam_data(self, r: np.ndarray, turn: int = 0) -> None:
        """
        Fill :attr:`beam_data` by walking ``r`` down :attr:`tracked`.

        Parameters
        ----------
        r: np.ndarray
            The beam entering the pass; left alone
        turn: int
            AT's turn count, which ``element_track`` cannot take
        """
        import at

        refs = np.arange(len(self.tracked) + 1)
        chunk = max(1, 2**28 // (48 * r.shape[1]))
        stats = []
        for start in range(0, len(refs), chunk):
            rout, *_ = at.lattice_track(
                self.tracked, r, refpts=refs[start:start + chunk], turn=turn
            )
            stats += [self.bunch_statistics(rout[:, :, k, 0]) for k in range(rout.shape[2])]
        # one row per element of the ring, and the end
        self.beam_data = stats[: len(self.ring)] + [stats[-1]]

    def bunch_statistics(self, r: np.ndarray) -> dict:
        """
        Tracked (not ``get_optics``) bunch statistics at one point, for the twiss file.

        Parameters
        ----------
        r: np.ndarray
            6xN pyAT coordinates; lost particles are NaN and left out

        Returns
        -------
        dict
            One row of :attr:`beam_data`, in Xsuite's column names
        """
        x, px, y, py, delta, ct = r[:, ~np.isnan(r).any(axis=0)]
        p0c, mass = self.reference_p0c, self.rest_energy
        betagamma = p0c * (1 + np.mean(delta)) / mass

        def emittance(u, pu, corrected):
            cov = np.cov([u, pu, delta])
            block = cov[:2, :2]
            if corrected and cov[2, 2] > 0:
                block = block - np.outer(cov[:2, 2], cov[:2, 2]) / cov[2, 2]
            return betagamma * np.sqrt(max(np.linalg.det(block), 0.0))

        return {
            "mean_x": np.mean(x),
            "mean_y": np.mean(y),
            "sigma_x": np.std(x),
            "sigma_px": np.std(px),
            "sigma_y": np.std(y),
            "sigma_py": np.std(py),
            "sigma_zeta": np.std(ct),
            "sigma_delta": np.std(delta),
            "momentum": np.mean(np.hypot(p0c * (1 + delta), mass)) - mass,
            "emit_xn": emittance(x, px, False),
            "emit_yn": emittance(y, py, False),
            "emit_xn_corrected": emittance(x, px, True),
            "emit_yn_corrected": emittance(y, py, True),
        }

    def optics_ring(self):
        """:attr:`ring`, 4D unless its cavities have voltage, as Xsuite's twiss chooses."""
        if self.ring.is_6d and self.rf_voltage:
            return self.ring
        return self.ring.disable_6d(copy=True)

    def periodic_optics(self):
        """
        ``get_optics`` around :meth:`optics_ring`, with chromaticity, once a run.

        Returns
        -------
        tuple | None
            ``(ld0, rd, ld)`` at every element and the end, or None if the ring
            has no periodic solution
        """
        import at

        if self._optics is None:
            ring = self.optics_ring()
            try:
                self._optics = at.get_optics(ring, range(len(ring) + 1), get_chrom=True)
            except (at.AtError, np.linalg.LinAlgError) as error:
                warn(f"AT found no periodic solution for {self.objectname}: {error}")
                self._optics = ()
        return self._optics or None

    def _twiss(self):
        """
        ``get_optics`` at every element and the end: periodic if :attr:`periodic`,
        else from the beam's twiss.

        Returns
        -------
        tuple
            ``(ld0, rd, ld)``
        """
        import at

        if self.periodic:
            optics = self.periodic_optics()
            if optics is not None:
                return optics
            warn(
                f"Line '{self.objectname}' asks for the periodic solution, but "
                "AT found none. Falling back to the beam's twiss, which is not "
                "the ring's."
            )
        twiss = self.global_parameters["beam"].twiss
        return at.get_optics(
            self.ring.disable_6d(copy=True),
            range(len(self.ring) + 1),
            twiss_in={
                "beta": [twiss.beta_x.val, twiss.beta_y.val],
                "alpha": [twiss.alpha_x.val, twiss.alpha_y.val],
            },
        )

    def read_one_turn_map(self):
        """
        ``find_m66`` about the closed orbit.

        Returns
        -------
        numpy.ndarray | None
            The 6x6 map, or None without a periodic solution.
        """
        import at

        optics = self.periodic_optics()
        if optics is None:
            return None
        matrix, _ = at.find_m66(self.optics_ring(), orbit=optics[0].closed_orbit)
        return matrix[np.ix_(AT_ORDER, AT_ORDER)]

    def read_closed_orbit(self):
        """The periodic solution's orbit at the start."""
        optics = self.periodic_optics()
        if optics is None:
            return None
        return np.asarray(optics[0].closed_orbit, dtype=float)[AT_ORDER]

    def read_optics_summary(self) -> dict:
        """Total tunes off the phase advance, and ``get_optics``'s chromaticity."""
        optics = self.periodic_optics()
        if optics is None:
            return {}
        _, rd, ld = optics
        return {
            "tune_x_total": float(ld.mu[-1, 0]) / (2 * np.pi),
            "tune_y_total": float(ld.mu[-1, 1]) / (2 * np.pi),
            "chromaticity_x": float(rd.chromaticity[0]),
            "chromaticity_y": float(rd.chromaticity[1]),
        }

    def apply_programs(self, turn: int) -> None:
        """Set each programmed element to its value for ``turn`` (1-based); ``name[i]`` sets one entry."""
        for program in self.programs:
            matches = [e for e in self.ring if e.FamName == program.element]
            attribute = self.program_attribute(program, type(matches[0]).__name__ if matches else None)
            if attribute is None:
                continue
            value = program.value_at(turn)
            name, _, index = attribute.partition("[")
            for element in matches:
                if index:
                    # magnets have no KickAngle until one is given
                    vector = np.array(getattr(element, name, np.zeros(2)), dtype=float)
                    vector[int(index.rstrip("]"))] = value
                    setattr(element, name, vector)
                else:
                    setattr(element, name, value)

    def _cavities(self, name: str) -> list:
        return [e for e in self.ring if e.FamName == name and type(e).__name__ == "RFCavity"]

    def cavity_phase(self, name: str) -> float | None:
        """A cavity's LAURA phase, in degrees, from its ``TimeLag``."""
        cavities = self._cavities(name)
        return cavity_phase(cavities[0].TimeLag, cavities[0].Frequency) if cavities else None

    def set_cavity_phase(self, name: str, phase: float) -> None:
        """Set a cavity's ``TimeLag`` for a LAURA phase in degrees; the next pass sees it."""
        for element in self._cavities(name):
            element.TimeLag = time_lag(phase, element.Frequency)

    def _probes(self, points) -> np.ndarray:
        """One particle per ``(x, y)``, as pyAT coordinates."""
        r = np.zeros((6, len(points)), order="F")
        r[0], r[2] = np.asarray(points, dtype=float).T
        return r

    def track_reference_particle(self) -> dict:
        """
        Track one particle launched just off the closed orbit, sampled after each turn.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py``, each of length :attr:`turns`.
        """
        import at

        orbit = self.read_closed_orbit()
        orbit = np.zeros(6) if orbit is None else orbit[AT_ORDER]
        nudge = float((self.da_settings or {}).get("x_max", 1e-3)) / 100.0
        r = np.asfortranarray((orbit + [nudge, 0, nudge, 0, 0, 0])[:, None])
        stride = self.passes_per_turn
        rout, *_ = at.lattice_track(
            self.ring, r, nturns=self.turns * stride, refpts=len(self.ring)
        )
        track = rout[:4, 0, 0, stride - 1::stride]
        return dict(zip(("x", "px", "y", "py"), track))

    def run_dynamic_aperture(self) -> list:
        """
        Track one particle per point of :meth:`da_rays` and record survival.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per start.
        """
        import at

        points = self.da_rays()
        passes = self.turns * self.passes_per_turn
        *_, losses = at.lattice_track(
            self.ring, self._probes(points), nturns=passes, refpts=[], losses=True
        )
        lost = losses["loss_map"]
        turns = np.where(lost["islost"], lost["turn"], passes) // self.passes_per_turn
        self.dynamic_aperture = [
            (float(x), float(y), int(turn)) for (x, y), turn in zip(points, turns)
        ]
        return self.dynamic_aperture

    def run_frequency_map(self) -> list:
        """
        Tune footprint over the aperture grid, recorded at the start of every turn.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y, diffusion)`` per surviving grid point.
        """
        import at

        xs, ys = self.da_grid()
        points = [(x, y) for y in ys for x in xs]
        rout, _, losses = at.lattice_track(
            self.ring, self._probes(points), nturns=self.turns * self.passes_per_turn,
            refpts=[0], losses=True,
        )
        coords = rout[:4, :, 0, :: self.passes_per_turn]
        lost = losses["loss_map"]["islost"]
        return self._footprint(
            (x, y, *coords[:, i]) for i, (x, y) in enumerate(points) if not lost[i]
        )

    def postProcess(self) -> None:
        """
        Read back the ring's maps and write the twiss CSV.
        """
        super().postProcess()
        self._write_twiss_csv()

    def _write_twiss_csv(self) -> None:
        """:attr:`tws` in Xsuite's columns, with the tracked beam statistics beside it."""
        import pandas as pd

        ld = self.tws[2]
        s = ld.s_pos + self.entrance_s
        svals = np.array(self.getSValues(at_entrance=False)) + s[0]
        zvals = [a[-1] for a in self.getZValues()]
        beta, alpha, mu, eta = ld.beta, ld.alpha, ld.mu, ld.dispersion
        df = pd.DataFrame({
            "name": [e.FamName for e in self.ring] + ["_end_point"],
            "s": s,
            "z": np.interp(s, svals, zvals),
            "betx": beta[:, 0],
            "alfx": alpha[:, 0],
            "gamx": (1 + alpha[:, 0] ** 2) / beta[:, 0],
            "bety": beta[:, 1],
            "alfy": alpha[:, 1],
            "gamy": (1 + alpha[:, 1] ** 2) / beta[:, 1],
            "mux": mu[:, 0] / (2 * np.pi),
            "muy": mu[:, 1] / (2 * np.pi),
            "dx": eta[:, 0],
            "dpx": eta[:, 1],
            "dy": eta[:, 2],
            "dpy": eta[:, 3],
        })
        for key in self.beam_data[0] if self.beam_data else []:
            df[key] = [row[key] for row in self.beam_data]
        df.to_csv(f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.at.csv')
