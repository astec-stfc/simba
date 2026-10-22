"""
`ImpactX`_ backend, tracked in-process.

AMReX starts once per process and ImpactX cannot start again after
``finalize()``, so every line shares one :func:`simulation`, never finalized.

.. _ImpactX: https://impactx.readthedocs.io
"""

from typing import Any, ClassVar
from warnings import warn

import numpy as np
from laura.translator.conversion_rules.codes.impactx_conversion import lattice

from ... import exceptions
from ...Framework_objects import frameworkLattice
from ...Modules.Beams import impactx as rbf_impactx
from ...Modules.constants import speed_of_light

_simulation = None


def simulation():
    """The process's one ImpactX simulation, made on first use."""
    global _simulation
    if _simulation is None:
        import amrex.space3d as amr
        import impactx

        sim = impactx.ImpactX()
        amr.ParmParse("diag").add("enable", False)
        sim.particle_shape = 2
        sim.space_charge = "false"
        sim.poisson_solver = "multigrid"
        sim.n_cell = [32, 32, 32]
        sim.max_level = 1
        sim.prob_relative = [3.0, 1.1]
        sim.blocking_factor_x, sim.blocking_factor_y, sim.blocking_factor_z = [16], [16], [4]
        sim.dynamic_size = True
        sim.verbose = 0
        sim.tiny_profiler = False
        sim.init_grids()
        _simulation = sim
    return _simulation


class impactxLattice(frameworkLattice):
    """A line converted to ImpactX element dicts and tracked by :func:`simulation`."""

    code: str = "impactx"
    """The lattice object type."""

    supports_turns: ClassVar[bool] = True
    """One ``track_particles`` a pass, which carries on from the last."""

    supports_periodic: ClassVar[bool] = True
    """By probes about the closed orbit; see :meth:`periodic_solution`."""

    supports_nsuperperiods: ClassVar[bool] = True
    """As N passes per turn."""

    supports_radiation: ClassVar[bool] = True
    """``sim.isr``, which is quantum: ``mean`` gets the excitation too."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """See :meth:`run_dynamic_aperture`."""

    supports_frequency_map: ClassVar[bool] = True
    """See :meth:`run_frequency_map`."""

    supports_programs: ClassVar[bool] = True
    """On :attr:`dicts`, which every pass is built from."""

    supports_ramp: ClassVar[bool] = True
    """The beam is re-referenced at each turn's start; strengths are normalised, so follow it."""

    program_attributes: ClassVar[dict] = {"Kicker": ("xkick", "ykick")}
    """``(horizontal, vertical)`` dict key an unqualified program sets, per ImpactX type."""

    aperture_limit: ClassVar[float] = 1.0
    """A scanned particle past this ``|x|`` or ``|y|`` [m] is lost, as Ocelot's default."""

    native_rf: ClassVar[str] = "synchronous"
    """``ShortRF`` keeps time from the reference particle."""

    otm_convention: ClassVar[str] = "x, px, y, py, zeta, delta"
    """Converted from ImpactX's ``(t, pt)``, which no rescale reaches: ``zeta = -beta0 t``, ``delta = -pt / beta0``."""

    otm_longitudinal_scale: ClassVar[int | None] = 0
    """Already Xsuite's."""

    native_time: ClassVar[tuple[str, str]] = ("t", "m")
    """``c (t - reference_time)``: positive behind the reference."""

    space_charge_modes_off: ClassVar[tuple] = ("", "false", "none", "0", "off")
    """``charge: space_charge_mode`` values that mean no space charge."""

    space_charge_step: float = 0.1
    """Metres between space-charge kicks, as Xsuite's: each thick element's ``nslice``."""

    particle_definition: str = None
    """Name of the initial particle distribution."""

    dicts: list = []
    """The line as ImpactX element dicts; see ``to_impactx``."""

    pin: Any = None
    """The input beam as :func:`~simba.Modules.Beams.impactx.write_impactx_beam` gives it."""

    beam_data: list = []
    """Beam moments at the start and after every element of the last pass: the twiss CSV's rows."""

    _periodic: Any = None
    """This run's :meth:`periodic_solution`, found once."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition

    @classmethod
    def native_time_scale(cls, beta0: float) -> float:
        """``t = c (t - reference_time)``."""
        return speed_of_light

    @property
    def field_maps(self) -> bool:
        """
        ``tracking: field_maps``: cavities with a standing-wave field map as
        thick ``RFCavity``, crested by :meth:`autophase`::

            files:
              LINE:
                code: impactx
                tracking: {field_maps: true}

        Off by default, as ImpactX pushes the beam through them with a linear
        map: RF focusing, but no RF curvature.
        """
        return bool((self.file_block.get("tracking") or {}).get("field_maps", False))

    @property
    def space_charge_mode(self) -> str:
        """``charge: space_charge_mode``, lower-cased; ``""`` if not given."""
        charge = self.file_block.get("charge") or {}
        return str(charge.get("space_charge_mode", "")).lower()

    def writeElements(self) -> None:
        """Build :attr:`dicts`, crested by :meth:`autophase` with :attr:`field_maps`."""
        self.dicts = self.section.to_impactx(
            energy=self.reference_energy,
            particle=self.global_parameters["beam"].species,
            field_maps=self.field_maps,
        )
        if self.field_maps:
            self.autophase()

    def write(self) -> None:
        """Build the lattice via :meth:`writeElements`."""
        self.writeElements()

    def preProcess(self) -> None:
        """Load the input beam, convert it to :attr:`pin` and check the collective settings."""
        super().preProcess()
        self.load_input_beam(self.get_prefix(), self.particle_definition)
        self.pin = rbf_impactx.write_impactx_beam(
            self.global_parameters["beam"], p0c=self.reference_p0c, t0=self.reference_t0
        )
        mode = self.space_charge_mode
        if mode not in self.space_charge_modes_off + ("3d",):
            warn(exceptions.SpaceChargeModeWarning(self.objectname, "ImpactX", mode, "'3d'"))
        if self.wakefield_enable and self.wakefields_and_cavity_wakefields:
            warn(f"Line '{self.objectname}': ImpactX has no structure wakefields; they are dropped.")
        if self.field_maps and self.fixed_reference and self.cavities:
            # ponytail: no fixed-reference RFCavity; thin ShortRF is the ring default
            warn(
                f"Line '{self.objectname}': a field-map RFCavity moves ImpactX's reference and is "
                "crested for the first pass only; leave field_maps off on a ring."
            )
        if self.radiation == "mean":
            warn(f"Line '{self.objectname}': ImpactX's ISR is quantum; 'mean' radiation gets its excitation too.")

    def _set_reference(self, ref) -> None:
        """Reset ``ref`` to the line's reference particle at its entrance."""
        ref.reset()
        ref.set_charge_qe(float(self.reference_charge)).set_mass_MeV(self.rest_energy / 1e6)
        ref.set_kin_energy_MeV((self.reference_energy - self.rest_energy) / 1e6)

    @property
    def _qm(self) -> float:
        """The reference's charge over mass, in ``1/eV``, as ``add_n_particles`` takes it."""
        return self.reference_charge / (self.rest_energy / 1e6) * 1e-6

    def _lattice(self, dicts: list, beta_gamma: float | None = None):
        """
        ``dicts`` as ImpactX elements; with a :attr:`fixed_reference`, each ``ShortRF`` by :meth:`_fixed_reference_kick`.

        ``beta_gamma``, the reference's now, scales each ``ChrAcc``'s ``bz`` from the
        design's, as it is a field rather than a strength and so would not follow a ramp.
        """
        if beta_gamma is not None:
            ratio = beta_gamma * self.rest_energy / self.reference_p0c
            dicts = [{**d, "bz": d["bz"] * ratio} if d["type"] == "ChrAcc" else d for d in dicts]
        elements = lattice(dicts)
        if self.fixed_reference:
            for i, d in enumerate(dicts):
                if d["type"] == "ShortRF":
                    elements[i] = self._fixed_reference_kick(d)
        return elements

    @staticmethod
    def _fixed_reference_kick(d: dict):
        """
        ``ShortRF``'s kick to the particles, leaving the reference alone: its own
        moves the reference by ``V cos(phase)``, and the magnets with it.
        """
        from impactx import elements

        k, phase = 2 * np.pi * d["freq"] / speed_of_light, np.radians(d["phase"])

        def kick(pti, ref):
            soa = pti.soa().to_xp()
            soa.real["momentum_t"][:] -= (
                d["V"] * np.cos(k * soa.real["position_t"] + phase) / ref.beta_gamma
            )

        element = elements.Programmable(name=d["name"])
        element.beam_particles = kick
        element.ref_particle = lambda ref: None
        return element

    def _track_reference(self, ref, dicts: list) -> None:
        """Track ``ref`` alone through ``dicts``."""
        if dicts:
            sim = simulation()
            sim.lattice.clear()
            sim.lattice.extend(lattice(dicts))
            sim.track_reference(ref)

    def autophase(self) -> dict:
        """
        Crest every ``RFCavity`` in :attr:`dicts` by tracking the reference
        particle alone, upstream to downstream, each with those before it crested.

        Returns
        -------
        dict
            ``{name: crest}``, the ImpactX phase of most energy gain [deg]
        """
        import impactx
        from scipy.optimize import minimize_scalar

        ref = impactx.RefPart()
        self._set_reference(ref)
        crests, done = {}, 0
        for i, d in enumerate(self.dicts):
            if d["type"] != "RFCavity":
                continue
            self._track_reference(ref, self.dicts[done:i])

            def loss(phase, d=d):
                trial = ref.copy()
                self._track_reference(trial, [{**d, "phase": phase}])
                return -trial.kin_energy_MeV

            coarse = np.arange(0.0, 360.0, 10.0)
            best = coarse[np.argmin([loss(p) for p in coarse])]
            crest = minimize_scalar(
                loss, bounds=(best - 10, best + 10), method="bounded", options={"xatol": 1e-4}
            ).x
            d["phase"] += crest - float(getattr(self.elementObjects[d["name"]], "crest", 0.0) or 0.0)
            crests[d["name"]] = crest % 360
            done = i
        return crests

    def cavity_phase(self, name: str) -> float | None:
        """A ``ShortRF``'s ``phase`` [deg] in :attr:`dicts`, which every pass is built from."""
        return next(
            (d["phase"] for d in self.dicts if d["type"] == "ShortRF" and d["name"] == name), None
        )

    def set_cavity_phase(self, name: str, phase: float) -> None:
        """Set a ``ShortRF``'s ``phase`` [deg] for the next pass."""
        for d in self.dicts:
            if d["type"] == "ShortRF" and d["name"] == name:
                d["phase"] = float(phase)

    def apply_programs(self, turn: int) -> None:
        """Set each programmed element's dict key (``program.parameter``, else :attr:`program_attributes`) for ``turn``."""
        for program in self.programs:
            matches = [d for d in self.dicts if d["name"] == program.element]
            attribute = self.program_attribute(program, matches[0]["type"] if matches else None)
            if attribute is None:
                continue
            for d in matches:
                d[attribute] = float(program.value_at(turn))

    def apply_ramp(self, turn: int) -> None:
        """
        Re-reference the beam to the ramp's momentum for ``turn``, keeping each
        particle's absolute momentum; ``t`` is a time difference at this ``s``, so stays.
        """
        p0c_new = self.ramp_p0c(turn)
        if p0c_new is None:
            return
        beam = simulation().beam
        ref = beam.ref
        p0c_old = ref.beta_gamma * self.rest_energy
        energy_old, energy_new = np.hypot(p0c_old, self.rest_energy), np.hypot(p0c_new, self.rest_energy)
        for level in range(beam.finest_level + 1):
            for pti in beam.iterator(level=level):
                real = pti.soa().to_xp().real
                real["momentum_x"][:] *= p0c_old / p0c_new
                real["momentum_y"][:] *= p0c_old / p0c_new
                real["momentum_t"][:] = (real["momentum_t"] * p0c_old + energy_new - energy_old) / p0c_new
        ref.set_kin_energy_MeV((energy_new - self.rest_energy) / 1e6)

    def _sliced(self, dicts: list) -> list:
        """``dicts`` with a kick every :attr:`space_charge_step`, for space charge or ISR."""
        # LinearMap alone cannot be sliced
        return [
            {**d, "nslice": max(1, int(np.ceil(d["ds"] / self.space_charge_step - 1e-9)))}
            if d.get("ds", 0) > 0 and d["type"] != "LinearMap"
            else d
            for d in dicts
        ]

    def run(self) -> None:
        """
        Track :attr:`pin` for :attr:`turns`, writing beams at the screens, markers,
        BPMs and end on :meth:`output_turns`, and filling :attr:`beam_data`.
        """
        import impactx

        sim = simulation()
        self._periodic = None
        space_charge = self.space_charge_mode == "3d"
        sim.space_charge = "3D" if space_charge else "false"
        sim.isr = self.radiation in ("mean", "quantum")
        sim.isr_on_ref_part = False
        slices = space_charge or sim.isr
        csr = self.csr_enable and bool(self.dipoles)
        if csr and not impactx.Config.have_fft:
            warn(f"Line '{self.objectname}': this ImpactX was built without FFT, so has no CSR.")
        sim.csr = csr and impactx.Config.have_fft
        if self.csr_bins:
            sim.csr_bins = int(self.csr_bins)
        beam = sim.beam
        beam.clear()
        self._set_reference(beam.ref)
        pin = dict(self.pin)
        beam.add_n_particles(
            pin.pop("x"), pin.pop("y"), pin.pop("t"), pin.pop("px"), pin.pop("py"), pin.pop("pt"),
            self._qm, **pin,
        )
        ids = np.sort(beam.to_df(local=True)["idcpu"].to_numpy())
        names = {d["name"] for d in self.dicts}
        s_in = self.getSValues(as_dict=True, at_entrance=True)
        points = {
            e.name: (e.physical.start.z, self.entrance_s + s_in[e.name])
            for e in self.screens_and_markers_and_bpms
            if e.name in names and e.name != self.end and self.writes_output(e.name)
        }
        end = (self.endObject.physical.end.z, self.entrance_s + self.getSValues(as_dict=True)[self.end])
        last = self.last_pass(self.turns)
        now = {}

        def before(sim):
            name = sim.tracking_element.name
            if now["record"] and name in points:
                self._write_beam(sim, ids, name, *points[name], now["name_turn"], now["turn"])

        def after(sim):
            if now["index"] == last:
                self.beam_data.append(self._moments(sim, sim.tracking_element.name, now["s"]))

        def track_pass(turn, index, name_turn, record):
            now.update(turn=turn, index=index, name_turn=name_turn, record=record, s=sim.beam.ref.s)
            if index == last:
                self.beam_data = [self._moments(sim, "_start_point", now["s"])]
            sim.lattice.clear()
            # rebuilt each pass, for any phase or program run_turns moved
            sim.lattice.extend(
                self._lattice(self._sliced(self.dicts) if slices else self.dicts, sim.beam.ref.beta_gamma)
            )
            sim.track_particles()
            if record:
                self._write_beam(sim, ids, self.end, *end, name_turn, turn)

        sim.hook["before_element"], sim.hook["after_element"] = before, after
        try:
            self.run_turns(track_pass, self.apply_ramp)
        finally:
            # the hooks hold this line; the simulation outlives it
            del sim.hook["before_element"], sim.hook["after_element"]

    def _write_beam(
        self, sim, ids, name: str, zstart: float, s: float, name_turn: int | None, turn: int
    ) -> None:
        """Write the beam as it is now, at ``name`` on ``turn``, via :meth:`write_beam_file`."""
        from copy import deepcopy

        ref = sim.beam.ref
        beam = deepcopy(self.global_parameters["beam"])
        rbf_impactx.read_impactx_beam(
            beam, sim.beam.to_df(local=True), ids,
            p0c=ref.beta_gamma * self.rest_energy,
            t_reference=self.reference_t0 + ref.t / speed_of_light,
            zstart=zstart, s=s, ref_index=self.ref_idx,
        )
        beam.turn = turn
        self.write_beam_file(beam, name, name_turn)

    def _moments(self, sim, name: str, s0: float = 0.0) -> dict:
        """One :attr:`beam_data` row, in Xsuite's twiss columns; ``s`` from ``s0``, the pass's start."""
        ref, moments = sim.beam.ref, sim.beam.beam_moments()
        p0c = ref.beta_gamma * self.rest_energy
        return {
            "name": name,
            "s": ref.s - s0,
            "betx": moments["beta_x"],
            "alfx": moments["alpha_x"],
            "bety": moments["beta_y"],
            "alfy": moments["alpha_y"],
            # ImpactX's dispersion is in pt, about -beta0 delta
            "dx": ref.beta * moments["dispersion_x"],
            "dpx": ref.beta * moments["dispersion_px"],
            "dy": ref.beta * moments["dispersion_y"],
            "dpy": ref.beta * moments["dispersion_py"],
            "mean_x": moments["mean_x"],
            "mean_y": moments["mean_y"],
            "sigma_x": moments["sigma_x"],
            "sigma_px": moments["sigma_px"],
            "sigma_y": moments["sigma_y"],
            "sigma_py": moments["sigma_py"],
            "sigma_zeta": moments["sigma_t"],
            "sigma_delta": moments["sigma_pt"] / ref.beta,
            "momentum": np.hypot(p0c, self.rest_energy) - moments["mean_pt"] * p0c - self.rest_energy,
            "emit_xn": moments["emittance_xn"],
            "emit_yn": moments["emittance_yn"],
        }

    def _track_probes(self, orbit, rows: list | None = None, dicts: list | None = None):
        """
        Track a :func:`~simba.Modules.Matrices.probe_grid` about ``orbit``, in
        ImpactX's coordinates, through one pass with no collective effects.

        Parameters
        ----------
        orbit: array-like
            ``(x, px, y, py, t, pt)``
        rows: list, optional
            Appended ``(name, coordinates)`` at the start and after each element
        dicts: list, optional
            The line; :attr:`dicts` if not given

        Returns
        -------
        numpy.ndarray
            The probes' ``6 x 13`` coordinates at the end; NaN if any was lost
        """
        from ...Modules.Matrices import probe_grid

        sim = simulation()
        sim.space_charge, sim.csr, sim.isr = "false", False, False
        sim.beam.clear()
        self._set_reference(sim.beam.ref)
        x, px, y, py, t, pt = probe_grid(orbit)
        sim.beam.add_n_particles(x, y, t, px, py, pt, self._qm, w=np.ones(13))
        columns = [f"{kind}_{u}" for u in "xyt" for kind in ("position", "momentum")]

        def coordinates(sim):
            df = sim.beam.to_df(local=True)
            if len(df) < 13:
                return np.full((6, 13), np.nan)
            return df.iloc[np.argsort(df["idcpu"].to_numpy())][columns].to_numpy().T

        if rows is not None:
            rows.append(("_start_point", coordinates(sim)))
            sim.hook["after_element"] = lambda sim: rows.append(
                (sim.tracking_element.name, coordinates(sim))
            )
        try:
            sim.lattice.clear()
            sim.lattice.extend(self._lattice(self.dicts if dicts is None else dicts))
            sim.track_particles()
        finally:
            if rows is not None:
                del sim.hook["after_element"]
        return coordinates(sim)

    def _closed_orbit(self, orbit, planes: int, dicts: list | None = None):
        """
        Newton's method on probe maps from ``orbit``, closing its first ``planes``
        coordinates and holding the rest.

        Returns
        -------
        tuple | None
            ``(orbit, matrix)``; None if a probe was lost or it did not converge
        """
        from ...Modules.Matrices import map_from_probes

        orbit = np.array(orbit, dtype=float)
        for _ in range(20):
            out, matrix = map_from_probes(self._track_probes(orbit, dicts=dicts))
            if out is None:
                return None
            residual = (out - orbit)[:planes]
            if np.max(np.abs(residual)) < 1e-12:
                return orbit, matrix
            orbit[:planes] += np.linalg.solve(np.eye(planes) - matrix[:planes, :planes], residual)
        return None

    def periodic_solution(self):
        """
        The closed orbit and one-pass map in ImpactX's coordinates, by
        :meth:`_closed_orbit`, then the probes on it after each element; once a run.
        Without RF, ``t`` never closes, so only the transverse orbit is solved.

        Returns
        -------
        tuple | None
            ``(orbit, matrix, rows)``, rows as :meth:`_track_probes`; None if a
            probe was lost or the orbit did not converge
        """
        if self._periodic is None:
            found = self._closed_orbit(np.zeros(6), 6 if self.rf_voltage else 4)
            if found is None:
                warn(f"Line '{self.objectname}': ImpactX found no closed orbit.")
                self._periodic = False
            else:
                rows = []
                self._track_probes(found[0], rows)
                self._periodic = (*found, rows)
        return self._periodic or None

    def chromaticity(self, delta: float = 1e-4):
        """
        ``dQ/ddelta`` from the tunes on the closed orbits at ``+-delta``, with the RF
        off, as Bmad's finite difference.

        Returns
        -------
        numpy.ndarray | None
            ``(chromaticity_x, chromaticity_y)``; None if either orbit was not found
        """
        from ...Modules.Matrices import fractional_tune

        dicts = [
            {**d, "V": 0.0} if d["type"] == "ShortRF" else {**d, "escale": 0.0} if d["type"] == "RFCavity" else d
            for d in self.dicts
        ]
        energy, p0c = self.reference_energy, self.reference_p0c
        tunes = []
        for sign in (1, -1):
            start = np.zeros(6)
            start[5] = (energy - np.hypot(p0c * (1 + sign * delta), self.rest_energy)) / p0c
            found = self._closed_orbit(start, 4, dicts)
            if found is None:
                return None
            tunes.append([fractional_tune(found[1], u) for u in "xy"])
        return ((np.subtract(*tunes) + 0.5) % 1 - 0.5) / (2 * delta)

    def _scan(self, x, y, px=0.0, py=0.0):
        """
        Track one particle per start for :attr:`turns`, with no collective effects.

        A particle is lost once it is gone (an aperture), not finite, or past
        :attr:`aperture_limit`.

        Returns
        -------
        tuple
            ``(survived, coordinates)``: turns each completed (:attr:`turns` if it
            survived), and ``(x, px, y, py)`` at the start of every turn and the end,
            ``4 x n x (turns + 1)``, NaN once lost
        """
        import pandas as pd

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        n, zero = len(x), np.zeros(len(x))
        sim = simulation()
        sim.space_charge, sim.csr, sim.isr = "false", False, False
        sim.beam.clear()
        self._set_reference(sim.beam.ref)
        sim.beam.add_n_particles(x, y, zero, zero + px, zero + py, zero, self._qm, w=np.ones(n))
        ids = np.sort(sim.beam.to_df(local=True)["idcpu"].to_numpy())
        columns = ["position_x", "momentum_x", "position_y", "momentum_y"]

        def coordinates():
            df = sim.beam.to_df(local=True)
            values = pd.DataFrame(df[columns].to_numpy(), index=df["idcpu"].to_numpy()).reindex(ids).to_numpy().T
            lost = ~np.isfinite(values).all(axis=0) | (np.abs(values[[0, 2]]) > self.aperture_limit).any(axis=0)
            values[:, lost] = np.nan
            return values

        survived = np.full(n, self.turns)
        tracks = np.full((4, n, self.turns + 1), np.nan)
        tracks[..., 0] = coordinates()
        sim.lattice.clear()
        sim.lattice.extend(self._lattice(self.dicts))
        for turn in range(self.turns):
            for _ in range(self.passes_per_turn):
                sim.track_particles()
            tracks[..., turn + 1] = coordinates()
            survived[np.isnan(tracks[0, :, turn + 1]) & (survived == self.turns)] = turn
            if np.isnan(tracks[0, :, turn + 1]).all():
                break
        return survived, tracks

    def run_dynamic_aperture(self) -> list:
        """
        One particle per point of :meth:`da_rays`, by :meth:`_scan`.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per start.
        """
        x, y = (np.array(v) for v in zip(*self.da_rays()))
        survived, _ = self._scan(x, y)
        self.dynamic_aperture = [(float(a), float(b), int(t)) for a, b, t in zip(x, y, survived)]
        return self.dynamic_aperture

    def run_frequency_map(self) -> list:
        """
        Tune footprint over :meth:`da_grid`, from :meth:`_scan`'s turn-by-turn coordinates.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y, diffusion)`` per surviving grid point.
        """
        xs, ys = self.da_grid()
        x, y = (np.array(v) for v in zip(*[(a, b) for b in ys for a in xs]))
        survived, tracks = self._scan(x, y)
        return self._footprint(
            (x[i], y[i], *tracks[:, i, :-1]) for i in range(len(x)) if survived[i] == self.turns
        )

    def track_reference_particle(self) -> dict:
        """
        One particle launched just off the closed orbit, sampled after each turn.

        Returns
        -------
        dict
            ``x``/``px``/``y``/``py``, each of length :attr:`turns`.
        """
        orbit = self.read_closed_orbit()
        if orbit is None:
            orbit = np.zeros(6)
        nudge = float((self.da_settings or {}).get("x_max", 1e-3)) / 100.0
        survived, tracks = self._scan([orbit[0] + nudge], [orbit[2] + nudge], orbit[1], orbit[3])
        if survived[0] < self.turns:
            warn(f"Line '{self.objectname}': the reference particle did not survive, so there is no trajectory.")
            return {}
        return dict(zip(("x", "px", "y", "py"), tracks[:, 0, 1:]))

    @property
    def _xsuite_scale(self) -> np.ndarray:
        """Each coordinate's factor to Xsuite's, to first order: ``zeta = -beta0 t``, ``delta = -pt / beta0``."""
        beta0 = self.reference_p0c / self.reference_energy
        return np.array([1, 1, 1, 1, -beta0, -1 / beta0])

    def read_one_turn_map(self):
        """:meth:`periodic_solution`'s map, in Xsuite's coordinates."""
        solution = self.periodic_solution()
        if solution is None:
            return None
        scale = self._xsuite_scale
        return scale[:, None] * solution[1] / scale[None, :]

    def read_closed_orbit(self):
        """:meth:`periodic_solution`'s orbit, in Xsuite's coordinates."""
        solution = self.periodic_solution()
        if solution is None:
            return None
        orbit = solution[0] * self._xsuite_scale
        beta0 = self.reference_p0c / self.reference_energy
        pt = solution[0][5]
        orbit[5] = np.sqrt(1 - 2 * pt / beta0 + pt**2) - 1
        return orbit

    def periodic_optics(self) -> dict | None:
        """
        The periodic twiss after each :meth:`periodic_solution` row, in Xsuite's
        columns, from each row's map: ``mux``/``muy`` in turns, dispersion in ``delta``.
        Each plane is taken as uncoupled.
        """
        from ...Modules.Matrices import map_from_probes, periodic_twiss

        solution = self.periodic_solution()
        if solution is None:
            return None
        scale = self._xsuite_scale
        maps = [scale[:, None] * map_from_probes(c)[1] / scale[None, :] for _, c in solution[2]]
        one_turn = self.read_one_turn_map()
        optics = {}
        for u, i in (("x", 0), ("y", 2)):
            start = periodic_twiss(one_turn, u)
            beta0, alpha0 = start["beta"], start["alpha"]
            block = slice(i, i + 2)
            eta0 = np.linalg.solve(np.eye(2) - one_turn[block, block], one_turn[block, 5])
            m11, m12, m21, m22 = (np.array([m[i + a, i + b] for m in maps]) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
            c, d = m11 * beta0 - m12 * alpha0, m21 * beta0 - m22 * alpha0
            optics[f"bet{u}"] = (c**2 + m12**2) / beta0
            optics[f"alf{u}"] = -(c * d + m12 * m22) / beta0
            optics[f"mu{u}"] = np.unwrap(np.arctan2(m12, c)) / (2 * np.pi)
            eta = np.array([m[block, block] @ eta0 + m[block, 5] for m in maps])
            optics[f"d{u}"], optics[f"dp{u}"] = eta[:, 0], eta[:, 1]
        return optics

    def read_optics_summary(self) -> dict:
        """The full tunes, from :meth:`periodic_optics`, and :meth:`chromaticity`."""
        optics = self.periodic_optics()
        if optics is None:
            return {}
        summary = {"tune_x_total": float(optics["mux"][-1]), "tune_y_total": float(optics["muy"][-1])}
        chromaticity = self.chromaticity()
        if chromaticity is not None:
            summary["chromaticity_x"], summary["chromaticity_y"] = (float(c) for c in chromaticity)
        return summary

    def postProcess(self) -> None:
        """Write :attr:`beam_data` as the twiss CSV; its optics periodic if :attr:`periodic`."""
        import pandas as pd

        super().postProcess()
        df = pd.DataFrame(self.beam_data)
        optics = self.periodic_optics() if self.periodic else None
        if optics is not None:
            for key, values in optics.items():
                df[key] = values
        df["s"] += self.entrance_s
        for u in "xy":
            df[f"gam{u}"] = (1 + df[f"alf{u}"] ** 2) / df[f"bet{u}"]
            if optics is None:
                # beam moments give no phase advance
                df[f"mu{u}"] = np.nan
        svals = np.array(self.getSValues(at_entrance=False)) + self.entrance_s
        df["z"] = np.interp(df["s"], svals, [a[-1] for a in self.getZValues()])
        df.to_csv(f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.impactx.csv')
