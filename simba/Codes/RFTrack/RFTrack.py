"""
SIMBA RF-Track Module

Defines :class:`~simba.Codes.RFTrack.RFTrack.rftrackLattice`, which converts the
:class:`~simba.Framework_objects.frameworkObject` s defined in a
:class:`~simba.Framework_objects.frameworkLattice` into an RF-Track ``Lattice``
Python object and tracks a beam through it.

Unlike ASTRA/Elegant/GPT, RF-Track is a pure-Python in-process library (no
input-deck file, no external executable) — architecturally this class follows
the same shape as :class:`~simba.Codes.Ocelot.Ocelot.ocelotLattice`, not
``astraLattice``. See ``laura/RFTrack/PLAN.md`` ("Architecture decision") for
why there is no ``RFTrackRules.py``, no ``Codes/Generators/rftrack.py``, and no
``Executables.py``/``Executables.yaml`` entry for this code.
"""
from typing import Any

import numpy as np

from ...Framework_objects import frameworkLattice


class rftrackLattice(frameworkLattice):
    """
    Class for defining the RF-Track lattice object, used for converting the
    :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an RF-Track
    ``Lattice`` object, and for tracking a beam through it.
    """

    code: str = "rftrack"
    """String indicating the lattice object type. The class must be named
    ``rftrackLattice`` for SIMBA's introspection-based code registry
    (``Framework_lattices.py`` / ``Framework.py:159,774``) to find it."""

    lat_obj: Any = None
    """The built ``RF_Track.Lattice`` object (set by :func:`write`)."""

    pin: Any = None
    """Initial particle distribution as an ``RF_Track.Bunch6d``."""

    pout: Any = None
    """Final particle distribution as an ``RF_Track.Bunch6d``, after tracking."""

    sc_engine: Any = None
    """The global ``SpaceCharge_PIC_FreeSpace`` engine for this section (set by
    :func:`_setup_space_charge`). Held here **on purpose**: ``RF_Track.cvar.SC_engine``
    only stores a C pointer (SWIG keeps no Python reference), so without a live
    Python reference the engine is garbage-collected after ``_setup_space_charge``
    returns and ``track`` segfaults on the dangling pointer -- verified against
    RF_Track 2.6.3."""

    tws: Any = None
    """Raw RF-Track transport table (set by :func:`postProcess`), mirrors
    ``ocelotLattice.tws`` — kept as the native RF-Track result, not auto-converted
    into SIMBA's generic twiss object (no other code's ``postProcess`` does that
    either; ``framework.twiss.read_rftrack_transport_table(...)`` is the opt-in
    path a user/notebook can call separately, same as
    ``framework.twiss.read_astra_twiss_files(...)``)."""

    tracking_type: str = "auto"
    """Which RF-Track tracking environment to use for this section:

    - ``"lattice"`` (or ``"bunch6d"``): space-integration ``Lattice`` tracking a
      ``Bunch6d`` -- the default for downstream, space-charge-light sections.
    - ``"volume"`` (or ``"bunch6dt"``): time-integration ``Volume`` tracking a
      ``Bunch6dT`` -- required for a photocathode section so mirror-charge space
      charge is modelled correctly (manual §5.1.1/§7.5), analogous to ASTRA's
      ``Lmirror``.
    - ``"auto"`` (default): ``Volume`` for a cathode section, ``Lattice``
      otherwise. Overridable per-section here or via the ``charge`` settings
      block's ``tracking`` key. See :func:`_tracking_type`."""

    def _tracking_type(self) -> str:
        """
        Resolve :attr:`tracking_type` to ``"volume"`` or ``"lattice"``, honouring
        an explicit override (attribute or ``charge.tracking`` setting) and
        otherwise defaulting to ``"volume"`` for a cathode section.
        """
        explicit = str(getattr(self, "tracking_type", "auto") or "auto").lower()
        if explicit == "auto":
            charge = (self.file_block.get("charge") or {}) | (
                self.globalSettings.get("charge") or {}
            )
            explicit = str(charge.get("tracking", "auto") or "auto").lower()
        if explicit in ("volume", "bunch6dt"):
            return "volume"
        if explicit in ("lattice", "bunch6d"):
            return "lattice"
        return "volume" if self._space_charge_settings()["cathode"] else "lattice"

    def _space_charge_settings(self) -> dict:
        """
        Resolve this section's space-charge configuration from the **same**
        ``charge`` settings block ASTRA reads (``space_charge_mode``,
        ``cathode``, ``mirror_charge``, ``sample_interval``); see
        ``simba/Codes/ASTRA/ASTRA.py`` and ``laura``'s ``astra_charge``.

        RF-Track's PIC solver is intrinsically 3D, so the ASTRA ``2D``/``3D``
        distinction only toggles space charge on/off here. ``sc_nsteps`` (kicks
        per element, manual §5.1.2) and the emission options (§7.4) are extra
        RF-Track-only keys read from the same block, with sensible defaults.
        """
        charge = (self.file_block.get("charge") or {}) | (
            self.globalSettings.get("charge") or {}
        )
        mode = charge.get("space_charge_mode", False)
        enabled = mode not in (False, None, 0, "False", "None", "false", "none")
        cathode = bool(charge.get("cathode", False)) or (
            (self.file_block.get("input") or {}).get("particle_definition")
            == "initial_distribution"
        )
        return {
            "enabled": enabled,
            "cathode": cathode,
            # Mirror charges default on for a cathode section (matches ASTRA,
            # where cathode -> Lmirror), overridable via mirror_charge.
            "mirror": bool(charge.get("mirror_charge", cathode)),
            "sample_interval": int(charge.get("sample_interval", 1) or 1),
            "sc_nsteps": int(charge.get("sc_nsteps", 10) or 10),
            "emission_nsteps": int(charge.get("emission_nsteps", 10) or 10),
            "emission_range": float(charge.get("emission_range", 2.0) or 2.0),
            # Volume (time-integration) tracking options [mm/c] (§3.2, §5.1.1):
            # dt_mm integration step, sc_dt_mm space-charge kick interval,
            # tt_dt_mm transport-table sampling interval.
            "dt_mm": float(charge.get("dt_mm", 0.1) or 0.1),
            "sc_dt_mm": float(charge.get("sc_dt_mm", 1.0) or 1.0),
            "tt_dt_mm": float(charge.get("tt_dt_mm", 10.0) or 10.0),
        }

    def write(self) -> None:
        """
        Build :attr:`lat_obj` from the LAURA section via
        ``SectionLatticeTranslator.to_rftrack()``, and save it as a standalone
        Python script to ``master_subdir`` (``save=True``), mirroring
        ``ocelotLattice.write()``/``to_ocelot(save=True)``.

        Passes the beam's reference momentum-over-charge as ``P_Q`` — required
        for correct dipole (``SBend``) bending; see
        ``Modules/Beams/rftrack.get_P_Q`` and
        ``laura.translator.conversion_rules.codes.rftrack_conversion.build_sbend``.

        For ``Lattice`` tracking with space charge enabled, each element is given
        ``sc_nsteps`` space-charge kicks (manual §5.1.2). For ``Volume`` tracking
        (:func:`_tracking_type` ``== "volume"``, e.g. a cathode section) an
        RF-Track ``Volume`` is built instead and its integration/transport-table
        steps set; the engine/grid, kick interval and cathode mirror charges are
        configured at track time by :func:`_setup_space_charge`.
        """
        from ...Modules.Beams import rftrack as rbf_rftrack

        sc = self._space_charge_settings()
        P_Q = rbf_rftrack.get_P_Q(self.global_parameters["beam"])
        if self._tracking_type() == "volume":
            self.lat_obj = self.section.to_rftrack_volume(P_Q=P_Q, save=True)
            self.lat_obj.dt_mm = sc["dt_mm"]
            self.lat_obj.tt_dt_mm = sc["tt_dt_mm"]
        else:
            self.lat_obj = self.section.to_rftrack(
                P_Q=P_Q, save=True, sc_nsteps=sc["sc_nsteps"] if sc["enabled"] else 0
            )

    def _setup_space_charge(self) -> None:
        """
        Configure RF-Track's global space-charge engine for this section
        (manual §5.1.3), sizing the ``SpaceCharge_PIC_FreeSpace`` grid exactly
        as ASTRA sizes its ``&CHARGE`` grid, and — for a cathode section —
        activating mirror charges at the cathode plane (§7.5) and the
        photo-emission tracking options (§7.4).

        For ``Volume`` tracking the per-kick interval is set via ``sc_dt_mm``
        (manual §5.1.1); for ``Lattice`` tracking the kicks are per-element
        ``sc_nsteps`` applied in :func:`write`.

        No-op when space charge is disabled: with ``sc_nsteps == 0`` /
        ``sc_dt_mm`` unset no space-charge kicks are applied, so a stale global
        engine from a previous section is harmless.
        """
        from laura.translator.conversion_rules.codes import rftrack_conversion

        sc = self._space_charge_settings()
        if not sc["enabled"]:
            return
        rft = rftrack_conversion.get_rftrack()
        npart = len(self.global_parameters["beam"].x)
        mirror_z = (
            self.startObject.physical.start.z
            if (sc["cathode"] and sc["mirror"])
            else None
        )
        # Keep the engine alive on the instance (see :attr:`sc_engine`) -- a
        # bare local would be GC'd before track and segfault.
        self.sc_engine = rftrack_conversion.space_charge_engine(
            npart, sample_interval=sc["sample_interval"], mirror_z=mirror_z
        )
        # NOTE: real RF_Track exposes the settings singleton as ``cvar`` (SWIG),
        # not ``cvars`` as the manual (§5.1.3) prints -- verified against
        # RF_Track 2.6.3. Setting the wrong name silently no-ops space charge.
        rft.cvar.SC_engine = self.sc_engine
        if self._tracking_type() == "volume":
            # Volume (time integration): enable SC via the kick interval, and
            # (for a cathode) the photo-emission options (§5.1.1, §7.4). These
            # are all TrackingOptions the Volume object exposes directly.
            self.lat_obj.sc_dt_mm = sc["sc_dt_mm"]
            if sc["cathode"]:
                self.lat_obj.emission_nsteps = sc["emission_nsteps"]
                self.lat_obj.emission_range = sc["emission_range"]
        elif sc["cathode"]:
            # Lattice (space integration): emission options are guarded because
            # they primarily apply to time-integrated emission tracking.
            for opt, val in (
                ("emission_nsteps", sc["emission_nsteps"]),
                ("emission_range", sc["emission_range"]),
            ):
                if hasattr(self.lat_obj, opt):
                    setattr(self.lat_obj, opt, val)

    def preProcess(self) -> None:
        """
        Read the initial particle distribution defined in
        ``file_block['input']['prefix']`` and convert it to an ``RF_Track.Bunch6d``
        (for ``Lattice`` tracking) or ``RF_Track.Bunch6dT`` (for ``Volume``
        tracking; see :func:`_tracking_type`).

        Also (re-)writes that starting beam to ``<master_subdir>/<start>.openpmd.hdf5``,
        for consistency with the per-screen/end snapshots :func:`postProcess`
        writes -- useful when ``prefix`` points outside ``master_subdir`` (e.g.
        a generator's own output directory), so every lattice's full set of
        diagnostic snapshots (start, screens, end) ends up in one place.
        """
        super().preProcess()
        from ...Modules.Beams import rftrack as rbf_rftrack

        prefix = self.get_prefix()
        # For "initial_distribution", load from "laser.openpmd.hdf5" (like ASTRA
        # does with "laser.astra"), not the element name. This matches ASTRA's
        # convention for photocathode beams generated by a preceding generator.
        particle_def = self.start
        if (self.file_block.get("input") or {}).get("particle_definition") == "initial_distribution":
            particle_def = "laser"
        self.read_input_file(prefix, particle_def)
        beam = self.global_parameters["beam"]
        self.pin = (
            rbf_rftrack.beam_to_bunch6dt(beam)
            if self._tracking_type() == "volume"
            else rbf_rftrack.beam_to_bunch6d(beam)
        )

    def run(self) -> None:
        """
        Track :attr:`pin` through :attr:`lat_obj`, setting :attr:`pout`.
        ``lat_obj`` is either an RF-Track ``Lattice`` (``Bunch6d``) or ``Volume``
        (``Bunch6dT``); both expose ``track``.

        Sets up the global space-charge engine / cathode mirror charges first
        (see :func:`_setup_space_charge`) so the space-charge kicks take effect.
        """
        self._setup_space_charge()
        self.pout = self.lat_obj.track(self.pin)

    def _bunch_to_beam(self, beam, bunch, zstart: float, s: float) -> None:
        """
        Convert a tracked RF-Track bunch back into SIMBA's generic ``beam``,
        dispatching on the bunch's **actual** type rather than the tracking
        mode. This matters because a ``Volume`` that wraps a ``Lattice`` returns
        its own end bunch as a ``Bunch6dT`` but the snapshots at the wrapped
        Lattice's ``Screen`` elements come back as ``Bunch6d`` (verified against
        RF_Track 2.6.3) -- so the end and the screens of one Volume section need
        different converters.
        """
        from ...Modules.Beams import rftrack as rbf_rftrack

        if type(bunch).__name__ == "Bunch6dT":
            rbf_rftrack.bunch6dt_to_beam(beam, bunch, s=s)
        else:
            rbf_rftrack.bunch6d_to_beam(beam, bunch, zstart=zstart, s=s)

    def _transport_table_dict(self) -> dict:
        """
        Read this section's RF-Track transport table into a canonical
        ``{identifier: column}`` dict keyed the way
        ``Modules/Twiss/rftrack.interpret_rftrack_data`` expects (``S``,
        ``mean_x``, ``beta_x``, ``sigma_t`` ...), storing the raw table on
        :attr:`tws`.

        A ``Volume`` accepts a *different* identifier set from a ``Lattice``
        (manual Table 4.2 vs 4.1: ``%mean_Z`` not ``%S``; capitalised
        ``%mean_X``/``%sigma_X`` -- verified against RF_Track 2.6.3, which raises
        ``unknown identifier '%S'`` for a Volume). The Volume columns are
        requested and then relabelled to the shared keys so the downstream
        twiss reader/writer is identical for both. ``%mean_Z``/``%sigma_Z`` are
        [mm]; the longitudinal ``sigma_t`` slot is [mm/c], numerically equal to
        ``sigma_Z`` [mm] for the bunch length, so the value carries over.
        """
        from ...Modules.Twiss import rftrack as rtf_rftrack

        s0 = self.startObject.physical.s
        keys = [
            "S", "mean_x", "mean_y", "beta_x", "beta_y", "alpha_x", "alpha_y",
            "emitt_x", "emitt_y", "sigma_x", "sigma_y", "sigma_t", "mean_P",
        ]
        if self._tracking_type() == "volume":
            cols = (
                "%mean_Z %mean_X %mean_Y %beta_x %beta_y %alpha_x %alpha_y "
                "%emitt_x %emitt_y %sigma_X %sigma_Y %sigma_Z %mean_P"
            )
            self.tws = self.lat_obj.get_transport_table(cols)
            arr = np.asarray(self.tws)
            col = {k: arr[:, i] for i, k in enumerate(keys)}
            col["S"] = col["S"] * 1e-3 + s0  # mean_Z [mm] -> S [m] (+ arc-length)
            return col
        cols = (
            "%S %mean_x %mean_y %beta_x %beta_y %alpha_x %alpha_y "
            "%emitt_x %emitt_y %sigma_x %sigma_y %sigma_t %mean_P"
        )
        self.tws = self.lat_obj.get_transport_table(cols)
        return rtf_rftrack.transport_table_to_dict(
            np.asarray(self.tws), cols.split(), zstart=s0
        )

    def postProcess(self) -> None:
        """
        Convert :attr:`pout` back into SIMBA's generic beam object, store the
        raw transport table on :attr:`tws`, and write the beam to an openPMD
        HDF5 file so the next section's lattice can read it as its input
        (mirrors ``astraLattice.astra_to_hdf5`` /
        ``Ocelot``'s ``SaveBeamOpenPMD``) — the filename (``<end>.openpmd.hdf5``
        in ``master_subdir``) matches what
        ``frameworkLattice.read_input_file`` looks for via ``self.start`` of
        the following section.

        Deliberately does **not** auto-populate a generic *in-memory* twiss
        object here — neither ``astraLattice`` nor ``ocelotLattice`` do that
        in their own ``postProcess`` either (that's an opt-in step a
        user/notebook does separately, e.g.
        ``framework.twiss.read_astra_twiss_files(...)``); the equivalent for
        RF-Track is ``framework.twiss.read_rftrack_transport_table(self.lat_obj,
        name)`` (live, in-process) or ``framework.twiss.read_rftrack_twiss_files(...)``
        (from the file written below).

        Also writes one openPMD HDF5 snapshot per non-disabled ``screen``
        element (mirrors ASTRA/Ocelot writing a beam file at every screen).
        BPMs are deliberately excluded: RF-Track's native ``Bpm`` only exposes
        noisy X/Y readings (``get_bpm_readings()``), not a full 6D phase
        space, so there is no bunch to write for them (unlike ASTRA, which
        has no real BPM model and treats every diagnostic uniformly).

        Finally writes :attr:`tws` to ``<master_subdir>/<objectname>_twiss.rftrack.hdf5``
        via ``Modules/Twiss/rftrack.save_rftrack_twiss_hdf`` (mirrors
        ``Ocelot``'s ``<objectname>_twiss.oh5``) -- picked up automatically by
        ``Framework.save_summary_files()``'s ``Twiss_Summary.hdf5`` via
        ``Modules/Twiss/__init__.py``'s ``code_signatures``/``codes["rftrack_h5"]``,
        the same way every other code's twiss output is.
        """
        super().postProcess()
        from ...Modules import Beams as rbf
        from ...Modules.Twiss import rftrack as rtf_rftrack

        for screen, bunch in zip(self.screens, self.lat_obj.get_bunch_at_screens()):
            screen_beam = rbf.beam()
            self._bunch_to_beam(
                screen_beam, bunch, zstart=screen.physical.end.z, s=screen.physical.s
            )
            rbf.openpmd.write_openpmd_beam_file(
                screen_beam,
                f'{self.global_parameters["master_subdir"]}/{screen.name}.openpmd.hdf5',
            )
        beam = self.global_parameters["beam"]
        self._bunch_to_beam(
            beam, self.pout, zstart=self.endObject.physical.end.z, s=self.endObject.physical.s
        )
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            f'{self.global_parameters["master_subdir"]}/{self.end}.openpmd.hdf5',
        )
        rtf_rftrack.save_rftrack_twiss_hdf(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.rftrack.hdf5',
            self._transport_table_dict(),
        )
