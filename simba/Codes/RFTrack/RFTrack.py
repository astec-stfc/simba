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

    tws: Any = None
    """Raw RF-Track transport table (set by :func:`postProcess`), mirrors
    ``ocelotLattice.tws`` — kept as the native RF-Track result, not auto-converted
    into SIMBA's generic twiss object (no other code's ``postProcess`` does that
    either; ``framework.twiss.read_rftrack_transport_table(...)`` is the opt-in
    path a user/notebook can call separately, same as
    ``framework.twiss.read_astra_twiss_files(...)``)."""

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

        When space charge is enabled (see :func:`_space_charge_settings`), each
        element is given ``sc_nsteps`` space-charge kicks (manual §5.1.2); the
        engine/grid and cathode mirror charges are set up at track time by
        :func:`_setup_space_charge`.
        """
        from ...Modules.Beams import rftrack as rbf_rftrack

        sc = self._space_charge_settings()
        P_Q = rbf_rftrack.get_P_Q(self.global_parameters["beam"])
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

        No-op when space charge is disabled: with ``sc_nsteps == 0`` on every
        element (see :func:`write`) no space-charge kicks are applied, so a
        stale global engine from a previous section is harmless.
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
        engine = rftrack_conversion.space_charge_engine(
            npart, sample_interval=sc["sample_interval"], mirror_z=mirror_z
        )
        rft.cvars.SC_engine = engine
        if sc["cathode"]:
            # Emission tracking options (§7.4) — settable on a Lattice element
            # per RF-Track's TrackingOptions; guarded because they primarily
            # apply to time-integrated (Bunch6dT/Volume) emission tracking.
            for opt, val in (
                ("emission_nsteps", sc["emission_nsteps"]),
                ("emission_range", sc["emission_range"]),
            ):
                if hasattr(self.lat_obj, opt):
                    setattr(self.lat_obj, opt, val)

    def preProcess(self) -> None:
        """
        Read the initial particle distribution defined in
        ``file_block['input']['prefix']`` and convert it to an ``RF_Track.Bunch6d``.

        Also (re-)writes that starting beam to ``<master_subdir>/<start>.openpmd.hdf5``,
        for consistency with the per-screen/end snapshots :func:`postProcess`
        writes -- useful when ``prefix`` points outside ``master_subdir`` (e.g.
        a generator's own output directory), so every lattice's full set of
        diagnostic snapshots (start, screens, end) ends up in one place.
        """
        super().preProcess()
        from ...Modules import Beams as rbf
        from ...Modules.Beams import rftrack as rbf_rftrack

        prefix = self.get_prefix()
        self.read_input_file(prefix, self.start)
        # rbf.openpmd.write_openpmd_beam_file(
        #     self.global_parameters["beam"],
        #     f'{self.global_parameters["master_subdir"]}/{self.start}.openpmd.hdf5',
        # )
        self.pin = rbf_rftrack.beam_to_bunch6d(self.global_parameters["beam"])

    def run(self) -> None:
        """
        Track :attr:`pin` through :attr:`lat_obj`, setting :attr:`pout`.

        Sets up the global space-charge engine / cathode mirror charges first
        (see :func:`_setup_space_charge`) so the per-element ``sc_nsteps`` kicks
        applied in :func:`write` take effect.
        """
        self._setup_space_charge()
        self.pout = self.lat_obj.track(self.pin)

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
        from ...Modules.Beams import rftrack as rbf_rftrack
        from ...Modules.Twiss import rftrack as rtf_rftrack

        for screen, bunch in zip(self.screens, self.lat_obj.get_bunch_at_screens()):
            screen_beam = rbf.beam()
            rbf_rftrack.bunch6d_to_beam(
                screen_beam, bunch, zstart=screen.physical.end.z, s=screen.physical.s
            )
            rbf.openpmd.write_openpmd_beam_file(
                screen_beam,
                f'{self.global_parameters["master_subdir"]}/{screen.name}.openpmd.hdf5',
            )
        beam = self.global_parameters["beam"]
        rbf_rftrack.bunch6d_to_beam(
            beam, self.pout, zstart=self.endObject.physical.end.z, s=self.endObject.physical.s
        )
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            f'{self.global_parameters["master_subdir"]}/{self.end}.openpmd.hdf5',
        )
        columns = (
            "%S %mean_x %mean_y %beta_x %beta_y %alpha_x %alpha_y "
            "%emitt_x %emitt_y %sigma_x %sigma_y %sigma_t %mean_P"
        )
        self.tws = self.lat_obj.get_transport_table(columns)
        rtf_rftrack.save_rftrack_twiss_hdf(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.rftrack.hdf5',
            rtf_rftrack.transport_table_to_dict(
                np.asarray(self.tws), columns.split(),
                zstart=self.startObject.physical.s,
            ),
        )
