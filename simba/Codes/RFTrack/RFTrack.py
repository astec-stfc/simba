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

    def write(self) -> None:
        """
        Build :attr:`lat_obj` from the LAURA section via
        ``SectionLatticeTranslator.to_rftrack()``.

        Passes the beam's reference momentum-over-charge as ``P_Q`` — required
        for correct dipole (``SBend``) bending; see
        ``Modules/Beams/rftrack.get_P_Q`` and
        ``laura.translator.conversion_rules.codes.rftrack_conversion.build_sbend``.
        """
        from ...Modules.Beams import rftrack as rbf_rftrack

        P_Q = rbf_rftrack.get_P_Q(self.global_parameters["beam"])
        self.lat_obj = self.section.to_rftrack(P_Q=P_Q)

    def preProcess(self) -> None:
        """
        Read the initial particle distribution defined in
        ``file_block['input']['prefix']`` and convert it to an ``RF_Track.Bunch6d``.
        """
        super().preProcess()
        from ...Modules.Beams import rftrack as rbf_rftrack

        prefix = self.get_prefix()
        self.read_input_file(prefix, self.start)
        self.pin = rbf_rftrack.beam_to_bunch6d(self.global_parameters["beam"])

    def run(self) -> None:
        """
        Track :attr:`pin` through :attr:`lat_obj`, setting :attr:`pout`.
        """
        self.pout = self.lat_obj.track(self.pin)

    def postProcess(self) -> None:
        """
        Convert :attr:`pout` back into SIMBA's generic beam object, and store
        the raw transport table on :attr:`tws`.

        Deliberately does **not** auto-populate a generic twiss object here —
        neither ``astraLattice`` nor ``ocelotLattice`` do that in their own
        ``postProcess`` either (twiss reading is an opt-in step a user/notebook
        does separately, e.g. ``framework.twiss.read_astra_twiss_files(...)``);
        ``framework.twiss.read_rftrack_transport_table(self.lat_obj, name)`` is
        the equivalent opt-in call for RF-Track, wired into
        ``Modules/Twiss/__init__.py``'s ``codes`` dict.
        """
        super().postProcess()
        from ...Modules.Beams import rftrack as rbf_rftrack

        beam = self.global_parameters["beam"]
        rbf_rftrack.bunch6d_to_beam(beam, self.pout)
        self.tws = self.lat_obj.get_transport_table(
            "%S %mean_x %mean_y %beta_x %beta_y %alpha_x %alpha_y "
            "%emitt_x %emitt_y %sigma_x %sigma_y %sigma_t %mean_P"
        )
