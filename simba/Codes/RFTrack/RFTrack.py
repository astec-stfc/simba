"""
SIMBA RF-Track module: builds an RF-Track ``Lattice`` (or, from a cathode, a ``Volume``)
through LAURA and tracks the beam through it in-process. See `RF-Track`_.

    .. _RF-Track: https://gitlab.cern.ch/rf-track
"""
import math
from typing import Any
from warnings import warn

import numpy as np

from ... import exceptions
from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules.Beams import rftrack as rbf_rftrack


class rftrackLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` built and tracked by RF-Track.

    A beam starting mid-line is tracked in ``s`` through a ``Lattice``. A cathode beam
    (``laser``) is emitted and tracked in time through a ``Volume``, every element placed
    in 3D; its cavities are phased and its magnets scaled on a reference particle
    by ``autophase_volume``.
    """

    code: str = "rftrack"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    lattice: Any | None = None
    """The lattice as an ``RF_Track.Lattice``, or ``RF_Track.Volume`` from a cathode"""

    pin: Any | None = None
    """Incoming beam as an ``RF_Track.Bunch6d`` (``Bunch6dT`` from a cathode)"""

    pout: Any | None = None
    """Outgoing beam as an ``RF_Track.Bunch6d``"""

    space_charge_step: float = 0.1
    """Path length between space-charge kicks [m], when ``charge: space_charge_mode`` is ``3D``."""

    time_step: float = 0.1
    """``Volume`` time step [mm/c]."""

    integrator_tolerance: float = 1e-6
    """``Volume`` integrator's absolute tolerance (``odeint_epsabs``). RF-Track's default,
    1e-3, kicks identical particles through a hard-edged magnet differently."""

    space_charge_time_step: float = 1.0
    """``Volume`` time between space-charge kicks [mm/c], with ``space_charge_mode: 3D``."""
    wake_time_step: float = 10.0
    """``Volume`` time between wake kicks [mm/c]; a ``Volume`` applies no wake without it."""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition

    @property
    def space_charge_mode(self) -> str:
        """``charge: space_charge_mode``, lower-cased; ``""`` if not given."""
        charge = (self.file_block or {}).get("charge") or {}
        return str(charge.get("space_charge_mode", "")).lower()

    @property
    def space_charge(self) -> bool:
        """True if the file block asks for 3D space charge."""
        return self.space_charge_mode == "3d"

    @property
    def volume(self) -> bool:
        """True from a cathode, which is tracked through a ``Volume``."""
        return self.particle_definition == "laser"

    def writeElements(self) -> bool:
        """
        Build :attr:`lattice` from the section, and save it as ``<section>.py`` in `master_subdir`.

        Returns
        -------
        bool
            Always True
        """
        if self.volume:
            self.lattice = self.section.to_rftrack_volume(save=True)
        else:
            self.lattice = self.section.to_rftrack(save=True)
        return True

    def write(self) -> None:
        """Build :attr:`lattice`; see :meth:`writeElements`."""
        self.writeElements()

    def preProcess(self) -> None:
        """Load the input beam and convert it to :attr:`pin`."""
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.load_input_beam(prefix, self.particle_definition)
        beam = self.global_parameters["beam"]
        if self.volume:
            self.pin = rbf_rftrack.write_rftrack_cathode_bunch(beam, self.ref_idx)
        else:
            self.pin = rbf_rftrack.write_rftrack_bunch(beam)
        mode = self.space_charge_mode
        if mode not in ("", "false", "none", "0", "off", "3d"):
            warn(exceptions.SpaceChargeModeWarning(self.objectname, "RF-Track", mode, "'3d'"))

    def run(self) -> None:
        """Phase the cavities (and scale the magnets) to :attr:`pin`, then track it to :attr:`pout`."""
        if self.volume:
            self._run_volume()
            return
        self.lattice.autophase(self.pin)
        if self.space_charge:
            import RF_Track as rft
            from laura.translator.conversion_rules.codes.rftrack_conversion import (
                space_charge_engine,
            )

            # RF-Track 2.7.4 tracks with its global engine; Lattice.set_sc_engine is ignored
            rft.cvar.SC_engine = space_charge_engine(self.pin.size())
            length = self.lattice.get_length()
            self.lattice.distribute_sc_nsteps(max(1, math.ceil(length / self.space_charge_step)))
        self.pout = self.lattice.track(self.pin)

    def _run_volume(self) -> None:
        """Phase and scale the ``Volume`` on a reference particle emitted on axis at ``t = 0``
        with the beam's mean momentum, then track :attr:`pin`."""
        import RF_Track as rft
        from laura.translator.conversion_rules.codes.rftrack_conversion import (
            autophase_volume,
            space_charge_engine,
        )

        options = rft.TrackingOptions()
        options.dt_mm = self.time_step
        options.odeint_epsabs = self.integrator_tolerance
        options.cfx_dt_mm = self.wake_time_step
        mass, charge, momentum = self.pin.get_phase_space("%m %Q %Pc").mean(axis=0)
        reference = rft.Bunch6dT(np.array([[0, 0, 0, 0, 0, momentum, mass, charge, 1.0, 0.0]]))
        autophase_volume(self.lattice, reference, options)
        if self.space_charge:
            rft.cvar.SC_engine = space_charge_engine(self.pin.size(), mirror_z=0.0)
            options.sc_dt_mm = self.space_charge_time_step
        self.lattice.track(self.pin, options)
        self.pout = self.lattice.get_bunch_at_s1()

    def _write_beam(self, bunch, name: str, z: float) -> rbf.beam:
        """Write ``bunch`` as ``<name>.openpmd.hdf5``, with its reference time at ``z``."""
        beam = rbf.beam()
        rbf_rftrack.interpret_rftrack_bunch(
            beam, bunch, zstart=z, s=bunch.get_S(), ref_index=self.ref_idx
        )
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            f'{self.global_parameters["master_subdir"]}/{self.output_basename(name)}.openpmd.hdf5',
        )
        return beam

    def postProcess(self) -> None:
        """Write the beam at every screen and at the end as openPMD; the end's becomes the framework beam."""
        screens = [s.get_name() for s in self.lattice.get_screens()]
        for name, bunch in zip(screens, self.lattice.get_bunch_at_screens()):
            if name in (self.start, self.end) or name not in self.elementObjects or not bunch.size():
                continue
            self._write_beam(bunch, name, self.elementObjects[name].physical.middle.z)
        self.global_parameters["beam"] = self._write_beam(
            self.pout, self.end, self.endObject.physical.end.z
        )
