from copy import deepcopy

from ...Modules import Beams as rbf
from ocelot.cpbd.physics_proc import PhysProc, SaveBeam, _logger


class PendingBeams(list):
    """``SaveBeamOpenPMD(pending=...)``'s list, still shared after the Navigator deep-copies its processes."""

    def __deepcopy__(self, memo):
        return self

class SaveBeamOpenPMD(SaveBeam):

    def __init__(self, filename: str, global_parameters: dict = {}, zstart: float = 0,
                 sstart: float = None, ref_idx: int = 0, beam_turn: int | None = None,
                 t_reference: float | None = None, file_turn: int | None = None,
                 start: tuple = None, pending: list | None = None):
        PhysProc.__init__(self)
        self.energy = None
        self.global_parameters = global_parameters
        self.filename = filename
        self.beam_turn = beam_turn
        self.file_turn = file_turn
        """The turn of a multi-turn file this is, or None for a single beam."""
        self.zstart = zstart
        self.s = zstart if sstart is None else sstart
        self.ref_idx = ref_idx
        self.t_reference = t_reference
        self.start = start
        self.pending = pending
        """If given, ``(self, particles)`` goes here for :meth:`write` later, instead of writing now."""

    def apply(self, p_array, dz):
        self.s += dz
        _logger.debug(" SaveBeam applied, dz =" + str(dz))
        if self.pending is not None:
            self.pending.append((self, deepcopy(p_array)))
        else:
            self.write(p_array, self.global_parameters["beam"])

    def write(self, p_array, beam):
        """Convert ``p_array`` into ``beam`` and write it to :attr:`filename`."""
        rbf.ocelot.particle_array_to_beam(
            beam,
            p_array,
            zstart=self.zstart,
            s=self.s,
            ref_index=self.ref_idx,
            t_reference=self.t_reference,
            start=self.start,
        )
        beam.turn = self.beam_turn
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            self.filename,
            turn=self.file_turn,
        )
