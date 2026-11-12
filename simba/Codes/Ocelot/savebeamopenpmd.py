from ...Modules import Beams as rbf
from ocelot.cpbd.physics_proc import PhysProc, SaveBeam, _logger

class SaveBeamOpenPMD(SaveBeam):

    def __init__(self, filename: str, global_parameters: dict = {}, zstart: float = 0,
                 sstart: float = None, ref_idx: int = 0, beam_turn: int | None = None,
                 t_reference: float | None = None, file_turn: int | None = None):
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

    def apply(self, p_array, dz):
        self.s += dz
        _logger.debug(" SaveBeam applied, dz =" + str(dz))
        rbf.ocelot.particle_array_to_beam(
            self.global_parameters["beam"],
            p_array,
            zstart=self.zstart,
            s=self.s,
            ref_index=self.ref_idx,
            t_reference=self.t_reference,
        )
        self.global_parameters["beam"].turn = self.beam_turn
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.filename,
            turn=self.file_turn,
        )
