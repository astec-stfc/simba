from ...Modules import Beams as rbf
from ocelot.cpbd.physics_proc import PhysProc, SaveBeam, _logger


class SaveBeamOpenPMD(SaveBeam):

    def __init__(
        self,
        filename: str,
        global_parameters: dict = {},
        zstart: float = 0,
        s_start: float = None,
        ref_idx: int = 0,
    ):
        PhysProc.__init__(self)
        self.energy = None
        self.global_parameters = global_parameters
        self.filename = filename
        self.zstart = zstart
        # s_start: LAURA-resolved arc-length physical.s anchor for the running
        # `self.s` accumulator below, kept distinct from `zstart` (Cartesian,
        # used for beam.z) -- defaults to zstart when not given.
        self.s = zstart if s_start is None else s_start
        self.ref_idx = ref_idx

    def apply(self, p_array, dz):
        self.s += dz
        _logger.debug(" SaveBeam applied, dz =" + str(dz))
        rbf.ocelot.particle_array_to_beam(
            self.global_parameters["beam"],
            p_array,
            zstart=self.zstart,
            s=self.s,
            ref_index=self.ref_idx,
        )
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.filename,
        )
