from ...Modules import Beams as rbf
from ocelot.cpbd.physics_proc import PhysProc, SaveBeam, _logger

class SaveBeamOpenPMD(SaveBeam):

    def __init__(self, filename: str, global_parameters: dict = {}, zstart: float = 0,
                 sstart: float = None, ref_idx: int = 0):
        PhysProc.__init__(self)
        self.energy = None
        self.global_parameters = global_parameters
        self.filename = filename
        self.zstart = zstart
        # zstart is a lab z (used for the particle z/t); s is measured along the
        # reference trajectory, which is longer wherever anything upstream bends
        self.s = zstart if sstart is None else sstart
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
        # save_particle_array(filename=self.filename, p_array=p_array)