import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import (
    genesis4_par_to_data,
    write_genesis4_distribution,
    write_genesis4_beam,
)
from .openpmd import read_openpmd_beam_file, write_openpmd_beam_file
from ..units import UnitValue

def read_genesis_beam_file(
        self,
        filename: str,
        zoffset: float = 0,
        steady_state: bool = False,
        bunch_length: float | None = None,
        n_slices: int | None = None,
        resample: int = None,
):
    if steady_state and any([bunch_length is None and n_slices is None]):
        raise ValueError("bunch_length or n_slices must be provided for steady-state beams.")
    data = genesis4_par_to_data(filename)
    opmdfn = filename.replace('_BEAM.par.h5', '.openpmd.hdf5')
    if steady_state:
        newdata = {
            "x": [],
            "y": [],
            "z": [],
            "px": [],
            "py": [],
            "pz": [],
            "t": [],
            "weight": [],
            "status": [],
        }
        zvals = np.linspace(-bunch_length / 2, bunch_length / 2, n_slices)
        for z in zvals:
            for k, v in data.items():
                if not k in ["species", "z"]:
                    newdata[k].append(v)
            newdata["z"].append(data["z"] + z + zoffset)
        pg = ParticleGroup(data=newdata)
    else:
        pg = ParticleGroup(data=data)
        pg.z += zoffset
    pg.write(opmdfn)
    read_openpmd_beam_file(self, opmdfn)
    if isinstance(resample, int):
        postbeam = self.Particles.kde(resample)
        self.Particles.x = UnitValue(postbeam[0], "m")
        self.Particles.y = UnitValue(postbeam[1], "m")
        self.Particles.z = UnitValue(postbeam[2], "m")
        self.Particles.px = UnitValue(postbeam[3], "kg*m/s")
        self.Particles.py = UnitValue(postbeam[4], "kg*m/s")
        self.Particles.pz = UnitValue(postbeam[5], "kg*m/s")

def write_genesis_beam_distribution(self, filename: str):
    pg = write_openpmd_beam_file(self, filename)
    write_genesis4_distribution(pg, filename.replace('openpmd', 'genesis'))

def write_genesis_beam_file(self, filename: str, n_slice: int | None = None):
    pg = write_openpmd_beam_file(self, filename)
    pg.t = np.zeros(len(pg.z))
    pg.z -= np.mean(pg.z)
    write_genesis4_beam(pg, filename.replace('openpmd', 'genesis'), n_slice=n_slice)
