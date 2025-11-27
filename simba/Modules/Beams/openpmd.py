import os
from pmd_beamphysics import ParticleGroup, pmd_init, particle_paths
from h5py import File
from .. import constants
from ..units import UnitValue


openpmd_coords = [
    "x",
    "y",
    "z",
    "px",
    "py",
    "pz",
    "t",
    "weight",
    "status",
]


def read_openpmd_beam_file(self, filename):
    self.filename = filename
    fname = os.path.expandvars(filename)
    h5file = File(fname, "r")
    pp = particle_paths(h5file)
    bunch_data = h5file[pp[0]]
    particles = ParticleGroup(h5=bunch_data)
    self._beam.x = UnitValue(particles.x, units="m")
    self._beam.y = UnitValue(particles.y, units="m")
    self._beam.t = UnitValue(particles.t, units="s")
    self._beam.z = UnitValue(particles.z, units="m")
    self._beam.px = UnitValue(particles.px * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(particles.py * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(particles.pz * self.q_over_c, units="kg*m/s")
    self._beam.charge = UnitValue(particles.weight, units="C")
    self._beam.total_charge = UnitValue(particles.charge, units="C")
    self._beam.nmacro = UnitValue(particles.weight / constants.elementary_charge)
    self._beam.status = UnitValue(particles.status)
    self.set_species(particles.species)
    bunch_species = bunch_data[particles.species]
    self._beam.s = UnitValue(bunch_species["s"], units="m") if "s" in bunch_species else None
    self.longitudinal_reference = "t"
    if "reference_particle" in bunch_data[particles.species]:
        ref_particle = bunch_data[particles.species]["reference_particle"]
        self.reference_particle = [ref_particle[coord][()] for coord in openpmd_coords]
        self.reference_particle_index = int(ref_particle["index"][()])
        # print(f"OpenPMD We have a reference particle idx = {self.reference_particle_index} pz = {self.reference_particle[5]}")
    else:
        self.reference_particle = None
        self.reference_particle_index = None
    h5file.close()


def write_openpmd_beam_file(
    self,
    filename,
    pos=[0, 0, 0],
    toffset=0,
):
    xoffset = pos[0]
    yoffset = pos[1]
    zoffset = pos[2]
    fname = os.path.expandvars(filename)
    h5file = File(fname, "w")
    pmd_init(h5file, basePath="/", particlesPath="particles")
    h5file_particles = h5file.create_group("particles")
    data = {
        "x": self.x + UnitValue(xoffset, units="m"),
        "y": self.y + UnitValue(yoffset, units="m"),
        "z": self.z + UnitValue(zoffset, units="m"),
        "px": self.cpx,
        "py": self.cpy,
        "pz": self.cpz,
        "t": self.t + UnitValue(toffset, units="s"),
        "weight": abs(self.charge),
        "status": self._beam.status,
        "species": [self.species],
    }
    particles = ParticleGroup(data=data)
    particles.write(h5file_particles)
    h5file_species = h5file_particles[self.species]
    if self.s is not None:
        h5file_species["s"] = self.s
    if hasattr(self, "reference_particle") and self.reference_particle is not None:
        write_openpmd_reference_particle(self, h5file_species)
    h5file.close()


def write_openpmd_reference_particle(self, h5: File):
    ref_particle = self.reference_particle
    h5file_reference_particle = h5.create_group("reference_particle")
    for i, coord in enumerate(openpmd_coords):
        h5file_reference_particle[coord] = UnitValue(ref_particle[i])
    h5file_reference_particle['index'] = int(self.reference_particle_index)
    # print(f"OpenPMD Saving reference particle idx = {self.reference_particle_index} z = {self.reference_particle[2]}")
