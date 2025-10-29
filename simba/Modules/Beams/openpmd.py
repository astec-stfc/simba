import numpy as np
from pmd_beamphysics import ParticleGroup
from .. import constants
from ..units import UnitValue


def read_openpmd_beam_file(self, filename):
    self.filename = filename
    particles = ParticleGroup(h5=filename)
    self._beam.x = UnitValue(particles.x, units="m")
    self._beam.y = UnitValue(particles.y, units="m")
    self._beam.t = UnitValue(particles.t, units="s")
    self._beam.z = UnitValue(particles.z + np.mean(particles.t * constants.speed_of_light), units="m")
    self._beam.px = UnitValue(particles.px * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(particles.py * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(particles.pz * self.q_over_c, units="kg*m/s")

    self._beam.charge = UnitValue(particles.weight, units="C")
    self._beam.total_charge = UnitValue(particles.charge, units="C")
    self._beam.nmacro = UnitValue(particles.weight / constants.elementary_charge, units="")
    self.set_species(particles.species)
    self.longitudinal_reference = "t"
    # print(filename, 'weight', particles.weight)
    # print(filename, 'charge', particles.charge)
    # print(filename, 'nmacro', self._beam["nmacro"])


def write_openpmd_beam_file(self, filename):
    data = {
        "x":  np.array(self.x),
        "y":  np.array(self.y),
        "z":  np.array(self.z),
        "px": np.array(self.cpx),
        "py": np.array(self.cpy),
        "pz": np.array(self.cpz),
        "t":  np.array(self.t),
        "weight": np.array(abs(self.charge)),
        "status": np.array(self.status),
        "species": [self.species],
    }
    particles = ParticleGroup(data=data)
    try:
        particles.write(filename)
    except OSError as e:
        print(f"Error writing OpenPMD beam file {filename}: {e}")
