import numpy as np
from .. import constants
from ..units import UnitValue

def particle_bunch_to_beam(self, bunch, zpos=0):
    self._beam.particle_mass = UnitValue(
        np.full(len(bunch.x), constants.m_e),
        units="kg",
    )
    self._beam.charge = UnitValue(bunch.q, "C")
    self._beam.total_charge = UnitValue(sum(bunch.q), "C")
    self._beam.x = UnitValue(bunch.x, "m")
    self._beam.y = UnitValue(bunch.y, "m")
    self._beam.z = UnitValue(zpos + bunch.xi, "m")
    self._beam.t = UnitValue(self._beam.z.val / (self.Bz * constants.speed_of_light), "s")
    self.set_momenta(
        bunch.px * self.particle_rest_energy_eV,
        bunch.py * self.particle_rest_energy_eV,
        bunch.pz * self.particle_rest_energy_eV,
    )
    self._beam.nmacro = np.full(len(self._beam.x), 1)

def beam_to_particle_bunch(self, zstart=0):
    """Convert the internal beam representation to a Wake-T ParticleBunch."""
    from wake_t import ParticleBunch

    mass = self._beam.particle_mass
    if isinstance(mass, UnitValue):
        mass = mass.val
    if not isinstance(mass, float):
        mass = mass[0]
    pxval = self._beam.px.val if isinstance(self._beam.px, UnitValue) else self._beam.px
    pyval = self._beam.py.val if isinstance(self._beam.py, UnitValue) else self._beam.py
    pzval = self._beam.pz.val if isinstance(self._beam.pz, UnitValue) else self._beam.pz
    px = pxval / self.q_over_c / self.particle_rest_energy_eV.val
    py = pyval / self.q_over_c / self.particle_rest_energy_eV.val
    pz = pzval / self.q_over_c / self.particle_rest_energy_eV.val
    xval = self._beam.x.val if isinstance(self._beam.x, UnitValue) else self._beam.x
    yval = self._beam.y.val if isinstance(self._beam.y, UnitValue) else self._beam.y
    zval = self._beam.z.val if isinstance(self._beam.z, UnitValue) else self._beam.z
    qval = self._beam.charge.val if isinstance(self._beam.charge, UnitValue) else self._beam.charge
    xi = (zval - np.mean(zval))# * constants.speed_of_light
    return ParticleBunch(
        np.array(qval / constants.elementary_charge),
        x=xval,
        y=yval,
        xi=xi,
        px=px,
        py=py,
        pz=pz,
        m_species=mass,
    )
