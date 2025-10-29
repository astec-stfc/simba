import numpy as np
from .. import constants
from ..units import UnitValue
import h5py
from os.path import basename

def particle_bunch_to_beam(self, bunch, zpos=0):
    self._beam.particle_mass = UnitValue(
        np.full(len(bunch.x), constants.m_e),
        units="kg",
    )
    self._beam.charge = UnitValue(bunch.q, "C")
    self._beam.total_charge = UnitValue(sum(bunch.q), "C")
    # self._beam.gamma = UnitValue(
    #     np.sqrt(
    #         1 + (bunch.px ** 2 + bunch.py ** 2 + bunch.pz ** 2)
    #     ),
    #     ""
    # )
    self._beam.x = UnitValue(bunch.x, "m")
    self._beam.y = UnitValue(bunch.y, "m")
    self._beam.z = UnitValue(zpos + bunch.xi, "m")
    self._beam.t = UnitValue(self._beam.z.val / (self.Bz * constants.speed_of_light), "s")
    self._beam.px = UnitValue(bunch.px * self.particle_rest_energy_eV * self.q_over_c, "kg*m/s")
    self._beam.py = UnitValue(bunch.py * self.particle_rest_energy_eV * self.q_over_c, "kg*m/s")
    self._beam.pz = UnitValue(bunch.pz * self.particle_rest_energy_eV * self.q_over_c, "kg*m/s")
    self._beam.nmacro = np.full(len(self._beam.x), 1)

def beam_to_particle_bunch(self, zstart=0):
    """Convert the internal beam representation to a Wake-T ParticleBunch."""
    from wake_t import ParticleBunch

    mass = self._beam.particle_mass
    if isinstance(mass, UnitValue):
        mass = mass.val
    if not isinstance(mass, float):
        mass = mass[0]
    # charge = self._beam.get("charge", np.full(len(self.x), -constants.elementary_charge)).val
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
    bunch = ParticleBunch(
        np.array(qval / constants.elementary_charge),
        x=xval,
        y=yval,
        xi=xi,
        px=px,
        py=py,
        pz=pz,
        m_species=mass,
        # prop_distance=zstart
    )
    return bunch

def read_wake_t_beam_file(self, filename, z_offset=0, charge=None):
    self.code = "waket"
    self._beam.particle_rest_energy_eV = self.E0_eV
    self.filename = filename
    with h5py.File(filename, 'r') as f:
        namestrip = basename(filename).strip('data').strip('.h5').lstrip('0')
        if namestrip == '':
            try:
                particles = f[f"/data/0/particles/electron/"]
            except KeyError as e:
                raise KeyError(
                    f"Could not find '/data/0/particles/electron/' in {filename}. "
                    f"Available keys: {list(f.keys())}"
                ) from e
        else:
            particles = f[f"/data/{namestrip}/particles/electron/"]
        # assuming you have your file object as f
        # particles = f["/data/0/particles/elec_bunch/"]
        # print(f['data']['0']['particles']['elec_bunch']['momentum'].keys())
        self._beam.x = UnitValue(particles['position/x'][:], "m")
        self._beam.y = UnitValue(particles['position/y'][:], "m")
        self._beam.z = UnitValue(f['position/z'][:] + z_offset, "m")
        self._beam.t = UnitValue(self._beam.z / constants.speed_of_light, "s")
        self._beam.px = UnitValue(f['momentum/px'][:], "kg*m/s")
        self._beam.py = UnitValue(f['particles/py'][:], "kg*m/s")
        self._beam.pz = UnitValue(f['particles/pz'][:], "kg*m/s")
        self._beam.nmacro = np.full(len(self._beam.x), 1)
        self._beam.particle_mass = UnitValue(
            np.full(len(self._beam.x), constants.m_e),
            units="kg",
        )
        if charge is not None:
            self._beam.charge = UnitValue(np.full(len(self._beam["x"]), charge / len(self._beam["x"])), "C")
            self._beam.total_charge = UnitValue(charge, "C")
        else:
            if not hasattr(self, "charge"):
                raise AttributeError("Bunch charge must be part of the beam object or provided as an argument.")
