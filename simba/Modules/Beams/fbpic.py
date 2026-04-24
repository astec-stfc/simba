import numpy as np
from ..units import UnitValue
from .. import constants
import h5py
from os.path import basename

def beam_to_particles(
        self,
        simulation: "Simulation",
        boost: "BoostConverter" = None,
        zstart: float=0,
) -> "Particles":
    """
    Convert the internal beam representation to an FBPIC Particles object.
    The function `add_particle_bunch_from_arrays` is used; see `FBPIC bunch utils`_.

    .. _FBPIC bunch utils: https://github.com/fbpic/fbpic/blob/dev/fbpic/lpa_utils/bunch.py

    Parameters
    ----------
    self: :class:`~SimulationFramework.Modules.Beams.beam`
        The beam object
    simulation: FBPIC `Simulation` object
        The FBPIC `Simulation` class
    boost: FBPIC `BoostConverter` class, optional
        Lorentz-boosted frame object
    zstart: float
        Initial z-position (defaults to zero; other values not yet tested)

    Returns
    -------
    fbpic.particles.particles.Particles
        FBPIC Particles object
    """
    from fbpic.lpa_utils.bunch import add_particle_bunch_from_arrays
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
    zval = (zval - zstart)# * constants.speed_of_light
    total_npart_actual = int(self._beam.total_charge.val / self._beam.particle_charge.val[0])
    npart_actual = np.full(len(self._beam.x), abs(int(total_npart_actual / len(self._beam.x))))
    bunch = add_particle_bunch_from_arrays(
        simulation,
        self._beam.particle_charge.val[0],
        self._beam.particle_mass.val[0],
        xval,
        yval,
        zval,
        px,
        py,
        pz,
        npart_actual,
        boost=boost,
    )
    return bunch

def read_fbpic_beam_file(self, filename, z_offset=0, charge=None):
    self.code = "fbpic"
    self._beam.particle_rest_energy_eV = self.E0_eV
    self.filename = filename
    with h5py.File(filename, 'r') as f:
        namestrip = basename(filename).strip('data').strip('.h5').lstrip('0')
        if namestrip == '':
            try:
                particles = f[f"/data/0/particles/elec_bunch/"]
            except KeyError as e:
                raise KeyError(
                    f"Could not find '/data/0/particles/bunch/' in {filename}. "
                    f"Available keys: {list(f.keys())}"
                ) from e
        else:
            particles = f[f"/data/{namestrip}/particles/bunch/"]
        # assuming you have your file object as f
        # particles = f["/data/0/particles/elec_bunch/"]
        # print(f['data']['0']['particles']['elec_bunch']['momentum'].keys())
        self._beam.x = UnitValue(particles['position/x'][:], "m")
        self._beam.y = UnitValue(particles['position/y'][:], "m")
        self._beam.z = UnitValue(particles['position/z'][:] + z_offset, "m")
        self._beam.t = UnitValue(self._beam.z / constants.speed_of_light, "s")
        self._beam.px = UnitValue(particles['momentum/x'][:], "kg*m/s")
        self._beam.py = UnitValue(particles['momentum/y'][:], "kg*m/s")
        self._beam.pz = UnitValue(particles['momentum/z'][:], "kg*m/s")
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