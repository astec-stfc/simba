import numpy as np
from .. import constants
from ..units import UnitValue


def read_vsim_h5_beam_file(self, filename, charge=70e-12, interval=1):
    self.reset_dicts()
    with h5py.File(filename, "r") as h5file:
        data = np.array(h5file.get("/BeamElectrons"))[1:-1:interval]
        z, y, x, cpz, cpy, cpx = data.transpose()
    self.filename = fileName
    self.code = "VSIM"
    self["longitudinal_reference"] = "z"
    cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    self._beam.x = UnitValue(x, units="m")
    self._beam.y = UnitValue(y, units="m")
    self._beam.z = UnitValue(z, units="m")
    self._beam.px = UnitValue(cpx * self.particle_mass, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.particle_mass, units="kg*m/s")
    self._beam.pz = UnitValue(cpz * self.particle_mass, units="kg*m/s")
    self._beam.t = UnitValue(
        [
            (z / (-1 * Bz * constants.speed_of_light)) for z, Bz in zip(self.z, self.Bz)
        ],
        units="s",
    )
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam.set_total_charge(charge)


def write_vsim_beam_file(self, file, normaliseT=False):
    if len(self._beam.charge) == len(self.x):
        chargevector = self._beam.charge
    else:
        chargevector = np.full(len(self.x), self._beam.total_charge / len(self.x))
    if normaliseT:
        tvector = self.t - np.mean(self.t)
        zvector = self.z - np.mean(self.z)
    else:
        tvector = self.t
        zvector = self.z
    zvector = [
        t * (1 * Bz * constants.speed_of_light) if z == 0 else z
        for z, t, Bz in zip(zvector, tvector, self.Bz)
    ]
    """ this is the VSIM array in all it's glory """
    array = np.array(
        [
            zvector,
            self.y,
            self.x,
            self.Bz * self.gamma * constants.speed_of_light,
            self.By * self.gamma * constants.speed_of_light,
            self.Bx * self.gamma * constants.speed_of_light,
        ]
    ).transpose()
    """ take the rms - if the rms is 0 set it to 1, so we don't get a divide by error """
    np.savetxt(file, array, fmt=("%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e"))
