import numpy as np
import h5py
from warnings import warn
from .. import constants
from ..units import UnitValue

emass_eV = constants.m_e * (constants.speed_of_light ** 2) / constants.elementary_charge
emass_MeV = emass_eV * 1e-6
emass_GeV = emass_eV * 1e-9

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_opal_s_positions(filename, spos, tolerance=0.1):
    file = h5py.File(filename, 'r')
    file_s_pos = [file[f"Step#{i}"].attrs["SPOS"][0] for i in range(len(file.keys()))]
    elem_indices = {}
    for name, s in spos.items():
        sval = find_nearest(file_s_pos, s)
        if abs(sval - s) < tolerance:
            elem_indices.update({name: file_s_pos.index(find_nearest(file_s_pos, s))})
        else:
            warn(f"Could not find beam output within {tolerance} tolerance for {name}")
    file.close()
    return elem_indices

def read_opal_beam_file(self, filename, step=0):
    self.filename = filename
    self["code"] = "OPAL"
    file = h5py.File(filename, 'r')
    if step == -1:
        beamdata = file[f"Step#{len(file.keys())-1}"]
    else:
        beamdata = file[f"Step#{step}"]
    try:
        if np.isclose(beamdata.attrs["MASS"], emass_GeV):
            mass = constants.m_e
        else:
            mass = constants.m_p
    except:
        mass = beamdata.attrs["TotalCharge"] / beamdata.attrs["TotalMass"] / 1e6 / constants.speed_of_light
        if np.isclose(constants.m_e, mass, atol=1e-28):
            mass = constants.m_e
        elif np.isclose(constants.m_p, mass, atol=1e-28):
            mass = constants.m_p
        else:
            warn("Could not determine if particle is electron or proton; setting electron mass.")
            mass = constants.m_e
    self._beam.particle_mass = UnitValue(
        np.full(len(beamdata["x"][()]), mass), units="kg"
    )
    # print('SDDS', self._beam["particle_mass"])
    self._beam.particle_rest_energy = UnitValue(
        (self._beam.particle_mass * constants.speed_of_light ** 2),
        units="J",
    )
    # print('SDDS', self._beam["particle_rest_energy"])
    self._beam.particle_rest_energy_eV = UnitValue(
        (self._beam.particle_rest_energy / constants.elementary_charge),
        units="eV/c",
    )
    # print('SDDS', self._beam["particle_rest_energy_eV"])
    self._beam.particle_charge = UnitValue(
        np.full(len(beamdata["x"][()]), constants.elementary_charge),
        units="C",
    )
    self._beam.x = UnitValue(beamdata["x"][()], units="m")
    self._beam.y = UnitValue(beamdata["y"][()], units="m")
    try:
        self._beam.t = UnitValue(beamdata["time"][()], units="s")
    except:
        t0 = beamdata.attrs["TIME"]
        self._beam.t = UnitValue(t0 - beamdata["z"][()]/constants.speed_of_light, units="s")
    # self._beam.z = UnitValue((np.mean(self._beam.t) - self._beam.t) * constants.speed_of_light, units="m")

    gammax = beamdata["px"][()]
    gammay = beamdata["py"][()]
    gammaz = beamdata["pz"][()]
    pfac = 0 if "MONI" in filename else constants.m_e * constants.speed_of_light
    # pfac = 0 if "generator" in filename else pfac
    self._beam.px = UnitValue(gammax * self._beam.particle_rest_energy_eV * self.q_over_c, "kg*m/s")
    self._beam.py = UnitValue(gammay * self._beam.particle_rest_energy_eV * self.q_over_c, "kg*m/s")
    self._beam.pz = UnitValue(gammaz * self._beam.particle_rest_energy_eV * self.q_over_c, "kg*m/s")
    self._beam.z = UnitValue((-1 * self._beam.Bz * constants.speed_of_light)
         * (self._beam.t - np.mean(self._beam.t)),
         units="m",
     )  # np.full(len(self.t), 0)
    # for b in ["px", "py", "pz"]:
    #     self._beam[b] += constants.m_e * constants.speed_of_light

    # if "time" in list(beamdata.keys()):
    #     self._beam.t = UnitValue(beamdata["time"][()], units="s")
    #     if "MONI" in filename:
    #         self._beam.t += UnitValue(-np.mean(self._beam.t), units="s")
    #
    #     #self._beam["z"] += np.mean(self.beam["t"]) * constants.speed_of_light
    # else:
    #     self._beam["t"] = -self._beam["z"] / constants.speed_of_light

    if "TotalCharge" in list(beamdata.attrs.keys()):
        self._beam.total_charge = UnitValue(beamdata.attrs['TotalCharge'][0], units="C")
    else:
        self._beam.total_charge = UnitValue(np.sum(beamdata["q"][()]), units="C")
    self._beam.charge = UnitValue(
        np.full(
            len(self._beam.x), self._beam.total_charge / len(self._beam.x)
        ),
        units="C"
    )
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x.val), 1), units="")
    self._beam.particle_mass = UnitValue(
        np.full(len(self._beam.x.val), constants.m_e),
        units="kg",
    )
    self._beam.particle_charge = UnitValue([-constants.elementary_charge for _ in range(len(self._beam.x.val))], units="C")
    file.close()


def write_opal_beam_file(self, filename, subz=0, emitted=False):
    """Save a text file for opal."""
    x = self.x.val
    betax_gamma = self.cpx.val * self.gamma.val / self.energy.val
    y = self.y.val
    betay_gamma = self.cpy.val * self.gamma.val / self.energy.val
    if emitted:
        z = self.t.val
    else:
        z = self.z.val - subz
    betaz_gamma = self.cpz.val * self.gamma.val / self.energy.val
    beamdata = np.transpose([x, betax_gamma, y, betay_gamma, z, betaz_gamma])
    data = np.concatenate(
        [np.array([[str(len(x)), '', '', '', '', '']]), beamdata])
    with open(filename, 'w') as f:
        for d in data:
            f.write(' '.join([str(x) for x in d]) + '\n')
