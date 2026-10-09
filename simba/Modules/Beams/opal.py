import numpy as np
import h5py
from warnings import warn
from .. import constants
from ..units import UnitValue

emass_eV = constants.m_e * (constants.speed_of_light ** 2) / constants.elementary_charge
emass_MeV = emass_eV * 1e-6
emass_GeV = emass_eV * 1e-9
pmass_GeV = constants.m_p * (constants.speed_of_light ** 2) / constants.elementary_charge * 1e-9

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
    except KeyError:
        # rest energy per particle; OPAL labels TotalMass "MeV" but writes GeV
        mass_GeV = beamdata.attrs["TotalMass"][0] / abs(beamdata.attrs["TotalCharge"][0]) * constants.elementary_charge
        if np.isclose(mass_GeV, emass_GeV, rtol=1e-3):
            mass = constants.m_e
        elif np.isclose(mass_GeV, pmass_GeV, rtol=1e-3):
            mass = constants.m_p
        else:
            warn("Could not determine if particle is electron or proton; setting electron mass.")
            mass = constants.m_e
    self.set_mass_and_charge(mass, constants.elementary_charge, len(beamdata["x"][()]))
    self._beam.x = UnitValue(beamdata["x"][()], units="m")
    self._beam.y = UnitValue(beamdata["y"][()], units="m")
    try:
        self._beam.t = UnitValue(beamdata["time"][()], units="s")
    except Exception:
        t0 = beamdata.attrs["TIME"]
        self._beam.t = UnitValue(t0 - beamdata["z"][()]/constants.speed_of_light, units="s")

    gammax = beamdata["px"][()]
    gammay = beamdata["py"][()]
    gammaz = beamdata["pz"][()]
    # OPAL stores momenta as beta*gamma
    self._beam.px = UnitValue(gammax * mass * constants.speed_of_light, "kg*m/s")
    self._beam.py = UnitValue(gammay * mass * constants.speed_of_light, "kg*m/s")
    self._beam.pz = UnitValue(gammaz * mass * constants.speed_of_light, "kg*m/s")
    self._beam.z = UnitValue((-1 * self._beam.Bz * constants.speed_of_light)
         * (self._beam.t - np.mean(self._beam.t)),
         units="m",
     )  # np.full(len(self.t), 0)

    if "TotalCharge" in list(beamdata.attrs.keys()):
        self._beam.total_charge = UnitValue(beamdata.attrs['TotalCharge'][0], units="C")
    else:
        self._beam.total_charge = UnitValue(np.sum(beamdata["q"][()]), units="C")
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x.val), 1), units="")
    self._beam.set_total_charge(self._beam.total_charge)
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))
    file.close()


def write_opal_beam_file(self, filename, subz=0, emitted=False):
    """Save a text file for opal."""
    x = self.x.val
    betax_gamma = self.cpx.val * self.gamma.val / self.energy.val
    y = self.y.val
    betay_gamma = self.cpy.val * self.gamma.val / self.energy.val
    z = self.t.val if emitted else self.z.val - subz
    betaz_gamma = self.cpz.val * self.gamma.val / self.energy.val
    beamdata = np.transpose([x, betax_gamma, y, betay_gamma, z, betaz_gamma])
    data = np.concatenate(
        [np.array([[str(len(x)), '', '', '', '', '']]), beamdata])
    with open(filename, 'w') as f:
        for d in data:
            f.write(' '.join([str(x) for x in d]) + '\n')
