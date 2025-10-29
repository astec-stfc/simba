import os
import numpy as np
from .. import constants
from ..units import UnitValue
from ..SDDSFile import SDDSFile, SDDS_Types


def read_SDDS_beam_file(self, fileName, charge=None, ascii=False, page=-1, z0=0):
    self.reset_dicts()
    self.sddsindex += 1
    elegantObject = SDDSFile(index=None, ascii=ascii)
    elegantObject.read_file(fileName, page=page)
    elegantData = elegantObject.data
    required_keys = ["x", "y", "t", "xp", "yp", "p"]
    beamprops = {}
    for k, v in elegantData.items():
        # case handling for multiple ELEGANT runs per file
        # only extract the first run (in ELEGANT this is the fiducial run)
        if isinstance(v, np.ndarray):
            if v.ndim > 1:
                beamprops.update({k: v[0]})
            else:
                try:
                    beamprops.update({k: v})
                except Exception:
                    pass
        else:
            try:
                beamprops.update({k: np.array(v)})
            except Exception:
                pass
    for k in required_keys:
        if k not in beamprops.keys():
            raise ValueError(f"Could not find column {k} in SDDS file")
    self.filename = fileName
    self._beam.particle_mass = UnitValue(
        np.full(len(beamprops["x"]), constants.m_e), units="kg"
    )
    # print('SDDS', self._beam["particle_mass"])
    self._beam.particle_rest_energy = UnitValue(
        (self._beam.particle_mass * constants.speed_of_light**2),
        units="J",
    )
    # print('SDDS', self._beam["particle_rest_energy"])
    self._beam.particle_rest_energy_eV = UnitValue(
        (self._beam.particle_rest_energy / constants.elementary_charge),
        units="eV/c",
    )
    # print('SDDS', self._beam["particle_rest_energy_eV"])
    self._beam.particle_charge = UnitValue(
        np.full(len(beamprops["x"]), constants.elementary_charge),
        units="C",
    )
    # print('SDDS', self._beam["particle_charge"])

    self.code = "SDDS"
    self._beam.x = UnitValue(beamprops["x"], units="m")
    self._beam.y = UnitValue(beamprops["y"], units="m")
    self._beam.t = UnitValue(beamprops["t"] + z0/constants.speed_of_light, units="s")
    cp = beamprops["p"] * self.particle_rest_energy_eV
    cpz = cp / np.sqrt(beamprops["xp"] ** 2 + beamprops["yp"] ** 2 + 1)
    cpx = beamprops["xp"] * cpz
    cpy = beamprops["yp"] * cpz
    self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(cpz * self.q_over_c, units="kg*m/s")
    self._beam.z = UnitValue(z0 +
        (-1 * self._beam.Bz * constants.speed_of_light)
        * (self._beam.t - np.mean(self._beam.t)),
        units="m",
    )  # np.full(len(self.t), 0)
    if "Charge" in elegantData and len(elegantData["Charge"]) > 0:
        self._beam.set_total_charge(elegantData["Charge"][0])
    elif charge is None:
        self._beam.set_total_charge(self._beam.total_charge)
    else:
        self._beam.set_total_charge(charge)
    self._beam.nmacro = UnitValue(np.full(len(self._beam.z), 1), units="")


def write_SDDS_file(self, filename: str = None, ascii=False, xyzoffset=[0, 0, 0]):
    """Save an SDDS file using the SDDS class."""
    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].strip(".ocelot").strip(".openpmd") + ".sdds"
    xoffset = xyzoffset[0]
    yoffset = xyzoffset[1]
    zoffset = xyzoffset[2]
    self.sddsindex += 1
    x = SDDSFile(index=(self.sddsindex), ascii=ascii)

    Cnames = ["x", "xp", "y", "yp", "t", "p"]
    Ctypes = [
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
    ]
    Csymbols = ["", "x'", "", "y'", "", ""]
    Cunits = ["m", "", "m", "", "s", "m$be$nc"]
    Ccolumns = [
        np.array(self.x) - float(xoffset),
        self.xp,
        np.array(self.y) - float(yoffset),
        self.yp,
        (self.t - np.mean(self.t)) + float(zoffset) / constants.speed_of_light,
        self.cp / self.particle_rest_energy_eV,
    ]
    x.add_columns(Cnames, Ccolumns, Ctypes, Cunits, Csymbols)

    Pnames = ["pCentral", "Charge", "Particles"]
    Ptypes = [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE]
    Psymbols = ["p$bcen$n", "", ""]
    Punits = ["m$be$nc", "C", ""]
    parameterData = [
        np.mean(self.BetaGamma),
        abs(self._beam.total_charge),
        len(self.x),
    ]
    x.add_parameters(Pnames, parameterData, Ptypes, Punits, Psymbols)

    x.write_file(filename)


def set_beam_charge(self, charge):
    self._beam["total_charge"] = charge
