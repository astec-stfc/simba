import os
import easygdf
import numpy as np
from ..gdf_beam import gdf_beam
from .. import constants
from ..units import UnitValue
from warnings import warn


def write_gdf_beam_file(
    self,
    filename: str=None,
    normaliseX: bool=False,
    normaliseZ: bool=False,
    cathode: bool=False,
    charge=None,
    mass=None,
):
    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].strip(".openpmd") + ".gdf"
    q = self._beam.charge
    m = self._beam.particle_mass

    if len(self._beam.nmacro) == len(self.x):
        nmacro = abs(self._beam.nmacro)
    elif len(self._beam.charge) == len(self.x):
        nmacro = abs(self._beam.charge / q)
    else:
        nmacro = np.full(
            len(self.x),
            abs(self._beam.total_charge / constants.elementary_charge / len(self.x)),
        )
    x = (
        self.x
        if not normaliseX
        else (
            (self.x - normaliseX)
            if isinstance(normaliseX, (int, float))
            else (self.x - np.mean(self.x))
        )
    )
    z = (
        self.z
        if not normaliseZ
        else (
            (self.z - normaliseZ)
            if isinstance(normaliseZ, (int, float))
            else (self.z - np.mean(self.z))
        )
    )
    if cathode:
        z = np.full(len(self.x), 0)
    dataarray = {
        "x": x,
        "y": self.y,
        "z": z,
        "q": q,
        "m": m,
        "nmacro": nmacro,
        "GBx": self.gamma * self.Bx,
        "GBy": self.gamma * self.By,
        "GBz": self.gamma * self.Bz,
        "gamma": self.gamma,
        "t": self.t,
    }
    if cathode:
        dataarray["t"] = self.t
    easygdf.save_initial_distribution(filename, **dataarray)


def read_gdf_beam_file_object(file):
    if isinstance(file, str):
        gdfbeam = gdf_beam(file)
    elif isinstance(file, gdf_beam):
        gdfbeam = file
    else:
        raise Exception("file is not str or gdf_beam object!")
    return gdfbeam


def read_gdf_beam_file_info(self, file):
    self.reset_dicts()
    gdfbeam = gdf_beam(self, file)
    return gdfbeam


def read_gdf_beam_file(
    self,
    filename=None,
    position=None,
    time=None,
    charge=None,
    longitudinal_reference="t",
    gdfbeam=None,
):
    self.reset_dicts()
    if gdfbeam is None and filename is not None:
        gdfbeam = read_gdf_beam_file_object(filename)
    elif gdfbeam is None and filename is None:
        return None

    if position is not None:  # and (time is not None or block is not None):
        self.longitudinal_reference = "t"
        gdfbeamdata = gdfbeam.get_position(position)
        if gdfbeamdata is not None:
            time = None
        else:
            raise ValueError(f"GDF DID NOT find position {position}")
    elif position is None and time is not None:
        self.longitudinal_reference = "z"
        gdfbeamdata = gdfbeam.get_time(time)
        time = None
    elif position is None and time is None:
        try:
            gdfbeamdata = gdfbeam.positions[0]
        except KeyError:
            gdfbeam.single_position_data()
            gdfbeamdata = gdfbeam
    else:
        raise ValueError("Could not load gdfbeamdata; position or time not known!")
    self.filename = filename
    self.code = "GPT"
    if "m" in gdfbeamdata:
        self._beam.particle_mass = gdfbeamdata["m"]
    else:
        self._beam.particle_mass = np.full(
            len(gdfbeamdata["x"]), constants.electron_mass
        )
    self._beam.particle_rest_energy = UnitValue(
            self._beam.particle_mass * constants.speed_of_light**2,
        units="J",
    )
    self._beam.particle_rest_energy_eV = UnitValue(
            self._beam.particle_rest_energy / constants.elementary_charge,
        units="eV/c",
    )
    self._beam.particle_charge = UnitValue([self._beam.sign(q) for q in gdfbeamdata["q"]], units="C")

    self._beam.x = UnitValue(gdfbeamdata["x"], units="m")
    self._beam.y = UnitValue(gdfbeamdata["y"], units="m")

    if "Bx" in gdfbeamdata:
        if "G" in gdfbeamdata:
            gamma = UnitValue(gdfbeamdata["G"], units="")
        else:
            beta = np.sqrt(gdfbeamdata["Bx"]**2 + gdfbeamdata["By"]**2 + gdfbeamdata["Bz"]**2)
            gamma = UnitValue(1.0 / np.sqrt(1 - beta**2), units="")
        self._beam.px = UnitValue(
            (
                gamma
                * gdfbeamdata["Bx"]
                * self._beam.particle_rest_energy
                / constants.speed_of_light
            ),
            units="kg*m/s",
        )
        self._beam.py = UnitValue(
            (
                gamma
                * gdfbeamdata["By"]
                * self._beam.particle_rest_energy
                / constants.speed_of_light
            ),
            units="kg*m/s",
        )
        self._beam.pz = UnitValue(
            (
                gamma
                * gdfbeamdata["Bz"]
                * self._beam.particle_rest_energy
                / constants.speed_of_light
            ),
            units="kg*m/s",
        )
    elif "GBx" in gdfbeamdata:
        self._beam.px = UnitValue(
            (
                gdfbeamdata["GBx"]
                * self._beam.particle_rest_energy
                / constants.speed_of_light
            ),
            units="kg*m/s",
        )
        self._beam.py = UnitValue(
            (
                gdfbeamdata["GBy"]
                * self._beam.particle_rest_energy
                / constants.speed_of_light
            ),
            units="kg*m/s",
        )
        self._beam.pz = UnitValue(
            (
                gdfbeamdata["GBz"]
                * self._beam.particle_rest_energy
                / constants.speed_of_light
            ),
            units="kg*m/s",
        )
    else:
        raise Exception("GDF File does not have Bx or GBx!")

    if "t" in gdfbeamdata and longitudinal_reference == "t":
        self._beam.t = UnitValue(gdfbeamdata["t"], units="t")
        self._beam.z = UnitValue(
            (-1 * gdfbeamdata["Bz"] * constants.speed_of_light) * (
                    gdfbeamdata["t"] - np.mean(gdfbeamdata["t"])
            ) + gdfbeamdata["z"],
            units="m",
        )
    elif "z" in gdfbeamdata and longitudinal_reference == "z":
        self._beam.z = UnitValue(gdfbeamdata["z"], "m")
        self._beam.t = UnitValue(np.full(len(self.z), 0), units="s")
    else:
        if "t" in gdfbeamdata:
            self._beam.t = UnitValue(gdfbeamdata["t"], units="s")
            if "Bz" in gdfbeamdata:
                bz = gdfbeamdata["Bz"]
            else:
                bz = gdfbeamdata["GBz"] / self._beam.gamma
            self._beam.z = UnitValue(
                (-1 * bz * constants.speed_of_light) * (
                        gdfbeamdata["t"] - np.mean(gdfbeamdata["t"])
                ) + gdfbeamdata["z"],
                units="m",
            )
        elif "z" in gdfbeamdata:# and self.longitudinal_reference == "z":
            self._beam.z = UnitValue(gdfbeamdata["z"], units="m")
            if "Bz" in gdfbeamdata:
                bz = gdfbeamdata["Bz"]
            else:
                bz = gdfbeamdata["GBz"] / self._beam.gamma
            self._beam.t = UnitValue(
                gdfbeamdata["z"] / (
                        -1 * bz * constants.speed_of_light
                ),
                units="s",
            )
            self._beam.t[self._beam.t == np.inf] = 0

    if "q" in gdfbeamdata:
        self._beam.set_total_charge(np.sum(gdfbeamdata["q"]))
    else:
        if charge is None:
            warn("Bunch charge is zero")
            self._beam.set_total_charge(0)
        else:
            self._beam.set_total_charge(charge)

    if "nmacro" in gdfbeamdata:
        self._beam.nmacro = UnitValue(gdfbeamdata["nmacro"], units="")
    else:
        self._beam.nmacro = UnitValue(np.full(len(self.z), 1), units="")
    return gdfbeam
