import sys
import os
import numpy as np
import csv
from .. import constants
from ..units import UnitValue


def read_csv_file(self, filename, delimiter=" "):
    with open(filename, "r") as f:
        data = np.array(
            [
                line
                for line in csv.reader(
                    f,
                    delimiter=delimiter,
                    quoting=csv.QUOTE_NONNUMERIC,
                    skipinitialspace=True,
                )
            ]
        )
    return data


def write_csv_file(self, filename, data):
    if sys.version_info[0] > 2:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter=" ", quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True
            )
            [writer.writerow(line) for line in data]
    else:
        with open(filename, "wb") as f:
            writer = csv.writer(
                f, delimiter=" ", quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True
            )
            [writer.writerow(line) for line in data]


def read_astra_beam_file(self, filename, normaliseZ=False, keepLost=False):
    self.reset_dicts()
    data = read_csv_file(self, filename)
    self.filename = filename
    interpret_astra_data(self, data, normaliseZ=normaliseZ, keepLost=keepLost)


def interpret_astra_data(self, data, normaliseZ=False, keepLost=False):
    if not keepLost:
        data = [d for d in data if d[-1] >= 0]
    x, y, z, cpx, cpy, cpz, clock, charge, index, status = np.transpose(data)
    zref = z[0]
    # s-position of the beam is the z-position of the reference-particle (!)
    self._beam.s = UnitValue(zref, units="m")
    self.code = "ASTRA"
    self.reference_particle = data[0]
    self.reference_particle_index = 0
    self._beam.toffset = UnitValue(1e-9 * data[0][6], units="z")
    self.longitudinal_reference = "z"
    z = self.normalise_to_ref_particle(z, subtractmean=False)
    cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
    clock = self.normalise_to_ref_particle(clock, subtractmean=True)
    self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(cpz * self.q_over_c, units="kg*m/s")
    self._beam.clock = UnitValue(1.0e-9 * clock, units="s")
    self._beam.charge = UnitValue(1.0e-9 * charge, units="C")
    self._beam.status = UnitValue(status)
    self._beam.z = UnitValue(z, units="m")
    self._beam.particle_mass = UnitValue(
        [self.mass_index[i] for i in index], units="kg"
    )
    self._beam.particle_rest_energy = UnitValue(
        self._beam.particle_mass * constants.speed_of_light**2, units="J"
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        self._beam.particle_rest_energy / constants.elementary_charge, units="eV/c"
    )
    self._beam.particle_charge = UnitValue(
        [constants.elementary_charge * self.charge_sign_index[i] for i in index],
        units="C",
    )
    # print self.Bz
    self._beam.t = UnitValue(
        [
            (
                clock
                if status == -1
                else ((z - zref) / (-1 * Bz * constants.speed_of_light))
            )
            for status, z, Bz, clock in zip(
                self._beam.status, z, self.Bz, self._beam.clock
            )
        ],
        units="s",
    )
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam.x = UnitValue(x, units="m")  # - self.xp * (self.t - np.mean(self.t))
    self._beam.y = UnitValue(y, units="m")  # - self.yp * (self.t - np.mean(self.t))
    self._beam.total_charge = UnitValue(np.sum(1.0e-9 * charge), units="C")
    self._beam.nmacro = UnitValue(
        np.array(np.array(self._beam.charge) / self._beam.particle_charge)
    )


def read_csrtrack_beam_file(self, filename):
    self.reset_dicts()
    data = self.read_csv_file(filename)
    self.code = "CSRTrack"
    self.reference_particle = data[0]
    self.reference_particle_index = 0
    self.longitudinal_reference = "z"
    z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
    z = self.normalise_to_ref_particle(z, subtractmean=False)
    cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
    self._beam.x = UnitValue(x, units="m")
    self._beam.y = UnitValue(y, units="m")
    self._beam.z = UnitValue(z, units="m")
    self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(cpz * self.q_over_c, units="kg*m/s")
    self._beam.clock = UnitValue(np.full(len(self.x), 0), units="s")
    self._beam.clock[0] = UnitValue(data[0, 0] * 1e-9, units="s")
    self._beam.status = UnitValue(np.full(len(self.x), 5))
    self._beam.particle_mass = UnitValue([constants.m_e], units="kg")
    self._beam.particle_rest_energy = UnitValue(
        self._beam.particle_mass * constants.speed_of_light**2, units="J"
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        self._beam.particle_rest_energy / constants.elementary_charge,
        units="eV/c",
    )
    self._beam.particle_charge = UnitValue([constants.elementary_charge], units="C")
    self._beam.t = UnitValue(
        self.z / (-1 * self.Bz * constants.speed_of_light), units="s"
    )  # [time if status is -1 else 0 for time, status in zip(clock, self._beam['status'])]
    self._beam.charge = UnitValue(charge, units="C")
    self._beam.total_charge = UnitValue(np.sum(self._beam.charge), units="C")


def read_pacey_beam_file(self, filename, charge=250e-12):
    self.reset_dicts()
    data = self.read_csv_file(filename, delimiter="\t")
    self.filename = filename
    self.code = "TPaceyASTRA"
    self.longitudinal_reference = "z"
    x, y, z, cpx, cpy, cpz = np.transpose(data)
    # cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    self._beam.x = UnitValue(x, units="m")
    self._beam.y = UnitValue(y, units="m")
    self._beam.z = UnitValue(z, units="m")
    self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(cpz * self.q_over_c, units="kg*m/s")
    self._beam.t = UnitValue(
        (self.z / (-1 * self.Bz * constants.speed_of_light)),
        units="s",
    )
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam.total_charge = UnitValue(charge, units="C")
    self._beam.charge = UnitValue(np.full(len(x), charge / len(x)), units="C")


def convert_csrtrackfile_to_astrafile(self, csrtrackfile, astrafile):
    data = read_csv_file(self, csrtrackfile)
    z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
    charge = -charge * 1e9
    clock0 = (data[0, 0] / constants.speed_of_light) * 1e9
    clock = np.full(len(x), 0)
    clock[0] = clock0
    index = np.full(len(x), 1)
    status = np.full(len(x), 5)
    array = np.array([x, y, z, cpx, cpy, cpz, clock, charge, index, status]).transpose()
    write_csv_file(self, astrafile, array)


def cdist(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def find_nearest_vector(self, nodes, node):
    return cdist([node], nodes).argmin()


def rms(self, x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))


def create_ref_particle(self, array, index=0, subtractmean=False):
    array[1:] = array[0] + array[1:]
    if subtractmean:
        array = array - np.mean(array)
    return array


def write_astra_beam_file(
    self,
    filename: str = None,
    index: int = None,
    status: int = 5,
    normaliseZ: bool = False,
    zoffset: float = 0.0
):
    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].strip(".ocelot").strip(".openpmd") + ".astra"
    if not isinstance(index, (list, tuple, np.ndarray)):
        if len(self._beam.charge) == len(self._beam.x):
            chargevector = 1e9 * self._beam.charge
        else:
            chargevector = np.full(
                len(self._beam.x), 1e9 * self._beam.total_charge / len(self._beam.x)
            )
    if index is not None:
        indexvector = np.full(len(self._beam.x), index)
    else:
        # print('write_astra_beam_file: index not found')
        indexvector = self._beam.particle_index
    # print('write_astra_beam_file: index =', indexvector)
    # exit()
    statusvector = (
        self._beam.status
        if hasattr(self._beam, "status") and self._beam.status is not None
        else (
            status
            if isinstance(status, (list, tuple, np.ndarray))
            else np.full(len(self._beam.x), 5)
        )
    )
    # print("write_astra", self._beam.status, statusvector)
    """ if a particle is emitting from the cathode it's z value is 0 and it's clock value is finite, otherwise z is finite and clock is irrelevant (thus zero) """
    if self.longitudinal_reference == "t":
        zvector = np.array([
            0 if status == -1 and t == 0 else z
            for status, z, t in zip(statusvector, self._beam.z, self._beam.t)
        ])
    else:
        zvector = self._beam.z
    """ if the clock value is finite, we calculate it from the z value, using Betaz """
    # clockvector = [1e9*z / (1 * Bz * constants.speed_of_light) if status == -1 and t == 0 else 1.0e9*t for status, z, t, Bz in zip(statusvector, self.z, self.t, self.Bz)]
    clockvector = 1.0e9 * self._beam.t
    """ this is the ASTRA array in all it's glory """
    array = np.array(
        [
            self._beam.x,
            self._beam.y,
            zvector,
            self._beam.cpx,
            self._beam.cpy,
            self._beam.cpz,
            clockvector,
            chargevector,
            indexvector,
            statusvector,
        ]
    ).transpose()
    if self.reference_particle is not None:
        ref_particle = self.reference_particle
        # print(f'ASTRA Write we have a reference particle! idx = {self.reference_particle_index} pz = {ref_particle[5]}')
    #     # np.insert(array, 0, ref_particle, axis=0)
    else:
        """ Offset the z position to the nominal starting position """
        array[:, 2] += zoffset
        """take the rms - if the rms is 0 set it to 1, so we don't get a divide by error"""
        rms_vector = [a if abs(a) > 0 else 1 for a in rms(self, array, axis=0)]
        """ normalise the array """
        norm_array = array / rms_vector
        """ take the meen of the normalised array """
        mean_vector = np.mean(norm_array, axis=0)
        """ find the index of the vector that is closest to the mean - if you read in an ASTRA file, this should actually return the reference particle! """
        nearest_idx = find_nearest_vector(self, norm_array, mean_vector)
        ref_particle = array[nearest_idx]
        self.reference_particle = ref_particle
        self.reference_particle_index = nearest_idx
        """ set the closest mean vector to be in position 0 in the array """
        array = np.roll(array, -1 * nearest_idx, axis=0)

    """ normalise Z to the reference particle """
    array[1:, 2] = array[1:, 2] - ref_particle[2]
    """ should we leave Z as the reference value, set it to 0, or set it to be some offset? """
    if normaliseZ is not False:
        array[0, 2] = 0
    if isinstance(normaliseZ, (int, float)):
        # print('Setting z offset', normaliseZ)
        array[0, 2] += normaliseZ
    """ normalise pz and the clock """
    # print('Mean pz = ', np.mean(array[:,5]))
    array[1:, 5] = array[1:, 5] - ref_particle[5]
    array[0, 6] = array[0, 6] + ref_particle[6]
    np.savetxt(
        filename,
        array,
        fmt=(
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%d",
            "%d",
        ),
    )
