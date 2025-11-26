import os
from ..SDDSFile import SDDSFile
import numpy as np
from .. import constants


def read_elegant_floor_file(
    self, filename, offset=[0, 0, 0], rotation=[0, 0, 0], reset=True
):
    if reset:
        self.reset_dicts()
    elegantObject = SDDSFile(index=(self.sddsindex))
    elegantObject.read_file(filename)
    elegantData = elegantObject.data
    setattr(
        self,
        "x",
        [np.round(x + offset[0], decimals=6) for x in elegantData["X"]],
        units="m",
    )
    setattr(
        self,
        "y",
        [np.round(y + offset[1], decimals=6) for y in elegantData["Y"]],
        units="m",
    )
    setattr(
        self,
        "z",
        [np.round(z + offset[2], decimals=6) for z in elegantData["Z"]],
        units="m",
    )
    setattr(
        self,
        "theta",
        [np.round(theta + rotation[0], decimals=6) for theta in elegantData["theta"]],
        units="radians",
    )
    setattr(
        self,
        "phi",
        [np.round(phi + rotation[1], decimals=6) for phi in elegantData["phi"]],
        units="radians",
    )
    setattr(
        self,
        "psi",
        [np.round(psi + rotation[2], decimals=6) for psi in elegantData["psi"]],
        units="radians",
    )
    xyz = list(zip(self.x, self.y, self.z))
    thetaphipsi = list(zip(self.phi, self.psi, self.theta))
    return list(zip(elegantData["ElementName"], xyz[-1:] + xyz[:-1], xyz, thetaphipsi))[
        1:
    ]


def read_elegant_twiss_files(self, filename, startS=0, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            # print('reading new file', f)
            read_elegant_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        pre, ext = os.path.splitext(filename)
        lattice_name = os.path.basename(pre)
        self.sddsindex += 1
        elegantObject = SDDSFile(index=(self.sddsindex))
        elegantObject.read_file(pre + ".flr")
        elegantObject.read_file(pre + ".sig")
        elegantObject.read_file(pre + ".twi")
        elegantObject.read_file(pre + ".cen")
        elegantData = elegantObject.data
        for k in elegantData:
            # handling for multiple elegant runs per file (e.g. error simulations)
            # by default extract only the first run (in ELEGANT this is the fiducial)
            if isinstance(elegantData[k], np.ndarray) and (elegantData[k].ndim > 1):
                elegantData[k] = elegantData[k][0]
            else:
                elegantData[k] = np.array(elegantData[k])
        z = elegantData["Z"]
        # z += self.z.val[-1] if len(self.z.val) > 0 else 0
        self.z.val = np.append(self.z.val, z)
        self.s.val = np.append(self.s.val, elegantData["s"])
        cp = elegantData["pCentral0"] * self.E0
        # self.append('cp', cp)
        ke = np.array(
            (np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge
        )
        self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
        gamma = 1 + ke / self.E0_eV
        cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
        self.cp.val = np.append(self.cp.val, cp)
        self.gamma.val = np.append(self.gamma.val, gamma)
        self.p.val = np.append(self.p.val, cp * self.q_over_c)
        self.enx.val = np.append(self.enx.val, elegantData["enx"])
        self.ex.val = np.append(self.ex.val, elegantData["ex"])
        self.eny.val = np.append(self.eny.val, elegantData["eny"])
        self.ey.val = np.append(self.ey.val, elegantData["ey"])
        self.beta_x.val = np.append(self.beta_x.val, elegantData["betax"])
        self.alpha_x.val = np.append(self.alpha_x.val, elegantData["alphax"])
        self.beta_y.val = np.append(self.beta_y.val, elegantData["betay"])
        self.alpha_y.val = np.append(self.alpha_y.val, elegantData["alphay"])
        self.sigma_x.val = np.append(self.sigma_x.val, elegantData["Sx"])
        self.sigma_y.val = np.append(self.sigma_y.val, elegantData["Sy"])
        self.sigma_xp.val = np.append(self.sigma_xp.val, elegantData["Sxp"])
        self.sigma_yp.val = np.append(self.sigma_yp.val, elegantData["Syp"])
        self.sigma_t.val = np.append(self.sigma_t.val, elegantData["St"])
        self.mean_x.val = np.append(self.mean_x.val, elegantData["Cx"])
        self.mean_y.val = np.append(self.mean_y.val, elegantData["Cy"])
        self.eta_x.val = np.append(self.eta_x.val, elegantData["etax"])
        self.eta_xp.val = np.append(self.eta_xp.val, elegantData["etaxp"])
        self.eta_y.val = np.append(self.eta_y.val, elegantData["etay"])
        self.eta_yp.val = np.append(self.eta_yp.val, elegantData["etayp"])
        self.sigma_p.val = np.append(self.sigma_p.val, elegantData["Sdelta"])
        self.beta_x_beam.val = np.append(self.beta_x_beam.val, elegantData["betaxBeam"])
        self.beta_y_beam.val = np.append(self.beta_y_beam.val, elegantData["betayBeam"])
        self.alpha_x_beam.val = np.append(
            self.alpha_x_beam.val, elegantData["alphaxBeam"]
        )
        self.alpha_y_beam.val = np.append(
            self.alpha_y_beam.val, elegantData["alphayBeam"]
        )
        self.ecnx.val = np.append(self.ecnx.val, elegantData["ecnx"])
        self.ecny.val = np.append(self.ecny.val, elegantData["ecny"])
        self.enz.val = np.append(self.enz.val, np.zeros(len(elegantData["Z"])))
        self.ez.val = np.append(self.ez.val, np.zeros(len(elegantData["Z"])))

        self.gamma_x.val = np.append(
            self.gamma_x.val, (1 + elegantData["alphax"] ** 2) / elegantData["betax"]
        )
        self.gamma_y.val = np.append(
            self.gamma_y.val, (1 + elegantData["alphay"] ** 2) / elegantData["betay"]
        )
        self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(elegantData["Z"])))
        self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(elegantData["Z"])))
        self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(elegantData["Z"])))

        beta = np.sqrt(1 - (gamma**-2))
        # print 'len(z) = ', len(z), '  len(beta) = ', len(beta)
        self.t.val = np.append(self.t.val, z / (beta * constants.speed_of_light))
        self.sigma_z.val = np.append(
            self.sigma_z.val, elegantData["St"] * (beta * constants.speed_of_light)
        )
        # self.append('sigma_cp', elegantData['Sdelta'] * cp )
        self.sigma_cp.val = np.append(
            self.sigma_cp.val, elegantData["Sdelta"] * cp / constants.elementary_charge
        )
        self.mean_cp.val = np.append(
            self.mean_cp.val, elegantData["Cdelta"] * cp / constants.elementary_charge
        )
        # print('elegant = ', (elegantData['Sdelta'] * cp / constants.elementary_charge)[-1)

        self.mux.val = np.append(self.mux.val, elegantData["psix"] / (2 * constants.pi))
        self.muy.val = np.append(self.muy.val, elegantData["psiy"] / (2 * constants.pi))

        self.element_name.val = np.append(
            self.element_name.val, elegantData["ElementName"]
        )
        self.lattice_name.val = np.append(
            self.lattice_name.val,
            np.full(len(elegantData["ElementName"]), lattice_name),
        )
        # ## BEAM parameters

        self.eta_x_beam.val = np.append(
            self.eta_x_beam.val, elegantData["s16"] / (elegantData["s6"] ** 2)
        )
        self.eta_xp_beam.val = np.append(
            self.eta_xp_beam.val, elegantData["s26"] / (elegantData["s6"] ** 2)
        )
        self.eta_y_beam.val = np.append(
            self.eta_y_beam.val, elegantData["s36"] / (elegantData["s6"] ** 2)
        )
        self.eta_yp_beam.val = np.append(
            self.eta_yp_beam.val, elegantData["s46"] / (elegantData["s6"] ** 2)
        )
