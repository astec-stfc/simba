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
        cp = elegantData["pCentral0"] * self.E0
        ke = np.array(
            (np.sqrt(self.E0**2 + cp**2) - self.E0) / constants.elementary_charge
        )
        gamma = 1 + ke / self.E0_eV
        cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
        beta = np.sqrt(1 - (gamma**-2))
        self.append_columns(
            len(z),
            z=z,
            s=elegantData["s"],
            kinetic_energy=ke,
            cp=cp,
            gamma=gamma,
            p=cp * self.q_over_c,
            enx=elegantData["enx"],
            ex=elegantData["ex"],
            eny=elegantData["eny"],
            ey=elegantData["ey"],
            beta_x=elegantData["betax"],
            alpha_x=elegantData["alphax"],
            beta_y=elegantData["betay"],
            alpha_y=elegantData["alphay"],
            sigma_x=elegantData["Sx"],
            sigma_y=elegantData["Sy"],
            sigma_xp=elegantData["Sxp"],
            sigma_yp=elegantData["Syp"],
            sigma_t=elegantData["St"],
            mean_x=elegantData["Cx"],
            mean_y=elegantData["Cy"],
            eta_x=elegantData["etax"],
            eta_xp=elegantData["etaxp"],
            eta_y=elegantData["etay"],
            eta_yp=elegantData["etayp"],
            sigma_p=elegantData["Sdelta"],
            beta_x_beam=elegantData["betaxBeam"],
            beta_y_beam=elegantData["betayBeam"],
            alpha_x_beam=elegantData["alphaxBeam"],
            alpha_y_beam=elegantData["alphayBeam"],
            ecnx=elegantData["ecnx"],
            ecny=elegantData["ecny"],
            gamma_x=(1 + elegantData["alphax"] ** 2) / elegantData["betax"],
            gamma_y=(1 + elegantData["alphay"] ** 2) / elegantData["betay"],
            enz=0.0, ez=0.0, beta_z=0.0, gamma_z=0.0, alpha_z=0.0,
            t=z / (beta * constants.speed_of_light),
            sigma_z=elegantData["St"] * (beta * constants.speed_of_light),
            sigma_cp=elegantData["Sdelta"] * cp,
            mean_cp=cp * (1 + elegantData["Cdelta"]),
            mux=elegantData["psix"] / (2 * constants.pi),
            muy=elegantData["psiy"] / (2 * constants.pi),
            element_name=elegantData["ElementName"],
            lattice_name=lattice_name,
            # ## BEAM parameters
            eta_x_beam=elegantData["s16"] / (elegantData["s6"] ** 2),
            eta_xp_beam=elegantData["s26"] / (elegantData["s6"] ** 2),
            eta_y_beam=elegantData["s36"] / (elegantData["s6"] ** 2),
            eta_yp_beam=elegantData["s46"] / (elegantData["s6"] ** 2),
        )
