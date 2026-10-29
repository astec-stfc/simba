import os
import h5py
import numpy as np
import re
from .. import constants

def cumtrapz(x=[], y=[]):
    try:
        return [np.trapezoid(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]
    except AttributeError:
        return [np.trapz(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]

def geometric_twiss(sigma_u, sigma_pu, emit_n, corr, bg):
    """Geometric Twiss beta/alpha from OPAL's normalised stat columns."""
    cov = np.sqrt(np.clip(sigma_u**2 * sigma_pu**2 - emit_n**2, 0, None))
    beta = sigma_u**2 * bg / emit_n
    alpha = -np.sign(corr) * cov / emit_n
    return beta, alpha

def read_opal_twiss_files(self, filename, startS=0, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_opal_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = re.split(r" |\\|/", filename.split(".opal_twiss")[0])[-1]
        self.sddsindex += 1
        with h5py.File(filename, "r") as f:
            opalData = f
            z = opalData["s"][()]

            def column(name):
                """A stat column, or zeros if this file predates it."""
                return opalData[name][()] if name in opalData else np.zeros(len(z))
            bg = opalData["ref_pz"][()]
            cp = bg * self.E0
            ke = np.array(
                (np.sqrt(self.E0**2 + cp**2) - self.E0) / constants.elementary_charge
            )
            gamma = 1 + ke / self.E0_eV
            cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
            betax, alphax = geometric_twiss(
                opalData["rms_x"][()], opalData["rms_px"][()],
                opalData["emit_x"][()], opalData["xpx"][()], bg,
            )
            betay, alphay = geometric_twiss(
                opalData["rms_y"][()], opalData["rms_py"][()],
                opalData["emit_y"][()], opalData["ypy"][()], bg,
            )
            eta_x = column("Dx")
            eta_xp = column("Dxp")
            eta_y = column("Dy")
            eta_yp = column("Dyp")
            sigma_p = column("rms_ps") / bg
            beta = np.sqrt(1 - (gamma**-2))
            self.append_columns(
                len(z),
                z=z,
                s=z,
                kinetic_energy=ke,
                cp=cp,
                gamma=gamma,
                p=cp * self.q_over_c,
                enx=opalData["emit_x"][()],
                ex=opalData["emit_x"][()] / bg,
                eny=opalData["emit_y"][()],
                ey=opalData["emit_y"][()] / bg,
                beta_x=betax,
                alpha_x=alphax,
                beta_y=betay,
                alpha_y=alphay,
                sigma_x=opalData["rms_x"][()],
                sigma_y=opalData["rms_y"][()],
                # rms_p* are normalised momenta; sigma_*p is a divergence in rad.
                sigma_xp=opalData["rms_px"][()] / bg,
                sigma_yp=opalData["rms_py"][()] / bg,
                sigma_t=opalData["rms_s"][()] / constants.speed_of_light,
                mean_x=opalData["mean_x"][()],
                mean_y=opalData["mean_y"][()],
                eta_x=eta_x,
                eta_xp=eta_xp,
                eta_y=eta_y,
                eta_yp=eta_yp,
                sigma_p=sigma_p,
                beta_x_beam=betax,
                beta_y_beam=betay,
                alpha_x_beam=alphax,
                alpha_y_beam=alphay,
                ecnx=opalData["emit_x"][()],
                ecny=opalData["emit_y"][()],
                gamma_x=(1 + alphax ** 2) / betax,
                gamma_y=(1 + alphay ** 2) / betay,
                enz=0.0, ez=0.0, beta_z=0.0, gamma_z=0.0, alpha_z=0.0,
                t=z / (beta * constants.speed_of_light),
                sigma_z=opalData["rms_s"][()],
                sigma_cp=sigma_p * cp,
                mean_cp=cp,
                mux=cumtrapz(x=z, y=1 / betax),
                muy=cumtrapz(x=z, y=1 / betay),
                element_name=lattice_name,
                lattice_name=lattice_name,
                # ## BEAM parameters
                eta_x_beam=eta_x,
                eta_xp_beam=eta_xp,
                eta_y_beam=eta_y,
                eta_yp_beam=eta_yp,
            )
