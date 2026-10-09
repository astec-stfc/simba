import os
import numpy as np
from .. import constants
import h5py

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def cumtrapz(
        x: list | np.ndarray = [],
        y: list | np.ndarray = []
):
    return [_trapezoid(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]

def read_cheetah_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_cheetah_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        pre, ext = os.path.splitext(filename)
        lattice_name = os.path.basename(pre)
        with h5py.File(filename, 'r') as f:
            file = f["Twiss"]
            z = file["z"][()] if "z" in file else file["s"][()]
            cp = np.sqrt(file["energy"][()] ** 2 - self.E0_eV**2)
            ke = np.array(np.sqrt(self.E0_eV ** 2 + cp ** 2) - self.E0_eV)
            gamma = 1 + ke / self.E0_eV
            betagamma = np.sqrt(gamma**2 - 1)
            ex = (
                file["projected_emittance_x"][()]
                if "projected_emittance_x" in file
                else file["emittance_x"][()]
            )
            ey = (
                file["projected_emittance_y"][()]
                if "projected_emittance_y" in file
                else file["emittance_y"][()]
            )
            beta = np.sqrt(1 - (gamma ** -2))
            self.append_columns(
                len(file["s"][()]),
                z=z,
                s=file["s"][()],
                cp=cp,
                mean_cp=cp,
                kinetic_energy=ke,
                gamma=gamma,
                p=cp * self.q_over_c,
                enx=ex * betagamma,
                ex=ex,
                eny=ey * betagamma,
                ey=ey,
                enz=0.0, ez=0.0, beta_z=0.0, alpha_z=0.0, gamma_z=0.0,
                beta_x=file["beta_x"][()],
                alpha_x=file["alpha_x"][()],
                gamma_x=(1 + file["alpha_x"][()] ** 2) / file["beta_x"][()],
                beta_y=file["beta_y"][()],
                alpha_y=file["alpha_y"][()],
                gamma_y=(1 + file["alpha_y"][()] ** 2) / file["beta_y"][()],
                sigma_x=file["sigma_x"][()],
                sigma_xp=file["sigma_px"][()],
                sigma_y=file["sigma_y"][()],
                sigma_yp=file["sigma_py"][()],
                sigma_z=file["sigma_tau"][()],
                sigma_t=file["sigma_tau"][()] / constants.speed_of_light,
                mean_x=file["mu_x"][()],
                mean_y=file["mu_y"][()],
                t=file["s"][()] / (beta * constants.speed_of_light),
                sigma_cp=file["sigma_p"][()] * cp,
                sigma_p=file["sigma_p"][()],
                mux=cumtrapz(x=z, y=1 / (file["sigma_x"][()] ** 2 / ex)),
                muy=cumtrapz(x=z, y=1 / (file["sigma_y"][()] ** 2 / ey)),
                eta_x=0.0, eta_xp=0.0, eta_y=0.0, eta_yp=0.0,
                element_name="",
                lattice_name=lattice_name,
                ecnx=file["emittance_x"][()] * betagamma,
                ecny=file["emittance_y"][()] * betagamma,
                eta_x_beam=0.0, eta_xp_beam=0.0, eta_y_beam=0.0, eta_yp_beam=0.0,
                beta_x_beam=0.0, beta_y_beam=0.0, alpha_x_beam=0.0, alpha_y_beam=0.0,
            )
