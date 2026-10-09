import os
import numpy as np
import pandas as pd
from .. import constants


def read_xsuite_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_xsuite_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        if "csv" not in filename:
            raise ValueError("Only csv files are supported for xsuite twiss files.")
        lattice_name = os.path.basename(filename).split(".")[0]
        df = pd.read_csv(filename)
        interpret_xsuite_data(self, lattice_name, df)


def interpret_xsuite_data(self, lattice_name, fdat):
    ke = np.array(fdat["momentum"])
    gamma = 1 + ke / self.E0_eV
    cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
    bg = cp / self.E0_eV
    beta = np.sqrt(1 - (gamma**-2))
    self.append_columns(
        len(fdat["s"]),
        z=np.array(fdat["z"] if "z" in fdat else fdat["s"]),
        s=np.array(fdat["s"]),
        kinetic_energy=ke,
        cp=cp,
        gamma=gamma,
        p=cp * self.q_over_c,
        enx=fdat["emit_xn"],
        ex=fdat["emit_xn"] / bg,
        eny=fdat["emit_yn"],
        ey=fdat["emit_yn"] / bg,
        enz=0.0, ez=0.0, beta_z=0.0, gamma_z=0.0, alpha_z=0.0,
        beta_x=fdat["betx"],
        alpha_x=fdat["alfx"],
        gamma_x=fdat["gamx"],
        beta_y=fdat["bety"],
        alpha_y=fdat["alfy"],
        gamma_y=fdat["gamy"],
        sigma_x=fdat["sigma_x"],
        sigma_y=fdat["sigma_y"],
        sigma_xp=fdat["sigma_px"],
        sigma_yp=fdat["sigma_py"],
        sigma_t=fdat["sigma_zeta"] / constants.speed_of_light,
        mean_x=fdat["mean_x"],
        mean_y=fdat["mean_y"],
        t=fdat["s"] / (beta * constants.speed_of_light),
        sigma_z=fdat["sigma_zeta"],
        sigma_cp=fdat["sigma_delta"] * cp,
        mean_cp=cp,
        sigma_p=fdat["sigma_delta"],
        mux=fdat["mux"],
        muy=fdat["muy"],
        eta_x=fdat["dx"],
        eta_xp=fdat["dpx"],
        eta_y=fdat["dy"],
        eta_yp=fdat["dpy"],
        element_name=fdat["name"],
        lattice_name=lattice_name,
        ecnx=fdat["emit_xn_corrected"] if "emit_xn_corrected" in fdat else fdat["emit_xn"],
        ecny=fdat["emit_yn_corrected"] if "emit_yn_corrected" in fdat else fdat["emit_yn"],
        eta_x_beam=fdat["dx"],
        eta_xp_beam=fdat["dpx"],
        eta_y_beam=fdat["dy"],
        eta_yp_beam=fdat["dpy"],
        beta_x_beam=fdat["betx"],
        beta_y_beam=fdat["bety"],
        alpha_x_beam=fdat["alfx"],
        alpha_y_beam=fdat["alfy"],
    )
