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
        fdat = {}
        # print("loading ocelot twiss file", filename)
        df = pd.read_csv(filename)
        interpret_xsuite_data(self, lattice_name, df)


def interpret_xsuite_data(self, lattice_name, fdat):
    self.z.val = np.append(self.z.val, np.array(fdat["s"]))
    self.s.val = np.append(self.s.val, np.array(fdat["s"]))
    E = fdat["momentum"]
    ke = E - self.E0_eV
    gamma = E / self.E0_eV
    cp = E
    # self.append('cp', cp)
    self.cp.val = np.append(self.cp.val, cp / constants.elementary_charge)
    ke = np.array(
        (np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge
    )
    self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
    gamma = 1 + ke / self.E0_eV
    self.gamma.val = np.append(self.gamma.val, gamma)
    self.p.val = np.append(self.p.val, cp * self.q_over_c)
    self.enx.val = np.append(self.enx.val, fdat["emit_xn"])
    self.ex.val = np.append(self.ex.val, fdat["emit_xn"] / gamma)
    self.eny.val = np.append(self.eny.val, fdat["emit_yn"])
    self.ey.val = np.append(self.ey.val, fdat["emit_yn"] / gamma)
    self.enz.val = np.append(self.enz.val, np.zeros(len(fdat["s"])))
    self.ez.val = np.append(self.ez.val, np.zeros(len(fdat["s"])))
    self.beta_x.val = np.append(self.beta_x.val, fdat["betx"])
    self.alpha_x.val = np.append(self.alpha_x.val, fdat["alfx"])
    self.gamma_x.val = np.append(self.gamma_x.val, fdat["gamx"])
    self.beta_y.val = np.append(self.beta_y.val, fdat["bety"])
    self.alpha_y.val = np.append(self.alpha_y.val, fdat["alfy"])
    self.gamma_y.val = np.append(self.gamma_y.val, fdat["gamy"])
    self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(fdat["s"])))
    self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(fdat["s"])))
    self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(fdat["s"])))
    self.sigma_x.val = np.append(self.sigma_x.val, fdat["sigma_x"])
    self.sigma_y.val = np.append(self.sigma_y.val, fdat["sigma_y"])
    self.sigma_xp.val = np.append(self.sigma_xp.val, fdat["sigma_px"])
    self.sigma_yp.val = np.append(self.sigma_yp.val, fdat["sigma_py"])
    self.sigma_t.val = np.append(self.sigma_t.val, fdat["sigma_zeta"] / constants.speed_of_light)
    self.mean_x.val = np.append(self.mean_x.val, fdat["mean_x"])
    self.mean_y.val = np.append(self.mean_y.val, fdat["mean_y"])
    beta = np.sqrt(1 - (gamma**-2))
    self.t.val = np.append(self.t.val, fdat["s"] / (beta * constants.speed_of_light))
    self.sigma_z.val = np.append(self.sigma_z.val, fdat["sigma_zeta"])
    # self.append('sigma_cp', elegantData['Sdelta'] * cp )
    self.sigma_cp.val = np.append(self.sigma_cp.val, fdat["sigma_delta"])
    self.mean_cp.val = np.append(self.mean_cp.val, cp)
    # print('elegant = ', (elegantData['Sdelta'] * cp / constants.elementary_charge)[-1)
    self.sigma_p.val = np.append(self.sigma_p.val, fdat["sigma_delta"])
    self.mux.val = np.append(self.mux.val, fdat["mux"])
    self.muy.val = np.append(self.muy.val, fdat["muy"])
    self.eta_x.val = np.append(self.eta_x.val, fdat["dx"])
    self.eta_xp.val = np.append(self.eta_xp.val, fdat["dpx"])
    self.eta_y.val = np.append(self.eta_y.val, fdat["dy"])
    self.eta_yp.val = np.append(self.eta_yp.val, fdat["dpy"])
    self.element_name.val = np.append(self.element_name.val, fdat["name"])
    self.lattice_name.val = np.append(
        self.lattice_name.val, np.full(len(fdat["s"]), lattice_name)
    )
    # ## BEAM parameters
    self.ecnx.val = np.append(self.ecnx.val, fdat["emit_xn"])
    self.ecny.val = np.append(self.ecny.val, fdat["emit_yn"])
    self.eta_x_beam.val = np.append(self.eta_x_beam.val, fdat["dx"])
    self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, fdat["dpx"])
    self.eta_y_beam.val = np.append(self.eta_y_beam.val, fdat["dy"])
    self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, fdat["dpy"])
    self.beta_x_beam.val = np.append(self.beta_x_beam.val, fdat["betx"])
    self.beta_y_beam.val = np.append(self.beta_y_beam.val, fdat["bety"])
    self.alpha_x_beam.val = np.append(self.alpha_x_beam.val, fdat["alfx"])
    self.alpha_y_beam.val = np.append(self.alpha_y_beam.val, fdat["alfy"])
    # self.cp_eV = self.cp
    # self.cp_eV = self.cp
    # for k in self.__dict__.keys():
    #     try:
    #         if len(getattr(self, k)) < len(getattr(self, "z")):
    #             self.append(k, np.zeros(len(fdat["s"])))
    #     except Exception:
    #         pass
