import os
import numpy as np
from .. import constants
import h5py

def cumtrapz(
        x: list | np.ndarray = [],
        y: list | np.ndarray = []
):
    return [np.trapz(x=x[:n], y=y[:n]) for n in range(len(x))]

def read_cheetah_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            # print('reading new file', f)
            read_cheetah_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        pre, ext = os.path.splitext(filename)
        lattice_name = os.path.basename(pre)
        with h5py.File(filename, 'r') as f:
            file = f["Twiss"]
            self.z.val = np.append(self.z.val, file["s"][()])
            self.s.val = np.append(self.s.val, file["s"][()])
            cp = file["energy"][()] - self.E0_eV
            self.cp.val = np.append(self.cp.val, cp)
            self.mean_cp.val = np.append(self.mean_cp.val, cp)
            ke = np.array(
                (np.sqrt(self.E0 ** 2 + cp ** 2) - self.E0 ** 2)
            )
            self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
            gamma = 1 + ke / self.E0_eV
            self.gamma.val = np.append(self.gamma.val, gamma)
            # self.mean_gamma.val = np.append(self.mean_gamma.val, gamma)
            # self.p.val = np.append(self.p, cp * self.q_over_c)
            self.enx.val = np.append(self.enx.val, file["emittance_x"][()] * gamma)
            self.ex.val = np.append(self.ex.val, file["emittance_x"][()])
            self.eny.val = np.append(self.eny.val, file["emittance_y"][()] * gamma)
            self.ey.val = np.append(self.ey.val, file["emittance_y"][()])
            self.enz.val = np.append(self.enz.val, np.zeros(len(file["s"][()])))
            self.ez.val = np.append(self.ez.val, np.zeros(len(file["s"][()])))
            self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(file["s"][()])))
            self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(file["s"][()])))
            self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(file["s"][()])))
            self.beta_x.val = np.append(self.beta_x.val, file["beta_x"][()])
            self.alpha_x.val = np.append(self.alpha_x.val, file["alpha_x"][()])
            self.gamma_x.val = np.append(self.gamma_x.val, (1 + file["alpha_x"][()] ** 2) / file["beta_x"][()])
            self.beta_y.val = np.append(self.beta_y.val, file["beta_y"][()])
            self.alpha_y.val = np.append(self.alpha_y.val, file["alpha_y"][()])
            self.gamma_y.val = np.append(self.gamma_y.val, (1 + file["alpha_y"][()] ** 2) / file["beta_y"][()])
            self.sigma_x.val = np.append(self.sigma_x.val, file["sigma_x"][()])
            self.sigma_xp.val = np.append(self.sigma_xp.val, file["sigma_px"][()])
            self.sigma_y.val = np.append(self.sigma_y.val, file["sigma_y"][()])
            self.sigma_yp.val = np.append(self.sigma_yp.val, file["sigma_py"][()])
            self.sigma_z.val = np.append(self.sigma_z.val, file["sigma_tau"][()])
            self.sigma_t.val = np.append(self.sigma_t.val, file["sigma_tau"][()] / constants.speed_of_light)
            self.mean_x.val = np.append(self.mean_x.val, file["mu_x"][()])
            self.mean_y.val = np.append(self.mean_y.val, file["mu_y"][()])
            beta = np.sqrt(1 - (gamma ** -2))
            self.t.val = np.append(self.t.val, file["s"][()] / (beta * constants.speed_of_light))
            self.sigma_cp.val = np.append(
                self.sigma_cp.val, file["sigma_p"][()] * cp
            )
            self.sigma_p.val = np.append(self.sigma_p.val, file["sigma_p"][()])
            self.mux.val = np.append(self.mux.val, cumtrapz(x=self.z.val, y=1 / (self.sigma_x.val ** 2 / self.ex.val)))
            self.muy.val = np.append(self.muy.val, cumtrapz(x=self.z.val, y=1 / (self.sigma_y.val ** 2 / self.ey.val)))
            self.eta_x.val = np.append(self.eta_x.val, np.zeros(len(file["s"][()])))
            self.eta_xp.val = np.append(self.eta_xp.val, np.zeros(len(file["s"][()])))
            self.eta_y.val = np.append(self.eta_y.val, np.zeros(len(file["s"][()])))
            self.eta_yp.val = np.append(self.eta_yp.val, np.zeros(len(file["s"][()])))
            self.element_name.val = np.append(self.element_name.val, np.full(len(file["s"][()]), ""))
            self.lattice_name.val = np.append(self.lattice_name.val, np.full(len(file["s"][()]), lattice_name))
            self.ecnx.val = np.append(self.ecnx.val, file["emittance_x"][()] / gamma)
            self.ecny.val = np.append(self.ecny.val, file["emittance_y"][()] / gamma)
            self.eta_x_beam.val = np.append(self.eta_x_beam.val, np.zeros(len(file["s"][()])))
            self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, np.zeros(len(file["s"][()])))
            self.eta_y_beam.val = np.append(self.eta_y_beam.val, np.zeros(len(file["s"][()])))
            self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, np.zeros(len(file["s"][()])))
            self.beta_x_beam.val = np.append(self.beta_x_beam.val, np.zeros(len(file["s"][()])))
            self.beta_y_beam.val = np.append(self.beta_y_beam.val, np.zeros(len(file["s"][()])))
            self.alpha_x_beam.val = np.append(self.alpha_x_beam.val, np.zeros(len(file["s"][()])))
            self.alpha_y_beam.val = np.append(self.alpha_y_beam.val, np.zeros(len(file["s"][()])))
            self.cp_eV = self.cp
            # self["sigma_cp_eV"] = self["sigma_cp"]
