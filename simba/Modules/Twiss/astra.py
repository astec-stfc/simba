import os
import numpy as np
from .. import constants


def cumtrapz(x=[], y=[]):
    return [np.trapz(x=x[:n], y=y[:n]) for n in range(len(x))]


def read_astra_twiss_files(self, filename, reset=True) -> None:
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_astra_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        if "xemit" not in filename.lower():
            filename = filename.replace("Yemit", "Xemit").replace("Zemit", "Xemit")
        xemit = (
            np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
        )
        if "yemit" not in filename.lower():
            filename = filename.replace("Xemit", "Yemit").replace("Zemit", "Yemit")
        yemit = (
            np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
        )
        if "zemit" not in filename.lower():
            filename = filename.replace("Xemit", "Zemit").replace("Yemit", "Zemit")
        zemit = (
            np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
        )
        interpret_astra_data(self, lattice_name, xemit, yemit, zemit)


def interpret_astra_data(self, lattice_name, xemit, yemit, zemit) -> None:
    z, t, mean_x, rms_x, rms_xp, exn, mean_xxp = np.transpose(xemit)
    z, t, mean_y, rms_y, rms_yp, eyn, mean_yyp = np.transpose(yemit)
    z, t, e_kin, rms_z, rms_e, ezn, mean_zep = np.transpose(zemit)
    e_kin = 1e6 * e_kin
    t = 1e-9 * t
    exn = 1e-6 * exn
    eyn = 1e-6 * eyn
    mean_x, mean_y, mean_xxp, mean_yyp, mean_zep = 1e-3 * np.array(
        [mean_x, mean_y, mean_xxp, mean_yyp, mean_zep]
    )
    rms_x, rms_xp, rms_y, rms_yp, rms_z, rms_e = 1e-3 * np.array(
        [rms_x, rms_xp, rms_y, rms_yp, rms_z, rms_e]
    )

    self.z.val = np.append(self.z.val, z)
    self.s.val = np.append(self.s.val, z)
    self.t.val = np.append(self.t.val, t)
    self.kinetic_energy.val = np.append(self.kinetic_energy.val, e_kin)
    gamma = 1 + (e_kin / self.E0_eV)
    self.gamma.val = np.append(self.gamma.val, gamma)
    cp = np.sqrt(e_kin * (2 * self.E0_eV + e_kin))
    self.cp.val = np.append(self.cp.val, cp)
    self.mean_cp.val = np.append(self.mean_cp.val, cp)
    p = cp * constants.elementary_charge * self.q_over_c
    self.p.val = np.append(self.p.val, p)
    self.enx.val = np.append(self.enx.val, exn)
    ex = exn / gamma
    self.ex.val = np.append(self.ex.val, ex)
    self.eny.val = np.append(self.eny.val, eyn)
    ey = eyn / gamma
    self.ey.val = np.append(self.ey.val, ey)
    self.enz.val = np.append(self.enz.val, ezn)
    ez = ezn / gamma
    self.ez.val = np.append(self.ez.val, ez)
    self.beta_x.val = np.append(self.beta_x.val, rms_x**2 / ex)
    self.gamma_x.val = np.append(self.gamma_x.val, rms_xp**2 / ex)
    self.alpha_x.val = np.append(
        self.alpha_x.val, (-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex
    )
    self.beta_y.val = np.append(self.beta_y.val, rms_y**2 / ey)
    self.gamma_y.val = np.append(self.gamma_y.val, rms_yp**2 / ey)
    self.alpha_y.val = np.append(
        self.alpha_y.val, (-1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey
    )
    self.beta_z.val = np.append(self.beta_z.val, rms_z**2 / ez)
    self.gamma_z.val = np.append(self.gamma_z.val, rms_e**2 / ez)
    self.alpha_z.val = np.append(
        self.alpha_z.val, (-1 * np.sign(mean_zep) * rms_z * rms_e) / ez
    )
    self.sigma_x.val = np.append(self.sigma_x.val, rms_x)
    self.sigma_xp.val = np.append(self.sigma_xp.val, rms_xp)
    self.sigma_y.val = np.append(self.sigma_y.val, rms_y)
    self.sigma_yp.val = np.append(self.sigma_yp.val, rms_yp)
    self.sigma_z.val = np.append(self.sigma_z.val, rms_z)
    self.mean_x.val = np.append(self.mean_x.val, mean_x)
    self.mean_y.val = np.append(self.mean_y.val, mean_y)
    beta = np.sqrt(1 - (gamma**-2))
    self.sigma_t.val = np.append(
        self.sigma_t.val, rms_z / (beta * constants.speed_of_light)
    )
    self.sigma_p.val = np.append(self.sigma_p.val, (rms_e / (e_kin + self.E0_eV)))
    self.sigma_cp.val = np.append(self.sigma_cp.val, (0.5e6 * (rms_e / e_kin) * cp))
    self.mux.val = np.append(self.mux.val, cumtrapz(x=z, y=1 / (rms_x**2 / ex)))
    self.muy.val = np.append(self.muy.val, cumtrapz(x=z, y=1 / (rms_y**2 / ey)))
    self.eta_x.val = np.append(self.eta_x.val, np.zeros(len(z)))
    self.eta_xp.val = np.append(self.eta_xp.val, np.zeros(len(z)))
    self.eta_y.val = np.append(self.eta_y.val, np.zeros(len(z)))
    self.eta_yp.val = np.append(self.eta_yp.val, np.zeros(len(z)))
    self.eta_x_beam.val = np.append(self.eta_x_beam.val, np.zeros(len(z)))
    self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, np.zeros(len(z)))
    self.eta_y_beam.val = np.append(self.eta_y_beam.val, np.zeros(len(z)))
    self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, np.zeros(len(z)))
    self.ecnx.val = np.append(self.ecnx.val, exn)
    self.ecny.val = np.append(self.ecny.val, eyn)
    self.element_name.val = np.append(self.element_name.val, z)
    self.lattice_name.val = np.append(
        self.lattice_name.val, np.full(len(z), lattice_name)
    )
    self.beta_x_beam.val = np.append(self.beta_x_beam.val, rms_x**2 / ex)
    self.beta_y_beam.val = np.append(self.beta_y_beam.val, rms_y**2 / ey)
    self.alpha_x_beam.val = np.append(
        self.alpha_x_beam.val, (-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex
    )
    self.alpha_y_beam.val = np.append(
        self.alpha_y_beam.val, (1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey
    )
    self.cp_eV.val = self.cp.val
