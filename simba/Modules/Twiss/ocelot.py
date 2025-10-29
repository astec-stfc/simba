import os
import numpy as np
from .. import constants


def cumtrapz(x=[], y=[]):
    return [np.trapz(x=x[:n], y=y[:n]) for n in range(len(x))]


def read_ocelot_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_ocelot_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        fdat = {}
        # print("loading ocelot twiss file", filename)
        with np.load(filename, allow_pickle=True) as data:
            for key, value in data.items():
                try:
                    data[key]
                    fdat.update({key: value})
                except ValueError:
                    pass
        interpret_ocelot_data(self, lattice_name, fdat)


def interpret_ocelot_data(self, lattice_name, fdat):
    self.z.val = np.append(self.z.val, fdat["s"])
    self.s.val = np.append(self.s.val, fdat["s"])
    E = fdat["_E"] * 1e9
    ke = E - self.E0_eV
    gamma = E / self.E0_eV
    cp = np.sqrt(E**2 - self.E0_eV**2)
    # self.append('cp', cp)
    ke = np.array(
        (np.sqrt(self.E0**2 + cp**2) - self.E0**2)
    )
    self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
    gamma = 1 + ke / self.E0_eV
    cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
    self.cp.val = np.append(self.cp.val, cp)
    self.gamma.val = np.append(self.gamma.val, gamma)
    self.p.val = np.append(self.p.val, cp * self.q_over_c)
    self.enx.val = np.append(self.enx.val, fdat["_emit_xn"])
    self.ex.val = np.append(self.ex.val, fdat["eigemit_1"])
    self.eny.val = np.append(self.eny.val, fdat["_emit_yn"])
    self.ey.val = np.append(self.ey.val, fdat["eigemit_2"])
    self.enz.val = np.append(self.enz.val, np.zeros(len(fdat["s"])))
    self.ez.val = np.append(self.ez.val, np.zeros(len(fdat["s"])))
    self.beta_x.val = np.append(self.beta_x.val, fdat["_beta_x"])
    self.alpha_x.val = np.append(self.alpha_x.val, fdat["_alpha_x"])
    self.gamma_x.val = np.append(
        self.gamma_x.val, (1 + fdat["_alpha_x"] ** 2) / fdat["_beta_x"]
    )
    self.beta_y.val = np.append(self.beta_y.val, fdat["_beta_y"])
    self.alpha_y.val = np.append(self.alpha_y.val, fdat["_alpha_y"])
    self.gamma_y.val = np.append(
        self.gamma_y.val, (1 + fdat["_alpha_y"] ** 2) / fdat["_beta_y"]
    )
    self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(fdat["s"])))
    self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(fdat["s"])))
    self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(fdat["s"])))
    self.sigma_x.val = np.append(self.sigma_x.val, np.sqrt(fdat["xx"]))
    self.sigma_y.val = np.append(self.sigma_y.val, np.sqrt(fdat["yy"]))
    self.sigma_xp.val = np.append(self.sigma_xp.val, np.sqrt(fdat["pxpx"]))
    self.sigma_yp.val = np.append(self.sigma_yp.val, np.sqrt(fdat["pypy"]))
    self.sigma_t.val = np.append(
        self.sigma_t.val, np.sqrt(fdat["tautau"]) / constants.speed_of_light
    )
    self.mean_x.val = np.append(self.mean_x.val, fdat["x"])
    self.mean_y.val = np.append(self.mean_y.val, fdat["y"])
    beta = np.sqrt(1 - (gamma**-2))
    self.t.val = np.append(self.t.val, fdat["s"] / (beta * constants.speed_of_light))
    self.sigma_z.val = np.append(self.sigma_z.val, np.sqrt(fdat["tautau"]) * beta)
    # self.append('sigma_cp', elegantData['Sdelta'] * cp )
    self.sigma_cp.val = np.append(
        self.sigma_cp.val, np.sqrt(fdat["pp"]) * cp / constants.elementary_charge
    )
    self.mean_cp.val = np.append(self.mean_cp.val, cp)
    # print('elegant = ', (elegantData['Sdelta'] * cp / constants.elementary_charge)[-1)
    self.sigma_p.val = np.append(self.sigma_p.val, np.sqrt(fdat["pp"]))
    self.mux.val = np.append(self.mux.val, fdat["mux"])
    self.muy.val = np.append(self.muy.val, fdat["muy"])
    self.eta_x.val = np.append(self.eta_x.val, fdat["Dx"])
    self.eta_xp.val = np.append(self.eta_xp.val, fdat["Dxp"])
    self.eta_y.val = np.append(self.eta_y.val, fdat["Dy"])
    self.eta_yp.val = np.append(self.eta_yp.val, fdat["Dyp"])
    self.element_name.val = np.append(self.element_name.val, np.zeros(len(fdat["s"])))
    self.lattice_name.val = np.append(
        self.lattice_name.val, np.full(len(fdat["s"]), lattice_name)
    )
    # ## BEAM parameters
    self.ecnx.val = np.append(self.ecnx.val, fdat["_emit_xn"])
    self.ecny.val = np.append(self.ecny.val, fdat["_emit_yn"])
    self.eta_x_beam.val = np.append(self.eta_x_beam.val, fdat["Dx"])
    self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, fdat["Dxp"])
    self.eta_y_beam.val = np.append(self.eta_y_beam.val, fdat["Dy"])
    self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, fdat["Dyp"])
    self.beta_x_beam.val = np.append(
        self.beta_x_beam.val, fdat["xx"] / fdat["eigemit_1"]
    )
    self.beta_y_beam.val = np.append(
        self.beta_y_beam.val, fdat["yy"] / fdat["eigemit_2"]
    )
    self.alpha_x_beam.val = np.append(
        self.alpha_x_beam.val,
        -1
        * np.sign(fdat["xpx"])
        * np.sqrt(fdat["xx"])
        * np.sqrt(fdat["pxpx"])
        / fdat["eigemit_1"],
    )
    self.alpha_y_beam.val = np.append(
        self.alpha_y_beam.val,
        -1
        * np.sign(fdat["ypy"])
        * np.sqrt(fdat["yy"])
        * np.sqrt(fdat["pypy"])
        / fdat["eigemit_2"],
    )
    self.cp_eV = self.cp
