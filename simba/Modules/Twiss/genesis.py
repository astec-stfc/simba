def get_mean(data, is_array: bool):
    if is_array:
        return np.mean(data, axis=1)
    return data
def read_genesis_twiss_files(self, filename, startS: float = 0, reset = True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            # print('reading new file', f)
            read_genesis_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        file = h5py.File(filename, "r")
        s = np.array(file["/Lattice/z"][()] + startS)
        s = np.append(s, file["/Lattice/z"][-1] + startS)
        if file["/Beam/energy"].shape[1] > 1:
            is_array = True
        else:
            is_array = False
        self.z.val = np.append(self.z.val, s)
        self.s.val = np.append(self.s.val, s)
        cp = get_mean(file["/Beam/energy"], is_array) * self.E0
        # self.append('cp', cp)
        ke = np.array(
            (np.sqrt(self.E0 ** 2 + cp ** 2) - self.E0 ** 2) / constants.elementary_charge
        )
        self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
        gamma = 1 + ke / self.E0_eV
        cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
        self.cp.val = np.append(self.cp.val, cp)
        self.gamma.val = np.append(self.gamma.val, gamma)
        self.p.val = np.append(self.p.val, cp * self.q_over_c)
        self.enx.val = np.append(self.enx.val, np.full(len(s), get_mean(file["/Beam/emitx"], is_array)))
        self.ex.val = np.append(self.ex.val, np.full(len(s), get_mean(file["/Beam/emitx"], is_array) / gamma))
        self.ecnx.val = np.append(self.ecnx.val, np.full(len(s), get_mean(file["/Beam/emitx"], is_array)))
        self.eny.val = np.append(self.eny.val, np.full(len(s), get_mean(file["/Beam/emitx"], is_array)))
        self.ey.val = np.append(self.ey.val, np.full(len(s), get_mean(file["/Beam/emitx"], is_array) / gamma))
        self.ecny.val = np.append(self.ecny.val, np.full(len(s), get_mean(file["/Beam/emitx"], is_array)))
        self.beta_x.val = np.append(self.beta_x.val, np.full(len(s), get_mean(file["/Beam/betax"], is_array)))
        self.alpha_x.val = np.append(self.alpha_x.val, np.full(len(s), get_mean(file["/Beam/alphax"], is_array)))
        self.beta_y.val = np.append(self.beta_y.val, np.full(len(s), get_mean(file["/Beam/betay"], is_array)))
        self.alpha_y.val = np.append(self.alpha_y.val, np.full(len(s), get_mean(file["/Beam/alphax"], is_array)))
        self.beta_x_beam.val = np.append(self.beta_x_beam.val, np.full(len(s), get_mean(file["/Beam/betax"], is_array)))
        self.beta_y_beam.val = np.append(self.beta_y_beam.val, np.full(len(s), get_mean(file["/Beam/betay"], is_array)))
        self.alpha_x_beam.val = np.append(self.beta_x_beam.val, np.full(len(s), get_mean(file["/Beam/alphax"], is_array)))
        self.alpha_y_beam.val = np.append(self.beta_y_beam.val, np.full(len(s), get_mean(file["/Beam/alphay"], is_array)))
        self.sigma_x.val = np.append(self.sigma_x.val, get_mean(file["/Beam/xsize"][()], is_array))
        self.sigma_y.val = np.append(self.sigma_y.val, get_mean(file["/Beam/ysize"][()], is_array))
        px = get_mean(file["/Beam/pxposition"][()], is_array)
        py = get_mean(file["/Beam/pyposition"][()], is_array)
        self.sigma_xp.val = np.append(self.sigma_xp.val, py)
        self.sigma_yp.val = np.append(self.sigma_yp.val, py)
        self.mean_x.val = np.append(self.mean_x.val, get_mean(file["/Beam/xposition"][()], is_array))
        self.mean_x.val = np.append(self.mean_x.val, get_mean(file["/Beam/yposition"][()], is_array))
        if len(self.sigma_t.val) > 0:
            self.sigma_t.val = np.append(self.sigma_t.val, np.full(len(s), self.sigma_t.val[-1]))
        else:
            self.sigma_t.val = np.append(self.sigma_t.val, np.zeros(len(s)))
        self.eta_x.val = np.append(self.eta_x.val, np.zeros(len(s)))
        self.eta_xp.val = np.append(self.eta_xp.val, np.zeros(len(s)))
        self.eta_y.val = np.append(self.eta_y.val, np.zeros(len(s)))
        self.eta_yp.val = np.append(self.eta_yp.val, np.zeros(len(s)))
        self.sigma_p.val = np.append(self.sigma_p.val, get_mean(file["/Beam/energyspread"][()], is_array) / get_mean(
            file["/Beam/energy"][()], is_array))
        self.enz.val = np.append(self.enz.val, np.zeros(len(s)))
        self.ez.val = np.append(self.ez.val, np.zeros(len(s)))

        self.gamma_x.val = np.append(
            self.gamma_x.val, (1 + np.full(len(s), get_mean(file["/Beam/alphax"], is_array)) ** 2) / np.full(len(s), get_mean(file["/Beam/betax"], is_array))
        )
        self.gamma_y.val = np.append(
            self.gamma_x.val, (1 + np.full(len(s), get_mean(file["/Beam/alphay"], is_array)) ** 2) / np.full(len(s), get_mean(file["/Beam/betay"], is_array))
        )
        self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(s)))
        self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(s)))
        self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(s)))

        beta = np.sqrt(1 - (gamma ** -2))
        self.t.val = np.append(self.t.val, s / (beta * constants.speed_of_light))
        self.sigma_z.val = np.append(self.sigma_z.val, np.zeros(len(s)))
        self.sigma_cp.val = np.append(
            self.sigma_cp.val, get_mean(file["/Beam/energyspread"][()], is_array) * 0.511 * 1e6
        )
        self.mean_cp.val = np.append(
            self.mean_cp.val, get_mean(file["/Beam/energy"][()], is_array)
        )

        self.mux.val = np.append(self.mux.val, np.zeros(len(s)))
        self.muy.val = np.append(self.muy.val, np.zeros(len(s)))

        self.element_name.val = np.append(
            self.element_name.val, np.full(len(s), "")
        )
        self.lattice_name.val = np.append(
            self.lattice_name.val,
            np.full(len(s), ""),
        )
        # ## BEAM parameters

        self.eta_x_beam.val = np.append(self.eta_x_beam.val, np.zeros(len(s)))
        self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, np.zeros(len(s)))
        self.eta_y_beam.val = np.append(self.eta_y_beam.val, np.zeros(len(s)))
        self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, np.zeros(len(s)))
        file.close()