import os
import h5py
import numpy as np
from .. import constants

def get_mean(data, is_array: bool):
    if is_array:
        return np.mean(data, axis=1)
    return np.asarray(data[()]).reshape(-1)

def read_genesis_twiss_files(self, filename, startS: float = 0, reset = True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_genesis_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        file = h5py.File(filename, "r")
        s = np.array(file["/Lattice/z"][()] + startS)
        s = np.append(s, file["/Lattice/z"][-1] + startS)
        is_array = file["/Beam/energy"].shape[1] > 1
        cp = get_mean(file["/Beam/energy"], is_array) * self.E0
        ke = np.array(
            (np.sqrt(self.E0 ** 2 + cp ** 2) - self.E0) / constants.elementary_charge
        )
        gamma = 1 + ke / self.E0_eV
        cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
        px = get_mean(file["/Beam/pxposition"][()], is_array)
        py = get_mean(file["/Beam/pyposition"][()], is_array)
        beta = np.sqrt(1 - (gamma ** -2))
        self.append_columns(
            len(s),
            z=s,
            s=s,
            kinetic_energy=ke,
            cp=cp,
            gamma=gamma,
            p=cp * self.q_over_c,
            enx=np.full(len(s), get_mean(file["/Beam/emitx"], is_array)),
            ex=np.full(len(s), get_mean(file["/Beam/emitx"], is_array) / gamma),
            ecnx=np.full(len(s), get_mean(file["/Beam/emitx"], is_array)),
            eny=np.full(len(s), get_mean(file["/Beam/emitx"], is_array)),
            ey=np.full(len(s), get_mean(file["/Beam/emitx"], is_array) / gamma),
            ecny=np.full(len(s), get_mean(file["/Beam/emitx"], is_array)),
            beta_x=np.full(len(s), get_mean(file["/Beam/betax"], is_array)),
            alpha_x=np.full(len(s), get_mean(file["/Beam/alphax"], is_array)),
            beta_y=np.full(len(s), get_mean(file["/Beam/betay"], is_array)),
            alpha_y=np.full(len(s), get_mean(file["/Beam/alphay"], is_array)),
            beta_x_beam=np.full(len(s), get_mean(file["/Beam/betax"], is_array)),
            beta_y_beam=np.full(len(s), get_mean(file["/Beam/betay"], is_array)),
            alpha_x_beam=np.full(len(s), get_mean(file["/Beam/alphax"], is_array)),
            alpha_y_beam=np.full(len(s), get_mean(file["/Beam/alphay"], is_array)),
            sigma_x=get_mean(file["/Beam/xsize"][()], is_array),
            sigma_y=get_mean(file["/Beam/ysize"][()], is_array),
            sigma_xp=px,
            sigma_yp=py,
            mean_x=get_mean(file["/Beam/xposition"][()], is_array),
            mean_y=get_mean(file["/Beam/yposition"][()], is_array),
            # carry the previous section's bunch length forward
            sigma_t=self.sigma_t.val[-1] if len(self.sigma_t.val) > 0 else 0.0,
            sigma_p=get_mean(file["/Beam/energyspread"][()], is_array) / get_mean(
                file["/Beam/energy"][()], is_array),
            gamma_x=(1 + np.full(len(s), get_mean(file["/Beam/alphax"], is_array)) ** 2) / np.full(len(s), get_mean(file["/Beam/betax"], is_array)),
            gamma_y=(1 + np.full(len(s), get_mean(file["/Beam/alphay"], is_array)) ** 2) / np.full(len(s), get_mean(file["/Beam/betay"], is_array)),
            t=s / (beta * constants.speed_of_light),
            sigma_cp=get_mean(file["/Beam/energyspread"][()], is_array) * 0.511 * 1e6,
            mean_cp=get_mean(file["/Beam/energy"][()], is_array),
            enz=0.0, ez=0.0, beta_z=0.0, gamma_z=0.0, alpha_z=0.0,
            eta_x=0.0, eta_xp=0.0, eta_y=0.0, eta_yp=0.0,
            sigma_z=0.0, mux=0.0, muy=0.0,
            element_name="",
            lattice_name="",
            # ## BEAM parameters
            eta_x_beam=0.0, eta_xp_beam=0.0, eta_y_beam=0.0, eta_yp_beam=0.0,
        )
        file.close()