import os
import h5py
import numpy as np
import re
from .. import constants

def cumtrapz(x=[], y=[]):
    return [np.trapz(x=x[:n], y=y[:n]) for n in range(len(x))]

def read_opal_twiss_files(self, filename, startS=0, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            # print('reading new file', f)
            read_opal_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = re.split(r" |\\|/", filename.split(".opal_twiss")[0])[-1]
        self.sddsindex += 1
        with h5py.File(filename, "r") as f:
            opalData = f
            z = opalData["s"][()]
            # z += self.z.val[-1] if len(self.z.val) > 0 else 0
            self.z.val = np.append(self.z.val, z)
            self.s.val = np.append(self.s.val, z)
            cp = opalData["ref_pz"][()]
            # self.append('cp', cp)
            ke = np.array(
                (np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge
            )
            self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
            gamma = 1 + ke / self.E0_eV
            cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
            self.cp.val = np.append(self.cp.val, cp)
            self.gamma.val = np.append(self.gamma.val, gamma)
            self.p.val = np.append(self.p.val, cp * self.q_over_c)
            self.enx.val = np.append(self.enx.val, opalData["emit_x"][()])
            self.ex.val = np.append(self.ex.val, opalData["emit_x"][()] / cp)
            self.eny.val = np.append(self.eny.val, opalData["emit_y"][()])
            self.ey.val = np.append(self.ey.val, opalData["emit_y"][()] / cp)
            betax = opalData["rms_x"][()] / opalData["emit_x"][()] / opalData["ref_pz"][()]
            alphax = (-1 * np.sign(opalData["xpx"][()]) * opalData["rms_x"][()] * opalData["rms_px"][()]) / opalData["emit_x"][()] / \
                     opalData["ref_pz"][()]
            betay = opalData["rms_y"][()] / opalData["emit_y"][()] / opalData["ref_pz"][()]
            alphay = (-1 * np.sign(opalData["ypy"][()]) * opalData["rms_y"][()] * opalData["rms_py"][()]) / opalData["emit_y"][()] / \
                     opalData["ref_pz"][()]
            self.beta_x.val = np.append(self.beta_x.val, betax)
            self.alpha_x.val = np.append(self.alpha_x.val, alphax)
            self.beta_y.val = np.append(self.beta_y.val, betay)
            self.alpha_y.val = np.append(self.alpha_y.val, alphay)
            self.sigma_x.val = np.append(self.sigma_x.val, opalData["rms_x"][()])
            self.sigma_y.val = np.append(self.sigma_y.val, opalData["rms_y"][()])
            self.sigma_xp.val = np.append(self.sigma_xp.val, opalData["rms_px"][()])
            self.sigma_yp.val = np.append(self.sigma_yp.val, opalData["rms_py"][()])
            self.sigma_t.val = np.append(self.sigma_t.val, opalData["rms_s"][()] / constants.speed_of_light)
            self.mean_x.val = np.append(self.mean_x.val, opalData["mean_x"][()])
            self.mean_y.val = np.append(self.mean_y.val, opalData["mean_y"][()])
            eta_x = opalData["mean_x"][()]
            eta_xp = opalData["mean_x"][()]
            eta_y = opalData["mean_y"][()]
            eta_yp = opalData["mean_y"][()]
            self.eta_x.val = np.append(self.eta_x.val, eta_x)
            self.eta_xp.val = np.append(self.eta_xp.val, eta_xp)
            self.eta_y.val = np.append(self.eta_y.val, eta_y)
            self.eta_yp.val = np.append(self.eta_yp.val, eta_yp)
            self.sigma_p.val = np.append(self.sigma_p.val, np.zeros(len(opalData["rms_x"][()])))
            self.beta_x_beam.val = np.append(self.beta_x_beam.val, betax)
            self.beta_y_beam.val = np.append(self.beta_y_beam.val, betay)
            self.alpha_x_beam.val = np.append(self.alpha_x_beam.val, alphax)
            self.alpha_y_beam.val = np.append(self.alpha_y_beam.val, alphay)
            self.ecnx.val = np.append(self.ecnx.val, opalData["emit_x"][()])
            self.ecny.val = np.append(self.ecny.val, opalData["emit_y"][()])
            self.enz.val = np.append(self.enz.val, np.zeros(len(opalData["rms_x"][()])))
            self.ez.val = np.append(self.ez.val, np.zeros(len(opalData["rms_x"][()])))

            self.gamma_x.val = np.append(
                self.gamma_x.val, (1 + alphax ** 2) / betax
            )
            self.gamma_y.val = np.append(
                self.gamma_y.val, (1 + alphay ** 2) / betay
            )
            self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(opalData["rms_x"][()])))
            self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(opalData["rms_x"][()])))
            self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(opalData["rms_x"][()])))

            beta = np.sqrt(1 - (gamma**-2))
            # print 'len(z) = ', len(z), '  len(beta) = ', len(beta)
            self.t.val = np.append(self.t.val, z / (beta * constants.speed_of_light))
            self.sigma_z.val = np.append(self.sigma_z.val, opalData["rms_s"][()])
            # self.append('sigma_cp', elegantData['Sdelta'] * cp )
            self.sigma_cp.val = np.append(self.sigma_cp.val, np.zeros(len(opalData["rms_x"][()])))
            self.mean_cp.val = np.append(self.mean_cp.val, cp)
            # print('elegant = ', (elegantData['Sdelta'] * cp / constants.elementary_charge)[-1)

            self.mux.val = np.append(self.mux.val, cumtrapz(x=z, y=1 / (opalData["rms_x"][()]**2 / opalData["emit_x"][()] / opalData["ref_pz"][()])))
            self.muy.val = np.append(self.muy.val, cumtrapz(x=z, y=1 / (opalData["rms_y"][()]**2 / opalData["emit_y"][()] / opalData["ref_pz"][()])))

            self.element_name.val = np.append(self.element_name.val, np.full(len(z), lattice_name))
            self.lattice_name.val = np.append(self.lattice_name.val, np.full(len(z), lattice_name))
            # ## BEAM parameters

            self.eta_x_beam.val = np.append(self.eta_x_beam.val, eta_x)
            self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, eta_xp)
            self.eta_y_beam.val = np.append(self.eta_y_beam.val, eta_y)
            self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, eta_yp)
