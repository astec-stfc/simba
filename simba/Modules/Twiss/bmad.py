import os
import numpy as np
import h5py
from .. import constants


def save_bmad_twiss_hdf(filename: str, twiss: dict = {}):
    """
    Write the twiss data extracted from Tao to an HDF5 file.

    Parameters
    ----------
    filename: str
        Name of the file to write.
    twiss: dict
        Twiss data, as produced by
        :func:`~simba.Codes.Bmad.Bmad.bmadLattice._twiss_data`.
    """
    with h5py.File(filename, "w") as f:
        for grp_name, values in twiss.items():
            values = np.asarray(values)
            if values.dtype.kind in ("U", "S", "O"):
                f.create_dataset(
                    grp_name,
                    data=values.astype(str).astype(object),
                    dtype=h5py.string_dtype(),
                )
            else:
                f.create_dataset(grp_name, data=values)
    return


def read_bmad_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_bmad_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        fdat = {}
        with h5py.File(filename, "r") as data:
            for key, value in data.items():
                if h5py.check_string_dtype(value.dtype):
                    fdat.update({key: value.asstr()[:]})
                else:
                    fdat.update({key: np.array(value)})
        interpret_bmad_data(self, lattice_name, fdat)


def interpret_bmad_data(self, lattice_name, fdat):
    """
    Populate the twiss object from the contents of a Bmad twiss file.
    """
    self.z.val = np.append(self.z.val, fdat["z"])
    self.s.val = np.append(self.s.val, fdat["s"])
    cp = fdat["p0c"]
    ke = fdat["e_tot"] - self.E0_eV
    gamma = fdat["e_tot"] / self.E0_eV
    beta = np.sqrt(1 - (gamma**-2))
    self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
    self.gamma.val = np.append(self.gamma.val, gamma)
    self.cp.val = np.append(self.cp.val, cp)
    self.p.val = np.append(self.p.val, cp * self.q_over_c)
    self.t.val = np.append(self.t.val, fdat["beam_t"])
    self.enx.val = np.append(self.enx.val, fdat["beam_norm_emit_x"])
    self.ex.val = np.append(self.ex.val, fdat["beam_emit_x"])
    self.eny.val = np.append(self.eny.val, fdat["beam_norm_emit_y"])
    self.ey.val = np.append(self.ey.val, fdat["beam_emit_y"])
    longitudinal = cp / (beta * constants.speed_of_light)
    self.enz.val = np.append(self.enz.val, fdat["beam_norm_emit_z"] * longitudinal)
    self.ez.val = np.append(self.ez.val, fdat["beam_emit_z"] * longitudinal)
    self.beta_x.val = np.append(self.beta_x.val, fdat["beam_beta_x"])
    self.alpha_x.val = np.append(self.alpha_x.val, fdat["beam_alpha_x"])
    self.gamma_x.val = np.append(self.gamma_x.val, fdat["beam_gamma_x"])
    self.beta_y.val = np.append(self.beta_y.val, fdat["beam_beta_y"])
    self.alpha_y.val = np.append(self.alpha_y.val, fdat["beam_alpha_y"])
    self.gamma_y.val = np.append(self.gamma_y.val, fdat["beam_gamma_y"])
    self.beta_z.val = np.append(self.beta_z.val, fdat["beam_beta_z"])
    self.alpha_z.val = np.append(self.alpha_z.val, fdat["beam_alpha_z"])
    self.gamma_z.val = np.append(self.gamma_z.val, fdat["beam_gamma_z"])
    self.sigma_x.val = np.append(self.sigma_x.val, fdat["beam_sigma_x"])
    self.sigma_y.val = np.append(self.sigma_y.val, fdat["beam_sigma_y"])
    self.sigma_xp.val = np.append(self.sigma_xp.val, fdat["beam_sigma_xp"])
    self.sigma_yp.val = np.append(self.sigma_yp.val, fdat["beam_sigma_yp"])
    self.sigma_t.val = np.append(self.sigma_t.val, fdat["beam_sigma_t"])
    self.sigma_z.val = np.append(self.sigma_z.val, fdat["beam_sigma_z"])
    self.mean_x.val = np.append(self.mean_x.val, fdat["beam_x"])
    self.mean_y.val = np.append(self.mean_y.val, fdat["beam_y"])
    self.sigma_p.val = np.append(self.sigma_p.val, fdat["beam_sigma_delta"])
    self.sigma_cp.val = np.append(
        self.sigma_cp.val, fdat["beam_sigma_delta"] * fdat["beam_p0c"]
    )
    self.mean_cp.val = np.append(
        self.mean_cp.val, fdat["beam_p0c"] * (1 + fdat["beam_delta"])
    )
    self.mux.val = np.append(self.mux.val, fdat["mu_x"] / (2 * constants.pi))
    self.muy.val = np.append(self.muy.val, fdat["mu_y"] / (2 * constants.pi))
    self.eta_x.val = np.append(self.eta_x.val, fdat["beam_eta_x"])
    self.eta_xp.val = np.append(self.eta_xp.val, fdat["beam_etap_x"])
    self.eta_y.val = np.append(self.eta_y.val, fdat["beam_eta_y"])
    self.eta_yp.val = np.append(self.eta_yp.val, fdat["beam_etap_y"])
    self.element_name.val = np.append(self.element_name.val, fdat["element_name"])
    self.lattice_name.val = np.append(
        self.lattice_name.val, np.full(len(fdat["s"]), lattice_name)
    )
    self.ecnx.val = np.append(self.ecnx.val, fdat["beam_norm_emit_a"])
    self.ecny.val = np.append(self.ecny.val, fdat["beam_norm_emit_b"])
    self.eta_x_beam.val = np.append(self.eta_x_beam.val, fdat["beam_eta_x"])
    self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, fdat["beam_etap_x"])
    self.eta_y_beam.val = np.append(self.eta_y_beam.val, fdat["beam_eta_y"])
    self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, fdat["beam_etap_y"])
    self.beta_x_beam.val = np.append(self.beta_x_beam.val, fdat["beam_beta_a"])
    self.beta_y_beam.val = np.append(self.beta_y_beam.val, fdat["beam_beta_b"])
    self.alpha_x_beam.val = np.append(self.alpha_x_beam.val, fdat["beam_alpha_a"])
    self.alpha_y_beam.val = np.append(self.alpha_y_beam.val, fdat["beam_alpha_b"])
