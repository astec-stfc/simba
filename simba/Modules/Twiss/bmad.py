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
    cp = fdat["p0c"]
    ke = fdat["e_tot"] - self.E0_eV
    gamma = fdat["e_tot"] / self.E0_eV
    beta = np.sqrt(1 - (gamma**-2))
    longitudinal = cp / (beta * constants.speed_of_light)
    self.append_columns(
        len(fdat["s"]),
        z=fdat["z"],
        s=fdat["s"],
        kinetic_energy=ke,
        gamma=gamma,
        cp=cp,
        p=cp * self.q_over_c,
        t=fdat["beam_t"],
        enx=fdat["beam_norm_emit_x"],
        ex=fdat["beam_emit_x"],
        eny=fdat["beam_norm_emit_y"],
        ey=fdat["beam_emit_y"],
        enz=fdat["beam_norm_emit_z"] * longitudinal,
        ez=fdat["beam_emit_z"] * longitudinal,
        beta_x=fdat["beam_beta_x"],
        alpha_x=fdat["beam_alpha_x"],
        gamma_x=fdat["beam_gamma_x"],
        beta_y=fdat["beam_beta_y"],
        alpha_y=fdat["beam_alpha_y"],
        gamma_y=fdat["beam_gamma_y"],
        beta_z=fdat["beam_beta_z"],
        alpha_z=fdat["beam_alpha_z"],
        gamma_z=fdat["beam_gamma_z"],
        sigma_x=fdat["beam_sigma_x"],
        sigma_y=fdat["beam_sigma_y"],
        sigma_xp=fdat["beam_sigma_xp"],
        sigma_yp=fdat["beam_sigma_yp"],
        sigma_t=fdat["beam_sigma_t"],
        sigma_z=fdat["beam_sigma_z"],
        mean_x=fdat["beam_x"],
        mean_y=fdat["beam_y"],
        sigma_p=fdat["beam_sigma_delta"],
        sigma_cp=fdat["beam_sigma_delta"] * fdat["beam_p0c"],
        mean_cp=fdat["beam_p0c"] * (1 + fdat["beam_delta"]),
        mux=fdat["mu_x"] / (2 * constants.pi),
        muy=fdat["mu_y"] / (2 * constants.pi),
        eta_x=fdat["beam_eta_x"],
        eta_xp=fdat["beam_etap_x"],
        eta_y=fdat["beam_eta_y"],
        eta_yp=fdat["beam_etap_y"],
        element_name=fdat["element_name"],
        lattice_name=lattice_name,
        ecnx=fdat["beam_norm_emit_a"],
        ecny=fdat["beam_norm_emit_b"],
        eta_x_beam=fdat["beam_eta_x"],
        eta_xp_beam=fdat["beam_etap_x"],
        eta_y_beam=fdat["beam_eta_y"],
        eta_yp_beam=fdat["beam_etap_y"],
        beta_x_beam=fdat["beam_beta_a"],
        beta_y_beam=fdat["beam_beta_b"],
        alpha_x_beam=fdat["beam_alpha_a"],
        alpha_y_beam=fdat["beam_alpha_b"],
    )
