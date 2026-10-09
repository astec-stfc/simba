import os
import numpy as np
import h5py
from .. import constants


def cumtrapz(x=[], y=[]):
    try:
        return [np.trapezoid(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]
    except AttributeError:
        return [np.trapz(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]


def save_ocelot_twiss_hdf(self, filename: str, twiss: dict = {}):
    f = h5py.File(filename, 'w')
    for grp_name in twiss:
        try:
            f.create_dataset(grp_name, data=twiss[grp_name])
        except Exception:
            pass
    f.close()
    return


def read_ocelot_twiss_files_hdf(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_ocelot_twiss_files_hdf(self, f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        fdat = {}
        with h5py.File(filename, 'r') as data:
            for key, value in data.items():
                try:
                    value = np.array(data[key])
                    fdat.update({key: value})
                except ValueError as e:
                    print(f"Failed to interpret {key} for {filename}, {e}")
        interpret_ocelot_data(self, lattice_name, fdat)


def read_ocelot_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_ocelot_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        fdat = {}
        with np.load(filename, allow_pickle=True) as data:
            for key, value in data.items():
                try:
                    data[key]
                    fdat.update({key: value})
                except ValueError:
                    pass
        interpret_ocelot_data(self, lattice_name, fdat)


def _uncorrected_emittance(uu, pp_, up, disp, dispp, varp, betagamma):
    """
    Add the dispersive contribution back into Ocelot's dispersion-subtracted
    second moments and form the uncorrected emittance.

    Parameters
    ----------
    uu, pp_, up: np.ndarray
        Dispersion-subtracted second moments <u^2>, <u'^2> and <u u'>
    disp, dispp: np.ndarray
        Dispersion and its derivative for this plane
    varp: np.ndarray
        Momentum-spread variance <dp^2>
    betagamma: np.ndarray
        Relativistic beta*gamma, for the normalisation

    Returns
    -------
    tuple
        (normalised, geometric) uncorrected emittance
    """
    uu = uu + disp**2 * varp
    pp_ = pp_ + dispp**2 * varp
    up = up + disp * dispp * varp
    geometric = np.sqrt(np.clip(uu * pp_ - up**2, 0, None))
    return geometric * betagamma, geometric


def interpret_ocelot_data(self, lattice_name, fdat):
    E = fdat["_E"] * 1e9
    ke = E - self.E0_eV
    gamma = E / self.E0_eV
    cp = np.sqrt(E**2 - self.E0_eV**2)
    ke = np.array(np.sqrt(self.E0_eV**2 + cp**2) - self.E0_eV)
    gamma = 1 + ke / self.E0_eV
    cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
    betagamma = np.sqrt(np.clip(gamma**2 - 1, 0, None))
    enx, ex = _uncorrected_emittance(
        fdat["xx"], fdat["pxpx"], fdat["xpx"],
        fdat["Dx"], fdat["Dxp"], fdat["pp"], betagamma,
    )
    eny, ey = _uncorrected_emittance(
        fdat["yy"], fdat["pypy"], fdat["ypy"],
        fdat["Dy"], fdat["Dyp"], fdat["pp"], betagamma,
    )
    beta = np.sqrt(1 - (gamma**-2))
    if "id" in fdat:
        names = np.array(
            [
                v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                for v in np.asarray(fdat["id"]).ravel()
            ],
            dtype="U",
        )
    else:
        names = np.full(len(fdat["s"]), "", dtype="U")
    self.append_columns(
        len(fdat["s"]),
        z=fdat["z"] if "z" in fdat else fdat["s"],
        s=fdat["s"],
        kinetic_energy=ke,
        cp=cp,
        gamma=gamma,
        p=cp * self.q_over_c,
        enx=enx,
        ex=ex,
        eny=eny,
        ey=ey,
        enz=0.0, ez=0.0, beta_z=0.0, gamma_z=0.0, alpha_z=0.0,
        beta_x=fdat["_beta_x"],
        alpha_x=fdat["_alpha_x"],
        gamma_x=(1 + fdat["_alpha_x"] ** 2) / fdat["_beta_x"],
        beta_y=fdat["_beta_y"],
        alpha_y=fdat["_alpha_y"],
        gamma_y=(1 + fdat["_alpha_y"] ** 2) / fdat["_beta_y"],
        sigma_x=np.sqrt(fdat["xx"] + fdat["Dx"]**2 * fdat["pp"]),
        sigma_y=np.sqrt(fdat["yy"] + fdat["Dy"]**2 * fdat["pp"]),
        sigma_xp=np.sqrt(fdat["pxpx"]),
        sigma_yp=np.sqrt(fdat["pypy"]),
        sigma_t=np.sqrt(fdat["tautau"]) / constants.speed_of_light,
        mean_x=fdat["x"],
        mean_y=fdat["y"],
        t=fdat["s"] / (beta * constants.speed_of_light),
        sigma_z=np.sqrt(fdat["tautau"]) * beta,
        sigma_cp=np.sqrt(fdat["pp"]) * cp,
        mean_cp=cp,
        sigma_p=np.sqrt(fdat["pp"]),
        mux=fdat["mux"],
        muy=fdat["muy"],
        eta_x=fdat["Dx"],
        eta_xp=fdat["Dxp"],
        eta_y=fdat["Dy"],
        eta_yp=fdat["Dyp"],
        element_name=names,
        lattice_name=lattice_name,
        # ## BEAM parameters
        ecnx=fdat["_emit_x"] / gamma if "_emit_x" in fdat else fdat["_emit_xn"],
        ecny=fdat["_emit_y"] / gamma if "_emit_y" in fdat else fdat["_emit_yn"],
        eta_x_beam=fdat["Dx"],
        eta_xp_beam=fdat["Dxp"],
        eta_y_beam=fdat["Dy"],
        eta_yp_beam=fdat["Dyp"],
        beta_x_beam=fdat["xx"] / fdat["eigemit_1"],
        beta_y_beam=fdat["yy"] / fdat["eigemit_2"],
        alpha_x_beam=-1
        * np.sign(fdat["xpx"])
        * np.sqrt(fdat["xx"])
        * np.sqrt(fdat["pxpx"])
        / fdat["eigemit_1"],
        alpha_y_beam=-1
        * np.sign(fdat["ypy"])
        * np.sqrt(fdat["yy"])
        * np.sqrt(fdat["pypy"])
        / fdat["eigemit_2"],
    )
