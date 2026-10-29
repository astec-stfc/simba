import os
import numpy as np
from .. import constants


def cumtrapz(x=[], y=[]):
    try:
        return [np.trapezoid(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]
    except AttributeError:
        return [np.trapz(x=x[: n + 1], y=y[: n + 1]) for n in range(len(x))]


def read_s_offset(filename, lattice_name) -> float:
    """
    Distance between this lattice's s and z origins, as written by
    :meth:`~simba.Codes.ASTRA.ASTRA.astraLattice.write_s_offset`.

    Non-zero once anything upstream bends; 0.0 if the file is absent, so older output reads as s == z.
    """
    path = os.path.join(os.path.dirname(filename), lattice_name + ".s_offset")
    try:
        with open(path) as f:
            return float(f.read())
    except (OSError, ValueError):
        return 0.0


def read_astra_twiss_files(self, filename, reset=True) -> None:
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_astra_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        s_offset = read_s_offset(filename, lattice_name)
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
        interpret_astra_data(self, lattice_name, xemit, yemit, zemit, s_offset)


def interpret_astra_data(self, lattice_name, xemit, yemit, zemit, s_offset=0.0) -> None:
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

    gamma = 1 + (e_kin / self.E0_eV)
    cp = np.sqrt(e_kin * (2 * self.E0_eV + e_kin))
    ex = exn / gamma
    ey = eyn / gamma
    ez = ezn / gamma
    beta = np.sqrt(1 - (gamma**-2))
    self.append_columns(
        len(z),
        z=z,
        # ASTRA only knows lab z; s additionally carries the path length of upstream bends
        s=z + s_offset,
        t=t,
        kinetic_energy=e_kin,
        gamma=gamma,
        cp=cp,
        mean_cp=cp,
        p=cp * constants.elementary_charge * self.q_over_c,
        enx=exn,
        ex=ex,
        eny=eyn,
        ey=ey,
        enz=ezn,
        ez=ez,
        beta_x=rms_x**2 / ex,
        gamma_x=rms_xp**2 / ex,
        alpha_x=(-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex,
        beta_y=rms_y**2 / ey,
        gamma_y=rms_yp**2 / ey,
        alpha_y=(-1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey,
        beta_z=rms_z**2 / ez,
        gamma_z=rms_e**2 / ez,
        alpha_z=(-1 * np.sign(mean_zep) * rms_z * rms_e) / ez,
        sigma_x=rms_x,
        sigma_xp=rms_xp,
        sigma_y=rms_y,
        sigma_yp=rms_yp,
        sigma_z=rms_z,
        mean_x=mean_x,
        mean_y=mean_y,
        sigma_t=rms_z / (beta * constants.speed_of_light),
        sigma_p=(rms_e / (e_kin + self.E0_eV)),
        sigma_cp=(0.5e6 * (rms_e / e_kin) * cp),
        mux=cumtrapz(x=z, y=1 / (rms_x**2 / ex)),
        muy=cumtrapz(x=z, y=1 / (rms_y**2 / ey)),
        eta_x=0.0, eta_xp=0.0, eta_y=0.0, eta_yp=0.0,
        eta_x_beam=0.0, eta_xp_beam=0.0, eta_y_beam=0.0, eta_yp_beam=0.0,
        ecnx=exn,
        ecny=eyn,
        element_name=z,
        lattice_name=lattice_name,
        beta_x_beam=rms_x**2 / ex,
        beta_y_beam=rms_y**2 / ey,
        alpha_x_beam=(-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex,
        alpha_y_beam=(1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey,
    )
