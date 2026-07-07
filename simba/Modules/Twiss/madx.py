import os
import numpy as np
from .. import constants


def read_madx_twiss_files(self, filename, startS=0, reset=True):
    """
    Read a MAD-X ``TWISS`` output (TFS) file and populate the
    :class:`~simba.Modules.Twiss.twiss` object, interpreting it in the same way
    as the other codes' twiss readers.

    A MAD-X twiss carries the *optics* functions (``BETX``, ``ALFX``, ``MUX``,
    ``DX`` ...) about a constant reference momentum rather than beam moments, so
    the beam sizes are reconstructed for a matched beam from the optics and the
    beam emittances / energy spread in the file header
    (``σ_x = sqrt(β_x ε_x + (η_x σ_δ)²)`` etc.).

    Parameters
    ----------
    filename: str | list | tuple
        Path to the MAD-X ``.tfs`` twiss file (or a list of them).
    startS: float, optional
        Unused, kept for signature parity with the other readers.
    reset: bool, optional
        Whether to reset the twiss dictionaries before reading.
    """
    import tfs

    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_madx_twiss_files(self, f, reset=False)
        return
    if not os.path.isfile(filename):
        return

    lattice_name = os.path.basename(os.path.splitext(filename)[0])
    self.sddsindex += 1

    madxData = tfs.read(filename)
    headers = {k.upper(): v for k, v in dict(madxData.headers).items()}
    madxData.columns = [c.upper() for c in madxData.columns]

    def col(name, default=0.0):
        if name in madxData.columns:
            return np.asarray(madxData[name], dtype=float)
        return np.full(len(madxData), default, dtype=float)

    # Rest energy / reference energy. MAD-X MASS, ENERGY, PC are in GeV; set the
    # object's rest mass from the file so non-electron species are handled too.
    E0_eV = float(headers["MASS"]) * 1e9
    self.set_E0(E0_eV * constants.elementary_charge / constants.speed_of_light**2)

    n = len(madxData)
    z = col("S")
    # ENERGY column is the per-row reference total energy [GeV] (constant across a
    # TWISS, which uses a fixed reference momentum); fall back to the header.
    E_tot = col("ENERGY", float(headers["ENERGY"])) * 1e9
    E_tot = np.where(E_tot > 0, E_tot, float(headers["ENERGY"]) * 1e9)
    gamma = E_tot / E0_eV
    beta_rel = np.sqrt(1.0 - gamma**-2)
    cp = np.sqrt(np.clip(E_tot**2 - E0_eV**2, 0.0, None))
    ke = E_tot - E0_eV

    betx, alfx, mux = col("BETX"), col("ALFX"), col("MUX")
    bety, alfy, muy = col("BETY"), col("ALFY"), col("MUY")
    gamx = (1.0 + alfx**2) / betx
    gamy = (1.0 + alfy**2) / bety

    # MAD-X dispersion DX = dx/dpt (deviation w.r.t. pt = β·δ); convert to the
    # usual dx/(dp/p) dispersion used by the other readers by multiplying by β.
    eta_x = col("DX") * beta_rel
    eta_xp = col("DPX") * beta_rel
    eta_y = col("DY") * beta_rel
    eta_yp = col("DPY") * beta_rel

    # Geometric emittances and relative momentum spread from the beam header.
    ex = float(headers.get("EX", 0.0))
    ey = float(headers.get("EY", 0.0))
    sigma_dpp = float(headers.get("SIGE", 0.0))
    sigt = float(headers.get("SIGT", 0.0))

    sigma_x = np.sqrt(betx * ex + (eta_x * sigma_dpp) ** 2)
    sigma_y = np.sqrt(bety * ey + (eta_y * sigma_dpp) ** 2)
    sigma_xp = np.sqrt(gamx * ex + (eta_xp * sigma_dpp) ** 2)
    sigma_yp = np.sqrt(gamy * ey + (eta_yp * sigma_dpp) ** 2)

    zeros = np.zeros(n)
    element_name = (
        np.asarray(madxData["NAME"], dtype=str)
        if "NAME" in madxData.columns
        else np.full(n, lattice_name)
    )

    self.z.val = np.append(self.z.val, z)
    self.s.val = np.append(self.s.val, z)
    self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
    self.gamma.val = np.append(self.gamma.val, gamma)
    self.cp.val = np.append(self.cp.val, cp)
    self.p.val = np.append(self.p.val, cp * self.q_over_c)
    self.t.val = np.append(self.t.val, z / (beta_rel * constants.speed_of_light))

    self.ex.val = np.append(self.ex.val, np.full(n, ex))
    self.enx.val = np.append(self.enx.val, np.full(n, ex) * beta_rel * gamma)
    self.ecnx.val = np.append(self.ecnx.val, np.full(n, ex) * beta_rel * gamma)
    self.ey.val = np.append(self.ey.val, np.full(n, ey))
    self.eny.val = np.append(self.eny.val, np.full(n, ey) * beta_rel * gamma)
    self.ecny.val = np.append(self.ecny.val, np.full(n, ey) * beta_rel * gamma)
    self.ez.val = np.append(self.ez.val, zeros)
    self.enz.val = np.append(self.enz.val, zeros)

    self.beta_x.val = np.append(self.beta_x.val, betx)
    self.alpha_x.val = np.append(self.alpha_x.val, alfx)
    self.gamma_x.val = np.append(self.gamma_x.val, gamx)
    self.beta_y.val = np.append(self.beta_y.val, bety)
    self.alpha_y.val = np.append(self.alpha_y.val, alfy)
    self.gamma_y.val = np.append(self.gamma_y.val, gamy)
    self.beta_z.val = np.append(self.beta_z.val, zeros)
    self.gamma_z.val = np.append(self.gamma_z.val, zeros)
    self.alpha_z.val = np.append(self.alpha_z.val, zeros)

    self.sigma_x.val = np.append(self.sigma_x.val, sigma_x)
    self.sigma_y.val = np.append(self.sigma_y.val, sigma_y)
    self.sigma_xp.val = np.append(self.sigma_xp.val, sigma_xp)
    self.sigma_yp.val = np.append(self.sigma_yp.val, sigma_yp)
    self.sigma_z.val = np.append(self.sigma_z.val, np.full(n, sigt))
    self.sigma_t.val = np.append(
        self.sigma_t.val, np.full(n, sigt / constants.speed_of_light)
    )
    self.sigma_cp.val = np.append(self.sigma_cp.val, sigma_dpp * cp)
    self.sigma_p.val = np.append(self.sigma_p.val, sigma_dpp * cp * self.q_over_c)

    self.mean_x.val = np.append(self.mean_x.val, col("X"))
    self.mean_y.val = np.append(self.mean_y.val, col("Y"))
    self.mean_cp.val = np.append(self.mean_cp.val, cp)

    self.mux.val = np.append(self.mux.val, mux)
    self.muy.val = np.append(self.muy.val, muy)

    self.eta_x.val = np.append(self.eta_x.val, eta_x)
    self.eta_xp.val = np.append(self.eta_xp.val, eta_xp)
    self.eta_y.val = np.append(self.eta_y.val, eta_y)
    self.eta_yp.val = np.append(self.eta_yp.val, eta_yp)

    self.element_name.val = np.append(self.element_name.val, element_name)
    self.lattice_name.val = np.append(
        self.lattice_name.val, np.full(n, lattice_name)
    )

    # MAD-X twiss has no separate beam-based optics/dispersion; mirror the lattice
    # values into the *_beam fields (as the OPAL reader does) so downstream code
    # that reads them still works.
    self.beta_x_beam.val = np.append(self.beta_x_beam.val, betx)
    self.beta_y_beam.val = np.append(self.beta_y_beam.val, bety)
    self.alpha_x_beam.val = np.append(self.alpha_x_beam.val, alfx)
    self.alpha_y_beam.val = np.append(self.alpha_y_beam.val, alfy)
    self.eta_x_beam.val = np.append(self.eta_x_beam.val, eta_x)
    self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, eta_xp)
    self.eta_y_beam.val = np.append(self.eta_y_beam.val, eta_y)
    self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, eta_yp)
