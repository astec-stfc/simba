"""
Convert RF-Track's transport table into SIMBA's generic ``twiss`` object.

ponytail: covers the core Twiss/beam-size quantities SIMBA's other Twiss
readers populate (s, beta, alpha, emittance, sigma, mean position/momentum).
Extend with more transport-table identifiers (see
``laura/RFTrack/RFTrack_API_notes.md`` §8 Table 4.1) only if a real use case
needs them — don't replicate every field speculatively.
"""
import numpy as np


def interpret_rftrack_data(self, lattice_name: str, table: np.ndarray, columns: list) -> None:
    """
    Append one RF-Track transport-table result onto this SIMBA ``twiss`` object.

    Parameters
    ----------
    lattice_name: str
        Name of the lattice/section the table was generated from.
    table: np.ndarray
        Result of ``Lattice.get_transport_table(...)``, one row per sampled point.
    columns: list
        The identifiers requested from ``get_transport_table``, in column order
        (e.g. ``["%S", "%beta_x", "%beta_y", ...]``), so this function knows
        which column is which regardless of the exact identifier string used.
    """
    col = {name.lstrip("%"): table[:, i] for i, name in enumerate(columns)}
    n = table.shape[0]

    def _append(attr, key, default=0.0, scale=1.0):
        values = col[key] * scale if key in col else np.full(n, default)
        current = getattr(self, attr)
        current.val = np.append(current.val, values)

    _append("s", "S")
    _append("z", "S")
    _append("mean_x", "mean_x", scale=1e-3)  # mm -> m
    _append("mean_y", "mean_y", scale=1e-3)
    _append("beta_x", "beta_x")
    _append("beta_y", "beta_y")
    _append("alpha_x", "alpha_x")
    _append("alpha_y", "alpha_y")
    _append("enx", "emitt_x")
    _append("eny", "emitt_y")
    _append("sigma_x", "sigma_x", scale=1e-3)
    _append("sigma_y", "sigma_y", scale=1e-3)
    _append("sigma_t", "sigma_t")
    _append("mean_cp", "mean_P")
    self.lattice_name.val = np.append(self.lattice_name.val, np.full(n, lattice_name))


def read_rftrack_transport_table(
    self,
    lattice,
    lattice_name: str,
    identifiers: str = "%S %mean_x %mean_y %beta_x %beta_y %alpha_x %alpha_y "
    "%emitt_x %emitt_y %sigma_x %sigma_y %sigma_t %mean_P",
    reset: bool = True,
) -> None:
    """
    Read an RF-Track ``Lattice``'s (or ``Volume``'s) transport table directly
    (in-process — no file I/O, unlike ``Twiss/astra.py`` which parses
    ``Xemit``/``Yemit``/``Zemit`` files off disk) into this SIMBA ``twiss`` object.

    Parameters
    ----------
    lattice: RF_Track.Lattice
        Lattice that has already been tracked (``get_transport_table`` requires
        ``element.set_tt_nsteps(N)`` to have been called on at least one element).
    lattice_name: str
        Name recorded against every sampled point.
    identifiers: str
        Space-separated RF-Track transport-table identifiers to request.
    reset: bool
        Clear existing twiss arrays before appending, mirroring
        ``read_astra_twiss_files``/``read_ocelot_twiss_files``.
    """
    if reset:
        self.reset_dicts()
    columns = identifiers.split()
    table = np.asarray(lattice.get_transport_table(identifiers))
    interpret_rftrack_data(self, lattice_name, table, columns)
