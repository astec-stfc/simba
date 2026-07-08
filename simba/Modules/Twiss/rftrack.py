"""
Convert RF-Track's transport table into SIMBA's generic ``twiss`` object.

ponytail: covers the core Twiss/beam-size quantities SIMBA's other Twiss
readers populate (s, beta, alpha, emittance, sigma, mean position/momentum).
Extend with more transport-table identifiers (see
``laura/RFTrack/RFTrack_API_notes.md`` §8 Table 4.1) only if a real use case
needs them — don't replicate every field speculatively.
"""
import os

import h5py
import numpy as np

from .. import constants


def transport_table_to_dict(table: np.ndarray, columns: list, zstart: float = 0.0) -> dict:
    """
    Turn a raw ``Lattice.get_transport_table(...)`` result into a
    ``{identifier: column}`` dict -- the shape both :func:`interpret_rftrack_data`
    and :func:`save_rftrack_twiss_hdf` work with.

    Parameters
    ----------
    table: np.ndarray
        Result of ``Lattice.get_transport_table(...)``, one row per sampled point.
    columns: list
        The identifiers requested from ``get_transport_table``, in column order
        (e.g. ``["%S", "%beta_x", "%beta_y", ...]``).
    zstart: float
        Added onto ``S`` [m] -- RF-Track's own transport table starts at 0 for
        every ``Lattice`` (an in-process Python object with no absolute
        coordinate, unlike ASTRA/Elegant's own z, which is already absolute
        because it comes from a namelist ``Zstart``). Pass the physical z
        position of this lattice's first element (e.g.
        ``self.startObject.physical.start.z``) so RF-Track's twiss ``s`` lines
        up with every other code's, instead of restarting from 0 per section.
    """
    col = {name.lstrip("%"): table[:, i] for i, name in enumerate(columns)}
    if "S" in col:
        col["S"] = col["S"] + zstart
    return col


def save_rftrack_twiss_hdf(filename: str, twiss: dict) -> None:
    """
    Write a transport-table dict (see :func:`transport_table_to_dict`) to an
    HDF5 file, one top-level dataset per column -- mirrors
    ``Modules/Twiss/ocelot.py``'s ``save_ocelot_twiss_hdf``, so
    :func:`read_rftrack_twiss_files` (and therefore
    ``Framework.save_summary_files()``'s ``Twiss_Summary.hdf5``) can read it
    back the same way.
    """
    with h5py.File(filename, "w") as f:
        for key, value in twiss.items():
            try:
                f.create_dataset(key, data=value)
            except Exception:
                pass


def interpret_rftrack_data(self, lattice_name: str, col: dict) -> None:
    """
    Append one RF-Track transport-table result onto this SIMBA ``twiss`` object.

    Parameters
    ----------
    lattice_name: str
        Name of the lattice/section the table was generated from.
    col: dict
        ``{identifier: column}``, as built by :func:`transport_table_to_dict`
        (live tracking) or read back from an HDF5 file (see
        :func:`read_rftrack_twiss_files`).
    """
    n = len(next(iter(col.values())))

    def _append(attr, key, default=0.0, scale=1.0):
        values = col[key] * scale if key in col else np.full(n, default)
        current = getattr(self, attr)
        current.val = np.append(current.val, values)

    # Scales convert RF-Track's own transport-table units (RFTrack_API_notes.md
    # §8 Table 4.1: mean_x/y [mm], beta_x/y [m] (already SI, no scale),
    # alpha_x/y [-] (dimensionless, no scale), emitt_x/y [mm*mrad], sigma_x/y
    # [mm], sigma_t [mm/c], mean_P [MeV/c]) into what `twiss_defaults` (see
    # Modules/Twiss/__init__.py) declares for each SIMBA field (enx/eny
    # [m-rad], sigma_t [s], mean_cp [eV/c]) -- verified against a real
    # end-to-end simba.Framework().track() run, where Twiss_Summary.hdf5 came
    # out with emittance/sigma_t/mean_cp orders of magnitude off from the
    # per-lattice *_twiss.rftrack.hdf5 file (which stores RF-Track's raw,
    # unconverted units and is unaffected by this).
    _append("s", "S")
    _append("z", "S")
    _append("mean_x", "mean_x", scale=1e-3)  # mm -> m
    _append("mean_y", "mean_y", scale=1e-3)
    _append("beta_x", "beta_x")
    _append("beta_y", "beta_y")
    _append("alpha_x", "alpha_x")
    _append("alpha_y", "alpha_y")
    _append("enx", "emitt_x", scale=1e-6)  # mm*mrad -> m-rad
    _append("eny", "emitt_y", scale=1e-6)
    _append("sigma_x", "sigma_x", scale=1e-3)
    _append("sigma_y", "sigma_y", scale=1e-3)
    _append("sigma_t", "sigma_t", scale=1e-3 / constants.speed_of_light)  # mm/c -> s
    _append("mean_cp", "mean_P", scale=1e6)  # MeV/c -> eV/c
    self.lattice_name.val = np.append(self.lattice_name.val, np.full(n, lattice_name))


def read_rftrack_transport_table(
    self,
    lattice,
    lattice_name: str,
    identifiers: str = "%S %mean_x %mean_y %beta_x %beta_y %alpha_x %alpha_y "
    "%emitt_x %emitt_y %sigma_x %sigma_y %sigma_t %mean_P",
    reset: bool = True,
    zstart: float = 0.0,
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
    zstart: float
        LAURA-resolved arc-length ``physical.s`` [m] of this lattice's first
        element, added onto ``S`` -- see :func:`transport_table_to_dict`.
    """
    if reset:
        self.reset_dicts()
    columns = identifiers.split()
    table = np.asarray(lattice.get_transport_table(identifiers))
    interpret_rftrack_data(self, lattice_name, transport_table_to_dict(table, columns, zstart=zstart))


def read_rftrack_twiss_files(self, filename, reset: bool = True) -> None:
    """
    Read one or more RF-Track twiss HDF5 files (written by
    :func:`save_rftrack_twiss_hdf`, e.g. by
    ``rftrackLattice.postProcess()``) into this SIMBA ``twiss`` object.

    File-based counterpart of :func:`read_rftrack_transport_table` -- this is
    the one wired into ``Modules/Twiss/__init__.py``'s ``codes``/
    ``code_signatures`` so RF-Track lattices are picked up by
    ``twiss.load_directory()``/``Framework.save_summary_files()``'s
    ``Twiss_Summary.hdf5`` the same way ASTRA/Ocelot/etc. are, instead of only
    being readable via the live in-memory ``Lattice`` object.

    Parameters
    ----------
    filename: str or list
        Path (or list of paths) to a ``*_twiss.rftrack.hdf5`` file.
    reset: bool
        Clear existing twiss arrays before appending.
    """
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_rftrack_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        col = {}
        with h5py.File(filename, "r") as f:
            for key in f.keys():
                col[key] = np.array(f[key])
        interpret_rftrack_data(self, lattice_name, col)
