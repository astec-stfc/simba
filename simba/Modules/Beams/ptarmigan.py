"""
Beams to and from Ptarmigan's HDF5 particle files.

Ptarmigan (https://github.com/tgblackburn/ptarmigan) imports one species from
``final-state/<species>/{position, momentum, weight}``: four-positions (ct, x, y, z)
and four-momenta (E, px, py, pz), with their units under ``config/unit``. It writes
every species it ends with in the same layout.
"""

import h5py
import numpy as np

from .. import constants
from ..units import UnitValue

PTARMIGAN_SPECIES = ("electron", "positron", "photon")
"""The species Ptarmigan tracks."""


def write_ptarmigan_beam_file(beam, filename: str, laser_offset=(0.0, 0.0), laser_delay: float = 0.0) -> None:
    """
    Write a beam of one Ptarmigan species as a Ptarmigan import file.

    Positions are relative to the bunch's centre (its reference particle, or else its
    mean), all at one instant, for Ptarmigan to collide at the laser focus with
    ``distance_bt_ips: 0`` and ``auto_timing: true``. Transversely they are relative
    to the laser focus: the beam's own x, y, less ``laser_offset``.
    Ptarmigan's own ``offset`` is not used.

    Parameters
    ----------
    beam: :class:`~simba.Modules.Beams.beam`
        The beam, of one species in :data:`PTARMIGAN_SPECIES`.
    filename: str
        File to write.
    laser_offset: (float, float)
        Transverse position of the laser focus off the beam axis, x and y [m].
    laser_delay: float
        How much later than the bunch centre the laser peak reaches the focus [s]. The file
        holds the bunch as it is when the laser peak passes the focus, so a late laser finds
        it ``v laser_delay`` downstream.
    """
    if beam.species not in PTARMIGAN_SPECIES:
        raise ValueError(f"Ptarmigan tracks {PTARMIGAN_SPECIES}, not {beam.species}")
    index = beam.reference_particle_index
    z = np.asarray(beam.z.val, dtype=float)
    centre = z[index] if index is not None and 0 <= index < len(z) else np.mean(z)
    n = len(z)
    shift = np.mean(beam.Bz.val) * constants.speed_of_light * laser_delay
    position = np.column_stack(
        [np.zeros(n), beam.x.val - laser_offset[0], beam.y.val - laser_offset[1], z - centre + shift]
    )
    momentum = np.column_stack(
        [beam.energy.val, beam.cpx.val, beam.cpy.val, beam.cpz.val]
    ) / 1e6  # MeV(/c)
    nmacro = beam.Particles.nmacro
    if nmacro is None or np.size(nmacro) != n or not np.all(np.asarray(nmacro) > 0):
        nmacro = np.abs(np.asarray(beam.Particles.charge, dtype=float)) / constants.elementary_charge
    with h5py.File(filename, "w") as f:
        f["beam_axis"] = "+z"
        f["config/unit/momentum"] = "MeV/c"
        f["config/unit/position"] = "m"
        group = f.create_group(f"final-state/{beam.species}")
        group["position"] = position
        group["momentum"] = momentum
        group["weight"] = np.broadcast_to(np.asarray(nmacro, dtype=float), (n,)).copy()


def read_ptarmigan_output(filename: str) -> dict:
    """
    The final-state particles of a Ptarmigan output file, by species. The file must
    be in the beam's frame (``output: {coordinate_system: beam}``), so the beam travels along +z.

    Returns
    -------
    dict
        ``{species: {"position": (N, 4) [m], "momentum": (N, 4) [eV(/c)], "weight": (N,)}}``
        for each species with any particles: positions (ct, x, y, z), momenta (E, p).
    """
    scale_momentum = {"MeV/c": 1e6, "MeV": 1e6, "GeV/c": 1e9, "GeV": 1e9, "eV/c": 1.0, "eV": 1.0}
    scale_position = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "micron": 1e-6}
    particles = {}
    with h5py.File(filename, "r") as f:
        units = {key: f[f"config/unit/{key}"][()] for key in ("momentum", "position")}
        units = {k: v.decode() if isinstance(v, bytes) else str(v) for k, v in units.items()}
        if not bool(f["config/output/beam_defines_positive_z"][()]):
            raise ValueError(f"{filename} is not in the beam's frame; set output: coordinate_system: beam")
        for species in PTARMIGAN_SPECIES:
            group = f.get(f"final-state/{species}")
            if group is None or len(group["weight"]) == 0:
                continue
            particles[species] = {
                "position": np.asarray(group["position"][()], dtype=float).reshape(-1, 4)
                * scale_position[units["position"]],
                "momentum": np.asarray(group["momentum"][()], dtype=float).reshape(-1, 4)
                * scale_momentum[units["momentum"]],
                "weight": np.asarray(group["weight"][()], dtype=float),
            }
    return particles


def ptarmigan_particles_to_beam(
    self, particles: dict, primary: str, z_out: float, t_offset: float, laser_offset=(0.0, 0.0)
) -> None:
    """
    Fill a :class:`~simba.Modules.Beams.beam` from :func:`read_ptarmigan_output`, at a plane.

    Ptarmigan leaves the particles at one instant; each is drifted in a straight line
    to the plane ``z_out`` metres past the laser focus, and takes its arrival time there.
    Every species goes into this one beam, each particle with its own mass and charge,
    ``primary`` first; the openPMD writer splits them
    (:meth:`~simba.Modules.Beams.beam.split_species`).

    Parameters
    ----------
    particles: dict
        As returned by :func:`read_ptarmigan_output`.
    primary: str
        The species Ptarmigan was given, which becomes :attr:`species`.
    z_out: float
        Distance of the output plane from the laser focus along the beam [m].
    t_offset: float
        Added to Ptarmigan's times (whose zero is the laser peak at focus) [s].
    laser_offset: (float, float)
        As given to :func:`write_ptarmigan_beam_file`, to put x and y back on the beam axis
        (measured: Ptarmigan's beam-frame output has the same x as its input file).
    """
    from . import SPECIES

    order = [primary] + [s for s in particles if s != primary]
    columns = {k: [] for k in ("x", "y", "t", "cp", "name", "weight")}
    for species in order:
        if species not in particles:
            continue
        p = particles[species]
        ct, x, y, z = p["position"].T
        energy, cpx, cpy, cpz = p["momentum"].T
        ahead = cpz > 0  # only what moves towards the plane can reach it
        speed = constants.speed_of_light / energy[ahead]  # v / (pc) per component
        dt = (z_out - z[ahead]) / (cpz[ahead] * speed)
        columns["x"].append(x[ahead] + cpx[ahead] * speed * dt + laser_offset[0])
        columns["y"].append(y[ahead] + cpy[ahead] * speed * dt + laser_offset[1])
        columns["t"].append(ct[ahead] / constants.speed_of_light + dt + t_offset)
        columns["cp"].append(np.column_stack([cpx, cpy, cpz])[ahead])
        columns["name"].append(np.full(np.count_nonzero(ahead), species, dtype=object))
        columns["weight"].append(p["weight"][ahead])
    data = {k: np.concatenate(v) for k, v in columns.items()}
    cp = data["cp"]
    self.reset_dicts()
    self._beam.x = UnitValue(data["x"], units="m")
    self._beam.y = UnitValue(data["y"], units="m")
    self._beam.t = UnitValue(data["t"], units="s")
    self.set_momenta(cp[:, 0], cp[:, 1], cp[:, 2])
    self._beam.nmacro = UnitValue(data["weight"])
    sign = np.array([SPECIES[n][1] for n in data["name"]], dtype=float)
    self._beam.charge = UnitValue(sign * constants.elementary_charge * data["weight"], units="C")
    self._beam.status = UnitValue(np.full(len(data["x"]), 5))
    self.set_particle_species(data["name"], primary=primary)
    self.code = "Ptarmigan"
