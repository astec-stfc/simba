import numpy as np
from typing import Dict, List
from ..units import UnitValue
from .. import constants
import h5py

def beam_to_particles(
        self,
        simulation: "Simulation",
        boost: "BoostConverter" = None,
        zstart: float=0,
) -> "Particles":
    """
    Convert the internal beam representation to an FBPIC Particles object.
    The function `add_particle_bunch_from_arrays` is used; see `FBPIC bunch utils`_.

    .. _FBPIC bunch utils: https://github.com/fbpic/fbpic/blob/dev/fbpic/lpa_utils/bunch.py

    Parameters
    ----------
    self: :class:`~SimulationFramework.Modules.Beams.beam`
        The beam object
    simulation: FBPIC `Simulation` object
        The FBPIC `Simulation` class
    boost: FBPIC `BoostConverter` class, optional
        Lorentz-boosted frame object
    zstart: float
        Lattice z that maps to z = 0 in the FBPIC box

    Returns
    -------
    fbpic.particles.particles.Particles
        FBPIC Particles object
    """
    from fbpic.lpa_utils.bunch import add_particle_bunch_from_arrays
    mass = self._beam.particle_mass
    if isinstance(mass, UnitValue):
        mass = mass.val
    if not isinstance(mass, float):
        mass = mass[0]
    # charge = self._beam.get("charge", np.full(len(self.x), -constants.elementary_charge)).val
    pxval = self._beam.px.val if isinstance(self._beam.px, UnitValue) else self._beam.px
    pyval = self._beam.py.val if isinstance(self._beam.py, UnitValue) else self._beam.py
    pzval = self._beam.pz.val if isinstance(self._beam.pz, UnitValue) else self._beam.pz
    px = pxval / self.q_over_c / self.particle_rest_energy_eV.val
    py = pyval / self.q_over_c / self.particle_rest_energy_eV.val
    pz = pzval / self.q_over_c / self.particle_rest_energy_eV.val
    xval = self._beam.x.val if isinstance(self._beam.x, UnitValue) else self._beam.x
    yval = self._beam.y.val if isinstance(self._beam.y, UnitValue) else self._beam.y
    zval = self._beam.z.val if isinstance(self._beam.z, UnitValue) else self._beam.z
    zval = zval - zstart
    # One weight (number of real particles) per macroparticle, so bunches of
    # different charge per macroparticle -- a driver and a witness -- keep it.
    charge = self._beam.charge.val if isinstance(self._beam.charge, UnitValue) else self._beam.charge
    weights = np.abs(np.asarray(charge, dtype=float) / self._beam.particle_charge.val)
    if boost is False:
        boost = None
    bunch = add_particle_bunch_from_arrays(
        simulation,
        self._beam.particle_charge.val[0],
        self._beam.particle_mass.val[0],
        xval,
        yval,
        zval,
        px,
        py,
        pz,
        weights,
        boost=boost,
    )
    return bunch

def _species_arrays(species: "Particles", uz_min: float | None = None) -> Dict:
    """
    Copy one species' particle arrays to numpy, keeping uz > uz_min.

    ``Simulation.step`` already hands the particle data back from the GPU when
    it returns, so they are normally numpy arrays here; ``.get()`` covers any
    left on the device (cupy).
    """
    def host(a):
        return np.array(a.get() if hasattr(a, "get") else a, dtype=float)

    arrays = {k: host(getattr(species, k)) for k in ("x", "y", "z", "ux", "uy", "uz", "w")}
    if uz_min is not None:
        keep = arrays["uz"] > uz_min
        arrays = {k: v[keep] for k, v in arrays.items()}
    n = len(arrays["x"])
    arrays["q"] = np.full(n, float(species.q))
    arrays["m"] = np.full(n, float(species.m))
    return arrays


def particles_to_beam(
    self,
    species: "Particles | List[Particles]",
    zpos: float = 0,
    uz_min: "List[float | None] | None" = None,
) -> None:
    """
    Convert FBPIC `Particles` objects back into the internal beam representation.

    Parameters
    ----------
    self: :class:`~simba.Modules.Beams.beam`
        The beam object to populate
    species: fbpic.particles.particles.Particles, or a list of them
        The FBPIC species to read back; several are concatenated into one beam,
        e.g. the incoming bunch and the electrons injected into the wake
    zpos: float
        Longitudinal position to add to the particle z co-ordinates
    uz_min: list of float or None, optional
        Per species, keep only particles with ``uz`` above this (None keeps
        all), e.g. to drop injected electrons that were not trapped
    """
    species = list(species) if isinstance(species, (list, tuple)) else [species]
    uz_min = uz_min or [None] * len(species)
    parts = [_species_arrays(sp, cut) for sp, cut in zip(species, uz_min)]
    a = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    x, y, z, ux, uy, uz, w, q, m = (a[k] for k in ("x", "y", "z", "ux", "uy", "uz", "w", "q", "m"))

    self.code = "fbpic"
    self._beam.particle_rest_energy_eV = self.E0_eV
    self._beam.x = UnitValue(x, "m")
    self._beam.y = UnitValue(y, "m")
    self._beam.z = UnitValue(z + zpos, "m")
    self._beam.t = UnitValue(_arrival_time(z, ux, uy, uz), "s")
    self._beam.px = UnitValue(ux * m * constants.speed_of_light, "kg*m/s")
    self._beam.py = UnitValue(uy * m * constants.speed_of_light, "kg*m/s")
    self._beam.pz = UnitValue(uz * m * constants.speed_of_light, "kg*m/s")
    self.set_mass_and_charge(m, q, len(x))
    self._beam.charge = UnitValue(q * w, "C")
    self._beam.total_charge = UnitValue(float(np.sum(q * w)), "C")
    self._beam.nmacro = w
    self._beam.status = UnitValue(np.full(len(x), 5))


def _arrival_time(z, ux, uy, uz) -> np.ndarray:
    """
    Arrival time of each particle relative to the bunch centroid, from a
    snapshot at fixed time: the head (larger z) arrives first, matching the
    ASTRA reader's ``t = -(z - zref)/(beta_z*c)``.
    """
    beta_z = uz / np.sqrt(1 + ux**2 + uy**2 + uz**2)
    return -(z - np.mean(z)) / (beta_z * constants.speed_of_light)


def _openpmd_record(group, path: str, npart: int) -> np.ndarray:
    """
    Read one openPMD record, whether it is stored as a dataset or as a constant.

    Parameters
    ----------
    group: h5py.Group
        The species group, e.g. ``/data/10/particles/bunch``
    path: str
        Record path within the species group, e.g. ``position/x``
    npart: int
        Number of macroparticles, used to broadcast constant records

    Returns
    -------
    np.ndarray
        The record, in SI units
    """
    record = group[path]
    unit_si = record.attrs.get("unitSI", 1.0)
    if isinstance(record, h5py.Dataset) and record.shape:
        return np.asarray(record[:], dtype=float) * unit_si
    # Constant record: the value lives in the attributes.
    return np.full(npart, float(record.attrs["value"]) * unit_si)


def read_fbpic_beam_file(self, filename, z_offset=0, charge=None, species=None):
    """
    Read an FBPIC openPMD particle dump into the internal beam representation.

    This is how a boosted-frame run is read back: the raw
    `fbpic.particles.particles.Particles` arrays are in the boosted frame, so the
    lab-frame distribution has to come from the snapshots written by
    `BackTransformedParticleDiagnostic` instead.

    Parameters
    ----------
    self: :class:`~simba.Modules.Beams.beam`
        The beam object to populate
    filename: str
        Path to the openPMD HDF5 file
    z_offset: float
        Longitudinal position to add to the particle z co-ordinates
    charge: float, optional
        Total bunch charge in C. If given it overrides the per-particle
        weighting stored in the file.
    species: str, optional
        Name of the species to read. Defaults to ``bunch`` when present, and
        otherwise to the only species in the file.

    Raises
    ------
    KeyError
        If the requested species is not in the file
    """
    self.code = "fbpic"
    self._beam.particle_rest_energy_eV = self.E0_eV
    self.filename = filename
    with h5py.File(filename, "r") as f:
        iteration = next(iter(f["data"]))
        particles = f[f"data/{iteration}/particles"]
        if species is None:
            species = "bunch" if "bunch" in particles else next(iter(particles))
        if species not in particles:
            raise KeyError(
                f"Species {species!r} not found in {filename}. "
                f"Available species: {list(particles)}."
            )
        group = particles[species]
        npart = group["position/z"].shape[0]

        x = _openpmd_record(group, "position/x", npart)
        y = _openpmd_record(group, "position/y", npart)
        z = _openpmd_record(group, "position/z", npart)
        if "positionOffset" in group:
            x = x + _openpmd_record(group, "positionOffset/x", npart)
            y = y + _openpmd_record(group, "positionOffset/y", npart)
            z = z + _openpmd_record(group, "positionOffset/z", npart)
        px = _openpmd_record(group, "momentum/x", npart)
        py = _openpmd_record(group, "momentum/y", npart)
        pz = _openpmd_record(group, "momentum/z", npart)
        w = _openpmd_record(group, "weighting", npart)
        q = _openpmd_record(group, "charge", npart)
        m = _openpmd_record(group, "mass", npart)

    self._beam.x = UnitValue(x, "m")
    self._beam.y = UnitValue(y, "m")
    self._beam.z = UnitValue(z + z_offset, "m")
    mc = m * constants.speed_of_light
    self._beam.t = UnitValue(_arrival_time(z, px / mc, py / mc, pz / mc), "s")
    self._beam.px = UnitValue(px, "kg*m/s")
    self._beam.py = UnitValue(py, "kg*m/s")
    self._beam.pz = UnitValue(pz, "kg*m/s")
    self.set_mass_and_charge(m, q, npart)
    self._beam.nmacro = w
    self._beam.status = UnitValue(np.full(npart, 5))
    if charge is not None:
        self._beam.charge = UnitValue(np.full(npart, charge / npart), "C")
        self._beam.total_charge = UnitValue(charge, "C")
    else:
        self._beam.charge = UnitValue(q * w, "C")
        self._beam.total_charge = UnitValue(float(np.sum(q * w)), "C")
