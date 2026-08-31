import numpy as np
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
        Initial z-position (defaults to zero; other values not yet tested)

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
    zval = (zval - zstart)# * constants.speed_of_light
    total_npart_actual = int(self._beam.total_charge.val / self._beam.particle_charge.val[0])
    npart_actual = np.full(len(self._beam.x), abs(int(total_npart_actual / len(self._beam.x))))
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
        npart_actual,
        boost=boost,
    )
    return bunch

def particles_to_beam(self, species: "Particles", zpos: float = 0) -> None:
    """
    Convert an FBPIC `Particles` object back into the internal beam representation.

    Parameters
    ----------
    self: :class:`~simba.Modules.Beams.beam`
        The beam object to populate
    species: fbpic.particles.particles.Particles
        The FBPIC species to read back
    zpos: float
        Longitudinal position to add to the particle z co-ordinates
    """
    self.code = "fbpic"
    self._beam.particle_rest_energy_eV = self.E0_eV
    on_gpu = getattr(species, "use_cuda", False)
    if on_gpu:
        species.receive_particles_from_gpu()
    try:
        x = np.asarray(species.x, dtype=float)
        y = np.asarray(species.y, dtype=float)
        z = np.asarray(species.z, dtype=float)
        ux = np.asarray(species.ux, dtype=float)
        uy = np.asarray(species.uy, dtype=float)
        uz = np.asarray(species.uz, dtype=float)
        w = np.asarray(species.w, dtype=float)
        q = float(species.q)
        m = float(species.m)
    finally:
        if on_gpu:
            species.send_particles_to_gpu()

    self._beam.x = UnitValue(x, "m")
    self._beam.y = UnitValue(y, "m")
    self._beam.z = UnitValue(z + zpos, "m")
    self._beam.t = UnitValue((z + zpos) / constants.speed_of_light, "s")
    self._beam.px = UnitValue(ux * m * constants.speed_of_light, "kg*m/s")
    self._beam.py = UnitValue(uy * m * constants.speed_of_light, "kg*m/s")
    self._beam.pz = UnitValue(uz * m * constants.speed_of_light, "kg*m/s")
    self._beam.particle_mass = UnitValue(np.full(len(x), m), units="kg")
    self._beam.charge = UnitValue(q * w, "C")
    self._beam.total_charge = UnitValue(float(np.sum(q * w)), "C")
    self._beam.nmacro = w
    self._beam.status = UnitValue(np.full(len(x), 5))


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
    self._beam.t = UnitValue((z + z_offset) / constants.speed_of_light, "s")
    self._beam.px = UnitValue(px, "kg*m/s")
    self._beam.py = UnitValue(py, "kg*m/s")
    self._beam.pz = UnitValue(pz, "kg*m/s")
    self._beam.particle_mass = UnitValue(m, units="kg")
    self._beam.nmacro = w
    self._beam.status = UnitValue(np.full(npart, 5))
    if charge is not None:
        self._beam.charge = UnitValue(np.full(npart, charge / npart), "C")
        self._beam.total_charge = UnitValue(charge, "C")
    else:
        self._beam.charge = UnitValue(q * w, "C")
        self._beam.total_charge = UnitValue(float(np.sum(q * w)), "C")
