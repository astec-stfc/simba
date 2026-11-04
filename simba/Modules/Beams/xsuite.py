import os
import numpy as np

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
from .. import constants
from ..units import UnitValue
from ...FrameworkHelperFunctions import reference_time

def read_xsuite_beam_file(
    self, filename, zstart=0, s=0, ref_index=None, t_reference=None, start=None, context=None
):
    """Read an Xsuite beam: a ``.json`` or ``.pkl`` file, an
    ``xpart.Particles``, or a ``Particles.to_dict()`` as the ``.json`` holds.

    ``context`` is the xobjects context to build the particles in; the lattice's,
    so that a run held to the CPU does not make GPU particles. Default: the GPU if
    ``cupy`` imports."""
    import xobjects as xo
    import xpart as xp
    if context is None:
        context = xo.ContextCupy() if has_cupy else xo.ContextCpu()

    def from_dict(payload):
        try:
            return xp.Particles.from_dict(payload, _context=context)
        except ValueError:
            if self.is_mixed:
                from . import SPECIES

                mass0 = SPECIES[self.species][0] * constants.speed_of_light**2 / constants.elementary_charge
            else:
                mass0 = float(np.mean(self.E0_eV.val))
            return xp.Particles.from_dict(payload, _context=context, mass0=mass0)

    if isinstance(filename, dict):
        particles = from_dict(filename)
        filename = None
    elif isinstance(filename, str):
        if ".json" in filename:
            import json
            with open(filename) as fid:
                particles = from_dict(json.load(fid))
        elif ".pkl" in filename:
            import pickle
            with open(filename, 'rb') as fid:
                data = pickle.load(fid)
            try:
                particles = xp.Particles.from_dict(
                    data,
                    _context=context,
                    mass0=self.E0_eV.val
                )
            except TypeError:
                particles = xp.Particles.from_dict(
                    data,
                    _context=context,
                )
        else:
            raise ValueError(f"File format not supported for xsuite beam file {filename}.")
    elif isinstance(filename, xp.Particles):
        particles = filename
    else:
        raise ValueError("Input must be a filename or an xpart.Particles instance.")
    self.filename = filename
    self.code = "Xsuite"
    macro_charge = None
    total = getattr(self._beam, "total_charge", None)
    if total is not None and len(getattr(self._beam, "x", [])) > 0:
        macro_charge = abs(float(total)) / len(self._beam.x)
    state = np.asarray(particles.state)
    particle_id = np.asarray(particles.particle_id)
    keep = np.flatnonzero(state > 0)
    keep = keep[np.argsort(particle_id[keep], kind="stable")]

    def alive(name):
        return np.asarray(getattr(particles, name))[keep]

    if ref_index is not None:
        found = np.flatnonzero(particle_id[keep] == int(ref_index))
        ref_index = int(found[0]) if found.size else None
    mass = alive("mass")
    self._beam.particle_rest_energy_eV = UnitValue(mass, units="eV/c")
    self._beam.particle_mass = UnitValue(
        mass * constants.e / (constants.speed_of_light**2),
        units="kg",
    )
    # q0 is in units of e; a particle of another species carries its own ratios to it
    charge_ratio = alive("charge_ratio")
    mass_ratio = alive("mass_ratio")
    mixed = bool(np.any(charge_ratio != 1) or np.any(mass_ratio != 1))
    self._beam.particle_charge = UnitValue(
        np.ravel(particles.q0)[0] * charge_ratio * constants.elementary_charge,
        units="C",
    )
    self._beam.particle_rest_energy = UnitValue(
        (
                self._beam.particle_mass * constants.speed_of_light ** 2
        ),
        units="J",
    )
    self._beam.x = UnitValue(alive("x"), units="m")
    self._beam.y = UnitValue(alive("y"), units="m")

    lag = -alive("zeta") / (alive("beta0") * constants.speed_of_light)
    if t_reference is None and start is not None:
        # open line: the reference's time of flight from start = (t0, s0, p0)
        t_reference = reference_time(s, float(alive("p0c")[0]), float(particles.mass0), start)
    if t_reference is None:
        t_reference = alive("s") / constants.speed_of_light
    self._beam.t = UnitValue(t_reference + lag, units="s")
    # Xsuite's momenta are per unit mass_ratio
    p0c = alive("p0c") * mass_ratio
    p_total = p0c * (1 + alive("delta"))
    cpx = alive("px") * p0c
    cpy = alive("py") * p0c
    cpz = np.sqrt(p_total**2 - cpx**2 - cpy**2)
    self.set_momenta(cpx, cpy, cpz)
    # q0 is the charge *state* (-1 for an electron), not a charge
    q0 = float(np.ravel(particles.q0)[0])
    weights = alive("weight")
    if mixed:
        self._beam.charge = UnitValue(
            weights * q0 * charge_ratio * constants.elementary_charge, units="C"
        )
        self._beam.total_charge = UnitValue(float(np.sum(self._beam.charge)), units="C")
    else:
        if np.any(weights != 1) or not macro_charge:
            total = abs(np.sum(weights * q0)) * constants.elementary_charge
        else:
            total = macro_charge * len(keep)
        self._beam.set_total_charge(np.sign(q0) * total)
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1), units="")
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))
    if ref_index is None:
        self.reference_particle_index = None
    self.set_z_from_t(zstart, ref_index)
    self._beam.s = UnitValue(s, units="m")


def write_xsuite_beam_file(
    self, filename: str = None, write: bool = True, s_start: float = 0,
    p0c: float = None, t0: float = None, context=None,
):
    """Save a json file for xsuite; ``context`` as for :func:`read_xsuite_beam_file`."""
    import xobjects as xo
    import xtrack as xt
    if context is None:
        context = xo.ContextCupy() if has_cupy else xo.ContextCpu()

    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].removesuffix(".xsuite") + ".xsuite.json"
    mixed = self.is_mixed
    if mixed:
        from . import SPECIES

        mass_kg, q0 = SPECIES[self.species]
        mass0 = mass_kg * constants.speed_of_light**2 / constants.elementary_charge
        mass_ratio = np.asarray(self._beam.particle_rest_energy_eV, dtype=float) / mass0
        charge_ratio = np.asarray(self._beam.particle_charge, dtype=float) / (q0 * constants.elementary_charge)
    else:
        mass0 = self._beam.particle_rest_energy_eV.val
        q0 = self._beam.chargesign[0]
        mass_ratio = 1.0
    if p0c is None:
        p0c = self._beam.centroids.mean_cp.val
    if t0 is None:
        t0 = np.mean(self.t.val)
    x = self.x.val
    y = self.y.val
    beta0 = p0c / np.sqrt(p0c**2 + mass0**2)
    zeta = -(self.t.val - t0) * beta0 * constants.speed_of_light
    # Xsuite's momenta are per unit mass_ratio (energy = (energy0 + ptau p0c) mass_ratio)
    px = self.cpx.val / (p0c * mass_ratio)
    py = self.cpy.val / (p0c * mass_ratio)
    delta = self.cp.val / (p0c * mass_ratio) - 1
    if mixed:
        weight = np.abs(np.asarray(self._beam.charge, dtype=float)) / constants.elementary_charge
    else:
        total = getattr(self._beam, "total_charge", None)
        total = 0.0 if total is None else abs(float(getattr(total, "val", total)))
        weight = np.full(len(x), total / (len(x) * constants.elementary_charge) if total > 0 else 1.0)
    species = {"mass_ratio": mass_ratio, "charge_ratio": charge_ratio} if mixed else {}

    particles = xt.Particles(
        _context=context,
        mass0=[mass0],
        q0=q0,
        p0c=[p0c],
        x=x,
        px=px,
        y=y,
        py=py,
        zeta=zeta,
        delta=delta,
        s=s_start,
        weight=weight,
        **species,
    )
    import json
    if write:
        with open(filename, 'w') as fid:
            json.dump(particles.to_dict(), fid, cls=xo.JEncoder)
    return particles
