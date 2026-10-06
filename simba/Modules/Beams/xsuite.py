import os
import numpy as np

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
from .. import constants
from ..units import UnitValue

def read_xsuite_beam_file(self, filename, zstart=0, s=0, ref_index=None, t_reference=None):
    import xobjects as xo
    import xpart as xp
    if has_cupy:
        context = xo.ContextCupy()
    else:
        context = xo.ContextCpu()
    if isinstance(filename, str):
        if ".json" in filename:
            import json
            try:
                with open(filename, 'r') as fid:
                    particles = xp.Particles.from_dict(
                        json.load(fid),
                        _context=context,
                    )
            except ValueError:
                with open(filename, 'r') as fid:
                    particles = xp.Particles.from_dict(
                        json.load(fid),
                        _context=context,
                        mass0=float(self.E0_eV.val)
                    )
        elif ".pkl" in filename:
            import pickle
            with open(filename, 'rb') as fid:
                try:
                    particles = xp.Particles.from_dict(
                        pickle.load(fid),
                        _context=context,
                        mass0=self.E0_eV.val
                    )
                except TypeError:
                    particles = xp.Particles.from_dict(
                        pickle.load(fid),
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
    # q0 is in units of e
    self._beam.particle_charge = UnitValue(
        np.full(len(mass), particles.q0 * constants.elementary_charge),
        units="C",
    )
    self._beam.particle_rest_energy = UnitValue(
        (
                self._beam.particle_mass * constants.speed_of_light ** 2
        ),
        units="J",
    )
    # self._beam.gamma = UnitValue(parray.gamma, units="")
    self._beam.x = UnitValue(alive("x"), units="m")
    self._beam.y = UnitValue(alive("y"), units="m")

    lag = -alive("zeta") / (alive("beta0") * constants.speed_of_light)
    if t_reference is None:
        t_reference = alive("s") / constants.speed_of_light
    self._beam.t = UnitValue(t_reference + lag, units="s")
    p0c = alive("p0c")
    p_total = p0c * (1 + alive("delta"))
    cpx = alive("px") * p0c
    cpy = alive("py") * p0c
    self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(
        np.sqrt(p_total**2 - cpx**2 - cpy**2) * self.q_over_c, units="kg*m/s"
    )
    # q0 is the charge *state* (-1 for an electron), not a charge
    q0 = float(np.ravel(particles.q0)[0])
    weights = alive("weight")
    if np.any(weights != 1) or not macro_charge:
        total = abs(np.sum(weights * q0)) * constants.elementary_charge
    else:
        total = macro_charge * len(keep)
    self._beam.set_total_charge(np.sign(q0) * total)
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1), units="")
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))
    if ref_index is not None:
        self.reference_particle_index = int(ref_index)
        """ If we have a reference particle, t=0 is relative to it """
        self._beam.z = UnitValue(zstart +
            (-1 * self._beam.Bz * constants.speed_of_light) * (
                self._beam.t - self._beam.t[self.reference_particle_index]
            ),
            units="m",
        )
        self.reference_particle = [
            getattr(self._beam, coord)[self.reference_particle_index]
            for coord in self.reference_particle_coords
        ]
    else:
        """ If we don't have a reference particle, t=0 is relative to mean(t) """
        self.reference_particle_index = None
        self._beam.z = UnitValue(zstart +
            (-1 * self._beam.Bz * constants.speed_of_light) * (
                self._beam.t - np.mean(self._beam.t)
            ),
            units="m",
        )
        self.reference_particle = None
    self._beam.s = UnitValue(s, units="m")


def write_xsuite_beam_file(
    self, filename: str = None, write: bool = True, s_start: float = 0,
    p0c: float = None, t0: float = None,
):
    """Save a json file for xsuite."""
    import xobjects as xo
    import xtrack as xt
    if has_cupy:
        context = xo.ContextCupy()
    else:
        context = xo.ContextCpu()

    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].strip(".xsuite") + ".xsuite.json"
    mass0 = self._beam.particle_rest_energy_eV.val
    q0 = self._beam.chargesign[0]
    if p0c is None:
        p0c = self._beam.centroids.mean_cp.val
    if t0 is None:
        t0 = np.mean(self.t.val)
    x = self.x.val
    y = self.y.val
    beta0 = p0c / np.sqrt(p0c**2 + mass0**2)
    zeta = -(self.t.val - t0) * beta0 * constants.speed_of_light
    px = self.cpx.val / p0c
    py = self.cpy.val / p0c
    delta = self.cp.val / p0c - 1
    s = self.t.val * constants.speed_of_light
    total = getattr(self._beam, "total_charge", None)
    total = 0.0 if total is None else abs(float(getattr(total, "val", total)))
    weight = total / (len(x) * constants.elementary_charge) if total > 0 else 1.0

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
        weight=np.full(len(x), weight),
    )
    import json
    if write:
        with open(filename, 'w') as fid:
            json.dump(particles.to_dict(), fid, cls=xo.JEncoder)
    return particles
