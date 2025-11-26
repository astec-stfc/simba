import os
import numpy as np
from sympy.liealgebras.type_e import TypeE

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
from .. import constants
from ..units import UnitValue

def read_xsuite_beam_file(self, filename, zstart=0, s=0, ref_index=None):
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
    self._beam.particle_rest_energy_eV = UnitValue(particles.mass, units="eV/c")
    self._beam.particle_mass = UnitValue(
        particles.mass * constants.e / (constants.speed_of_light**2),
        units="kg",
    )
    self._beam.particle_charge = UnitValue(np.full(len(particles.mass), particles.q0), units="C")
    self._beam.particle_rest_energy = UnitValue(
        (
                self._beam.particle_mass * constants.speed_of_light ** 2
        ),
        units="J",
    )
    # self._beam.gamma = UnitValue(parray.gamma, units="")
    self._beam.x = UnitValue(particles.x, units="m")
    self._beam.y = UnitValue(particles.y, units="m")

    self._beam.t = UnitValue((particles.s - particles.zeta) / constants.speed_of_light, units="s")
    # self._beam.p = UnitValue(parray.energies, units="eV/c")
    self._beam.px = UnitValue(particles.px * particles.p0c * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(particles.py * particles.p0c * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(particles.p0c * (1 + particles.delta) * self.q_over_c, units="kg*m/s")
    self._beam.set_total_charge(abs(np.sum(particles.q0)))
    self._beam.z = UnitValue(zstart +
        (-1 * self._beam.Bz * constants.speed_of_light) * (
            self._beam.t - np.mean(self._beam.t)
        ),
        units="m",
    )
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
        self._beam.z = UnitValue(zstart +
            (-1 * self._beam.Bz * constants.speed_of_light) * (
                self._beam.t - self._beam.t[self.reference_particle_index]
            ),
            units="m",
        )
        self.reference_particle = None
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1), units="")
    self._beam.s = UnitValue(s, units="m")


def write_xsuite_beam_file(self, filename: str=None, write: bool=True, s_start: float=0):
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
    p0c = self._beam.centroids.mean_cp.val
    x = self.x.val
    y = self.y.val
    zeta = (self.t.val - np.mean(self.t.val)) * constants.speed_of_light
    px = self.cpx.val / self.cp.val
    py = self.cpy.val / self.cp.val
    delta = self.deltap.val
    s = self.t.val * constants.speed_of_light


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
    )
    import json
    if write:
        with open(filename, 'w') as fid:
            json.dump(particles.to_dict(), fid, cls=xo.JEncoder)
    return particles
