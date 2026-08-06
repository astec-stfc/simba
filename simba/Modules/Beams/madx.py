import numpy as np

from ..units import UnitValue
from .. import constants

speed_of_light = constants.speed_of_light
elementary_charge = constants.elementary_charge


def beam_to_madx_coords(self, p0c: float) -> dict:
    """
    Convert this :class:`~simba.Modules.Beams.beam` object into MAD-X
    canonical coordinates (X, PX, Y, PY, T, PT) for a given reference
    momentum. The momenta are normalised to `p0c`, ``PT`` is the energy
    deviation ``(E - E0)/(p0*c)``, and ``T = -c(t - <t>)`` (T > 0 = bunch
    head), following the conventions in Chapter 1 of the MAD-X manual.

    Parameters
    ----------
    p0c: float
        Reference momentum in eV/c

    Returns
    -------
    Dict
        Dictionary with the canonical coordinate arrays, the mean time
        `tbar` and the reference beta/energy
    """
    m0 = np.mean(self.particle_rest_energy_eV.val)
    E0ref = np.sqrt(p0c**2 + m0**2)
    beta0 = p0c / E0ref
    tarr = np.array(self.t.val)
    tbar = float(np.mean(tarr))
    return {
        "x": np.array(self.x.val),
        "px": np.array(self.cpx.val) / p0c,
        "y": np.array(self.y.val),
        "py": np.array(self.cpy.val) / p0c,
        "t": -speed_of_light * (tarr - tbar),
        "pt": (np.array(self.energy.val) - E0ref) / p0c,
        "tbar": tbar,
        "beta0": beta0,
        "E0ref": E0ref,
    }


def madx_coords_to_beam(
    self,
    coords: dict,
    p0c: float,
    tbar0: float,
    s_local: float,
    zpos: float,
    spos: float,
    charge_total: float,
    ref_index: int = None,
):
    """
    Convert MAD-X canonical coordinates at an observation point back into
    a generic :class:`~simba.Modules.Beams.beam` object, so that the
    distributions can be interpreted in the same way as those produced by
    the other tracking codes. The mass, charge sign and species of the new
    beam are taken from this (reference) beam.

    Parameters
    ----------
    coords: Dict
        Dictionary with (x, px, y, py, t, pt) arrays from the MAD-X track
        table
    p0c: float
        Reference momentum in eV/c
    tbar0: float
        Mean arrival time of the bunch at the start of the segment [s]
    s_local: float
        Position of the observation point along the segment [m]
    zpos: float
        Global z position of the observation point [m]
    spos: float
        Global s position of the observation point [m]
    charge_total: float
        Total charge of the (surviving) bunch [C]
    ref_index: int, optional
        Index of the reference particle

    Returns
    -------
    :class:`~simba.Modules.Beams.beam`
        The output beam object
    """
    m0 = np.mean(self.particle_rest_energy_eV.val)
    E0ref = np.sqrt(p0c**2 + m0**2)
    beta0 = p0c / E0ref
    E = E0ref + np.array(coords["pt"]) * p0c
    cp = np.sqrt(E**2 - m0**2)
    cpx = np.array(coords["px"]) * p0c
    cpy = np.array(coords["py"]) * p0c
    cpz = np.sqrt(cp**2 - cpx**2 - cpy**2)
    t = tbar0 + s_local / (beta0 * speed_of_light) - np.array(coords["t"]) / speed_of_light
    npart = len(coords["x"])
    newbeam = type(self)()
    newbeam.code = "MADX"
    newbeam.filename = ""
    nb = newbeam._beam
    mass = self.particle_mass
    mass = np.mean(mass.val) if hasattr(mass, "val") else constants.m_e
    chargesign = np.sign(np.mean(self.Q.val)) or -1
    nb.particle_mass = UnitValue(np.full(npart, mass), units="kg")
    nb.particle_rest_energy = UnitValue(
        nb.particle_mass * speed_of_light**2, units="J"
    )
    nb.particle_rest_energy_eV = UnitValue(
        nb.particle_rest_energy / elementary_charge, units="eV/c"
    )
    nb.particle_charge = UnitValue(
        np.full(npart, chargesign * elementary_charge), units="C"
    )
    nb.x = UnitValue(np.array(coords["x"]), units="m")
    nb.y = UnitValue(np.array(coords["y"]), units="m")
    nb.t = UnitValue(t, units="s")
    q_over_c = elementary_charge / speed_of_light
    nb.px = UnitValue(cpx * q_over_c, units="kg*m/s")
    nb.py = UnitValue(cpy * q_over_c, units="kg*m/s")
    nb.pz = UnitValue(cpz * q_over_c, units="kg*m/s")
    nb.set_total_charge(chargesign * abs(charge_total))
    nb.nmacro = UnitValue(np.full(npart, 1))
    nb.status = UnitValue(np.full(npart, 5))
    if ref_index is not None:
        newbeam.reference_particle_index = int(ref_index)
        tref = nb.t[int(ref_index)]
    else:
        tref = np.mean(nb.t)
    nb.z = UnitValue(
        zpos + (-1 * nb.Bz * speed_of_light) * (nb.t - tref), units="m"
    )
    nb.s = UnitValue(spos, units="m")
    if ref_index is not None:
        newbeam.reference_particle = [
            getattr(nb, coord)[int(ref_index)]
            for coord in newbeam.reference_particle_coords
        ]
    newbeam.species = self.species
    return newbeam
