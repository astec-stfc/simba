import numpy as np
from .. import constants
from ..units import UnitValue


def _val(x):
    return x.val if isinstance(x, UnitValue) else x


def particle_bunch_to_beam(self, bunch, zpos=0):
    """
    Convert a Wake-T ParticleBunch into the internal beam representation.

    Parameters
    ----------
    self: :class:`~simba.Modules.Beams.beam`
        The beam object to populate
    bunch: wake_t.ParticleBunch
        The bunch to convert
    zpos: float
        Lattice position of ``xi = 0``; particle z is ``zpos + xi``
    """
    n = len(bunch.x)
    self.set_mass_and_charge(bunch.m_species, bunch.q_species, n)
    self._beam.charge = UnitValue(bunch.q, "C")
    self._beam.total_charge = UnitValue(float(np.sum(bunch.q)), "C")
    self._beam.x = UnitValue(bunch.x, "m")
    self._beam.y = UnitValue(bunch.y, "m")
    self._beam.z = UnitValue(zpos + bunch.xi, "m")
    rest_energy_eV = bunch.m_species * constants.speed_of_light ** 2 / constants.elementary_charge
    self.set_momenta(
        bunch.px * rest_energy_eV,
        bunch.py * rest_energy_eV,
        bunch.pz * rest_energy_eV,
    )
    # Arrival time at zpos: the head (xi > 0) arrives first, as in the ASTRA reader.
    self._beam.t = UnitValue(
        -bunch.xi / (self.Bz.val * constants.speed_of_light), "s"
    )
    self._beam.nmacro = np.full(n, 1)
    self._beam.status = UnitValue(np.full(n, 5))


def beam_to_particle_bunch(self, zstart=0):
    """
    Convert the internal beam representation to a Wake-T ParticleBunch.

    Parameters
    ----------
    self: :class:`~simba.Modules.Beams.beam`
        The beam object
    zstart: float
        Lattice z that maps to ``xi = 0`` in the Wake-T box

    Returns
    -------
    wake_t.ParticleBunch
        The bunch, with one weight (number of real particles) per macroparticle
    """
    from wake_t import ParticleBunch

    mass = _val(self._beam.particle_mass)
    mass = float(np.atleast_1d(mass)[0])
    rest_energy_eV = self.particle_rest_energy_eV.val
    px = _val(self._beam.px) / self.q_over_c / rest_energy_eV
    py = _val(self._beam.py) / self.q_over_c / rest_energy_eV
    pz = _val(self._beam.pz) / self.q_over_c / rest_energy_eV
    charge = np.asarray(_val(self._beam.charge), dtype=float)
    q_species = float(np.atleast_1d(_val(self._beam.particle_charge))[0])
    return ParticleBunch(
        charge / q_species,
        x=_val(self._beam.x),
        y=_val(self._beam.y),
        xi=_val(self._beam.z) - zstart,
        px=px,
        py=py,
        pz=pz,
        q_species=q_species,
        m_species=mass,
    )
