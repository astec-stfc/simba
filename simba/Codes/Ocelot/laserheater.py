from warnings import warn

from numpy import log, sqrt
from ocelot.common.globals import m_e_GeV
from ocelot.cpbd.physics_proc import LaserModulator

from ...Modules.constants import speed_of_light


class WigglerLaserModulator(LaserModulator):
    """
    Ocelot's ``LaserModulator`` for a LAURA wiggler with a ``laser`` (a laser
    heater or modulator), applied between the wiggler's entrance and exit.

    * The amplitude is LAURA's
      :meth:`~laura.models.element.Wiggler.laser_energy_modulation`
      at the beam's energy when it arrives, rather than a fixed ``dE``.
    * The modulation is at the laser's wavelength; a laser off resonance with
      the wiggler gets a warning, as its coupling is not reduced here.
    * The laser envelope is Gaussian in time and space: ``sigma_l`` and
      ``sigma_x/y`` are the rms widths of the field, from the pulse FWHM and
      ``waist / 2``.
    * The wiggler is tracked as an Ocelot ``Undulator``, whose map already
      carries its R56, so ``include_r56`` stays off.
    """

    def __init__(self, wiggler):
        """
        Parameters
        ----------
        wiggler: :class:`~laura.models.element.Wiggler`
            The wiggler, with its laser
        """
        laser = wiggler.laser
        super().__init__(
            Ku=wiggler.deflection_parameter,
            Lu=wiggler.undulator_length,
            lperiod=float(wiggler.magnetic.period),
            sigma_l=speed_of_light * laser.pulse_duration_fwhm / (2 * sqrt(log(2))),
            sigma_x=laser.waist / 2,
            sigma_y=laser.waist / 2,
            laser_peak_pos=-laser.initial_position,
        )
        self.wiggler = wiggler
        self.wavelength = laser.wavelength
        self._checked_resonance = False

    def lambda_ph(self, energy: float) -> float:
        """The laser's wavelength [m], whatever the beam energy."""
        return self.wavelength

    def apply(self, p_array, dz: float) -> None:
        gamma = p_array.E / m_e_GeV
        if not self._checked_resonance:
            self._checked_resonance = True
            resonant = self.wiggler.resonant_wavelength(gamma)
            periods = self.wiggler.undulator_length / self.wiggler.magnetic.period
            if abs(self.wavelength / resonant - 1) > 1 / (2 * periods):
                warn(
                    f"{self.wiggler.name}: laser at {self.wavelength * 1e9:.1f} nm, but the "
                    f"wiggler is resonant at {resonant * 1e9:.1f} nm for this beam; the "
                    "modulation is computed as if on resonance"
                )
        self.dE = self.wiggler.laser_energy_modulation(gamma) * 1e-9
        super().apply(p_array, dz)
