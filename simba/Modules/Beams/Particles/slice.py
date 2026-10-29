"""Slice properties of a particle distribution."""
import numpy as np

from ...units import UnitValue
from ... import constants
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
)
from typing import Dict


class slice(BaseModel):
    """Slice properties of a particle distribution, binned in time."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    _slicelength: int | float = 0
    """Slice length [s]."""

    _slices: int = 0
    """Number of slices."""

    time_binned: Dict = {"beam": None, "slices": None, "slice_length": None}
    """Beam and settings of the last time binning, so it is not repeated."""

    _hist: np.ndarray = None
    """Histogram counts of the last binning."""

    _cp_Bins: UnitValue = None
    """Momentum bin edges."""

    _cp_binned: np.ndarray = None
    """Momentum bin index of each particle."""

    _tfbins: list = None
    """Boolean mask of each bin."""

    _cpbins: UnitValue = None
    """Momenta of the particles in each bin."""

    _tbins: UnitValue = None
    """Times of the particles in each bin."""

    _t_Bins: UnitValue = None
    """Time bin edges."""

    _t_binned: np.ndarray = None
    """Time bin index of each particle."""

    def __init__(self, beam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam = beam

    def have_we_already_been_binned(self) -> bool:
        """
        True if the beam has already been binned with the current settings.

        Returns
        -------
        bool
        """
        return bool(
            self.time_binned["beam"] == self.beam
            and self.time_binned["slices"] == self._slices
            and self.time_binned["slice_length"] == self._slicelength
        )

    def update_binned_parameters(self) -> None:
        """Record the current beam and settings in :attr:`time_binned`."""
        self.time_binned = {
            "beam": self.beam,
            "slices": self._slices,
            "slice_length": self._slicelength,
        }

    @computed_field
    @property
    def slice_length(self) -> UnitValue:
        """Slice length [s]; setting it rebins."""
        return UnitValue(self._slicelength, "s")

    @slice_length.setter
    def slice_length(self, slicelength: UnitValue | float) -> None:
        self._slicelength = slicelength
        self.bin_time()

    @computed_field
    @property
    def slices(self) -> int:
        """Number of slices; setting it calls :meth:`set_slices`."""
        return self._slices

    @slices.setter
    def slices(self, slices: int):
        self.set_slices(slices)

    def set_slices(self, slices: int) -> None:
        """
        Split the bunch length into a number of slices and rebin.

        Parameters
        ----------
        slices: int
            Number of slices; 0 gives slices of 0.1 ps
        """
        twidth = np.ptp(self.beam.t, axis=0)
        if twidth == 0:
            t = self.beam.z / (-1 * self.beam.Bz * constants.speed_of_light)
            twidth = np.ptp(t, axis=0)
        if slices == 0:
            slices = int(twidth / 0.1e-12)
        self._slices = slices
        self._slicelength = twidth / slices
        self.bin_time()

    def bin_time(self) -> None:
        """
        Bin the particles in time by :attr:`slice_length`, unless already binned.

        Uses z / (beta_z c) if all particles share one t; with no slice length set, uses 20 slices.
        """
        if not self.have_we_already_been_binned():
            if len(self.beam.t) > 0:
                if not self.slice_length > 0:
                    self._slice_length = 0
                twidth = np.ptp(self.beam.t, axis=0)
                if twidth == 0:
                    t = self.beam.z / (-1 * self.beam.Bz * constants.speed_of_light)
                    twidth = np.ptp(t, axis=0)
                else:
                    t = self.beam.t
                if not self.slice_length > 0.0:
                    self.slice_length = twidth / 20.0
                nbins = max([1, int(np.ceil(twidth / self.slice_length))]) + 2
                self._hist, binst = np.histogram(
                    t,
                    bins=nbins,
                    range=(
                        np.min(t) - self.slice_length,
                        np.max(t) + self.slice_length,
                    ),
                )
                self._t_Bins = binst
                self._t_binned = np.digitize(t, self._t_Bins)
                self._tfbins = [[self._t_binned == i] for i in range(1, len(binst))]
                self._tbins = UnitValue(
                    [np.array(self.beam.t)[tuple(tbin)] for tbin in self._tfbins],
                    units="s",
                    dtype=np.ndarray,
                )
                self._cpbins = UnitValue(
                    [np.array(self.beam.cp)[tuple(tbin)] for tbin in self._tfbins],
                    units="eV/c",
                    dtype=np.ndarray,
                )
                self.update_binned_parameters()

    def bin_momentum(self, width: float=10**6) -> None:
        """
        Bin the particles in momentum; this overwrites the time bins.

        Parameters
        ----------
        width: float
            Bin width [eV/c]; None splits the momentum range into :attr:`slices` bins
        """
        pwidth = max(self.beam.cp) - min(self.beam.cp)
        slice_length_cp = pwidth / self.slices if width is None else width
        nbins = max([1, int(np.ceil(pwidth / slice_length_cp))]) + 2
        self._hist, binst = np.histogram(
            self.beam.cp,
            bins=nbins,
            range=(
                min(self.beam.cp) - slice_length_cp,
                max(self.beam.cp) + slice_length_cp,
            ),
        )
        self._cp_Bins = binst
        self._cp_binned = np.digitize(self.beam.cp, self._cp_Bins)
        self._tfbins = [np.array([self._cp_binned == i]) for i in range(1, len(binst))]
        self._cpbins = UnitValue(
                    [np.array(self.beam.cp)[tuple(cpbin)] for cpbin in self._tfbins],
                    units="eV/c",
                    dtype=np.ndarray,
                )
        self._tbins = UnitValue(
                    [np.array(self.beam.t)[tuple(tbin)] for tbin in self._tfbins],
                    units="s",
                    dtype=np.ndarray,
                )

    @computed_field
    @property
    def slice_bins(self) -> UnitValue:
        """Time bin centres [s]."""
        if not hasattr(self, "slice"):
            self.bin_time()
        bins = self._t_Bins
        return (bins[:-1] + bins[1:]) / 2

    @computed_field
    @property
    def slice_cpbins(self) -> UnitValue:
        """Momentum bin centres [eV/c]."""
        if self._cp_Bins is None:
            self.bin_momentum()
            # bin_momentum overwrites the time bins; restore them
            self.time_binned = {"beam": None, "slices": None, "slice_length": None}
            self.bin_time()
        bins = self._cp_Bins
        return (bins[:-1] + bins[1:]) / 2

    @computed_field
    @property
    def slice_momentum(self) -> UnitValue:
        """Mean momentum of each slice [eV/c]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        return UnitValue(
            [cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins],
            units="eV/c",
        )

    @computed_field
    @property
    def slice_momentum_spread(self) -> UnitValue:
        """RMS momentum spread of each slice [eV/c]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        return UnitValue(
            [cpbin.std() if len(cpbin) > 0 else 0 for cpbin in self._cpbins],
            units="eV/c",
        )

    @computed_field
    @property
    def slice_relative_momentum_spread(self) -> UnitValue:
        """Relative momentum spread of each slice [%]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        return UnitValue(
            [
                100 * cpbin.std() / cpbin.mean() if len(cpbin) > 0 else 0
                for cpbin in self._cpbins
            ],
            units="",
        )

    def slice_data(self, data: UnitValue | np.ndarray) -> UnitValue:
        """
        Split a per-particle array into the time slices.

        Parameters
        ----------
        data: UnitValue | np.ndarray

        Returns
        -------
        UnitValue
            Object array with one entry per slice
        """
        if self._tbins is None:
            self.bin_time()
        return UnitValue(
            [data[tuple(tbin)] for tbin in self._tfbins], units=data.units, dtype=object
        )

    def emitbins(self, x: UnitValue | np.ndarray, y: UnitValue | np.ndarray) -> np.ndarray:
        """
        Slice two arrays and pair them with the slice momenta.

        Parameters
        ----------
        x, y: UnitValue or np.ndarray

        Returns
        -------
        np.ndarray
            Rows of (x slice, y slice, momentum slice)
        """
        xbins = self.slice_data(x)
        ybins = self.slice_data(y)
        return np.array([xbins, ybins, self._cpbins]).T

    @computed_field
    @property
    def ex(self) -> UnitValue:
        """Alias of :attr:`slice_horizontal_emittance`."""
        return self.slice_ex

    @computed_field
    @property
    def ey(self) -> UnitValue:
        """Alias of :attr:`slice_vertical_emittance`."""
        return self.slice_ey

    @computed_field
    @property
    def enx(self) -> UnitValue:
        """Alias of :attr:`slice_normalized_horizontal_emittance`."""
        return self.slice_enx

    @computed_field
    @property
    def eny(self) -> UnitValue:
        """Alias of :attr:`slice_normalized_vertical_emittance`."""
        return self.slice_eny

    @computed_field
    @property
    def slice_ex(self) -> UnitValue:
        """Alias of :attr:`slice_horizontal_emittance`."""
        return self.slice_horizontal_emittance

    @computed_field
    @property
    def slice_ey(self) -> UnitValue:
        """Alias of :attr:`slice_vertical_emittance`."""
        return self.slice_vertical_emittance

    @computed_field
    @property
    def slice_enx(self) -> UnitValue:
        """Alias of :attr:`slice_normalized_horizontal_emittance`."""
        return self.slice_normalized_horizontal_emittance

    @computed_field
    @property
    def slice_eny(self) -> UnitValue:
        """Alias of :attr:`slice_normalized_vertical_emittance`."""
        return self.slice_normalized_vertical_emittance

    @computed_field
    @property
    def slice_t(self) -> np.ndarray:
        """Time bin centres [s] as a plain array."""
        return np.array(self.slice_bins)

    @computed_field
    @property
    def slice_z(self) -> np.ndarray:
        """:attr:`slice_t` reversed, for plotting along z; still in seconds."""
        return np.array(list(reversed(np.array(self.slice_bins))))

    @property
    def slice_horizontal_emittance(self) -> UnitValue:
        """Geometric horizontal emittance of each slice [m-rad]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        emitbins = self.emitbins(self.beam.x, self.beam.xp)
        return UnitValue(
            [
                self.beam.emittance.emittance_calc(xbin, xpbin) if len(cpbin) > 0 else 0
                for xbin, xpbin, cpbin in emitbins
            ],
            units="m-rad",
        )

    @property
    def slice_vertical_emittance(self) -> UnitValue:
        """Geometric vertical emittance of each slice [m-rad]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        emitbins = self.emitbins(self.beam.y, self.beam.yp)
        return UnitValue(
            [
                self.beam.emittance.emittance_calc(ybin, ypbin) if len(cpbin) > 0 else 0
                for ybin, ypbin, cpbin in emitbins
            ],
            units="m-rad",
        )

    @property
    def slice_normalized_horizontal_emittance(self) -> UnitValue:
        """Normalised horizontal emittance of each slice [m-rad]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        emitbins = self.emitbins(self.beam.x, self.beam.xp)
        return UnitValue(
            [
                (
                    self.beam.emittance.emittance_calc(xbin, xpbin, cpbin)
                    if len(cpbin) > 0
                    else 0
                )
                for xbin, xpbin, cpbin in emitbins
            ],
            units="m-rad",
        )

    @property
    def slice_normalized_vertical_emittance(self) -> UnitValue:
        """Normalised vertical emittance of each slice [m-rad]."""
        if self._tbins is None or self._cpbins is None:
            self.bin_time()
        emitbins = self.emitbins(self.beam.y, self.beam.yp)
        return UnitValue(
            [
                (
                    self.beam.emittance.emittance_calc(ybin, ypbin, cpbin)
                    if len(cpbin) > 0
                    else 0
                )
                for ybin, ypbin, cpbin in emitbins
            ],
            units="m-rad",
        )

    @computed_field
    @property
    def slice_current(self) -> UnitValue:
        """Current of each slice [A]; slices with fewer than two particles give 0."""
        if self._hist is None:
            self.bin_time()
        absQ = np.abs(self.beam.Q) / len(self.beam.t)
        bin_width = np.diff(self._t_Bins)
        f = lambda bin, width: absQ * (len(bin) / width) if len(bin) > 1 else 0
        return UnitValue(
            [f(bin, width) for bin, width in zip(self._tbins, bin_width)], units="A"
        )

    @computed_field
    @property
    def peak_current(self) -> UnitValue:
        """Maximum of :attr:`slice_current` [A]."""
        peakI = self.slice_current
        return UnitValue(max(abs(peakI)), units="A")

    @computed_field
    @property
    def slice_max_peak_current_slice(self) -> int:
        """Index of the slice with the peak current."""
        peakI = self.slice_current
        return list(abs(peakI)).index(max(abs(peakI)))

    @computed_field
    @property
    def beta_x(self) -> UnitValue:
        """Alias of :attr:`slice_beta_x`."""
        return self.slice_beta_x

    @computed_field
    @property
    def alpha_x(self) -> UnitValue:
        """Alias of :attr:`slice_alpha_x`."""
        return self.slice_alpha_x

    @computed_field
    @property
    def gamma_x(self) -> UnitValue:
        """Alias of :attr:`slice_gamma_x`."""
        return self.slice_gamma_x

    @computed_field
    @property
    def beta_y(self) -> UnitValue:
        """Alias of :attr:`slice_beta_y`."""
        return self.slice_beta_y

    @computed_field
    @property
    def alpha_y(self) -> UnitValue:
        """Alias of :attr:`slice_alpha_y`."""
        return self.slice_alpha_y

    @computed_field
    @property
    def gamma_y(self) -> UnitValue:
        """Alias of :attr:`slice_gamma_y`."""
        return self.slice_gamma_y

    @property
    def slice_beta_x(self) -> UnitValue:
        """Horizontal beta of each slice [m]."""
        xbins = self.slice_data(self.beam.x)
        exbins = self.slice_horizontal_emittance
        emitbins = list(zip(xbins, exbins))
        return UnitValue(
            [self.beam.covariance(x, x) / ex if ex > 0 else 0 for x, ex in emitbins],
            units="m",
        )

    @property
    def slice_alpha_x(self) -> UnitValue:
        """Horizontal alpha of each slice."""
        xbins = self.slice_data(self.beam.x)
        xpbins = self.slice_data(self.beam.xp)
        exbins = self.slice_horizontal_emittance
        emitbins = list(zip(xbins, xpbins, exbins))
        return UnitValue(
            [
                -1 * self.beam.covariance(x, xp) / ex if ex > 0 else 0
                for x, xp, ex in emitbins
            ],
            units="",
        )

    @property
    def slice_gamma_x(self) -> UnitValue:
        """Horizontal gamma of each slice [1/m]."""
        xpbins = self.slice_data(self.beam.xp)
        exbins = self.slice_horizontal_emittance
        emitbins = list(zip(xpbins, exbins))
        return UnitValue(
            [self.beam.covariance(xp, xp) / ex if ex > 0 else 0 for xp, ex in emitbins],
            units="rad/m",
        )

    @property
    def slice_beta_y(self) -> UnitValue:
        """Vertical beta of each slice [m]."""
        ybins = self.slice_data(self.beam.y)
        eybins = self.slice_vertical_emittance
        emitbins = list(zip(ybins, eybins))
        return UnitValue(
            [self.beam.covariance(y, y) / ey if ey > 0 else 0 for y, ey in emitbins],
            units="m",
        )

    @property
    def slice_alpha_y(self) -> UnitValue:
        """Vertical alpha of each slice."""
        ybins = self.slice_data(self.beam.y)
        ypbins = self.slice_data(self.beam.yp)
        eybins = self.slice_vertical_emittance
        emitbins = list(zip(ybins, ypbins, eybins))
        return UnitValue(
            [
                -1 * self.beam.covariance(y, yp) / ey if ey > 0 else 0
                for y, yp, ey in emitbins
            ],
            units="",
        )

    @property
    def slice_gamma_y(self) -> UnitValue:
        """Vertical gamma of each slice [1/m]."""
        ypbins = self.slice_data(self.beam.yp)
        eybins = self.slice_vertical_emittance
        emitbins = list(zip(ypbins, eybins))
        return UnitValue(
            [self.beam.covariance(yp, yp) / ey if ey > 0 else 0 for yp, ey in emitbins],
            units="rad/m",
        )

    def sliceAnalysis(self, density: bool=False) -> tuple:
        """
        Summary of the peak-current slice.

        Parameters
        ----------
        density: bool
            Also return the slice density from :class:`~simba.Modules.Beams.Particles.mve.MVE`

        Returns
        -------
        tuple
            At the peak-current slice: current, std of :attr:`slice_current`, relative
            momentum spread, normalised x and y emittances, momentum, and density (0 unless
            `density`).
        """
        self.bin_time()
        peakIPosition = self.slice_max_peak_current_slice
        slice_density = self.beam.mve.slice_density[peakIPosition] if density else 0
        return (
            self.slice_current[peakIPosition],
            np.std(self.slice_current),
            self.slice_relative_momentum_spread[peakIPosition],
            self.slice_normalized_horizontal_emittance[peakIPosition],
            self.slice_normalized_vertical_emittance[peakIPosition],
            self.slice_momentum[peakIPosition],
            slice_density,
        )

    @computed_field
    @property
    def chirp(self) -> UnitValue:
        """Momentum chirp across the slices within 75% of the peak current [eV/s]."""
        self.bin_time()
        slice_current_centroid_indices = []
        peakIPosition = self.slice_max_peak_current_slice
        peakI = self.slice_current[peakIPosition]
        slicemomentum = self.slice_momentum
        for index, slice_current in enumerate(self.slice_current):
            if abs(peakI - slice_current) < (peakI * 0.75):
                slice_current_centroid_indices.append(index)
        slice_momentum_centroid = [slicemomentum[index] for index in slice_current_centroid_indices]
        chirp = (slice_momentum_centroid[-1] - slice_momentum_centroid[0]) / (
                len(slice_momentum_centroid) * self.slice_length
        )
        return UnitValue(chirp, "eV/s")

    @computed_field
    @property
    def chirp_m1(self) -> UnitValue:
        """
        As :attr:`chirp`, but in fractional momentum.

        Labelled 1/m, but divided by the slice length in seconds, so really 1/s.
        """
        self.bin_time()
        slice_current_centroid_indices = []
        peakIPosition = self.slice_max_peak_current_slice
        peakI = self.slice_current[peakIPosition]
        centralmomentum = self.slice_momentum[peakIPosition]
        slicemomentum = (self.slice_momentum - centralmomentum) / centralmomentum
        for index, slice_current in enumerate(self.slice_current):
            if abs(peakI - slice_current) < (peakI * 0.75):
                slice_current_centroid_indices.append(index)
        slice_momentum_centroid = [slicemomentum[index] for index in slice_current_centroid_indices]
        chirp = (slice_momentum_centroid[-1] - slice_momentum_centroid[0]) / (
                len(slice_momentum_centroid) * self.slice_length
        )
        return UnitValue(chirp, "1/m")

    def get_chirp_coeffs(self, order=3) -> Dict:
        """
        Polynomial fit of fractional slice momentum against z [m], within 75% of the peak current.

        Parameters
        ----------
        order: int
            Polynomial order

        Returns
        -------
        Dict
            Coefficients keyed ``order_N``
        """
        peakIPosition = self.slice_max_peak_current_slice
        peakI = self.slice_current[peakIPosition]
        centralmomentum = self.slice_momentum[peakIPosition]

        # fractional momentum deviation δ
        slicemomentum = (self.slice_momentum - centralmomentum) / centralmomentum

        # longitudinal positions (assuming uniform spacing from slice_length)
        z = np.arange(len(slicemomentum)) * self.slice_length * constants.speed_of_light

        # restrict to region with significant current (same logic as chirp)
        mask = np.abs(self.slice_current - peakI) < (0.75 * peakI)
        z_masked = z[mask]
        delta_masked = slicemomentum[mask]

        # polynomial fit δ(z)
        coeffs = np.polyfit(z_masked, delta_masked, order)  # highest power first

        return {f"order_{order - i}": coeff for i, coeff in enumerate(coeffs)}
