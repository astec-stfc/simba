from simba.Modules.Beams.Particles.minimumVolumeEllipse import (
    getMinVolEllipse,
)
import numpy as np
from scipy.stats import gaussian_kde
from functools import partial
from scipy.spatial import ConvexHull
from ...units import UnitValue

class MVE:

    def __init__(self, beam):
        self.beam = beam
        self.sliceProperties = dict()

    def kde_bw_func(self, bandwidth, x, *args, **kwargs):
        return bandwidth / x.std(ddof=1)

    def kde_function(self, x, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scipy"""
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        # Taken from https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        if (
            not hasattr(self, "_kde_x")
            or not len(x) == len(self._kde_x)
            or not np.allclose(x, self._kde_x)
            or not bandwidth == self._kde_bandwidth
        ):
            self._kde_x = x
            self._kde_bandwidth = bandwidth
            bw = partial(self.kde_bw_func, bandwidth, x)
            self._kde_function = gaussian_kde(x, bw_method=bw, **kwargs)
        return self._kde_function

    def PDF(self, x, x_grid, bandwidth=0.2, **kwargs):
        kde = self.kde_function(x, bandwidth, **kwargs)
        return kde.evaluate(x_grid)

    def PDFI(self, x, x_grid, bandwidth=0.2, **kwargs):
        kde = self.kde_function(x, bandwidth, **kwargs)
        vals = kde.evaluate(x_grid)
        return self.beam.total_charge * vals

    def CDF(self, x, x_grid, bandwidth=0.2, **kwargs):
        kde = self.kde_function(x, bandwidth, **kwargs)
        cdf = np.vectorize(lambda e: kde.integrate_box_1d(x_grid[0], e))
        return cdf(x_grid)

    def FWHM(self, X, Y, frac=0.5):
        frac = 1.0 / frac if frac > 1 else frac
        d = Y - (max(Y) * frac)
        indexes = np.where(d > 0)[0]
        return abs(X[indexes][-1] - X[indexes][0]), indexes

    @property
    def volume(self):
        return self.volume6D(
            self.beam.x,
            self.beam.y,
            self.beam.z - np.mean(self.beam.z),
            self.beam.cpx / self.beam.cpz,
            self.beam.cpy / self.beam.cpz,
            ((self.beam.cpz / np.mean(self.beam.cp)) - 1),
        )

    @property
    def density(self):
        return len(self.beam.x) / self.volume

    def volume6D(self, x, y, t, xp, yp, cp):
        if len(x) < 10:
            return 1e6
        else:
            beam = list(zip(x, y, t, xp, yp, cp))
            return ConvexHull(beam, qhull_options="QJ").volume

    def mve_emittance(self, x, xp, p=None):
        (center, radii, rotation, hullP) = getMinVolEllipse(list(zip(x, xp)), 0.01)
        emittance = radii[0] * radii[1]
        if p is None:
            return UnitValue(emittance, "m-rad")
        else:
            gamma = np.mean(p) / self.beam.E0_eV
            return UnitValue(gamma * emittance, "m-rad")

    @property
    def normalized_mve_horizontal_emittance(self):
        return self.mve_emittance(self.beam.x, self.beam.xp, self.beam.cp)

    @property
    def normalized_mve_vertical_emittance(self):
        return self.mve_emittance(self.beam.y, self.beam.yp, self.beam.cp)

    @property
    def horizontal_mve_emittance(self):
        return self.mve_emittance(self.beam.x, self.beam.xp)

    @property
    def vertical_mve_emittance(self):
        return self.mve_emittance(self.beam.y, self.beam.yp)

    @property
    def slice_6D_Volume(self):
        if not hasattr(self, "_tbins") or not hasattr(self, "_cpbins"):
            self.beam.slice.bin_time()
        xbins = self.beam.slice.slice_data(self.beam.x)
        ybins = self.beam.slice.slice_data(self.beam.y)
        zbins = self.beam.slice.slice_data(self.beam.z - np.mean(self.beam.z))
        pxbins = self.beam.slice.slice_data(self.beam.cpx / self.beam.cpz)
        pybins = self.beam.slice.slice_data(self.beam.cpy / self.beam.cpz)
        pzbins = self.beam.slice.slice_data(((self.beam.cpz / np.mean(self.beam.cp)) - 1))
        emitbins = list(zip(xbins, ybins, zbins, pxbins, pybins, pzbins))
        self.sliceProperties["6D_Volume"] = np.array(
            [self.volume6D(*a) for a in emitbins]
        )
        return self.sliceProperties["6D_Volume"]

    @property
    def slice_density(self):
        if not hasattr(self, "_tbins") or not hasattr(self, "_cpbins"):
            self.beam.slice.bin_time()
        xbins = self.beam.slice.slice_data(self.beam.x)
        volume = self.slice_6D_Volume
        self.sliceProperties["Density"] = np.array(
            [len(x) / v for x, v in zip(xbins, volume)]
        )
        return self.sliceProperties["Density"]

    def mvesliceAnalysis(self):
        self.slice = {}
        self.bin_time()
        peakIPosition = self.beam.slice.slice_max_peak_current_slice
        return (
            self.beam.slice.slice_peak_current[peakIPosition],
            np.std(self.beam.slice.slice_peak_current),
            self.beam.slice.slice_relative_momentum_spread[peakIPosition],
            self.beam.slice.slice_normalized_mve_horizontal_emittance[peakIPosition],
            self.beam.slice.slice_normalized_mve_vertical_emittance[peakIPosition],
            self.beam.slice.slice_momentum[peakIPosition],
            self.beam.slice.slice_density[peakIPosition],
        )

    @property
    def slice_normalized_mve_horizontal_emittance(self):
        if not hasattr(self, "_tbins") or not hasattr(self, "_cpbins"):
            self.beam.slice.bin_time()
        emitbins = self.beam.slice.emitbins(self.beam.x, self.beam.xp)
        self.sliceProperties["Normalized_mve_Horizontal_Emittance"] = np.array(
            [
                self.mve_emittance(xbin, xpbin, cpbin) if len(cpbin) > 0 else 0
                for xbin, xpbin, cpbin in emitbins
            ]
        )
        return self.sliceProperties["Normalized_mve_Horizontal_Emittance"]

    @property
    def slice_normalized_mve_vertical_emittance(self):
        if not hasattr(self, "_tbins") or not hasattr(self, "_cpbins"):
            self.beam.slice.bin_time()
        emitbins = self.beam.slice.emitbins(self.beam.y, self.beam.yp)
        self.sliceProperties["Normalized_mve_Vertical_Emittance"] = np.array(
            [
                self.mve_emittance(ybin, ypbin, cpbin) if len(cpbin) > 0 else 0
                for ybin, ypbin, cpbin in emitbins
            ]
        )
        return self.sliceProperties["Normalized_mve_Vertical_Emittance"]
