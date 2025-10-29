import numpy as np

try:
    from scipy.stats import gaussian_kde
except ImportError:
    pass


class kde:

    def __init__(self, beam):
        self.beam = beam

    def get_vals_6d(self, beam):
        means = [np.mean(p) for p in beam]
        stds = [np.std(p) for p in beam]
        sliced_beam = [((p - np.mean(p)) / np.std(p)) for p in beam]
        return sliced_beam, means, stds

    def _kde_bw_func(self, bandwidth, *args, **kwargs):
        return bandwidth / x.std(ddof=1)

    def _kde_function(self, beam, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scipy"""
        # bw = partial(self._kde_bw_func, bandwidth, beam)
        return gaussian_kde(beam, bw_method=bandwidth, **kwargs)

    def resample(self, npart, bandwidth=0.2, **kwargs):
        beam = [getattr(self.beam, a) for a in ["x", "y", "z", "px", "py", "pz"]]
        prebeam, means, stds = self.get_vals_6d(beam)
        values = np.vstack(prebeam)
        kernel = self._kde_function(values, bandwidth, **kwargs)
        kdebeam = kernel.resample(npart)
        postbeam = [(p * s) + m for p, m, s in zip(*[kdebeam, means, stds])]
        return postbeam
