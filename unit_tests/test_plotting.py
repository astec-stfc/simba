import matplotlib

matplotlib.use("Agg")
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from simba.Modules.plotting import plotting
from simba.Modules.Twiss import plot as twiss_plot
from simba.Modules.Twiss import twiss


@pytest.fixture
def tws():
    t = twiss()
    for k in ["z", "s"]:
        t.append(k, [0.0, 1.0, 2.0, 3.0])
    for k in ["sigma_x", "sigma_y", "sigma_z"]:
        t.append(k, [1e-3, 2e-3, 3e-3, 4e-3])
    yield t
    plt.close("all")


def test_twiss_plot_with_xlim(tws):
    twiss_plot.plot(tws, xlim=(0.5, 2.5))
    twiss_plot.plot(tws, nice=False)


def test_plot_with_limits(tws):
    ax, _, _ = plotting.plot(SimpleNamespace(twiss=tws, beams=None), ykeys2=[], limits=(1.0, 2.0))
    assert len(ax.lines[0].get_xdata()) == 4  # the in-range points plus one either side


def test_general_plot_with_limits_and_grid(tws):
    plotting.general_plot(SimpleNamespace(twiss=tws), ykeys=["sigma_x"], limits=(0.5, 1.5), grid=True)
    assert len(tws.z.val) == 4
