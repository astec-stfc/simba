import os
import sys
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import copy
from ..units import nice_array, nice_scale_prefix

CMAP0 = copy(plt.get_cmap("viridis"))
CMAP0.set_under("white")
CMAP1 = copy(plt.get_cmap("plasma"))


def plot(
    twiss_object,
    ykeys=["sigma_x", "sigma_y"],
    ykeys2=["sigma_z"],
    xkey="z",
    xlim=None,
    nice=True,
    include_labels=True,
    include_legend=True,
    **kwargs,
):
    """
    Plots stat output multiple keys.

    If a list of ykeys2 is given, these will be put on the right hand axis. This can also be given as a single key.

    Logical switches, all default to True:
        nice: a nice SI prefix and scaling will be used to make the numbers reasonably sized.

        include_legend: The plot will include the legend

        include_labels: the layout will include element labels.

    Copied almost verbatim from lume-impact's Impact.plot.plot_stats_with_layout
    """
    I = twiss_object  # convenience
    I.sort()  # sort before plotting!

    fig, all_axis = plt.subplots(**kwargs)
    ax_plot = [all_axis]

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_plot.append(ax_plot[0].twinx())

    # No need for a legend if there is only one plot
    if len(ykeys) == 1 and not ykeys2:
        include_legend = False

    X = I.stat(xkey)

    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # X axis scaling
    units_x = str(I.stat(xkey).unit)
    if nice:
        X, factor_x, prefix_x = nice_array(X.val)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout
    for ax in ax_plot:
        ax.set_xlim(xlim[0] / factor_x, xlim[1] / factor_x)
        ax.set_xlabel(f"{xkey} ({units_x})")

    # Draw for Y1 and Y2

    linestyles = ["solid", "dashed"]

    ii = -1  # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [I.stat(key).unit for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [I.stat(key).val[good] for key in keys]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = "C" + str(ii)
            ax.plot(
                X,
                dat / factor,
                label=f"{key} ({unit})",
                color=color,
                linestyle=linestyle,
            )

        ax.set_ylabel(", ".join(keys) + f" ({unit})")

    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc="best")

    # return fig
