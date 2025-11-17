import os
import sys
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

# plt.rcParams["axes.axisbelow"] = False
from copy import copy

try:
    from ..units import nice_array, nice_scale_prefix, set_nice_array
except:
    pass

try:
    from fastkde import fastKDE

    fastKDE_installed = True
except ImportError as e:
    print("fastKDE missing - plotScreenImage will use SciPy")
    fastKDE_installed = False

try:
    from scipy import stats

    SciPy_installed = True
except:
    SciPy_installed = False
CMAP0 = copy(plt.get_cmap("viridis"))
CMAP0.set_under("white")
CMAP1 = copy(plt.get_cmap("plasma"))

# beamobject = rbf.beam()


def density_plot(
        particle_group,
        key="x",
        bins=None,
        filename=None,
        **kwargs,
):
    """
    1D density plot. Also see: marginal_plot

    Example:

        density_plot(P, 'x', bins=100)

    """

    if not bins:
        n = len(particle_group)
        bins = int(n / 100)
    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(getattr(particle_group, key))
    if key != "charge":
        w = abs(particle_group.charge)
    else:
        w = np.ones(len(getattr(particle_group, key)))
    u1 = ""  # particle_group.units(key).unitSymbol
    ux = p1 + u1

    labelx = f"{key} ({ux})"

    fig, ax = plt.subplots(**kwargs)
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist / hist_width)
    ax.bar(hist_x, hist_y, hist_width, color="grey")
    # Special label for C/s = A
    if u1 == "s":
        _, hist_prefix = nice_scale_prefix(hist_f / f1)
        ax.set_ylabel(f"{hist_prefix}A")
    else:
        ax.set_ylabel(f"{hist_prefix}C/{ux}")

    ax.set_xlabel(labelx)
    if isinstance(filename, str):
        plt.savefig(filename)



def slice_plot(
        particle_group,
        xkey="t",
        ykey="slice_current",
        xlim=None,
        nice=True,
        include_legend=True,
        subtract_mean=True,
        bins=None,
        filename=None,
        **kwargs,
):
    """
    slice plot. Also see: marginal_plot

    Example:

        slice plot(P, 'slice_current', bins=100)

    """

    P = particle_group

    fig, all_axis = plt.subplots(**kwargs)
    ax_plot = [all_axis]

    if not bins:
        n = len(particle_group)
        bins = int(n / 100)
    P.slice.slices = bins

    X = getattr(P.slice, "slice_" + xkey)
    if subtract_mean:
        X = X - np.mean(X)

    if isinstance(ykey, str):
        ykey = [ykey]
    if not isinstance(ykey, (list, tuple)):
        ykey = [ykey]
    if len(ykey) == 1:
        include_legend = False

    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # X axis scaling
    units_x = "s"  # str(P.units(xkey))
    if nice:
        X, factor_x, prefix_x = nice_array(X)
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
    for ix, keys in enumerate([ykey]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [getattr(P.slice, key).units for key in keys]  # [I.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [np.array(getattr(P.slice, key)[good]) for key in keys]

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

    if isinstance(filename, str):
        plt.savefig(filename)


def marginal_plot(
    particle_group,
    key1="t",
    key2="p",
    bins=None,
    units=["", ""],
    scale=[1, 1],
    subtract_mean=[False, False],
    cmap=None,
    limits=None,
    filename=None,
    **kwargs,
):
    """
    Density plot and projections

    Example:

        marginal_plot(P, 't', 'energy', bins=200)

    """

    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n / 2))

    cmap = CMAP0 if cmap is None else cmap

    if not isinstance(subtract_mean, (list, tuple)):
        subtract_mean = [subtract_mean, subtract_mean]
    if not isinstance(scale, (list, tuple)):
        scale = [scale, scale]

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(
        scale[0]
        * (getattr(particle_group, key1) - subtract_mean[0] * np.mean(getattr(particle_group, key1)))
    )
    y, f2, p2 = nice_array(
        scale[1]
        * (getattr(particle_group, key2) - subtract_mean[1] * np.mean(getattr(particle_group, key2)))
    )
    x = x / scale[0]
    y = y / scale[1]

    w = np.full(len(x), 1)  #
    charge = getattr(particle_group, "charge")

    u1, u2 = [getattr(particle_group, k).units for k in [key1, key2]]
    ux = p1 + u1
    uy = p2 + u2

    labelx = f"{key1} ({ux})"
    labely = f"{key2} ({uy})"

    fig = plt.figure(**kwargs)

    gs = GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_y = fig.add_subplot(gs[1:4, 3])
    # ax_info = fig.add_subplot(gs[0, 3:4])
    # ax_info.table(cellText=['a'])

    # Proper weighting
    ax_joint.hexbin(
        x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=cmap, vmin=1e-20
    )
    if limits is not None:
        ax_joint.axis(limits)

    # Manual histogramming version
    # H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # ax_joint.imshow(H.T, cmap=cmap, vmin=1e-16, origin='lower', extent=extent, aspect='auto')

    # Top histogram
    # Old method:
    # dx = x.ptp()/bins
    # ax_marg_x.hist(x, weights=w/dx/f1, bins=bins, color='gray')
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    # Special label for C/s = A
    if u1 == "s" and abs(np.sum(charge).val) > 0:
        hist_y, hist_f, hist_prefix = nice_array(
            -np.sum(charge).val * hist / hist_width / len(charge)
        )
        ax_marg_x.bar(hist_x, hist_y, hist_width, color="gray")
        _, hist_prefix = nice_scale_prefix(hist_f / f1)
        # print(np.sum(charge).val, hist_f, f1)
        ax_marg_x.set_ylabel(f"{hist_prefix}A")
    else:
        if abs(np.sum(charge).val) > 0:
            hist_y, hist_f, hist_prefix = nice_array(
                -np.sum(charge).val * hist / hist_width / len(charge)
            )
            ax_marg_x.bar(hist_x, hist_y, hist_width, color="gray")
            ax_marg_x.set_ylabel(f"{hist_prefix}C/{uy}")
        else:
            hist_y, hist_f, hist_prefix = nice_array(hist)
            ax_marg_x.bar(hist_x, hist_y, hist_width, color="gray")
            ax_marg_x.set_ylabel(f"{hist_prefix}Counts/{uy}")
    if limits is not None:
        ax_marg_x.set_xlim(limits[0:2])

    # Side histogram
    # Old method:
    # dy = y.ptp()/bins
    # ax_marg_y.hist(y, orientation="horizontal", weights=w/dy, bins=bins, color='gray')
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    if u1 == "s" and abs(np.sum(charge).val) > 0:
        hist_y, hist_f, hist_prefix = nice_array(
            -np.sum(charge).val * hist / hist_width / len(charge)
        )
        ax_marg_y.barh(hist_x, hist_y, hist_width, color="gray")
        ax_marg_y.set_xlabel(f"{hist_prefix}C/{uy}")
    else:
        if abs(np.sum(charge).val) > 0:
            hist_y, hist_f, hist_prefix = nice_array(
                -np.sum(charge).val * hist / hist_width / len(charge)
            )
            ax_marg_y.barh(hist_x, hist_y, hist_width, color="gray")
            ax_marg_y.set_xlabel(f"{hist_prefix}C/{uy}")
        else:
            hist_y, hist_f, hist_prefix = nice_array(hist)
            ax_marg_y.barh(hist_x, hist_y, hist_width, color="gray")
            ax_marg_y.set_xlabel(f"{hist_prefix}Counts/{uy}")
    if limits is not None:
        ax_marg_y.set_ylim(limits[2:])

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)

    if isinstance(filename, str):
        plt.savefig(filename)


def plot(self, keys=None, bins=None, type="density", **kwargs):

    if keys is not None and (
        (isinstance(keys, (list, tuple)) and len(keys) == 1) or isinstance(keys, str)
    ):
        if isinstance(keys, (list, tuple)):
            ykey = keys[0]
        if type == "slice" or "slice_" in ykey:
            return slice_plot(self, ykey=ykey, bins=bins, **kwargs)
        elif type == "density":
            return density_plot(self, key=ykey, bins=bins, **kwargs)
    else:
        xkey, ykey = keys
        return marginal_plot(self, key1=xkey, key2=ykey, bins=bins, **kwargs)


def plotScreenImage(
    beam,
    keys=["x", "y"],
    scale=[1, 1],
    iscale=1,
    colormap=plt.cm.jet,
    size=None,
    grid=False,
    marginals=False,
    limits=None,
    screen=False,
    use_scipy=False,
    subtract_mean=[False, False],
    title="",
    **kwargs,
):
    # Do the self-consistent density estimate
    key1, key2 = keys
    if not isinstance(subtract_mean, (list, tuple)):
        subtract_mean = [subtract_mean, subtract_mean]
    if not isinstance(scale, (list, tuple)):
        scale = [scale, scale]
    if not isinstance(size, (list, tuple)):
        size = [size, size]

    x, f1, p1 = nice_array(
        scale[0] * (getattr(beam, key1) - subtract_mean[0] * np.mean(getattr(beam, key1)))
    )
    y, f2, p2 = nice_array(
        scale[1] * (getattr(beam, key2) - subtract_mean[1] * np.mean(getattr(beam, key2)))
    )

    u1, u2 = [getattr(beam, k).units for k in keys]
    ux = p1 + u1
    uy = p2 + u2

    labelx = f"{key1} ({ux})"
    labely = f"{key2} ({uy})"

    if fastKDE_installed and not use_scipy:
        myPDF, axes = fastKDE.pdf(x, y, use_xarray=False, **kwargs)
        v1, v2 = axes
    elif SciPy_installed:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        v1, v2 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([v1.ravel(), v2.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        myPDF = np.reshape(kernel(positions).T, v1.shape)
    else:
        raise Exception("fastKDE or SciPy required")
    # normalise the PDF to 1
    myPDF = myPDF / myPDF.max() * iscale

    # Initialise the plot objects
    # start with a square Figure

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    if marginals:
        fig = plt.figure(figsize=(12.41, 12.41))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(8, 2),
            height_ratios=(2, 8),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.95,
            wspace=0.05,
            hspace=0.05,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    else:
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(top=0.95)
        ax = fig.add_subplot()

    # Define ticks
    # Major ticks every 5, minor ticks every 1
    if size[0] is None:
        use_size = False
        if not screen:
            xmin, xmax = [min(v1.flatten()), max(v1.flatten())]
            ymin, ymax = [min(v2.flatten()), max(v2.flatten())]
            size = [xmax - xmin, ymax - ymin]
        else:
            xmin, xmax = -15, 15
            ymin, ymax = -15, 15
            size = [15, 15]
        minvalx = xmin
        maxvalx = xmax
        meanvalx = (xmin + xmax) / 2.0 if not subtract_mean[0] else 0
        minvaly = ymin
        maxvaly = ymax
        meanvaly = (ymin + ymax) / 2.0 if not subtract_mean[1] else 0
    else:
        use_size = True
        maxvalx = size[0] / f1
        minvalx = -maxvalx
        meanvalx = (max(v1) + min(v1)) / 2.0 if not subtract_mean[0] else 0
        maxvaly = size[1] / f2
        minvaly = -maxvaly
        meanvaly = (max(v2) + min(v2)) / 2.0 if not subtract_mean[1] else 0
        size[0] = size[0] / f1
        size[1] = size[1] / f2

    # print(meanvaly, minvaly, maxvaly)
    # major_ticksx = meanvalx + np.arange(
    #     minvalx, maxvalx + (maxvalx - minvalx) / 100, (maxvalx - minvalx) / 4
    # )
    # minor_ticksx = meanvalx + np.arange(
    #     minvalx, maxvalx + (maxvalx - minvalx) / 100, (maxvalx - minvalx) / 40
    # )
    # ax.set_xticks(major_ticksx)
    # ax.set_xticks(minor_ticksx, minor=True)
    # major_ticksy = meanvaly + np.arange(
    #     minvaly, maxvaly + (maxvaly - minvaly) / 100, (maxvaly - minvaly) / 4
    # )
    # minor_ticksy = meanvaly + np.arange(
    #     minvaly, maxvaly + (maxvaly - minvaly) / 100, (maxvaly - minvaly) / 40
    # )
    # # print(minvaly, maxvaly, meanvaly, major_ticksy)
    # ax.set_yticks(major_ticksy)
    # ax.set_yticks(minor_ticksy, minor=True)

    if marginals:
        hist, bin_edges = myPDF.sum(axis=0)[:-1], v1
        hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
        hist_width = np.diff(bin_edges)
        hist_y, hist_f, hist_prefix = nice_array(hist / hist_width)
        ax_histx.bar(hist_x, hist_y, hist_width, color=colormap(hist_y / max(hist_y)))

        hist, bin_edges = myPDF.sum(axis=1)[:-1], v2
        hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
        hist_width = np.diff(bin_edges)
        hist_y, hist_f, hist_prefix = nice_array(hist / hist_width)
        ax_histy.barh(hist_x, hist_y, hist_width, color=colormap(hist_y / max(hist_y)))

    # Make a circle for the edges of the screen
    if screen:
        draw_circle = plt.Circle(
            (meanvalx, meanvaly),
            15,
            fill=True,
            ec="w",
            fc=colormap(0),
            zorder=-1,
        )
        ax.add_artist(draw_circle)
    if screen:
        ax.set_facecolor("k")
    else:
        ax.set_facecolor(colormap(0))

    # Make a circle to clip the PDF
    if screen:
        circ = plt.Circle((meanvalx, meanvaly), 15, facecolor="none")
    else:
        circ = plt.Circle((meanvalx, meanvaly), 3 * max(size), facecolor="none")
    # ax.add_patch(circ) # Plot the outline

    # Plot the PDF
    if grid:
        # Add a grid
        ax.grid(which="minor", color="w", alpha=0.3, clip_path=circ)
        ax.grid(which="major", color="w", alpha=0.55, clip_path=circ)
    # Set the image limits to slightly larger than the screen size
    if limits:
        if isinstance(limits, (int, float)):
            limits = (-limits, limits)
        if np.array(limits).shape == (2, 2):
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            bbox = plt.Rectangle(
                (min(limits[0]), min(limits[1])),
                max(limits[0]) - min(limits[0]),
                max(limits[1]) - min(limits[1]),
                facecolor="none",
                edgecolor="none",
            )
        elif np.array(limits).shape == (2,):
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            # make a bounding box for the limits
            bbox = plt.Rectangle(
                (min(limits), min(limits)),
                max(limits) - min(limits),
                max(limits) - min(limits),
                facecolor="none",
                edgecolor="none",
            )
    elif screen or use_size:
        ax.set_xlim([meanvalx - (size[0] + 0.5), meanvalx + (size[0] + 0.5)])
        ax.set_ylim([meanvaly - (size[1] + 0.5), meanvaly + (size[1] + 0.5)])
        bbox = plt.Rectangle(
            (-(size[0] + 0.5), -(size[1] + 0.5)),
            size[0] + 0,
            size[1] + 0,
            facecolor="none",
            edgecolor="none",
        )
    else:
        ax.set_xlim([min(v1), max(v1)])
        ax.set_ylim([min(v2), max(v2)])
        bbox = plt.Polygon(
            [
                (min(v1), min(v2)),
                (min(v1), max(v2)),
                (max(v1), max(v2)),
                (max(v1), min(v2)),
            ],
            facecolor="none",
            edgecolor="none",
        )
    # ax.add_artist(bbox)

    mesh = ax.pcolormesh(
        v1, v2, myPDF, cmap=colormap, zorder=1, shading="auto"
    )  # , clip_path=bbox)
    # if screen:
    #     mesh.set_clip_path(circ)
    if marginals:
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)
    # ax_histy.set_ylim([-(size + 0.5), (size + 0.5)])
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)

    # Extract the screen name
    file, ext = os.path.splitext(os.path.basename(beam.filename))
    # Set the screen name as the title
    if title == "":
        plt.suptitle(file)
    else:
        plt.suptitle(title)
    # Show the final image
    plt.draw()
    return ax

    if isinstance(filename, str):
        plt.savefig(filename)

def getScreenImage(
    beam,
    keys=["x", "y"],
    scale=[1, 1],
    iscale=1,
    colormap=plt.cm.jet,
    size=None,
    use_scipy=False,
    subtract_mean=[False, False],
    **kwargs,
):
    # Do the self-consistent density estimate
    key1, key2 = keys
    if not isinstance(subtract_mean, (list, tuple)):
        subtract_mean = [subtract_mean, subtract_mean]
    if not isinstance(scale, (list, tuple)):
        scale = [scale, scale]
    if not isinstance(size, (list, tuple)):
        size = [size, size]

    x, f1, p1 = nice_array(
        scale[0] * (beam[key1] - subtract_mean[0] * np.mean(beam[key1]))
    )
    y, f2, p2 = nice_array(
        scale[1] * (beam[key2] - subtract_mean[1] * np.mean(beam[key2]))
    )

    u1, u2 = [beam[k].units for k in keys]
    ux = p1 + u1
    uy = p2 + u2

    labelx = f"{key1} ({ux})"
    labely = f"{key2} ({uy})"

    if fastKDE_installed and not use_scipy:
        myPDF, axes = fastKDE.pdf(x, y, use_xarray=False, **kwargs)
        v1, v2 = axes
    elif SciPy_installed:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        v1, v2 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([v1.ravel(), v2.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        myPDF = np.reshape(kernel(positions).T, v1.shape)
    else:
        raise Exception("fastKDE or SciPy required")
    # normalise the PDF to 1
    myPDF = myPDF / myPDF.max() * iscale


    # Define ticks
    # Major ticks every 5, minor ticks every 1
    use_size = False
    xmin, xmax = [min(v1.flatten()), max(v1.flatten())]
    ymin, ymax = [min(v2.flatten()), max(v2.flatten())]
    return v1, v2, myPDF, colormap, labelx, labely
