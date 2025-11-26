import math
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
from ..units import nice_array, nice_scale_prefix
from mpl_axes_aligner import align
from ..Twiss import twissParameter, twiss_defaults
from nala.translator.converters.converter import translate_elements

# from units import nice_array, nice_scale_prefix

CMAP0 = copy(plt.get_cmap("viridis"))
CMAP0.set_under("white")
CMAP1 = copy(plt.get_cmap("plasma"))


def trans(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def ASTRA_TW_FieldMap(fielddat, start, stop, cells, p):
    zpos = list(fielddat[:, 0])
    startpos = zpos.index(start)
    stoppos = zpos.index(stop)
    halfcell1 = fielddat[:startpos]
    halfcell2 = fielddat[stoppos:]
    rfcell = fielddat[startpos:stoppos]
    n_cells = int(cells / p)
    cell_length = rfcell[-1, 0] - rfcell[0, 0]
    dat = list(halfcell1)
    for i in range(0, n_cells + 1, 1):
        dat += list(1.0 * rfcell)
        rfcell[:, 0] += cell_length
    halfcell2[:, 0] += n_cells * cell_length
    dat += list(halfcell2)
    dat = np.array(dat)
    return dat


def fieldmap_data(element, master_lattice):
    """
    Loads the fieldmap in absolute coordinates.

    If a fieldmaps dict is given, these will be used instead of loading the file.

    """

    # Position
    try:
        if element.field_reference_position == "start":
            offset = element.physical.start.z
        elif element.field_reference_position == "end":
            offset = element.physical.end.z
        else:
            offset = element.physical.middle.z
    except AttributeError:
        offset = element.phyiscal.start.z

    # Scaling
    try:
        scale = element.field_amplitude
    except AttributeError:
        scale = element.simulation.field_amplitude
    if element.hardware_type.lower() == "rfcavity":
        scale = scale / 1e6

    # file
    element = translate_elements(elements=[element], master_lattice_location=master_lattice)[element.name]
    element.update_field_definition()
    field = element.simulation.field_definition
    data = field.get_field_data(code="astra")

    if field.field_type == "1DElectroDynamic" and field.cavity_type == "TravellingWave":
        dat = ASTRA_TW_FieldMap(
            np.transpose([field.z.value.val, field.Ez.value.val]),
            field.start_cell_z,
            field.end_cell_z,
            element.n_cells,
            field.mode_denominator,
        )
    else:
        dat = data
    dat[:, 0] += offset
    x = dat[:, 1]
    normalise = max(x.min(), x.max(), key=abs)
    dat[:, 1] *= scale / normalise
    return dat


class magnet_plotting_data:

    def __init__(
        self,
        kinetic_energy=None,
    ):
        if kinetic_energy is not None:
            self.z, self.kinetic_energy = trans(kinetic_energy)
        else:
            self.z, self.kinetic_energy = ([0.0], [1.0])

    def half_rectangle(self, e, half_height):
        return np.array(
            [
                [e.physical.start.z, 0],
                [e.physical.start.z, half_height],
                [e.physical.end.z, half_height],
                [e.physical.end.z, 0],
            ]
        )

    def full_rectangle(self, e, half_height, width=0):
        return np.array(
            [
                [e.physical.start.z - width, -half_height],
                [e.physical.start.z - width, half_height],
                [e.physical.end.z + width, half_height],
                [e.physical.end.z + width, -half_height],
            ]
        )

    def quadrupole(self, e):
        # if e.gradient is None:
        strength = np.sign(e.k1l) * 0.5
        # else:
        #     idx = find_nearest(self.z, e.middle[2])
        #     ke = self.kinetic_energy[idx]
        #     strength = 1.0 / (3.3356 * ke / 1e6) * e.gradient
        return self.half_rectangle(e, strength), "red"

    def sextupole(self, e):
        # if e.gradient is None:
        strength = np.sign(e.k2l) * 0.5
        # else:
        #     idx = find_nearest(self.z, e.middle[2])
        #     ke = self.kinetic_energy[idx]
        #     strength = 1.0 / (3.3356 * ke / 1e6) * e.gradient
        return self.half_rectangle(e, strength), "green"

    def dipole(self, e):
        strength = np.sign(e.angle) * 0.4  # e.angle
        return self.half_rectangle(e, strength), "blue"

    def beam_position_monitor(self, e):
        strength = 0.1  # e.angle
        return self.full_rectangle(e, strength), "purple"

    def screen(self, e):
        strength = 0.33  # e.angle
        return self.full_rectangle(e, strength), "green"

    def aperture(self, e):
        strength = 0.15  # e.angle
        return self.full_rectangle(e, strength, width=0.01), "black"

    def wall_current_monitor(self, e):
        strength = 0.33  # e.angle
        return self.full_rectangle(e, strength), "brown"


def load_elements(
    lattice,
    bounds=None,
    sections="All",
    types=["RFCavity", "Solenoid"],
    kinetic_energy=None,
    verbose=False,
    scale=1,
):
    master_lattice = lattice.global_parameters["master_lattice_location"]
    fmap = {}
    mpd = magnet_plotting_data(kinetic_energy=kinetic_energy)
    for t in types:
        fmap[t] = {}
        if sections == "All":
            elements = [lattice[e["name"]] for e in lattice.getElementType(t)]
        else:
            elements = []
            for s in sections:
                elements += [lattice[e["name"]] for e in lattice[s].getElementType(t)]
        if bounds is not None:
            elements = [
                e
                for e in elements
                if e.physical.start.z <= bounds[1]
                and e.physical.end.z >= bounds[0] - 0.1
            ]
        for e in elements:
            if (
                (t == "RFCavity" or t == "Solenoid")
                and hasattr(e, "field_definition")
                and e.field_definition is not None
            ):
                fmap[t][e.name] = fieldmap_data(e, master_lattice)
            elif hasattr(mpd, t):
                fmap[t][e.name] = getattr(mpd, t)(e)
            else:
                print("Missing drawings for", t)
    return fmap


def add_fieldmaps_to_axes(
    lattice,
    axes,
    bounds=None,
    sections="All",
    fields=["RFCavity", "Solenoid"],
    include_labels=True,
    verbose=False,
):
    """
    Adds fieldmaps to an axes.

    """

    max_scale = 0

    fmaps = load_elements(
        lattice, bounds=bounds, sections=sections, verbose=verbose, types=fields
    )
    ax1 = axes

    ax1rhs = ax1.twinx()
    ax = [ax1, ax1rhs]

    ylabel = {"RFCavity": "$E_z$ (MV/m)", "Solenoid": "$B_z$ (T)"}
    color = {"RFCavity": "green", "Solenoid": "blue"}

    for i, section in enumerate(fields):
        a = ax[i]
        for name, data in fmaps[section].items():
            label = f"{section}_{name}"
            c = color[section]
            # if section == 'cavity':# and not section == 'solenoid':
            if section == fields[0]:
                max_scale = (
                    max(abs(data[:, 1]))
                    if max(abs(data[:, 1])) > max_scale
                    else max_scale
                )
            a.plot(*data.T, label=label, color=c)
            a.yaxis.label.set_color(c)
        a.set_ylabel(ylabel[section])

    if len(fields) < 1:
        for a in ax:
            a.set_yticks([])

    data = np.array([[0, 0], [100, 0]])
    ax[0].plot(*data.T, color="black")


def add_magnets_to_axes(
    lattice,
    axes,
    bounds=None,
    sections="All",
    magnets=["quadrupole", "dipole", "sextupole", "beam_position_monitor", "screen"],
    include_labels=True,
    kinetic_energy=None,
    verbose=False,
):
    """
    Adds magnets to an axes.

    """

    max_scale = 0

    fmaps = load_elements(
        lattice,
        bounds=bounds,
        sections=sections,
        verbose=verbose,
        types=magnets,
        scale=max_scale,
        kinetic_energy=kinetic_energy,
    )

    ax1 = axes
    ax1rhs = ax1.twinx()
    ax = [ax1, ax1rhs]

    ylabel = {
        "dipole": r"$\theta$ (rad)",
        "quadrupole": "$K_n$ (T/m)",
    }  # , "sextupole": "$K_2$ (T/$m^2$)"}
    axis = {"dipole": 0, "quadrupole": 1}
    color = {
        "dipole": "blue",
        "quadrupole": "red",
        "sextupole": "green",
        "beam_position_monitor": "purple",
    }

    for section, i in axis.items():
        a = ax[i]
        c = color[section]
        a.set_ylabel(ylabel[section])
        a.yaxis.label.set_color(c)

    for section in color.keys():
        if section in fmaps:
            for name, (data, c) in fmaps[section].items():
                a.fill(*data.T, color=c)

    data = np.array([[0, 0], [100, 0]])
    ax[0].plot(*data.T, color="black")
    if bounds:
        ax1.set_xlim(bounds[0], bounds[1])

    align.yaxes(ax[0], 0, ax[1], 0, 0.5)


def plot_fieldmaps(
    lattice,
    sections="All",
    include_labels=True,
    limits=None,
    figsize=(12, 4),
    fields=["RFCavity", "Solenoid"],
    magnets=["quadrupole", "dipole", "beam_position_monitor", "screen"],
    **kwargs,
):
    """
    Simple fieldmap plot
    """

    fig, axes = plt.subplots(figsize=figsize, **kwargs)

    add_fieldmaps_to_axes(
        lattice,
        axes,
        bounds=limits,
        include_labels=include_labels,
        sections=sections,
        fields=fields,
        # magnets=magnets,
    )


def plot(
    framework_object,
    ykeys=["sigma_x", "sigma_y"],
    ykeys2=["sigma_z"],
    xkey="z",
    limits=None,
    nice=True,
    include_layout=False,
    include_labels=True,
    include_legend=True,
    include_particles=False,
    fields=["RFCavity", "Solenoid"],
    magnets=[
        "quadrupole",
        "dipole",
        "beam_position_monitor",
        "screen",
        "wall_current_monitor",
        "aperture",
    ],
    grid=False,
    **kwargs,
):
    """
    Plots stat output multiple keys.

    If a list of ykeys2 is given, these will be put on the right hand axis. This can also be given as a single key.

    Logical switches, all default to True:
        nice: a nice SI prefix and scaling will be used to make the numbers reasonably sized.

        include_legend: The plot will include the legend

        include_layout: the layout plot will be displayed at the bottom

        include_labels: the layout will include element labels.

    Copied almost verbatim from lume-impact's Impact.plot.plot_stats_with_layout
    """
    twiss = framework_object.twiss  # convenience
    twiss.sort()  # sort before plotting!
    P = framework_object.beams

    if include_layout is not False:
        if "sharex" not in kwargs:
            kwargs["sharex"] = True
        fig, all_axis = plt.subplots(
            3,
            gridspec_kw={"height_ratios": [4, 1, 1]},
            subplot_kw=dict(frameon=False),
            **kwargs,
        )
        plt.subplots_adjust(hspace=0.0)
        ax_field_layout = all_axis[1]
        ax_magnet_layout = all_axis[2]
        ax_plot = [all_axis[0]]
    else:
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]

    if grid:
        ax_plot[0].grid(visible=True, which="major", color="#666666", linestyle="-")

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

    X = twiss.stat(xkey).val
    if not isinstance(X, twissParameter):
        if xkey in list(twiss_defaults.keys()):
            X = twissParameter(val=X, **twiss_defaults[xkey])
        else:
            X = twissParameter(val=X, name=xkey, unit="")

    # Only get the data we need
    if limits:
        good = np.logical_and(X.val >= limits[0], X.val <= limits[1])
        idx = list(np.where(good is True)[0])
        if len(idx) > 0:
            if idx[0] > 0:
                good[idx[0] - 1] = True
            if (idx[-1] + 1) < len(good):
                good[idx[-1] + 1] = True
            X = X[good]
        if min(X.val) > limits[0]:
            limits[0] = min(X.val)
        if max(X.val) < limits[1]:
            limits[1] = max(X.val)
    else:
        limits = min(X.val), max(X.val)
        good = slice(None, None, None)  # everything

    # Try particles within these bounds
    Pnames = []
    X_particles = []

    if include_particles:
        # try:
        for pname in range(len(P)):  # Modified from Impact
            xp = np.mean(np.array(getattr(P[pname], xkey)))
            if xp >= limits[0] and xp <= limits[1]:
                Pnames.append(pname)
                X_particles.append(xp)
        X_particles = np.array(X_particles)
    # except:
    #     Pnames = []
    else:
        Pnames = []

    # X axis scaling
    units_x = str(twiss.stat(xkey).unit)
    if nice:
        X.val, factor_x, prefix_x = nice_array(X.val)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout
    if include_layout is not False:
        ax_magnet_layout.set_xlim(limits[0] / factor_x, limits[1] / factor_x)
        ax_magnet_layout.set_xlabel(f"{xkey} ({units_x})")

    # Draw for Y1 and Y2
    linestyles = ["solid", "dashed"]

    legend_labels = []

    ii = -1  # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue

        ax = ax_plot[ix]
        ax.ticklabel_format(useOffset=False)
        linestyle = linestyles[ix]

        ulist = []
        for key in keys:
            Y = twiss.stat(key).val
            if not isinstance(Y, twissParameter):
                if xkey in list(twiss_defaults.keys()):
                    Y = twissParameter(val=Y, **twiss_defaults[key])
                else:
                    Y = twissParameter(val=Y, name=key, unit="")
            ulist.append(Y.unit)

        # Check that units are compatible
        # ulist = [twiss.stat(key).unit for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"

        # String Unit representation
        unit = str(ulist[0])

        # Data
        data = [twiss.stat(key).val[good] for key in keys]

        # Labels
        labels = [twiss.stat(key).label for key in keys]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        for key, dat, label in zip(keys, data, labels):
            legend_labels.append("$" + label.replace("sigma", r"\sigma") + "$")
            ii += 1
            color = "C" + str(ii)
            ax.plot(
                X.val,
                dat / factor,
                label=f"{label} ({unit})",
                color=color,
                linestyle=linestyle,
            )

            # Particles
            if Pnames:
                # try:
                # print(Pnames, [key in P._parameters['data'] for key in Pnames])
                Y_particles = np.array(
                    [
                        (
                            np.std(getattr(P[name], key))
                            if key in P._parameters["data"]
                            else getattr(P[name], key)
                        )
                        for name in Pnames
                    ]
                )
                if not all(v is None for v in Y_particles):
                    ax.scatter(
                        X_particles / factor_x, Y_particles / factor, color=color
                    )
            # except:
            #     pass
        labels = ["$" + k.replace("sigma", r"\sigma") + "$" for k in labels]
        ax.set_ylabel(", ".join(labels) + f" ({unit})")

    # Collect legend
    if include_legend:
        lines = []
        # labels = []
        for ax in ax_plot:
            a, _ = ax.get_legend_handles_labels()
            lines += a
            # labels += b
        ax_plot[0].legend(lines, legend_labels, loc="best")

    # Layout
    if include_layout is not False:

        # Gives some space to the top plot
        # ax_layout.set_ylim(-1, 1.5)

        if xkey == "z":
            # ax_layout.set_axis_off()
            ax_field_layout.set_xlim(limits[0], limits[1])
            ax_magnet_layout.set_xlim(limits[0], limits[1])
        # else:
        #     ax_layout.set_xlabel('mean_z')
        #     limits = (0, I.stop)
        add_fieldmaps_to_axes(
            framework_object.framework,
            ax_field_layout,
            bounds=limits,
            include_labels=include_labels,
            fields=fields,
        )
        add_magnets_to_axes(
            framework_object.framework,
            ax_magnet_layout,
            bounds=limits,
            include_labels=include_labels,
            magnets=magnets,
            kinetic_energy=list(
                zip(twiss.stat("z").val[good], twiss.stat("kinetic_energy").val[good])
            ),
        )
    return plt, fig, all_axis


def getattrsplit(self, attr):
    attrs = attr.split(".")
    for a in attrs:
        self = getattr(self, a)
    return self


def general_plot(
    framework_object,
    ykeys=[],
    ykeys2=[],
    xkey="z",
    limits=None,
    nice=True,
    include_layout=False,
    include_labels=True,
    include_legend=True,
    include_particles=False,
    fields=["RFCavity", "Solenoid"],
    magnets=[
        "quadrupole",
        "dipole",
        "beam_position_monitor",
        "screen",
        "wall_current_monitor",
        "aperture",
    ],
    grid=False,
    **kwargs,
):

    if include_layout is not False:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    else:
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]

    if grid:
        ax_plot[0].grid(b=True, which="major", color="#666666", linestyle="-")

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_plot.append(ax_plot[0].twinx())

    # Ensure we are using numpy arrays
    xdata = getattrsplit(framework_object.twiss, xkey)
    ydata = [getattrsplit(framework_object.twiss, y) for y in ykeys]
    ydata2 = [getattrsplit(framework_object.twiss, y) for y in ykeys2]

    # Split keys
    xkey = xkey.split(".")[-1]
    ykeys = [yk.split(".")[-1] for yk in ykeys]
    ykeys2 = [yk.split(".")[-1] for yk in ykeys2]

    # No need for a legend if there is only one plot
    if len(ydata) == 1 and not ydata2:
        include_legend = False

    X = xdata

    # Only get the data we need
    if limits:
        good = np.logical_and(X >= limits[0], X <= limits[1])
        idx = list(np.where(good is True)[0])
        if len(idx) > 0:
            if idx[0] > 0:
                good[idx[0] - 1] = True
            if (idx[-1] + 1) < len(good):
                good[idx[-1] + 1] = True
            X = X[good]
        if X.min() > limits[0]:
            limits[0] = X.min()
        if X.max() < limits[1]:
            limits[1] = X.max()
    else:
        limits = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # X axis scaling
    units_x = xdata.unit
    if nice:
        X, factor_x, prefix_x = nice_array(X.val)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout
    for ax in ax_plot:
        ax.set_xlim(limits[0] / factor_x, limits[1] / factor_x)
        ax.set_xlabel(f"{xkey} ({units_x})")

    # Draw for Y1 and Y2

    linestyles = ["solid", "dashed"]

    ii = -1  # counter for colors
    for ix, (d, keys) in enumerate([[ydata, ykeys], [ydata2, ykeys2]]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [dat.unit for dat in d]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [key.val[good] for key in d]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        keys = ["$" + k.replace("sigma", r"\sigma") + "$" for k in keys]
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

    # Layout
    if include_layout is not False:

        # Gives some space to the top plot
        # ax_layout.set_ylim(-1, 1.5)

        if xkey == "z":
            # ax_layout.set_axis_off()
            ax_layout.set_xlim(limits[0], limits[1])
        # else:
        #     ax_layout.set_xlabel('mean_z')
        #     limits = (0, I.stop)
        add_fieldmaps_to_axes(
            framework_object.framework,
            ax_layout,
            bounds=limits,
            include_labels=include_labels,
            fields=fields,
            # magnets=magnets,
        )
    return plt, fig, all_axis
