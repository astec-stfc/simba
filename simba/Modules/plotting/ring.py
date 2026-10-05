"""Plots for the two nonlinear ring studies: dynamic aperture and frequency map.

Both are *magnitude* plots -- turns survived, and a diffusion index -- so both
use a single sequential hue rather than a rainbow, which keeps the ordering
readable and survives colour-vision deficiency and greyscale printing.
``viridis`` is the default that `Modules.Beams.plot` already uses.

The one categorical distinction, survived against lost, is carried by **marker
shape and a legend entry**, never by colour alone: on a map whose whole point
is the boundary between the two, a reader who cannot separate the hues would
lose the result entirely.
"""

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

CMAP = copy(plt.get_cmap("viridis"))
"""Sequential, perceptually uniform and CVD-safe; matches `Beams.plot`."""

SURVIVED_COLOUR = "#2a4858"
"""A single dark step for survivors. They all survived the same number of
turns, so there is no magnitude among them to encode -- the shape and the
legend carry the distinction, and the colour axis is left for the losses."""


def _tidy(axes) -> None:
    """Recede the frame and grid so the data is the darkest thing present."""
    axes.grid(True, linewidth=0.4, alpha=0.3, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_linewidth(0.6)
        axes.spines[side].set_alpha(0.5)


def plot_dynamic_aperture(
    aperture,
    turns: int,
    axes=None,
    title: str = "Dynamic aperture",
    **kwargs,
):
    """Survival over the starting-amplitude grid.

    Parameters
    ----------
    aperture: list
        ``(x, y, turns_survived)`` as :meth:`run_dynamic_aperture` returns.
    turns: int
        Turns asked for, so survivors can be told from losses. A survivor
        reaches ``turns - 1``: Ocelot numbers turns from zero, and testing
        against ``turns`` would mark every particle lost.
    axes: matplotlib.axes.Axes | None
        Drawn into if given, otherwise a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(6, 5))
    data = np.asarray([(float(x), float(y), float(t)) for x, y, t in aperture])
    if not len(data):
        axes.set_title(f"{title} (no data)")
        return axes
    alive = data[:, 2] >= turns - 1
    # Colour carries *when* a particle was lost. Survivors all share one
    # value by definition, so colouring them would encode a constant and
    # throw away the only magnitude there is.
    if (~alive).any():
        points = axes.scatter(
            data[~alive, 0] * 1e3,
            data[~alive, 1] * 1e3,
            c=data[~alive, 2],
            cmap=CMAP,
            marker="X",
            s=34,
            linewidths=0,
            label="lost",
            zorder=2,
            **kwargs,
        )
        bar = axes.figure.colorbar(points, ax=axes)
        bar.set_label("turn lost")
        bar.outline.set_visible(False)
    if alive.any():
        axes.scatter(
            data[alive, 0] * 1e3,
            data[alive, 1] * 1e3,
            marker="o",
            s=26,
            linewidths=0,
            color=SURVIVED_COLOUR,
            label=f"survived {turns} turns",
            zorder=3,
        )
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_title(title)
    if (~alive).any() and alive.any():
        # Neutral handles, built by hand: letting matplotlib take the "lost"
        # swatch from the colour-mapped scatter picks an arbitrary step and
        # implies that one hue *means* lost, when the hue means lost-when.
        # Below the axes, because a full grid leaves no empty corner.
        handles = [
            Line2D(
                [], [], linestyle="none", marker="X", markersize=7,
                color="0.45", label="lost (colour: turn lost)",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=6,
                color=SURVIVED_COLOUR, label=f"survived {turns} turns",
            ),
        ]
        axes.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
        )
    _tidy(axes)
    return axes


def resonance_lines(axes, order: int = 4, **kwargs) -> None:
    """Overlay ``m*Qx + n*Qy = p`` up to ``|m| + |n| <= order``.

    Drawn recessively and thinner with increasing order, because a frequency
    map is read by *which* line a feature sits on -- the lines are reference,
    not data.
    """
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    for m in range(-order, order + 1):
        for n in range(-order, order + 1):
            total = abs(m) + abs(n)
            if total == 0 or total > order:
                continue
            for p in range(-2 * order, 2 * order + 1):
                style = {
                    "color": "0.55",
                    "linewidth": max(0.25, 0.9 - 0.15 * total),
                    "alpha": max(0.15, 0.6 - 0.1 * total),
                    "zorder": 1,
                }
                style.update(kwargs)
                if n == 0:
                    position = p / m
                    if x_min <= position <= x_max:
                        axes.axvline(position, **style)
                elif m == 0:
                    position = p / n
                    if y_min <= position <= y_max:
                        axes.axhline(position, **style)
                else:
                    xs = np.array([x_min, x_max])
                    axes.plot(xs, (p - m * xs) / n, **style)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)


def plot_frequency_map(
    footprint,
    axes=None,
    order: int = 4,
    title: str = "Frequency map",
    **kwargs,
):
    """Tune footprint coloured by the diffusion index.

    Parameters
    ----------
    footprint: list
        ``(x, y, tune_x, tune_y, D)`` as :meth:`run_frequency_map` returns.
    order: int
        Highest resonance order drawn; 0 draws none.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(6, 5))
    data = np.asarray(
        [(float(qx), float(qy), float(d)) for _, _, qx, qy, d in footprint]
    )
    data = data[np.isfinite(data).all(axis=1)] if len(data) else data
    if not len(data):
        axes.set_title(f"{title} (no data)")
        return axes
    points = axes.scatter(
        data[:, 0],
        data[:, 1],
        c=data[:, 2],
        cmap=CMAP,
        s=30,
        linewidths=0,
        zorder=3,
        **kwargs,
    )
    bar = axes.figure.colorbar(points, ax=axes)
    bar.set_label(r"$\log_{10}|\Delta Q|$  (more negative is more regular)")
    bar.outline.set_visible(False)
    axes.set_xlabel(r"$Q_x$")
    axes.set_ylabel(r"$Q_y$")
    axes.set_title(title)
    _tidy(axes)
    if order:
        resonance_lines(axes, order=order)
    return axes


def plot_amplitude_map(
    footprint,
    axes=None,
    title: str = "Amplitude map",
    **kwargs,
):
    """The same diffusion index against *starting amplitude* rather than tune.

    The companion to :func:`plot_frequency_map`: one says which resonance a
    particle is on, this says where in the machine aperture it started.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(6, 5))
    data = np.asarray(
        [(float(x), float(y), float(d)) for x, y, _, _, d in footprint]
    )
    data = data[np.isfinite(data).all(axis=1)] if len(data) else data
    if not len(data):
        axes.set_title(f"{title} (no data)")
        return axes
    points = axes.scatter(
        data[:, 0] * 1e3,
        data[:, 1] * 1e3,
        c=data[:, 2],
        cmap=CMAP,
        s=30,
        linewidths=0,
        zorder=3,
        **kwargs,
    )
    bar = axes.figure.colorbar(points, ax=axes)
    bar.set_label(r"$\log_{10}|\Delta Q|$")
    bar.outline.set_visible(False)
    axes.set_xlabel("initial x [mm]")
    axes.set_ylabel("initial y [mm]")
    axes.set_title(title)
    _tidy(axes)
    return axes
