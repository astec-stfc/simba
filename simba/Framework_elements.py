"""
SimFrame Elements Module

This module defines classes representing specific accelerator lattice elements, all of which inherit from
:class:`~simba.Framework_objects.frameworkElement`. Each element has a function for creating
strings or python objects representing that element for the codes supported, and is able to convert
the generic keywords associated with that class to names that are understood by each code.

Classes:
    - :class:`~simba.Elements.dipole.dipole`: Dipole magnet.
    - :class:`~simba.Elements.kicker.kicker`: Kicker magnet.
    - :class:`~simba.Elements.quadrupole.quadrupole`: Quadrupole magnet.
    - :class:`~simba.Elements.sextupole.sextupole`: Sextupole magnet.
    - :class:`~simba.Elements.octupole.octupole`: Octupole magnet.
    - :class:`~simba.Elements.cavity.cavity`: RF cavity.
    - :class:`~simba.Elements.wakefield.wakefield`: Wakefield.
    - :class:`~simba.Elements.rf_deflecting_cavity.rf_deflecting_cavity`: \
    RF deflecting cavity.
    - :class:`~simba.Elements.solenoid.solenoid`: Solenoid magnet.
    - :class:`~simba.Elements.aperture.aperture`: Aperture.
    - :class:`~simba.Elements.scatter.scatter`: Scatter object.
    - :class:`~simba.Elements.cleaner.cleaner`: Cleaner object.
    - :class:`~simba.Elements.wall_current_monitor.wall_current_monitor`: \
    Wall current monitor.
    - :class:`~simba.Elements.integrated_current_transformer.integrated_current_transformer`: \
    Integrated current transformer.
    - :class:`~simba.Elements.faraday_cup.faraday_cup`: Faraday cup.
    - :class:`~simba.Elements.screen.screen`: Diagnostics screen.
    - :class:`~simba.Elements.monitor.monitor`: Monitor object.
    - :class:`~simba.Elements.faraday_cup.faraday_cup`: Faraday cup.
    - :class:`~simba.Elements.watch_point.watch_point`: Watch point.
    - :class:`~simba.Elements.beam_position_monitor.beam_position_monitor`: Beam position monitor.
    - :class:`~simba.Elements.bunch_length_monitor.bunch_length_monitor`: Bunch length monitor.
    - :class:`~simba.Elements.beam_arrival_monitor.beam_arrival_monitor`: Beam arrival monitor.
    - :class:`~simba.Elements.collimator.collimator`: Collimator.
    - :class:`~simba.Elements.rcollimator.rcollimator`: Rectangular collimator.
    - :class:`~simba.Elements.apcontour.apcontour`: Contour.
    - :class:`~simba.Elements.center.center`: Center object.
    - :class:`~simba.Elements.marker.marker`: Marker object.
    - :class:`~simba.Elements.drift.drift`: Drift.
    - :class:`~simba.Elements.shutter.shutter`: Shutter.
    - :class:`~simba.Elements.valve.valve`: Vacuum valve.
    - :class:`~simba.Elements.bellows.bellows`: Bellows.
    - :class:`~simba.Elements.fel_modulator.fel_modulator`: FEL modulator.
    - :class:`~simba.Elements.wiggler.wiggler`: Wiggler.
    - :class:`~simba.Elements.gpt_ccs.gpt_ccs`: GPT coordinate system.
    - :class:`~simba.Elements.global_error.global_error`: Global error object.
    - :class:`~simba.Elements.charge.charge`: Bunch charge.
    - :class:`~simba.Elements.twiss.twiss`: Twiss matching.
"""

from simba.Framework_objects import (
    chicane,
    s_chicane,
    r56_group,
    element_group,
)  # noqa F401

disallowed_keywords = [
    "allowedkeywords",
    "conversion_rules",
    "objectdefaults",
    "global_parameters",
    "objectname",
    "beam",
]
