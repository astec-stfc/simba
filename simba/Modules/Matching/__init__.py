"""
Matching: element settings that meet a line's targets.

Each matcher returns its settings as ``{element: {dotted.field: value}}``, for the
caller to apply (see :meth:`~simba.Framework_objects.frameworkLattice.apply_settings`).
"""

from .longitudinal import energy_polynomial, longitudinal_match, rf_settings

__all__ = ["energy_polynomial", "longitudinal_match", "rf_settings"]
