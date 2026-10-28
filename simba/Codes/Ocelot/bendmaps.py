"""
Ocelot's second-order bend maps, made to agree with MAD-X, ELEGANT and Bmad.

Ocelot's default MAD-convention body map (``t_nnn_mad``) is wrong for a bend
with both k1 and k2. Its
TRANSPORT maps are right, except that the exit edge's ``T[3, 0, 2]`` has
``-k1 tan(e)`` where the entrance (and the MAD exit) has ``-2 k1 tan(e)``.
"""

import numpy as np

_PATCHED = False


def transport_bend_maps() -> None:
    """
    Switch Ocelot to its TRANSPORT second-order maps and restore the exit edge's
    missing ``-k1 tan(e)``. Run before any lattice is built: Ocelot caches maps.
    """
    global _PATCHED
    if _PATCHED:
        return
    import ocelot.cpbd.elements.bend_atom as bend_atom
    import ocelot.cpbd.high_order as high_order

    fringe_ext = bend_atom.fringe_ext

    def _fringe_ext(h, k1, e, h_pole=0.0, gap=0.0, fint=0.0):
        R, T = fringe_ext(h, k1, e, h_pole, gap, fint)
        T[3, 0, 2] -= k1 * np.tan(e)
        return R, T

    high_order.__MAD__ = False
    bend_atom.fringe_ext = _fringe_ext
    _PATCHED = True
