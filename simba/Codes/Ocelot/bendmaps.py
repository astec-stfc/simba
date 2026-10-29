"""
Ocelot's second-order maps for a bend with both k1 and k2, made to agree with
MAD-X, ELEGANT and Bmad.

Ocelot's default MAD-convention body map (``t_nnn_mad``) is wrong for such a
bend: slicing it into k1-only pieces with short sextupoles converges to MAD-X.
Its TRANSPORT maps are
right, except that the exit edge's ``T[3, 0, 2]`` has ``-k1 tan(e)`` where the
entrance (and the MAD exit) has ``-2 k1 tan(e)``. Every other bend keeps the
MAD maps.

TRANSPORT's maps are in ``x'``, MAD's in ``px = x' (1 + delta)``: a TRANSPORT
drift has no ``px delta`` term. So the TRANSPORT bend is converted to ``px`` at
both faces, to sit among MAD-convention elements.
"""

from functools import wraps

import numpy as np

_PATCHED = False


def _in_mad_momenta(R, T):
    """
    A TRANSPORT map's ``T`` in MAD's momenta: the map conjugated by
    ``px = x' (1 + delta)``, ``py`` alike, to second order.
    """
    T = T.copy()
    for p in (1, 3):
        T[p] += np.outer(R[p], R[5])
        T[:, p, 5] -= R[:, p]
    return T


def transport_bend_maps() -> None:
    """
    Build a k1-and-k2 bend's second-order maps with Ocelot's TRANSPORT
    convention, restoring the exit edge's missing ``-k1 tan(e)``. Run before any
    lattice is built: Ocelot caches maps.
    """
    global _PATCHED
    if _PATCHED:
        return
    import ocelot.cpbd.high_order as high_order
    from ocelot.cpbd.elements.bend_atom import BendAtom

    def transport(build, exit_edge=False):
        @wraps(build)
        def built(self, *args, **kwargs):
            if not (self.k1 and self.k2):
                return build(self, *args, **kwargs)
            mad, high_order.__MAD__ = high_order.__MAD__, False
            try:
                params = build(self, *args, **kwargs)
            finally:
                high_order.__MAD__ = mad
            if exit_edge:
                params.T[3, 0, 2] -= self.k1 * np.tan(self.e2)
            params.T = _in_mad_momenta(params.R, params.T)
            return params

        return built

    BendAtom.create_second_order_main_params = transport(
        BendAtom.create_second_order_main_params
    )
    BendAtom.create_second_order_entrance_params = transport(
        BendAtom.create_second_order_entrance_params
    )
    BendAtom.create_second_order_exit_params = transport(
        BendAtom.create_second_order_exit_params, exit_edge=True
    )
    _PATCHED = True
