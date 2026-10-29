"""Ocelot's lattice-file writer, without its quadratic search for repeated names."""

from bisect import insort
from collections import defaultdict

_PATCHED = False
ocelot_create_var_name = None
"""Ocelot's own, once :func:`fast_lattice_files` has replaced it."""


def _clean(name: str) -> str:
    return name.replace(".", "_").replace(":", "_").replace("-", "_")


def _create_var_name(objects):
    alphabet = "abcdefgiklmn"
    ids = [obj.id for obj in objects]
    where = defaultdict(list)
    for i, name in enumerate(ids):
        where[name].append(i)

    def rename(i, name):
        where[ids[i]].remove(i)
        insort(where[name], i)
        ids[i] = name

    for j, obj in enumerate(objects):
        inx = list(where[obj.id])  # the id, not the name it may have by now
        if len(inx) > 1:
            for n, i in enumerate(inx):
                rename(i, _clean(ids[i]) + alphabet[n])
        else:
            rename(j, _clean(ids[j]))
        obj.name = ids[j].lower()

    return objects


def fast_lattice_files() -> None:
    """Patch Ocelot's ``LatticeIO._create_var_name`` with the above. Idempotent."""
    global _PATCHED, ocelot_create_var_name
    if _PATCHED:
        return
    from ocelot.cpbd.latticeIO import LatticeIO

    ocelot_create_var_name = LatticeIO._create_var_name
    LatticeIO._create_var_name = staticmethod(_create_var_name)
    _PATCHED = True
