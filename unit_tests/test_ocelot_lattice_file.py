"""Ocelot's lattice-file variable names; see :mod:`simba.Codes.Ocelot.latticefile`.

The oracle is Ocelot's own ``LatticeIO._create_var_name``.
"""

import random
from types import SimpleNamespace

import pytest

pytest.importorskip("ocelot")

from simba.Codes.Ocelot import latticefile  # noqa: E402


def names(create, ids):
    return [obj.name for obj in create([SimpleNamespace(id=i) for i in ids])]


def ocelots(ids):
    latticefile.fast_lattice_files()
    return names(latticefile.ocelot_create_var_name, ids)


@pytest.mark.parametrize(
    "ids",
    [
        ["START", "Q1", "D", "Q2", "D", "END"],
        # a lettered repeat meeting an id that already has that name
        ["Q", "Qa", "Q", "Qb", "Q"],
        # ids that clean to the same name, but only once cleaned
        ["D.1", "D_1", "D-1", "D:1", "D.1"],
        ["M", "M", "M_a", "Ma", "M", "ma"],
    ],
)
def test_names_are_ocelots(ids):
    assert names(latticefile._create_var_name, ids) == ocelots(ids)


def test_names_are_ocelots_on_a_random_lattice():
    rng = random.Random(3)
    stems = ["Q", "D", "B.1", "B-1", "B_1", "S:F", "Da", "M"]
    for _ in range(50):
        ids = [rng.choice(stems) + rng.choice(["", "", "a", "b"]) for _ in range(30)]
        try:
            expected = ocelots(ids)
        except IndexError:  # Ocelot letters at most twelve repeats
            continue
        assert names(latticefile._create_var_name, ids) == expected


def test_the_patch_is_in_place():
    from ocelot.cpbd.latticeIO import LatticeIO

    latticefile.fast_lattice_files()
    assert LatticeIO._create_var_name is latticefile._create_var_name
