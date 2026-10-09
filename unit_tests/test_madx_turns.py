"""MAD-X multi-turn tracking: is a simba-side turn loop the same tracking?"""

import contextlib

import numpy as np
import pytest


LATTICE = """
QF: QUADRUPOLE, L=0.5, K1= 0.8;
QD: QUADRUPOLE, L=0.5, K1=-0.8;
D:  DRIFT, L=1.0;
CELL: LINE=(QF, D, QD, D);
RING: LINE=(CELL, CELL, CELL, CELL);
BEAM, PARTICLE=ELECTRON, PC=1.0;
USE, SEQUENCE=RING;
"""

START = {"x": 1e-4, "px": 0.0, "y": 5e-5, "py": 0.0, "t": 0.0, "pt": 0.0}
TURNS = 12
KEYS = ("x", "px", "y", "py", "pt")


@pytest.fixture(scope="module", autouse=True)
def _run_somewhere_disposable(tmp_path_factory):
    """MAD-X ``TRACK`` drops ``checkpoint_restart.dat`` into the working
    directory, so the working directory must not be the repository.

    Module-scoped to match ``both``: a function-scoped fixture is not applied
    when a module-scoped one does the work, which is how the stray file got
    into the repository in the first place."""
    with contextlib.chdir(tmp_path_factory.mktemp("madx")):
        yield


def thin_ring():
    """A sliced FODO ring, ready to track."""
    madx = pytest.importorskip("cpymad.madx")
    instance = madx.Madx(stdout=False)
    instance.input(LATTICE)
    instance.input("select, flag=makethin, slice=1; makethin, sequence=RING;")
    instance.use(sequence="RING")
    return instance


def start_command(coords):
    return "start, " + ", ".join(f"{k}={v:.15g}" for k, v in coords.items()) + ";"


def end_of_turn_rows(instance):
    """``trackone`` rows at the end of the sequence, de-duplicated by turn."""
    table = instance.table.trackone
    data = {k: np.array(table[k]) for k in ("turn", "s") + KEYS}
    at_end = (data["s"] == data["s"].max()) & (data["turn"] > 0)
    turns = data["turn"][at_end]
    _, first = np.unique(turns, return_index=True)
    keep = np.sort(first)
    return {k: data[k][at_end][keep] for k in ("turn",) + KEYS}


def native(n):
    """One ``TRACK`` call asked for ``n`` turns."""
    instance = thin_ring()
    instance.input("track, onepass, onetable;")
    instance.input(start_command(START))
    instance.input("observe, place=#e;")
    instance.input(f"run, turns={n};")
    instance.input("endtrack;")
    rows = end_of_turn_rows(instance)
    instance.exit()
    return rows


def looped(n):
    """``n`` one-turn ``TRACK`` calls, the exit coordinates fed back in."""
    instance = thin_ring()
    coords = dict(START)
    collected = {k: [] for k in ("turn",) + KEYS}
    for turn in range(1, n + 1):
        instance.input("track, onepass, onetable;")
        instance.input(start_command(coords))
        instance.input("observe, place=#e;")
        instance.input("run, turns=1;")
        instance.input("endtrack;")
        rows = end_of_turn_rows(instance)
        coords = {k: float(rows[k][-1]) for k in KEYS}
        coords["t"] = 0.0
        collected["turn"].append(turn)
        for k in KEYS:
            collected[k].append(coords[k])
    instance.exit()
    return {k: np.array(v) for k, v in collected.items()}


@pytest.fixture(scope="module")
def both():
    return native(TURNS), looped(TURNS)


def test_madx_tracks_the_turns_it_was_asked_for(both):
    """The premise: ``RUN, TURNS=N`` works, so MAD-X was never the limitation."""
    native_rows, _ = both
    assert list(native_rows["turn"].astype(int)) == list(range(1, TURNS + 1))


def test_the_simba_side_loop_is_the_same_tracking(both):
    """The question this file was written to answer."""
    native_rows, looped_rows = both
    for k in KEYS:
        assert native_rows[k] == pytest.approx(looped_rows[k], abs=1e-15)


def test_the_ring_is_actually_oscillating(both):
    """Guards the comparison above: two identical columns of zeros would also
    pass it. The betatron motion has to be large enough for the agreement to
    mean something."""
    native_rows, _ = both
    assert np.std(native_rows["x"]) > 1e-5
    assert np.min(native_rows["x"]) < 0 < np.max(native_rows["x"])
