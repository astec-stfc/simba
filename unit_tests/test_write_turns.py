"""A multi-turn run does not dump a beam file per turn unless asked.

Per-turn output was built with multipass in mind, where a handful of passes
means a handful of files. A ring is different: one screen over 10^6 turns is
10^6 files, ten screens is 10^7, and `test_summary_scaling.py` measures what
that costs just to *index*. Most ring studies -- tune, chromaticity, dynamic
aperture -- want none of those files.

So `write_turns` defaults off, and a multi-turn run writes what a single-turn
run writes: one file per screen, holding the last turn. `output_turns()` makes
that decision once, because all three multi-turn backends need it and would
otherwise each answer it differently.
"""

import pytest

from simba.Framework_objects import frameworkLattice


class FakeLine:
    """A lattice stub carrying only what the output decision reads."""

    def __init__(self, tracking=None):
        self.file_block = {"tracking": tracking or {}}
        self.objectname = "RING"

    turns = frameworkLattice.turns
    write_turns = frameworkLattice.write_turns
    output_turns = frameworkLattice.output_turns


# --- reading the flag ---------------------------------------------------


def test_the_default_is_off():
    assert FakeLine({"turns": 1000}).write_turns is False


def test_it_can_be_asked_for():
    assert FakeLine({"turns": 1000, "write_turns": True}).write_turns is True


# --- what gets written --------------------------------------------------


def test_a_single_turn_run_is_unchanged():
    """The flag must not touch the ordinary case: one file, unsuffixed."""
    assert FakeLine().output_turns() == [(None, None)]
    assert FakeLine({"turns": 1}).output_turns() == [(None, None)]


def test_the_flag_does_nothing_on_a_single_turn_run():
    assert FakeLine({"turns": 1, "write_turns": True}).output_turns() == [(None, None)]


def test_a_multi_turn_run_writes_only_the_last_turn():
    """And writes it under the *unsuffixed* name, so a ring run looks like
    any other run to everything downstream."""
    assert FakeLine({"turns": 1000}).output_turns() == [(1000, None)]


def test_asking_for_turns_writes_every_one():
    assert FakeLine({"turns": 3, "write_turns": True}).output_turns() == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]


def test_the_file_count_is_one_by_default_and_n_when_asked():
    """The whole point: 10^6 turns is one file, not 10^6, unless you say."""
    assert len(FakeLine({"turns": 1_000_000}).output_turns()) == 1
    assert len(FakeLine({"turns": 1000, "write_turns": True}).output_turns()) == 1000


# --- the two halves of each pair mean different things ------------------


def test_the_data_turn_is_one_based():
    """It selects a turn, so it counts from 1 like the turn count does."""
    turns = FakeLine({"turns": 4, "write_turns": True}).output_turns()
    assert [data for data, _ in turns] == [1, 2, 3, 4]


def test_the_name_turn_is_none_when_only_the_last_is_kept():
    """`output_basename(name, turn=None)` gives the plain name; passing the
    turn would suffix a file that is the only one there is."""
    (data_turn, name_turn), = FakeLine({"turns": 50}).output_turns()
    assert data_turn == 50
    assert name_turn is None


@pytest.mark.parametrize("turns", [2, 17, 1000])
def test_the_last_turn_selected_is_the_last_turn_tracked(turns):
    (data_turn, _), = FakeLine({"turns": turns}).output_turns()
    assert data_turn == turns
