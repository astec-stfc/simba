"""Dynamic aperture: how far off-axis a particle can start and survive."""

import math

import numpy as np
import pytest

from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice


class FakeLine:
    """A lattice stub carrying only what the scan setup reads."""

    def __init__(self, tracking=None):
        self.file_block = {"tracking": tracking or {}}
        self.objectname = "RING"
        self.code = "astra"

    codes_that_can = frameworkLattice.codes_that_can
    turns = frameworkLattice.turns
    da_settings = frameworkLattice.da_settings
    da_grid = frameworkLattice.da_grid
    dynamic_aperture_boundary = frameworkLattice.dynamic_aperture_boundary
    run_dynamic_aperture = frameworkLattice.run_dynamic_aperture


# --- which codes can do it ----------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [ocelotLattice, xsuiteLattice, madxLattice, elegantLattice, bmadLattice],
    ids=lambda c: c.__name__,
)
def test_every_ring_code_can(cls):
    """Ocelot `track_nturns`, Xsuite by tracking the grid and reading
    `state`/`at_turn`, MAD-X `DYNAP`, elegant `&find_aperture`, Bmad via
    Tao's own scan."""
    assert cls.supports_dynamic_aperture is True


def test_the_base_class_still_cannot():
    assert frameworkLattice.supports_dynamic_aperture is False


# --- Bmad's scan is configured by namelist, not at runtime --------------


class FakeBmad:
    """Enough of a lattice for the namelist builder."""

    def __init__(self, tracking=None):
        self.file_block = {"tracking": tracking or {}}

    turns = frameworkLattice.turns
    da_settings = frameworkLattice.da_settings
    da_grid = frameworkLattice.da_grid
    _tao_dynamic_aperture_namelist = bmadLattice._tao_dynamic_aperture_namelist


def test_the_namelist_carries_the_scan_settings():
    text = FakeBmad({"turns": 250, "dynamic_aperture": {"x_max": 0.02}})._tao_dynamic_aperture_namelist()
    assert "&tao_dynamic_aperture" in text
    assert "da_param%n_turn = 250" in text
    assert "da_param%x_init = 0.02" in text


def test_the_namelist_is_written_even_with_no_scan_asked_for():
    """It configures the search without starting it, so it is inert until
    `dynamic_aperture_calc` is switched on. Setting these at runtime with
    `set dynamic_aperture ...` core-dumped the Tao library, which is why
    the namelist is the route taken."""
    assert "&tao_dynamic_aperture" in FakeBmad()._tao_dynamic_aperture_namelist()


def test_the_angle_count_has_a_floor():
    """Fewer than three angles is not a boundary."""
    text = FakeBmad({"dynamic_aperture": {"n_angle": 1}})._tao_dynamic_aperture_namelist()
    assert "da_param%n_angle = 3" in text


def test_the_base_class_cannot():
    assert frameworkLattice.supports_dynamic_aperture is False


def test_the_base_class_returns_no_scan():
    """And says so, naming the codes that can: it returned [] in silence."""
    with pytest.warns(UserWarning, match="bmad, elegant, madx, ocelot and xsuite are the codes that can"):
        assert FakeLine().run_dynamic_aperture() == []


# --- the grid -----------------------------------------------------------


def test_the_grid_defaults_to_a_horizontal_scan():
    xs, ys = FakeLine().da_grid()
    assert len(xs) == 10
    assert len(ys) == 1


def test_the_grid_is_sized_by_the_settings():
    xs, ys = FakeLine({"dynamic_aperture": {"nx": 4, "ny": 3}}).da_grid()
    assert len(xs) == 4
    assert len(ys) == 3


def test_the_grid_never_starts_at_zero_amplitude():
    """A particle at zero amplitude survives any lattice, so a grid point
    there measures nothing."""
    xs, ys = FakeLine({"dynamic_aperture": {"nx": 5, "ny": 5}}).da_grid()
    assert xs[0] > 0
    assert ys[0] > 0


def test_the_grid_reaches_the_requested_maximum():
    xs, _ = FakeLine({"dynamic_aperture": {"nx": 7, "x_max": 0.05}}).da_grid()
    assert xs[-1] == pytest.approx(0.05)


def test_a_degenerate_grid_is_not_empty():
    xs, ys = FakeLine({"dynamic_aperture": {"nx": 0, "ny": -3}}).da_grid()
    assert len(xs) == 1 and len(ys) == 1


# --- the boundary -------------------------------------------------------


def test_the_boundary_is_the_largest_surviving_amplitude():
    line = FakeLine({"turns": 100})
    results = [(0.001, 0.0, 99), (0.002, 0.0, 99), (0.003, 0.0, 40)]
    assert line.dynamic_aperture_boundary(results) == [(0.0, 0.002)]


def test_a_full_survivor_is_turn_minus_one():
    """Ocelot numbers turns from zero, so testing `turn >= turns` would
    report an aperture of zero for a perfectly stable ring."""
    line = FakeLine({"turns": 100})
    assert line.dynamic_aperture_boundary([(0.005, 0.0, 99)]) == [(0.0, 0.005)]
    assert line.dynamic_aperture_boundary([(0.005, 0.0, 98)]) == []


def test_a_row_where_nothing_survives_is_absent():
    """Not reported as zero -- no aperture is not an aperture of nothing."""
    line = FakeLine({"turns": 100})
    results = [(0.01, 0.0, 99), (0.01, 0.02, 5)]
    assert line.dynamic_aperture_boundary(results) == [(0.0, 0.01)]


def test_the_boundary_covers_every_surviving_row():
    line = FakeLine({"turns": 50})
    results = [(0.01, 0.0, 49), (0.02, 0.0, 49), (0.005, 0.001, 49)]
    assert line.dynamic_aperture_boundary(results) == [(0.0, 0.02), (0.001, 0.005)]


# --- the preconditions, measured ----------------------------------------


def ocelot_ring(method, k2=500.0, ncell=8):
    import ocelot.cpbd.elements as oc
    from ocelot.cpbd.magnetic_lattice import MagneticLattice

    angle = 2 * math.pi / ncell
    cell = []
    for _ in range(ncell):
        cell += [
            oc.Quadrupole(l=0.3, k1=1.2),
            oc.Sextupole(l=0.2, k2=k2),
            oc.Drift(l=0.3),
            oc.SBend(l=1.0, angle=angle),
            oc.Drift(l=0.3),
            oc.Quadrupole(l=0.3, k1=-1.2),
            oc.Sextupole(l=0.2, k2=-k2),
            oc.Drift(l=0.3),
        ]
    return MagneticLattice(cell, method={"global": method})


def scan(lattice, turns=200, nx=12, x_max=0.02):
    from ocelot.cpbd.track import create_track_list, track_nturns

    track_list = create_track_list(
        np.linspace(x_max / nx, x_max, nx), [1e-4], [0.0], energy=1.0
    )
    track_list = track_nturns(
        lattice, turns, track_list, save_track=False, print_progress=False
    )
    return [(float(p.x), float(p.y), int(p.turn)) for p in track_list]


def test_a_linear_lattice_has_no_dynamic_aperture_to_find():
    """The trap: first-order maps make every amplitude survive, so the scan
    reports an enormous aperture and raises nothing."""
    from ocelot.cpbd.transformations import TransferMap

    results = scan(ocelot_ring(TransferMap))
    assert all(turn >= 199 for _, _, turn in results)


def test_second_order_maps_produce_a_real_boundary():
    from ocelot.cpbd.transformations import SecondTM

    results = scan(ocelot_ring(SecondTM))
    survived = [x for x, _, turn in results if turn >= 199]
    lost = [x for x, _, turn in results if turn < 199]
    assert survived and lost, "grid must straddle the aperture to test anything"
    assert max(survived) < min(lost)


def test_ocelot_single_particles_track_on_a_ring_with_rf():
    """Ocelot's own ``track_nturns`` asks ``twiss`` for its aperture limits with
    no energy, which Ocelot refuses once a cavity is in the lattice: the
    reference particle, dynamic aperture and frequency map failed on every
    ring with RF (CLIC DR's, after 23 minutes of bunch tracking)."""
    import ocelot.cpbd.elements as oc
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.transformations import SecondTM

    ring = ocelot_ring(SecondTM, k2=0.0)
    cells = list(ring.sequence) + [oc.Cavity(l=0.1, v=1e-4, freq=5e8, phi=90.0)]

    class FakeOcelot:
        track_reference_particle = ocelotLattice.track_reference_particle
        _track_nturns = ocelotLattice._track_nturns
        _ocelot_periodic = ocelotLattice._ocelot_periodic
        read_closed_orbit = ocelotLattice.read_closed_orbit
        lat_obj = MagneticLattice(cells, method={"global": SecondTM})
        objectname, turns, nsuperperiods, da_settings = "ring", 20, 1, {}
        reference_energy = 1e9
        _periodic = None

    trajectory = FakeOcelot().track_reference_particle()
    assert len(trajectory["x"]) == 20


def test_laura_builds_ocelot_lattices_second_order():
    """Which is why simba does not fall into the trap above. Pinned rather
    than guarded in code, because LAURA already does the right thing."""
    import inspect

    from laura.translator.converters import section

    source = inspect.getsource(section.SectionLatticeTranslator.to_ocelot)
    assert "SecondTM" in source
