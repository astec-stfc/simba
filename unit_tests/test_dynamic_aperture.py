"""Dynamic aperture: how far off-axis a particle can start and survive."""

import math
import shutil

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
    da_rays = frameworkLattice.da_rays
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


def test_the_boundary_is_the_last_survivor_before_the_first_loss():
    """As elegant's rays stop: the survivor past the loss is an island."""
    line = FakeLine({"turns": 100})
    results = [(0.001, 0.0, 99), (0.002, 0.0, 99), (0.003, 0.0, 40), (0.004, 0.0, 99)]
    assert line.dynamic_aperture_boundary(results) == [(0.002, 0.0)]


def test_a_full_survivor_is_turn_minus_one():
    """Ocelot numbers turns from zero, so testing `turn >= turns` would
    report an aperture of zero for a perfectly stable ring."""
    line = FakeLine({"turns": 100})
    assert line.dynamic_aperture_boundary([(0.005, 0.0, 99)]) == [(0.005, 0.0)]
    assert line.dynamic_aperture_boundary([(0.005, 0.0, 98)]) == []


def test_a_ray_lost_at_its_first_point_is_absent():
    """Not reported as zero -- no aperture is not an aperture of nothing."""
    line = FakeLine({"turns": 100})
    results = [(0.01, 0.0, 99), (0.0, 0.02, 5)]
    assert line.dynamic_aperture_boundary(results) == [(0.01, 0.0)]


def test_the_boundary_runs_from_plus_x_round_to_minus_x():
    line = FakeLine({"turns": 50, "dynamic_aperture": {"nx": 3, "n_lines": 3, "x_max": 0.02, "y_max": 0.01}})
    results = [(x, y, 49) for x, y in line.da_rays()]
    boundary = line.dynamic_aperture_boundary(results)
    assert np.allclose(boundary, [(0.02, 0.0), (0.0, 0.01), (-0.02, 0.0)], rtol=0, atol=1e-15)


# --- the rays -----------------------------------------------------------


def test_the_rays_are_elegants():
    """``find_aperture``'s ``n-line`` rays, read off its output for the toy
    ring of ``examples/ring_studies``: 11 rays, 18 degrees apart, each
    stepped in 1/19ths out to the ellipse of its ±20 x 10 mm box."""
    rays = FakeLine({"dynamic_aperture": {"nx": 20, "x_max": 0.02, "y_max": 0.01}}).da_rays()
    assert len(rays) == 11 * 19
    assert rays[14] == pytest.approx((0.02 * 15 / 19, 0.0))  # elegant's 15.79 mm on +x
    assert rays[19 + 18] == pytest.approx((0.02 * math.cos(math.pi / 10), 0.01 * math.sin(math.pi / 10)))
    assert rays[-1] == pytest.approx((-0.02, 0.0), abs=1e-15)


def test_the_rays_never_start_at_zero_amplitude():
    rays = FakeLine({"dynamic_aperture": {"nx": 5, "n_lines": 3}}).da_rays()
    assert min(math.hypot(x, y) for x, y in rays) > 0


def test_degenerate_rays_are_not_empty():
    assert FakeLine({"dynamic_aperture": {"nx": 0, "n_lines": 0}}).da_rays()


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


# --- elegant, through its real decks ------------------------------------

ELEGANT = shutil.which("elegant")


def elegant_ring_lattice(ncell=8, k2=500.0):
    """The sextupole ring above, as elegant ``.lte`` text.

    ``REFERENCE_CORRECTION``: at elegant's default four kicks, a 45-degree
    ``CSBEND`` misses its own reference by enough to put a millimetre of
    closed orbit round this ring, which the sextupoles turn into a tune
    shift of 0.03 at any starting amplitude.
    """
    angle = 2 * math.pi / ncell
    return (
        "QF: KQUAD, L=0.3, K1=1.2\n"
        f"SF: KSEXT, L=0.2, K2={k2}\n"
        "D: DRIF, L=0.3\n"
        f"B: CSBEND, L=1.0, ANGLE={angle}, REFERENCE_CORRECTION=1\n"
        "QD: KQUAD, L=0.3, K1=-1.2\n"
        f"SD: KSEXT, L=0.2, K2={-k2}\n"
        "CELL: LINE=(QF,SF,D,B,D,QD,SD,D)\n"
        f"RING: LINE=({ncell}*CELL)\n"
    )


class FakeElegantRing:
    """``elegantLattice``'s ring-study decks, run by elegant itself.

    Only the lattice file is hand-written; the decks, the run and the
    read-back are simba's.
    """

    _ring_study_deck = elegantLattice._ring_study_deck
    _without_watch_output = staticmethod(elegantLattice._without_watch_output)
    _da_bounds = elegantLattice._da_bounds
    run_dynamic_aperture = elegantLattice.run_dynamic_aperture
    run_frequency_map = elegantLattice.run_frequency_map
    da_settings = frameworkLattice.da_settings
    da_grid = frameworkLattice.da_grid
    turns = frameworkLattice.turns

    class executables(dict):
        @staticmethod
        def build_command(cmd, workdir):
            return cmd

    def __init__(self, directory, tracking):
        from simba.Modules import Beams as rbf

        self.objectname, self.code = "RING", "elegant"
        self.file_block = {"tracking": tracking}
        self.files = []
        self.fixed_reference = True
        self.rest_energy = 0.51099895e6
        self.reference_p0c = 1e9
        self.executables = self.executables(elegant=[ELEGANT])
        self.global_parameters = {
            "master_subdir": str(directory), "beam": rbf.beam(),
        }
        (directory / "RING.lte").write_text(elegant_ring_lattice())


@pytest.mark.skipif(ELEGANT is None, reason="elegant is not installed")
def test_elegant_finds_an_aperture_through_simbas_deck(tmp_path):
    """The deck had no ``&run_control``, so ``&find_aperture`` had no turn
    count: elegant wrote a valid header and no rows, and simba returned an
    empty boundary for every ring (CLIC DR's included)."""
    line = FakeElegantRing(
        tmp_path,
        {"turns": 64, "dynamic_aperture": {"nx": 12, "ny": 4, "x_max": 0.02, "y_max": 0.01}},
    )
    boundary = line.run_dynamic_aperture()
    assert boundary
    deck = (tmp_path / "RING_aperture" / "RING_aperture.ele").read_text()
    assert deck.index("&run_control") < deck.index("&find_aperture")
    assert "n_passes = 64" in deck


@pytest.mark.skipif(ELEGANT is None, reason="elegant is not installed")
def test_elegant_scans_write_no_screen_files(tmp_path):
    """Every screen wrote every pass of every grid point: on CLIC DR the scan
    was still running after ten minutes, and a ten-cell ring took as long."""
    line = FakeElegantRing(
        tmp_path, {"turns": 16, "dynamic_aperture": {"nx": 4, "ny": 2, "x_max": 0.004, "y_max": 0.002}}
    )
    lattice = tmp_path / "RING.lte"
    text = lattice.read_text().replace("CELL: LINE=(", "CELL: LINE=(W,")
    lattice.write_text('W: WATCH, FILENAME="screen.sdds"\n' + text)
    assert line.run_dynamic_aperture()
    assert line.run_frequency_map()
    for stem in ("RING_aperture", "RING_fma"):
        assert not (tmp_path / stem / "screen.sdds").exists()


def madx_ring_sequence(ncell=10):
    """A ten-cell FODO ring of thick quadrupoles and sextupoles, as MAD-X
    text: tunes 2.42 / 1.77 at 1 GeV, aperture near 13 mm in x."""
    lines = [
        "qf: quadrupole, l=0.3, k1=0.7/0.3;",
        "sf: sextupole, l=0.2, k2=8/0.2;",
        "qd: quadrupole, l=0.3, k1=-0.6/0.3;",
        "sd: sextupole, l=0.2, k2=-8/0.2;",
        # MAKETHIN slices only a sequence referred to centres, as simba writes them
        f"RING_seg_0: sequence, l={4.0 * ncell}, refer=centre;",
    ]
    for n in range(ncell):
        z = 4.0 * n
        lines += [f"qf, at={z + 0.15};", f"sf, at={z + 0.4};", f"qd, at={z + 2.15};", f"sd, at={z + 2.4};"]
    return "\n".join(lines + ["endsequence;"])


class FakeMadxRing:
    """``madxLattice``'s DYNAP, run by MAD-X on a hand-written sequence."""

    _track_grid = madxLattice._track_grid
    _turns_survived = madxLattice._turns_survived
    run_dynamic_aperture = madxLattice.run_dynamic_aperture
    run_frequency_map = madxLattice.run_frequency_map
    start_madx = madxLattice.start_madx
    stop_madx = madxLattice.stop_madx
    madx_beam_command = madxLattice.madx_beam_command
    makethin = madxLattice.makethin
    segment_name = madxLattice.segment_name
    da_settings = frameworkLattice.da_settings
    da_grid = frameworkLattice.da_grid
    turns = frameworkLattice.turns

    def __init__(self, directory, tracking):
        from types import SimpleNamespace

        self.objectname, self.code = "RING", "madx"
        self.file_block = {"tracking": tracking}
        self.seqstrings = [madx_ring_sequence()]
        self.section = SimpleNamespace(functional_definitions={})
        self.global_parameters = {
            "master_subdir": str(directory), "beam": SimpleNamespace(species="electron"),
        }
        self.reference_p0c, self.rest_energy, self.reference_charge = 1e9, 0.51099895e6, -1
        for name in ("nslice_quadrupole", "nslice_sbend", "nslice_sextupole", "makedipedge", "makethin_style"):
            setattr(self, name, madxLattice.model_fields[name].default)

    def da_rays(self):
        """The grid, so the aperture scan's survivors are the map's starts."""
        xs, ys = self.da_grid()
        return [(x, y) for y in ys for x in xs]


def test_madx_dynap_runs_on_a_thick_lattice(tmp_path):
    """Three faults, each hidden behind the one before:

    * ``DYNAP`` was given the sequence unsliced, and ``TRACK`` refuses thick
      elements: MAD-X stopped at the first one, and every MAD-X aperture
      and frequency map came back empty;
    * ``DYNAP``'s ``dktrturns`` is the full count for every particle (it
      skips the lost ones), so once sliced, every point survived;
    * ``fastune`` folds tunes into [0, 0.5]: Qy came back 0.23, not 0.77;
    * and it tunes particles the aperture scan counts as lost."""
    pytest.importorskip("cpymad")
    # 512 turns: (12, 2) and (12, 4) mm are lost late, at turns 428 and 414,
    # and DYNAP still gives both a tune
    tracking = {"turns": 512, "dynamic_aperture": {"nx": 10, "ny": 2, "x_max": 0.02, "y_max": 0.004}}
    aperture = FakeMadxRing(tmp_path, tracking).run_dynamic_aperture()
    turns = [turn for _, _, turn in aperture]
    assert len(aperture) == 20
    assert max(turns) >= 511 and min(turns) < 511, "grid must straddle the aperture"
    ring = FakeMadxRing(tmp_path, tracking)
    footprint = ring.run_frequency_map()
    assert ring.logfile is None, "both MAD-X sessions should close their log"

    def points(rows):
        return {(round(row[0], 9), round(row[1], 9)) for row in rows}

    assert points(footprint) == points(r for r in aperture if r[2] >= 511)
    assert footprint[0][2] == pytest.approx(0.42, abs=0.01)
    assert footprint[0][3] == pytest.approx(0.77, abs=0.01)


def test_madx_slicing_keeps_the_detuning_with_amplitude(tmp_path):
    """MAKETHIN gave sextupoles one slice, which put PTC's dQx/dεx and dQy/dεy
    ~10% out here, and flipped the sign of dQy/dεy on CLIC DR: its MAD-X
    frequency map sheared upward in Qy where elegant's and Xsuite's fell."""
    pytest.importorskip("cpymad")

    def detuning(thin):
        ring = FakeMadxRing(tmp_path, {"turns": 1})
        madx = ring.start_madx()
        madx.input(ring.seqstrings[0])
        ring.madx_beam_command(madx, ring.reference_p0c, "RING_seg_0")
        madx.input("use, sequence=RING_seg_0;")
        if thin:
            ring.makethin(madx, "RING_seg_0")
        madx.input(
            "ptc_create_universe;\n"
            "ptc_create_layout, model=2, method=6, nst=5, exact=true;\n"
            "select_ptc_normal, anhx=1,0,0, anhy=0,1,0;\n"
            "ptc_normal, closed_orbit, normal, icase=4, no=3;\n"
            "ptc_end;"
        )
        values = list(madx.table.normal_results["value"])
        ring.stop_madx(madx)
        return values

    assert detuning(thin=True) == pytest.approx(detuning(thin=False), rel=0.02)
