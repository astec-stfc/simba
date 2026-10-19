"""simba's Ocelot ``Navigator``; see :mod:`simba.Codes.Ocelot.navigator`.

The oracle is Ocelot's own ``Navigator``: a pass through it and through
:class:`PassNavigator` must step the same way and give the same particles, bit
for bit.
"""

import copy
from contextlib import nullcontext

import numpy as np
import pytest

pytest.importorskip("ocelot")

from ocelot.cpbd.beam import ParticleArray  # noqa: E402
from ocelot.cpbd.elements import Drift, Marker, Quadrupole, SBend  # noqa: E402
from ocelot.cpbd.magnetic_lattice import MagneticLattice  # noqa: E402
from ocelot.cpbd.navi import Navigator  # noqa: E402
from ocelot.cpbd.physics_proc import PhysProc  # noqa: E402
from ocelot.cpbd.track import track  # noqa: E402

from simba.Codes.Ocelot.navigator import PassNavigator, lattice_pass  # noqa: E402


class Recorder(PhysProc):
    """Logs where it is applied. The log is the class's: the Navigator applies
    deep copies of the processes it was given."""

    log = []

    def __init__(self, name, step=1):
        super().__init__(step=step)
        self.name = name

    def apply(self, p_array, dz):
        Recorder.log.append((self.name, float(p_array.s), float(dz)))


def lattice(cells=60):
    """Lengths that do not add up exactly, over more elements than numpy sums
    in one block."""
    sequence = [Marker(eid="START")]
    for i in range(cells):
        sequence += [
            Drift(l=0.1, eid=f"D{i}A"),
            Quadrupole(l=1 / 3, k1=0.7 * (-1) ** i, eid=f"Q{i}"),
            Drift(l=0.3, eid=f"D{i}B"),
            SBend(l=0.7, angle=0.01, eid=f"B{i}"),
            Marker(eid=f"M{i}"),
            Drift(l=0.2 + 1e-9 * i, eid=f"D{i}C"),
        ]
    sequence.append(Marker(eid="END"))
    return MagneticLattice(sequence)


def beam(n=40):
    p = ParticleArray(n=n)
    p.rparticles[:] = np.random.default_rng(7).normal(scale=1e-4, size=(6, n))
    p.q_array[:] = 1e-12 / n
    p.E = 1.0
    return p


def one_pass(navigator_class):
    lat = lattice()
    navi = navigator_class(lat)
    seq = lat.sequence
    navi.add_physics_processes(
        [Recorder("kick"), Recorder("range", step=2)],
        [seq[5 * 6 + 5], seq[20 * 6 + 1]],
        [seq[5 * 6 + 5], seq[40 * 6 + 3]],
    )
    Recorder.log = []
    navi.go_to_start()
    ours = navigator_class is PassNavigator
    with lattice_pass(lat) if ours else nullcontext():
        tws, p = track(lat, beam(), navi=navi, print_progress=False)
    return Recorder.log, [t.s for t in tws], p.rparticles.copy()


def test_a_pass_is_ocelots_bit_for_bit():
    log0, s0, x0 = one_pass(Navigator)
    log1, s1, x1 = one_pass(PassNavigator)
    assert log1 == log0
    assert s1 == s0
    assert np.array_equal(x1, x0)
    assert {name for name, *_ in log0} == {"kick", "range"}


def test_element_ends_are_ocelots_sums():
    lat = lattice()
    navi = PassNavigator(lat)
    seq = lat.sequence
    for n in range(len(seq)):
        ocelot = np.sum(np.array([elem.l for elem in seq[: n + 1]]))
        assert navi.element_end(n) == ocelot


def test_the_length_is_fixed_and_the_lattice_shared_only_within_a_pass():
    lat = lattice()
    length = lat.totalLen
    with lattice_pass(lat):
        assert lat.totalLen == length
        assert isinstance(lat, MagneticLattice)
        assert copy.deepcopy({"lat": lat})["lat"] is lat
        with lattice_pass(lat):
            assert lat.totalLen == length
        assert lat.totalLen == length
    assert type(lat) is MagneticLattice
    assert not hasattr(lat, "_pass_length")
    assert copy.deepcopy(lat) is not lat


def test_the_lattice_is_restored_when_a_pass_fails():
    lat = lattice()
    with pytest.raises(RuntimeError), lattice_pass(lat):
        raise RuntimeError
    assert type(lat) is MagneticLattice


def test_a_reset_rebuilds_the_element_ends():
    lat = lattice(cells=2)
    navi = PassNavigator(lat)
    before = navi.element_end(len(lat.sequence) - 1)
    lat.sequence[1].l = 0.5
    navi.go_to_start()
    assert navi.element_end(len(lat.sequence) - 1) == pytest.approx(before + 0.4)
