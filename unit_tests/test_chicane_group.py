"""`chicane.set_angle` on a lattice that does not run down the world z axis.

The arithmetic is laid out in the chicane's own frame -- LAURA's orientation matrix for
the first dipole, anchored on its entrance -- so a surveyed lattice such as LCLS's, whose
laser heater sits 0.61 rad off z and two kilometres from the origin, has to come out as
the same chicane, just turned.
"""

import types

import numpy as np
import pytest

from laura.models.element import Dipole, Marker
from laura.utils.rotation_matrix import euler_angles_to_rotation_matrix
from simba.Framework_objects import chicane

LZ = 0.2
DIPOLE_Z = (0.1, 1.3, 3.3, 4.5)
MARKER_Z = (2.0, 2.8)
NAMES = [f"D{i}" for i in range(1, 5)] + [f"M{i}" for i in range(1, 3)]


def _build(theta0):
    """A four-dipole chicane on the axis `theta0`, straight and on-crest to start with."""
    rotation = euler_angles_to_rotation_matrix(theta0, 0.0, 0.0)
    elements = {}
    for i, z in enumerate(DIPOLE_Z, start=1):
        d = Dipole(name=f"D{i}", hardware_class="Magnet", machine_area="TEST")
        d.physical.length = LZ
        d.magnetic.length = LZ
        d.magnetic.angle = 0.0
        elements[d.name] = d
    for i, z in enumerate(MARKER_Z, start=1):
        m = Marker(name=f"M{i}", hardware_class="Marker", hardware_type="Marker",
                   machine_area="TEST")
        elements[m.name] = m
    for name, z in zip(NAMES, DIPOLE_Z + MARKER_Z):
        p = elements[name].physical
        p.global_rotation.theta = theta0
        x, _, zz = rotation @ np.array([0.0, 0.0, z])
        p.middle = {"x": x, "y": 0.0, "z": zz}
    framework = types.SimpleNamespace(elementObjects=elements, groupObjects={})
    group = chicane("bc", framework, "chicane", list(NAMES[:4]))
    return group, elements, rotation


def _layout(elements):
    return {
        name: (
            np.asarray(e.physical.middle.array),
            float(e.physical.global_rotation.theta),
            float(e.physical.length),
        )
        for name, e in elements.items()
    }


@pytest.mark.parametrize("angle", [0.0, 0.1185])
def test_rotated_chicane_is_the_straight_one_turned(angle):
    straight_group, straight, _ = _build(0.0)
    turned_group, turned, rotation = _build(0.61)
    straight_group.set_angle(angle)
    turned_group.set_angle(angle)

    for name, (pos, theta, length) in _layout(straight).items():
        pos_turned, theta_turned, length_turned = _layout(turned)[name]
        assert rotation.T @ pos_turned == pytest.approx(pos, abs=1e-12)
        assert theta_turned == pytest.approx(theta + 0.61, abs=1e-12)
        assert length_turned == pytest.approx(length, abs=1e-12)


def test_zeroing_puts_everything_back_on_the_axis():
    """What `dipoleangle: 0` is for: no bend, and nothing left standing off the line."""
    group, elements, rotation = _build(0.61)
    group.set_angle(0.1185)
    group.set_angle(0.0)
    origin = group._design_axis[2]
    across = rotation.T[0]
    for name, e in elements.items():
        offset = np.dot(np.asarray(e.physical.middle.array) - origin, across)
        assert offset == pytest.approx(0.0, abs=1e-12), name
        assert e.physical.global_rotation.theta == pytest.approx(0.61, abs=1e-12)
    for name in NAMES[:4]:
        assert elements[name].magnetic.angle == pytest.approx(0.0, abs=1e-15)
        assert elements[name].physical.length == pytest.approx(LZ, abs=1e-12)
