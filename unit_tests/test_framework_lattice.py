import importlib
import inspect
import pkgutil
import re
import types

import pytest
import simba.Framework as sfw
import simba.Codes
import simba.Modules.Beams as rbf
from simba.Codes.ASTRA.ASTRA import astra_newrun, astraLattice
from simba.Framework_objects import frameworkLattice
from simba.Modules.Twiss.astra import read_s_offset
from simba.Modules.units import UnitValue
from laura.models.element import Quadrupole, Marker, Element, PhysicalBaseElement
from laura.models.physical import PhysicalElement

@pytest.fixture
def framework_with_elements(tmp_path):
    fw_obj = sfw.Framework(directory=str(tmp_path))
    quad1 = Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A1")
    quad2 = Quadrupole(name="Q2", hardware_class="Magnet", machine_area="A1")
    marker = Element(name="M1", hardware_class="Marker", hardware_type="Marker", machine_area="A1")
    fw_obj.elementObjects = {"Q1": quad1, "Q2": quad2, "M1": marker}
    return fw_obj

def test_get_element_returns_full_object(framework_with_elements):
    fw = framework_with_elements
    elem = fw.getElement("Q1")
    assert isinstance(elem, Quadrupole)
    assert elem.name == "Q1"

def test_get_element_returns_specific_param(framework_with_elements):
    fw = framework_with_elements
    param = fw.getElement("Q1", "hardware_class")
    assert param == "Magnet"

def test_get_element_nonexistent_returns_empty_dict(framework_with_elements):
    fw = framework_with_elements
    with pytest.warns(UserWarning):
        result = fw.getElement("NON_EXISTENT")
        assert result == {}

def test_get_element_type_returns_all(framework_with_elements):
    fw = framework_with_elements
    quads = fw.getElementType("Quadrupole")
    names = [e["name"] for e in quads]
    assert set(names) == {"Q1", "Q2"}

def test_get_element_type_with_param(framework_with_elements):
    fw = framework_with_elements
    classes = fw.getElementType("Quadrupole", param="hardware_class")
    assert all(c == "Magnet" for c in classes)

def test_set_element_type_updates_values(framework_with_elements):
    fw = framework_with_elements
    fw.setElementType("Quadrupole", "virtual_name", ["NewQ1", "NewQ2"])
    assert fw.elementObjects["Q1"].virtual_name == "NewQ1"
    assert fw.elementObjects["Q2"].virtual_name == "NewQ2"

def test_set_element_type_raises_on_length_mismatch(framework_with_elements):
    fw = framework_with_elements
    with pytest.raises(ValueError):
        fw.setElementType("Quadrupole", "virtual_name", ["OnlyOne"])

def test_set_lattice_prefix_sets_prefix(framework_with_elements):
    class MockLattice:
        def __init__(self):
            self.prefix = None
        def set_prefix(self, p):
            self.prefix = p
    fw = framework_with_elements
    fw.latticeObjects["L1"] = MockLattice()
    fw.set_lattice_prefix("L1", "prefix_value")
    assert fw.latticeObjects["L1"].prefix == "prefix_value"

def _all_lattice_classes():
    for m in pkgutil.walk_packages(simba.Codes.__path__, "simba.Codes."):
        try:
            importlib.import_module(m.name)
        except Exception:
            pass
    todo, seen = list(frameworkLattice.__subclasses__()), []
    while todo:
        cls = todo.pop()
        if cls not in seen:
            seen.append(cls)
            todo.extend(cls.__subclasses__())
    return seen

def test_no_property_setter_shadowed_by_inherited_field():
    """A property whose name is already a pydantic field on the parent is silently dropped
    by the metaclass, so assignment writes the field and the setter never runs. This broke
    astraLattice.sample_interval (n_red stayed 1 however it was set)."""
    broken = {}
    for cls in _all_lattice_classes():
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        setters = set(re.findall(r"@(\w+)\.setter", src))
        lost = sorted(p for p in setters if p not in cls.__dict__ and p in cls.model_fields)
        if lost:
            broken[cls.__name__] = lost
    assert not broken, f"property setters dropped by pydantic: {broken}"

def _astra_lattice_stub(tmp_path, zstart, zstop):
    """Enough of an astraLattice for find_ASTRA_filename, which only touches these three."""
    return types.SimpleNamespace(
        startObject=types.SimpleNamespace(physical=types.SimpleNamespace(start=types.SimpleNamespace(z=zstart))),
        zstop=zstop,
        global_parameters={"master_subdir": str(tmp_path)},
    )

def test_find_astra_filename_prefers_screen_position_over_lattice_end(tmp_path):
    """A screen must pick up its own ASTRA output. The end-of-lattice names are a last
    resort - when they were tried first every screen silently got the final distribution."""
    for name in ["S07.3837.001", "S07.4830.001"]:
        (tmp_path / name).write_text("")
    latt = _astra_lattice_stub(tmp_path, 38.2678, 48.2978)
    screen = types.SimpleNamespace(physical=types.SimpleNamespace(middle=types.SimpleNamespace(z=38.37299)))
    assert astraLattice.find_ASTRA_filename(latt, "S07", screen, 1, 100) == "S07.3837.001"

def test_find_astra_filename_falls_back_to_relative_mm_naming(tmp_path):
    """ASTRA switches from cm to mm-relative-to-zstart for sections under 1m (e.g. L4H),
    and `mult` comes from the screens, which a screenless lattice does not have."""
    (tmp_path / "L4H.0965.001").write_text("")
    latt = _astra_lattice_stub(tmp_path, 23.4572, 24.4222)
    end = types.SimpleNamespace(physical=types.SimpleNamespace(middle=types.SimpleNamespace(z=24.4222)))
    assert astraLattice.find_ASTRA_filename(latt, "L4H", end, 1, 100) == "L4H.0965.001"

@pytest.mark.parametrize(
    "entrance_s, start_z, expected",
    [
        (0.0, 0.0, 0.0),                    # injector400: nothing upstream
        (3.296930, 3.296930, 0.0),          # S02: nothing bending upstream yet
        (31.290855, 31.267800, 0.023055),   # S06: downstream of the VBC chicane
    ],
)
def test_astra_s_offset(entrance_s, start_z, expected):
    """s - z at the lattice entrance: the extra path length accumulated by upstream
    bends. ASTRA tracks in lab z, so its output needs this added back on."""
    latt = types.SimpleNamespace(
        entrance_s=entrance_s,
        startObject=types.SimpleNamespace(
            physical=types.SimpleNamespace(start=types.SimpleNamespace(z=start_z))
        ),
    )
    assert astraLattice.s_offset.fget(latt) == pytest.approx(expected)

@pytest.mark.parametrize(
    "start_s, start_length, expected",
    [(0.32, 0.32, 0.0), (3.296930, 0.0, 3.296930), (31.290855, 0.0, 31.290855)],
)
def test_entrance_s_backs_off_the_first_element(start_s, start_length, expected):
    """Per-element s values are measured from the lattice entrance, but start_s is the s
    at the *exit* of the first element. Equal for a zero-length marker; different for a
    lattice starting on a real element (injector400 starts on the 0.32m gun cavity)."""
    latt = types.SimpleNamespace(
        start_s=start_s,
        startObject=types.SimpleNamespace(
            physical=types.SimpleNamespace(length=start_length)
        ),
    )
    assert frameworkLattice.entrance_s.fget(latt) == pytest.approx(expected)

def test_read_s_offset_round_trip(tmp_path):
    (tmp_path / "S06.s_offset").write_text(repr(0.031444))
    assert read_s_offset(str(tmp_path / "S06.Xemit.001"), "S06") == pytest.approx(0.031444)

def test_read_s_offset_defaults_to_zero_when_absent(tmp_path):
    """Directories written before the offset existed, or by something else, read as s == z."""
    assert read_s_offset(str(tmp_path / "S06.Xemit.001"), "S06") == 0.0

def test_beam_s_must_be_written_to_the_particles_object():
    """astra_to_hdf5 assigns beam.Particles.s: the beam wrapper forwards attribute *reads*
    to the Particles object but not writes, so `beam.s = ...` is a silent no-op."""
    b = rbf.beam()
    b.Particles.s = UnitValue(12.34, units="m")
    assert float(b.s) == pytest.approx(12.34)

@pytest.mark.parametrize("interval", [1, 8, 64])
def test_astra_sample_interval_written_as_n_red(interval):
    """astraLattice.preProcess copies sample_interval into this header; guard the mapping."""
    header = astra_newrun(
        global_parameters={},
        input_particle_definition="test.astra",
        sample_interval=interval,
    )
    assert f"n_red = {interval}," in header.write_ASTRA()

def _lattice_stub(path, elements, output):
    """Enough of a frameworkLattice for the `start` property."""
    return types.SimpleNamespace(
        file_block={"output": output},
        elementObjects=elements,
        end=path[-1],
        machine=types.SimpleNamespace(elements_between=lambda end: path),
    )

def _element(z, length):
    return PhysicalBaseElement(
        name="E", hardware_class="RF", hardware_type="RFCavity", machine_area="HRG1",
        physical=PhysicalElement(middle=[0, 0, z + length / 2.0], length=length),
    )

def test_start_uses_explicit_start_element():
    latt = _lattice_stub(["A", "B"], {}, {"start_element": "B"})
    assert frameworkLattice.start.fget(latt) == "B"

def test_start_from_zstart_skips_zero_length_hardware_at_the_same_z():
    """The HRG1 section lists three laser shutters and an aperture -- all zero-length and
    all at z=0 -- ahead of the gun cavity. Picking by z alone made the answer depend on
    element ordering, which differs between the summary file and the YAML loader."""
    path = ["SHUT-01", "SHUT-02", "APER-01", "GUN-CAV-01", "SOL-01"]
    elements = {
        "SHUT-01": _element(0.0, 0.0),
        "SHUT-02": _element(0.0, 0.0),
        "APER-01": _element(0.0, 0.0),
        "GUN-CAV-01": _element(0.0, 0.32),
        "SOL-01": _element(0.00241, 0.32),
    }
    latt = _lattice_stub(path, elements, {"zstart": 0, "end_element": "SOL-01"})
    assert frameworkLattice.start.fget(latt) == "GUN-CAV-01"

def test_start_falls_back_to_first_element_on_the_beam_path():
    """Not the first key of elementObjects: that is the whole machine, and it contains
    off-beamline hardware such as the virtual cathode camera."""
    path = ["APER-01", "QUAD-01"]
    elements = {"CAMERA": _element(0.0, 0.0), "APER-01": _element(5.0, 0.0)}
    latt = _lattice_stub(path, elements, {})
    assert frameworkLattice.start.fget(latt) == "APER-01"


@pytest.fixture(scope="module")
def clara():
    import os
    ml = os.environ.get("CLARA_MASTER_LATTICE",
                        r"C:\Users\jkj62.CLRC\Documents\GitHub\laura-lattices\CLARA")
    if not os.path.isdir(ml):
        pytest.skip("CLARA master lattice not available")
    import tempfile
    f = sfw.Framework(directory=tempfile.mkdtemp(), clean=False, verbose=False,
                      master_lattice=ml, generator_defaults="clara.yaml")
    f.loadSettings("Lattices/clara400_v13.def")
    return f

@pytest.mark.parametrize("angle, arc, path", [(0.0, 0.200981, 5.442500),
                                              (0.1185, 0.201452, 5.465552)])
def test_variable_chicane_geometry(clara, angle, arc, path):
    """The VBC translates its middle two dipoles without ever rotating a magnet: the
    faces stay perpendicular to the 0mm axis, so each magnet spans a fixed z while the
    arc through it lengthens with the angle, and the edge angles carry the bend."""
    clara.groupObjects["bunch_compressor"].set_angle(angle)
    dips = [clara.elementObjects[f"CLA-VBC-MAG-DIP-0{i}"].physical for i in range(1, 5)]

    for p in dips:
        assert p.global_rotation.theta == pytest.approx(0.0, abs=1e-12)   # never rotated
        assert p.length == pytest.approx(arc, abs=1e-6)                   # arc grows
        assert p.end.z - p.start.z == pytest.approx(0.200981, abs=1e-6)   # z extent fixed

    # the offset leg is a straight line: no sideways step between dipoles 2 and 3
    assert dips[2].start.x - dips[1].end.x == pytest.approx(0.0, abs=1e-9)
    # and the chicane closes back onto the axis
    assert dips[3].end.x == pytest.approx(0.0, abs=1e-9)
    assert clara.latticeObjects["VBC"].getSValues(at_entrance=False)[-1] == pytest.approx(path, abs=1e-6)

def test_variable_chicane_set_angle_is_idempotent(clara):
    clara.groupObjects["bunch_compressor"].set_angle(0.1185)
    first = [clara.elementObjects[f"CLA-VBC-MAG-DIP-0{i}"].physical.length for i in range(1, 5)]
    clara.groupObjects["bunch_compressor"].set_angle(0.1185)
    again = [clara.elementObjects[f"CLA-VBC-MAG-DIP-0{i}"].physical.length for i in range(1, 5)]
    assert first == pytest.approx(again)
