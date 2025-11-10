import os
import pytest
from unittest.mock import MagicMock
import shutil

import simba.Framework as fw
from simba.Codes.Generators import frameworkGenerator
from nala.models.element import Quadrupole, Marker
from nala import NALA
from nala.Exporters.YAML import export_machine

@pytest.fixture
def simple_machine():
    outdir = f"{os.path.dirname(os.path.abspath(__file__))}/framework"

    m1 = Marker(
        name="M1",
        machine_area="FODO",
        hardware_class="Marker",
        physical={
            "middle": {"x": 0.0, "y": 0.0, "z": 0.0}
        }
    )

    q1f = Quadrupole(
        name="QUAD1F",
        machine_area="FODO",
        magnetic={"length": 1.0, "k1l": -1},
        physical={
            "length": 1.0,
            "middle": {"x": 0.0, "y": 0.0, "z": 0.75}
        }
    )

    q1d = Quadrupole(
        name="QUAD1D",
        machine_area="FODO",
        magnetic={"length": 1.0, "k1l": 1.0},
        physical={
            "length": 1.0,
            "middle": {"x": 0.0, "y": 0.0, "z": 3.25}
        }
    )

    m3 = Marker(
        name="M3",
        machine_area="FODO",
        hardware_class="Marker",
        physical={
            "middle": {"x": 0.0, "y": 0.0, "z": 4.0}
        }
    )

    sections = {"sections": {"FODO": ["M1", "QUAD1F", "QUAD1D", "M3"]}}
    layouts = {"default_layout": "line1", "layouts": {"line1": ["FODO"]}}

    machine = NALA(element_list=[m1, q1f, q1d, m3], layout=layouts, section=sections)
    export_machine(path=f"{outdir}/lattice", machine=machine, overwrite=True)
    return machine, outdir

@pytest.fixture
def simple_generator():
    gen = frameworkGenerator(
        global_parameters={
            "master_subdir": f"{os.path.dirname(os.path.abspath(__file__))}"
        },
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4,
        sigma_px=1e3,
        sigma_y=1e-4,
        sigma_py=1e3,
        sigma_z=1e-3,
        sigma_pz=1e3,
        gaussian_cutoff_x=3,
        gaussian_cutoff_y=3,
        gaussian_cutoff_z=3,
        gaussian_cutoff_px=3,
        gaussian_cutoff_py=3,
        gaussian_cutoff_pz=3,
        charge=100e-12,
    )
    return gen.write()

def test_framework_initialization(simple_machine):
    machine, outdir = simple_machine

    framework = fw.Framework(
        machine=machine,
        # simcodes="/path/to/simcodes",
        directory=os.path.join(outdir, "ocelot"),
        clean=True,
        verbose=True
    )

    assert framework.machine == machine
    assert framework.directory == os.path.join(outdir, "ocelot")

def test_framework_settings_and_tracking(simple_machine, simple_generator):
    machine, outdir = simple_machine

    gen = simple_generator

    settings = fw.FrameworkSettings()
    files = {}
    for sec, elems in machine.sections.items():
        files[sec] = {
            "code": "ocelot",
            "charge": {"space_charge_mode": "False"},
            "input": {
                "twiss": {
                    "beta_x": 3.2844606,
                    "alpha_x": 2.48956886,
                    "nemit_x": 1e-6,
                    "beta_y": 3.2846606,
                    "alpha_y": -2.48956886,
                    "nemit_y": 1e-6,
                }
            },
            "output": {
                "start_element": elems[0].name,
                "end_element": elems[-1].name,
            },
        }
    settings.files = files
    settings.layout = machine.layout
    settings.section = {"sections": {name: e.names for name, e in machine.sections.items()}}
    settings.element_list = os.path.join(outdir, "lattice")

    framework = fw.Framework(
        machine=machine,
        # simcodes="/path/to/simcodes",
        directory=os.path.join(outdir, "ocelot"),
        clean=True,
        verbose=True
    )

    framework.loadSettings(settings=settings)
    framework.global_parameters["beam"] = MagicMock()
    framework["FODO"].lsc_enable = False
    framework["FODO"].csr_enable = False
    framework.set_lattice_prefix(
        "FODO",
        f"{os.path.dirname(os.path.abspath(__file__))}/"
    )

    # Mock the track method to avoid running actual simulations
    framework.track = MagicMock()
    framework.track()
    # framework.track.assert_called_once()

    shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__))}/framework")
    os.remove(f"{os.path.dirname(os.path.abspath(__file__))}/M1.openpmd.hdf5")

# from SimulationFramework.Framework_elements import *
# from SimulationFramework.Framework_Settings import FrameworkSettings
# import SimulationFramework.Modules.Beams as rbf  # noqa E402
# from SimulationFramework.Framework import Framework, disallowed
# from SimulationFramework.FrameworkHelperFunctions import convert_numpy_types
# from SimulationFramework.Codes.Generators.astra import ASTRAGenerator
# from SimulationFramework.Codes.Generators.gpt import GPTGenerator
# from test_beam import simple_beam
# from test_track import test_fodo_elements
# import pytest
# import yaml
# import os
# import shutil

# @pytest.fixture
# def test_fodo_settings():
#     settings = FrameworkSettings()
#     settings.settingsFilename = "test.def"
#     settings.files = {
#         "FODO": {
#             "code": "elegant",
#             "output": {
#                 "start_element": "BEGIN",
#                 "end_element": "END",
#             },
#             "input": {
#                 "twiss": {
#                     "beta_x": 10,
#                     "alpha_x": 0,
#                 }
#             }
#         }
#     }
#     settings.elements = {
#         "filename": ["FODO.yaml"]
#     }
#     return settings
#
# @pytest.fixture
# def test_init_framework(test_fodo_elements):
#     dic = dict({"elements": dict()})
#     latticedict = dic["elements"]
#     for k, e in test_fodo_elements.items():
#         latticedict[k] = {
#             p[0].replace("object", ""): convert_numpy_types(getattr(e, p[0]))
#             for p in e
#             if p[0] not in disallowed and getattr(e, p[0]) is not None
#         }
#     with open("./FODO.yaml", "w") as yaml_file:
#         yaml.default_flow_style = True
#         yaml.dump(dic, yaml_file)
#     fw = Framework(directory='./fw_test/', generator_defaults="clara.yaml")
#     return fw
#
# def test_framework_functionality(test_init_framework, test_fodo_settings, simple_beam):
#     test_init_framework.loadSettings(settings=test_fodo_settings)
#     simple_beam.write_HDF5_beam_file(f'./fw_test/BEGIN.hdf5')
#     test_init_framework.track()
#     elemtypes = test_init_framework.getElementType(typ="quadrupole")
#     test_init_framework.setElementType(
#         "quadrupole",
#         "k1l", [1 for _ in range(len(elemtypes))],
#     )
#     test_init_framework.save_changes_file(typ="quadrupole")
#     test_init_framework.load_changes_file()
#     test_init_framework.check_lattice_drifts()
#     test_init_framework.change_Lattice_Code("FODO", "astra")
#     assert isinstance(test_init_framework["FODO"].__str__(), str)
#     with pytest.raises(NotImplementedError):
#         test_init_framework.change_Lattice_Code("FODO", "test")
#     assert isinstance(test_init_framework.getElement("QUAD1"), quadrupole)
#     assert isinstance(test_init_framework.getElement("QUAD1", "k1l"), float)
#     with pytest.warns(UserWarning):
#         assert test_init_framework.getElement("QUAD1", "test")
#         assert test_init_framework.getElement("test") == {}
#     test_init_framework.modifyElements(
#         elementNames="QUAD1",
#         parameter="k1l",
#         value=2.0,
#     )
#     assert test_init_framework.getElement("QUAD1", "k1l") == 2.0
#     assert isinstance(test_init_framework["FODO"].getZValues(), list)
#     assert isinstance(test_init_framework["FODO"].getZValues(as_dict=True), dict)
#     assert isinstance(test_init_framework["FODO"].getElems(), list)
#     assert isinstance(test_init_framework["FODO"].getElems(as_dict=True), dict)
#     assert isinstance(test_init_framework["FODO"].getSNamesElems(), tuple)
#     assert isinstance(test_init_framework["FODO"].getZNamesElems(), tuple)
#     test_init_framework["FODO"].file_block.update(
#         {
#             "output": {
#                 "zstart": 0.15,
#                 "zstop": 0.35,
#             }
#         }
#     )
#     assert test_init_framework["FODO"].start == "QUAD2"
#     assert test_init_framework["FODO"].end == "QUAD3"
#     test_init_framework.modifyElements(
#         elementNames="all",
#         parameter="centre",
#         value=[1.0, 2.0, 3.0],
#     )
#     for elem in test_init_framework.elementObjects.values():
#         assert elem.centre == [1.0, 2.0, 3.0]
#     test_init_framework.loadSettings(settings=test_fodo_settings)
#
#     test_init_framework.offsetElements(x=0.1, y=0.1)
#
#     for elem in test_init_framework.elementObjects.values():
#         assert elem.x == elem.y == 0.1
#
#     os.remove("test_changes.yaml")
#     os.remove("FODO.yaml")
#     shutil.rmtree("./fw_test")
#
#
# def test_generator(test_init_framework, test_fodo_settings):
#     test_init_framework.loadSettings(settings=test_fodo_settings)
#     with pytest.warns(UserWarning):
#         test_init_framework.add_Generator("clara_400_2ps_Gaussian")
#         assert isinstance(test_init_framework["generator"], ASTRAGenerator)
#     test_init_framework["generator"].write()
#     test_init_framework.change_generator(generator="gpt")
#     assert isinstance(test_init_framework["generator"], GPTGenerator)
#     test_init_framework["generator"].write()
#     with pytest.warns(UserWarning):
#         test_init_framework.change_generator(generator="test")
#         assert isinstance(test_init_framework["generator"], ASTRAGenerator)
#     shutil.rmtree("./fw_test")
