from SimulationFramework.Framework_elements import *
from SimulationFramework.Framework_Settings import FrameworkSettings
import SimulationFramework.Modules.Beams as rbf  # noqa E402
from SimulationFramework.Framework import Framework, disallowed
from SimulationFramework.FrameworkHelperFunctions import convert_numpy_types
from SimulationFramework.Codes.Generators.astra import ASTRAGenerator
from SimulationFramework.Codes.Generators.gpt import GPTGenerator
from test_beam import simple_beam
from test_track import test_fodo_elements
import pytest
import yaml
import os
import shutil

@pytest.fixture
def test_fodo_settings():
    settings = FrameworkSettings()
    settings.settingsFilename = "test.def"
    settings.files = {
        "FODO": {
            "code": "elegant",
            "output": {
                "start_element": "BEGIN",
                "end_element": "END",
            },
            "input": {
                "twiss": {
                    "beta_x": 10,
                    "alpha_x": 0,
                }
            }
        }
    }
    settings.elements = {
        "filename": ["FODO.yaml"]
    }
    return settings

@pytest.fixture
def test_init_framework(test_fodo_elements):
    dic = dict({"elements": dict()})
    latticedict = dic["elements"]
    for k, e in test_fodo_elements.items():
        latticedict[k] = {
            p[0].replace("object", ""): convert_numpy_types(getattr(e, p[0]))
            for p in e
            if p[0] not in disallowed and getattr(e, p[0]) is not None
        }
    with open("./FODO.yaml", "w") as yaml_file:
        yaml.default_flow_style = True
        yaml.dump(dic, yaml_file)
    fw = Framework(directory='./fw_test/', generator_defaults="clara.yaml")
    return fw

def test_framework_functionality(test_init_framework, test_fodo_settings, simple_beam):
    test_init_framework.loadSettings(settings=test_fodo_settings)
    simple_beam.write_HDF5_beam_file(f'./fw_test/BEGIN.hdf5')
    test_init_framework.track()
    elemtypes = test_init_framework.getElementType(typ="quadrupole")
    test_init_framework.setElementType(
        "quadrupole",
        "k1l", [1 for _ in range(len(elemtypes))],
    )
    test_init_framework.save_changes_file(typ="quadrupole")
    test_init_framework.load_changes_file()
    test_init_framework.check_lattice_drifts()
    test_init_framework.change_Lattice_Code("FODO", "astra")
    assert isinstance(test_init_framework["FODO"].__str__(), str)
    with pytest.raises(NotImplementedError):
        test_init_framework.change_Lattice_Code("FODO", "test")
    assert isinstance(test_init_framework.getElement("QUAD1"), quadrupole)
    assert isinstance(test_init_framework.getElement("QUAD1", "k1l"), float)
    with pytest.warns(UserWarning):
        assert test_init_framework.getElement("QUAD1", "test")
        assert test_init_framework.getElement("test") == {}
    test_init_framework.modifyElements(
        elementNames="QUAD1",
        parameter="k1l",
        value=2.0,
    )
    assert test_init_framework.getElement("QUAD1", "k1l") == 2.0
    assert isinstance(test_init_framework["FODO"].getZValues(), list)
    assert isinstance(test_init_framework["FODO"].getZValues(as_dict=True), dict)
    assert isinstance(test_init_framework["FODO"].getElems(), list)
    assert isinstance(test_init_framework["FODO"].getElems(as_dict=True), dict)
    assert isinstance(test_init_framework["FODO"].getSNamesElems(), tuple)
    assert isinstance(test_init_framework["FODO"].getZNamesElems(), tuple)
    test_init_framework["FODO"].file_block.update(
        {
            "output": {
                "zstart": 0.15,
                "zstop": 0.35,
            }
        }
    )
    assert test_init_framework["FODO"].start == "QUAD2"
    assert test_init_framework["FODO"].end == "QUAD3"
    test_init_framework.modifyElements(
        elementNames="all",
        parameter="centre",
        value=[1.0, 2.0, 3.0],
    )
    for elem in test_init_framework.elementObjects.values():
        assert elem.centre == [1.0, 2.0, 3.0]
    test_init_framework.loadSettings(settings=test_fodo_settings)

    test_init_framework.offsetElements(x=0.1, y=0.1)

    for elem in test_init_framework.elementObjects.values():
        assert elem.x == elem.y == 0.1

    os.remove("test_changes.yaml")
    os.remove("FODO.yaml")
    shutil.rmtree("./fw_test")


def test_generator(test_init_framework, test_fodo_settings):
    test_init_framework.loadSettings(settings=test_fodo_settings)
    with pytest.warns(UserWarning):
        test_init_framework.add_Generator("clara_400_2ps_Gaussian")
        assert isinstance(test_init_framework["generator"], ASTRAGenerator)
    test_init_framework["generator"].write()
    test_init_framework.change_generator(generator="gpt")
    assert isinstance(test_init_framework["generator"], GPTGenerator)
    test_init_framework["generator"].write()
    with pytest.warns(UserWarning):
        test_init_framework.change_generator(generator="test")
        assert isinstance(test_init_framework["generator"], ASTRAGenerator)
    shutil.rmtree("./fw_test")
