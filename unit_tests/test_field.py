import simba.Modules.Fields as rff  # noqa E402
from simba.Modules.Fields.FieldParameter import FieldParameter
from simba.Modules.units import UnitValue
import pytest
import numpy as np
import os

@pytest.fixture
def simple_field():
    field = rff.field()

    field_length = 1000
    field.length = field_length
    field.x = FieldParameter(
        name="x",
        value=UnitValue(np.linspace(0, 1, field_length), "m"),
    )
    field.y = FieldParameter(
        name="y",
        value=UnitValue(np.linspace(0, 1, field_length), "m"),
    )
    field.z = FieldParameter(
        name="z",
        value=UnitValue(np.linspace(0, 1, field_length), "m"),
    )
    field.r = FieldParameter(
        name="r",
        value=UnitValue(np.linspace(0, 1, field_length), "m"),
    )
    field.Ex = FieldParameter(
        name="Ex",
        value=UnitValue(np.linspace(0, 1, field_length), "V/m"),
    )
    field.Ey = FieldParameter(
        name="Ey",
        value=UnitValue(np.linspace(0, 1, field_length), "V/m"),
    )
    field.Ez = FieldParameter(
        name="Ez",
        value=UnitValue(np.linspace(0, 1, field_length), "V/m"),
    )
    field.Er = FieldParameter(
        name="Er",
        value=UnitValue(np.linspace(0, 1, field_length), "V/m"),
    )
    field.Bx = FieldParameter(
        name="Bx",
        value=UnitValue(np.linspace(0, 1, field_length), "T"),
    )
    field.By = FieldParameter(
        name="By",
        value=UnitValue(np.linspace(0, 1, field_length), "T"),
    )
    field.Bz = FieldParameter(
        name="Bz",
        value=UnitValue(np.linspace(0, 1, field_length), "T"),
    )
    field.Br = FieldParameter(
        name="Br",
        value=UnitValue(np.linspace(0, 1, field_length), "T"),
    )
    field.Wx = FieldParameter(
        name="Wx",
        value=UnitValue(np.linspace(0, 1, field_length), "V/C/m"),
    )
    field.Wy = FieldParameter(
        name="Wy",
        value=UnitValue(np.linspace(0, 1, field_length), "V/C/m"),
    )
    field.Wz = FieldParameter(
        name="Wz",
        value=UnitValue(np.linspace(0, 1, field_length), "V/C"),
    )
    field.Wr = FieldParameter(
        name="Wr",
        value=UnitValue(np.linspace(0, 1, field_length), "V/C/m"),
    )
    field.G = FieldParameter(
        name="G",
        value=UnitValue(np.linspace(0, 1, field_length), "T/m"),
    )
    field.filename = "test.hdf5"
    field.read = True
    field.frequency = 3e9
    field.cavity_type = "StandingWave"
    return field

def test_astra_field_conversion(simple_field):
    astra_field_types = {
        "LongitudinalWake": ["z", "Wz"],
        "3DWake": ["z", "Wx", "Wy", "Wz"],
        "1DMagnetoStatic": ["z", "Bz"],
        "1DElectroDynamic": ["z", "Ez"],
        "1DQuadrupole": ["z", "G"],
    }

    for name, params in astra_field_types.items():
        simple_field.filename = f"test_{name}.hdf5"
        simple_field.field_type = name
        astraname = simple_field.write_field_file(code="astra")
        newf = rff.field(
            astraname,
            field_type=simple_field.field_type,
            frequency=simple_field.frequency,
            cavity_type=simple_field.cavity_type,
        )
        for param in params:
            assert all(getattr(newf, param).value == getattr(simple_field, param).value)
        for mod in simple_field.model_fields_set:
            if isinstance(
                    getattr(simple_field, mod), FieldParameter
            ) and mod not in params and getattr(newf, mod).value is not None:
                raise AssertionError
        os.remove(astraname)

def test_gdf_field_conversion(simple_field):
    gdf_field_types = {
        "LongitudinalWake": ["z", "Wz"],
        "TransverseWake": ["z", "Wx", "Wy"],
        "3DWake": ["z", "Wx", "Wy", "Wz"],
        "1DMagnetoStatic": ["z", "Bz"],
        "3DMagnetoStatic": ["x", "y", "z", "Bx", "By", "Bz"],
        "1DElectroDynamic": ["z", "Ez"],
    }

    for name, params in gdf_field_types.items():
        simple_field.filename = f"test_{name}.hdf5"
        simple_field.field_type = name
        gdfname = simple_field.write_field_file(code="gdf")
        newf = rff.field(
            gdfname,
            field_type=simple_field.field_type,
            frequency=simple_field.frequency,
            cavity_type=simple_field.cavity_type,
            normalize_b=False
        )
        for param in params:
            assert all(getattr(newf, param).value == getattr(simple_field, param).value)
        for mod in simple_field.model_fields_set:
            if isinstance(
                    getattr(simple_field, mod), FieldParameter
            ) and mod not in params and getattr(newf, mod).value is not None:
                raise AssertionError
        os.remove(gdfname)

def test_hdf5_field_conversion(simple_field):
    field_types = rff.allowed_fields

    for name in field_types:
        simple_field.filename = f"test_{name}.hdf5"
        simple_field.field_type = name
        hdf5name = simple_field.write_field_file(code="hdf5")
        newf = rff.field(
            hdf5name,
            field_type=simple_field.field_type,
            frequency=simple_field.frequency,
            cavity_type=simple_field.cavity_type,
        )
        for mod in simple_field.model_fields_set:
            if isinstance(getattr(simple_field, mod), FieldParameter):
                assert all(getattr(newf, mod).value == getattr(simple_field, mod).value)
            elif type(getattr(simple_field, mod)) in [str, float, int]:
                assert getattr(newf, mod) == getattr(simple_field, mod)
        os.remove(hdf5name)

def test_sdds_field_conversion(simple_field):
    sdds_field_types = {
        "LongitudinalWake": ["z", "Wz"],
        "TransverseWake": ["z", "Wx", "Wy"],
        "3DWake": ["z", "Wx", "Wy", "Wz"],
    }

    for name, params in sdds_field_types.items():
        simple_field.filename = f"test_{name}.sdds"
        simple_field.field_type = name
        sddsname = simple_field.write_field_file(code="sdds")
        newf = rff.field(
            sddsname,
            field_type=simple_field.field_type,
            frequency=simple_field.frequency,
            cavity_type=simple_field.cavity_type,
        )
        for param in params:
            assert all(getattr(newf, param).value == getattr(simple_field, param).value)
        for mod in simple_field.model_fields_set:
            if isinstance(
                    getattr(simple_field, mod), FieldParameter
            ) and mod not in params and getattr(newf, mod).value is not None:
                raise AssertionError
        os.remove(sddsname)

def test_opal_field_conversion(simple_field):
    opal_field_types = {
        "LongitudinalWake": ["z", "Wz"],
        "TransverseWake": ["z", "Wx", "Wy"],
        "3DWake": ["z", "Wx", "Wy", "Wz"],
        "1DMagnetoStatic": ["z", "Bz"],
        # "2DMagnetoStatic": ["z", "r", "Bz", "Br"],
        "1DElectroDynamic": ["z", "Ez"],
        # "2DElectroDynamic": ["z", "r", "Ez", "Er"],
    }

    for name, params in opal_field_types.items():
        simple_field.filename = f"test_{name}.opal"
        simple_field.field_type = name
        if ("wake" in name.lower()) or name == "1DMagnetoStatic":
            with pytest.warns(UserWarning):
                opalname = simple_field.write_field_file(code="opal")
        else:
            opalname = simple_field.write_field_file(code="opal")
        newf = rff.field(
            opalname,
            field_type=simple_field.field_type,
            frequency=simple_field.frequency,
            cavity_type=simple_field.cavity_type,
        )
        for param in params:
            assert all(getattr(newf, param).value == getattr(simple_field, param).value)
        for mod in simple_field.model_fields_set:
            if isinstance(
                    getattr(simple_field, mod), FieldParameter
            ) and mod not in params and getattr(newf, mod).value is not None:
                raise AssertionError
        os.remove(opalname)