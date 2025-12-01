# from pydantic import ValidationError
#
# from SimulationFramework.Framework_elements import *
# from SimulationFramework.Framework_objects import frameworkElement
# import SimulationFramework.Modules.Beams as rbf  # noqa E402
# from test_field import simple_field
# import pytest
# import warnings
# import numpy as np
#
# def all_subclasses(cls):
#     subclasses = cls.__subclasses__()
#     for subclass in subclasses:
#         subclasses += all_subclasses(subclass)
#     return subclasses
#
# @pytest.fixture
# def simple_element():
#     return frameworkElement(
#         objectname="ELEMENT",
#         objecttype="marker",
#         centre=[0, 0, 0],
#     )
#
# def test_framework_object(simple_element):
#     with pytest.raises(NameError):
#         frameworkElement(
#             objectname="ELEMENT",
#             objecttype="element",
#             centre=[0, 0, 0],
#         )
#     with pytest.raises(ValueError):
#         frameworkElement(
#             objectname=1,
#             objecttype="element",
#             centre=[0, 0, 0],
#         )
#     with pytest.raises(ValueError):
#         frameworkElement(
#             objectname="ELEMENT",
#             objecttype=1,
#             centre=[0, 0, 0],
#         )
#     with pytest.raises(ValidationError):
#         simple_element.centre= -1
#
# def test_framework_object_positions(simple_element):
#     for param in ["x", "y", "z"]:
#         setattr(simple_element, param, 0.1)
#         setattr(simple_element, "d" + param, 0.1)
#         setattr(simple_element, param + "_rot", 0.1)
#         setattr(simple_element, "d" + param + "_rot", 0.1)
#     assert simple_element.position_errors == [0.1, 0.1, 0.1]
#     assert simple_element.centre == [0.1, 0.1, 0.1]
#     assert simple_element.global_rotation == [0.1, 0.1, 0.1]
#     assert simple_element.rotation_errors == [0.1, 0.1, 0.1]
#     assert simple_element.tilt == 0.1
#     assert simple_element.theta == 0.1
#     assert simple_element.get_field_amplitude is None
#
#
# def test_framework_elements(simple_field):
#     warnings.filterwarnings("ignore", category=UserWarning)
#     exclude = ["global", "gpt", "astra", "csrtrack"]
#     for elem in all_subclasses(frameworkElement):
#         if not any(x in elem.__name__ for x in exclude):
#             element = elem(
#                 name="ELEMENT",
#                 type=elem.__name__,
#                 global_parameters={"master_subdir": ".", "master_lattice_location": "."},
#                 length=1.0,
#             )
#             if elem.__name__ == "dipole":
#                 element.angle = 1.0
#                 for param in [
#                     "arc_middle",
#                     "line_middle",
#                     "TD_middle",
#                     "intersection",
#                     "position_start",
#                     "position_end",
#                     "astra_end",
#                 ]:
#                     assert isinstance(getattr(element, param), np.ndarray)
#             element.write_Elegant()
#             element.write_Ocelot()
#             if elem.__name__ in ["cavity", "solenoid", "rf_deflecting_cavity"]:
#                 if elem.__name__ in ["cavity", "rf_deflecting_cavity"]:
#                     element.frequency = 1e9
#                     element.wakefield_definition = simple_field
#                 element.field_amplitude = 1e6
#                 element.field_definition = simple_field
#             if elem.__name__ == "wakefield":
#                 element.field_definition = simple_field
#             if elem.__name__ in ["aperture", "collimator", "scraper", "rcollimator"]:
#                 for typ in ["circular", "planar", "rectangular"]:
#                     element.shape = typ
#                     element.horizontal_size = element.vertical_size = 1.0
#                     element.write_ASTRA(1)
#             element.write_ASTRA(1)
#             assert element.x == element.y == element.z == element.dx == element.dy == element.dz == 0