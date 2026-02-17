# from importlib.metadata import PackageNotFoundError
#
# from SimulationFramework.Framework_elements import *
# from SimulationFramework.Framework import runSetup, load_directory
# from SimulationFramework.Framework_Settings import FrameworkSettings
# import SimulationFramework.Modules.Beams as rbf  # noqa E402
# from SimulationFramework.Modules.Beams.plot import (
#     plotScreenImage,
#     density_plot,
#     marginal_plot,
#     slice_plot,
# )
# import SimulationFramework.Modules.Twiss as rtf
# import SimulationFramework.Codes.Executables as exes
# from SimulationFramework.Framework_lattices import (
#     elegantLattice,
#     astraLattice,
#     ocelotLattice,
#     gptLattice,
#     csrtrackLattice,
#     cheetahLattice,
# )
# from SimulationFramework.FrameworkHelperFunctions import convert_numpy_types
# from SimulationFramework.Modules.Beams import Particles
# from SimulationFramework.Modules.Beams.Particles.emittance import emittance as emittanceobject
# from SimulationFramework.Modules.Beams.Particles.twiss import twiss as twissobject
# from SimulationFramework.Modules.Beams.Particles.slice import slice as sliceobject
# from SimulationFramework.Modules.Beams.Particles.sigmas import sigmas as sigmasobject
# from SimulationFramework.Modules.Beams.Particles.centroids import centroids as centroidsobject
# from test_beam import simple_beam
# import pytest
# import yaml
# from copy import deepcopy
# import os
# import shutil
#
# @pytest.fixture
# def test_fodo_elements():
#     begin = {
#         "objectname": "BEGIN",
#         "objecttype": "marker",
#         "centre": [0, 0, 0.0],
#     }
#
#     q1 = {
#         "objectname": "QUAD1",
#         "objecttype": "quadrupole",
#         "centre": [0, 0, 0.05],
#         "length": 0.1,
#         "k1l": -2,
#     }
#
#     q2 = {
#         "objectname": "QUAD2",
#         "objecttype": "quadrupole",
#         "centre": [0, 0, 0.2],
#         "length": 0.1,
#         "k1l": 4,
#     }
#
#     q3 = {
#         "objectname": "QUAD3",
#         "objecttype": "quadrupole",
#         "centre": [0, 0, 0.35],
#         "length": 0.1,
#         "k1l": -1,
#         "bore": 0.037,
#     }
#
#     end = {
#         "objectname": "END",
#         "objecttype": "marker",
#         "centre": [0, 0, 0.4],
#     }
#
#     beginelem = marker(**begin)
#     q1elem = quadrupole(**q1)
#     q2elem = quadrupole(**q2)
#     q3elem = quadrupole(**q3)
#     endelem = marker(**end)
#
#     elementobjects = {
#         name: elem for name, elem in zip(
#             ["BEGIN"] + [f"QUAD{i}" for i in range(1, 4)] + ["END"],
#             [beginelem, q1elem, q2elem, q3elem, endelem, ],
#         )
#     }
#     return elementobjects
#
# # @pytest.mark.parametrize("code,lattice_class", [
# #     ("elegant", elegantLattice),
# #     ("astra", astraLattice),
# #     ("ocelot", ocelotLattice),
# #     # ("gpt", gptLattice),
# # ])
# def prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=True):
#     if not os.path.isdir(f'./fodo/{code}'):
#         os.makedirs(f'./fodo/{code}')
#     simple_beam.write_HDF5_beam_file(f'./fodo/{code}/BEGIN.hdf5')
#     groupobjects = {}
#     rs = runSetup()
#     settings = FrameworkSettings()
#     try:
#         import SimCodes  # type: ignore
#
#         SimCodesLocation = os.path.dirname(SimCodes.__file__) + "/"
#     except ImportError:
#         raise PackageNotFoundError("SimCodes package must be installed for test to work")
#     global_parameters = {
#         "beam": deepcopy(simple_beam),
#         "simcodes_location": SimCodesLocation,
#         "master_subdir": f'./fodo/{code}',
#         "master_lattice": f'./fodo/{code}',
#         "delete_tracking_files": False,
#     }
#     executables = exes.Executables(global_parameters)
#     latticedict = {
#         "FODO": {
#             "code": code,
#             "input": {},
#             "output": {
#                 "start_element": "BEGIN",
#                 "end_element": "END",
#             },
#             "charge": {
#                 "space_charge_mode": "3d",
#             },
#             "csr": {
#                 "start": "BEGIN",
#                 "end": "END",
#             },
#         }
#     }
#     for name, elem in test_fodo_elements.items():
#         setattr(elem, "global_parameters", global_parameters)
#     latattrs = {
#         "name": "FODO",
#         "file_block": latticedict["FODO"],
#         "elementObjects": test_fodo_elements,
#         "groupObjects": groupobjects,
#         "runSettings": rs,
#         "settings": settings,
#         "executables": executables,
#         "global_parameters": global_parameters,
#         "lscDrifts": False,
#         "csrDrifts": False,
#     }
#     lattice = lattice_class(**latattrs)
#     assert isinstance(lattice.getElement(element="QUAD1"), quadrupole)
#     assert lattice.getElement(element="QUAD1", param="k1l") == -2
#     with pytest.warns(UserWarning):
#         assert lattice.getElement(element="test") == {}
#     if remove:
#         shutil.rmtree(f'./fodo/{code}')
#         shutil.rmtree(f'../fodo/{code}')
#     return lattice
#
# @pytest.mark.parametrize("code,lattice_class", [
#     ("elegant", elegantLattice),
#     # ("gpt", gptLattice),
# ])
# def test_track_and_analyze(simple_beam, test_fodo_elements, code, lattice_class):
#     lattice = prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=False)
#     lattice.preProcess()
#     lattice.write()
#     lattice.run()
#     lattice.postProcess()
#
#     settings = lattice.settings.copy()
#     settings = convert_numpy_types(settings)
#     subdir = lattice.global_parameters["master_subdir"]
#     t = rtf.load_directory(subdir)
#     t.save_HDF5_twiss_file(f"{subdir}/Twiss_Summary.hdf5")
#     rbf.save_HDF5_summary_file(subdir, f"{subdir}/Beam_Summary.hdf5")
#     with open(lattice.global_parameters["master_subdir"] + "/settings.def", "w") as yaml_file:
#         yaml.default_flow_style = True
#         yaml.safe_dump(settings, yaml_file, sort_keys=False)
#     fwdir = load_directory(lattice.global_parameters["master_subdir"], beams=True)
#     plt1, fig1, ax1 = fwdir.plot(
#         include_layout=True,
#         include_particles=True,
#         ykeys=['sigma_x', 'sigma_y'],
#         ykeys2=['sigma_z'],
#     )
#     plt1.close()
#     plt2, fig2, ax2 = fwdir.general_plot(
#         include_layout=True,
#         include_particles=True,
#         ykeys=['sigma_x', 'sigma_y'],
#         ykeys2=['sigma_z'],
#     )
#     plt2.close()
#     b1 = fwdir.beams[0]
#     plotScreenImage(b1, keys=["z", "cpz"], subtract_mean=[True, False])
#     density_plot(b1, key="x", bins=20)
#     marginal_plot(b1, key1="t", key2="cpz", bins=50, subtract_mean=[True, False])
#     slice_plot(b1, bins=500)
#     assert isinstance(fwdir.beams.sigmas, rbf.particlesGroup)
#     assert isinstance(fwdir.beams.sigmas.particles[0], sigmasobject)
#     assert isinstance(fwdir.beams.centroids, rbf.particlesGroup)
#     assert isinstance(fwdir.beams.centroids.particles[0], centroidsobject)
#     assert isinstance(fwdir.beams.emittance, rbf.particlesGroup)
#     assert isinstance(fwdir.beams.emittance.particles[0], emittanceobject)
#     assert isinstance(fwdir.beams.twiss, rbf.particlesGroup)
#     assert isinstance(fwdir.beams.twiss.particles[0], twissobject)
#     assert isinstance(fwdir.beams.slice, rbf.particlesGroup)
#     assert isinstance(fwdir.beams.slice.particles[0], sliceobject)
#     assert isinstance(fwdir.beams.data, rbf.particlesGroup)
#     assert isinstance(fwdir.beams.data.particles[0], Particles)
#     scrnames = [e.objectname for e in lattice.screens_and_markers_and_bpms]
#     for name, file in fwdir.beams.getScreens().items():
#         assert name in scrnames
#         assert os.path.isfile(file)
#     shutil.rmtree(f"./fodo/{code}")
#     try:
#         shutil.rmtree(f"../fodo/{code}")
#     except FileNotFoundError:
#         pass
#
# @pytest.mark.parametrize("code,lattice_class", [
#     ("astra", astraLattice),
#     ("ocelot", ocelotLattice),
#     ("gpt", gptLattice),
#     ("csrtrack", csrtrackLattice),
#     ("cheetah", cheetahLattice)
# ])
# def test_track(simple_beam, test_fodo_elements, code, lattice_class):
#     lattice = prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=False)
#     if code == "ocelot":
#         lattice.mbi.update({"set_mbi": True})
#     lattice.preProcess()
#     lattice.write()
#     lattice.writeElements()
#     if code != "gpt":
#         lattice.run()
#         lattice.postProcess()
#     shutil.rmtree(f"./fodo/{code}")
#     try:
#         shutil.rmtree(f"../fodo/{code}")
#     except FileNotFoundError:
#         pass
#
# # @pytest.mark.parametrize("code,lattice_class", [
# #     ("elegant", elegantLattice),
# #     # ("gpt", gptLattice),
# # ])
# # def test_run_setup_errors(simple_beam, test_fodo_elements, code, lattice_class):
# #     lattice = prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=False)
# #     errors = {
# #         "elements": {
# #             "QUAD*": {"k1": [0.1]}
# #         }
# #     }
# #     lattice.runSettings.loadElementErrors(errors)
# #     lattice.preProcess()
# #     lattice.write()
# #     lattice.run()
# #     lattice.postProcess()
# #     shutil.rmtree(f"./fodo/{code}")
#
# # @pytest.mark.parametrize("code,lattice_class", [
# #     ("elegant", elegantLattice),
# #     # ("gpt", gptLattice),
# # ])
# # def test_run_setup_scan(simple_beam, test_fodo_elements, code, lattice_class):
# #     lattice = prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=False)
# #     lattice.runSettings.setElementScan(
# #         name="QUAD1",
# #         item="k1",
# #         scanrange=(1, 3)
# #     )
# #     lattice.runSettings.setNRuns(nruns=3)
# #     lattice.preProcess()
# #     lattice.write()
# #     lattice.run()
# #     lattice.postProcess()
# #     shutil.rmtree(f"./fodo/{code}")
# #     try:
# #         shutil.rmtree(f"../fodo/{code}")
# #     except FileNotFoundError:
# #         pass