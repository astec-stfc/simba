# import re
# import sys
# from scipy.optimize import curve_fit
# import numpy as np
# import shutil
#
# # sys.path.append(r"C:\Users\jkj62\Documents\GitHub\SimFrame")
# import SimulationFramework.Framework as fw  # noqa E402
# from SimulationFramework.Framework import load_directory  # noqa E402
# import SimulationFramework.Modules.Beams as rbf  # noqa E402
# import SimulationFramework.Modules.Twiss as rtf  # noqa E402
#
#
# def sub_element_test(startfile="generator", endfile="S02", scaling=3, sampling=1):
#     framework = fw.Framework(
#         directory="./subelement_test",
#         clean=False,
#         verbose=True,
#         generator_defaults="clara.yaml",
#     )
#     framework.loadSettings("Lattices/clara400_v13.def")
#     framework.change_Lattice_Code("All", "ASTRA", exclude=[])
#     framework.change_Lattice_Code("VBC", "Elegant")
#     framework.change_generator("ASTRA")
#     framework.generator.load_defaults("clara_400_2ps_Gaussian")
#     print("###########   Performing sub_element_test   ###########")
#
#
# def astra_track(startfile="generator", endfile="S02", scaling=3, sampling=1):
#     # Define a new framework instance, in directory 'example_ASTRA'.
#     #       "clean" will empty (delete everything!) the directory if true
#     #       "verbose" will print a progressbar if true
#     framework = fw.Framework(
#         directory="./example_ASTRA",
#         clean=False,
#         verbose=True,
#         generator_defaults="clara.yaml",
#     )
#     # Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
#     framework.loadSettings("Lattices/clara400_v13.def")
#     # Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector can not be done in Elegant)
#     framework.change_Lattice_Code("All", "ASTRA", exclude=[])
#     # Again, but put the VBC in Elegant for CSR
#     framework.change_Lattice_Code("VBC", "Elegant")
#     # This is the code that generates the laser distribution (ASTRA or GPT)
#     framework.change_generator("ASTRA")
#     # Load a starting laser distribution setting
#     framework.generator.load_defaults("clara_400_2ps_Gaussian")
#     # Set the thermal emittance for the generator
#     # framework.generator.thermal_emittance = 0.0005
#     # This is a scaling parameter
#     # This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
#     if startfile == "generator":
#         framework.generator.number_of_particles = 2 ** (3 * scaling)
#     else:
#         framework[startfile].sample_interval = 2 ** (3 * sampling)
#
#     framework["bunch_compressor"].set_angle(0.12)
#     # framework['CLA-L02-CAV'].crest = 100
#     # framework["L02"].headers["newrun"].auto_phase = True
#     # Track the whole lattice
#     print("###########   Performing ASTRA Track   ###########")
#     framework.track(startfile=startfile, endfile=endfile)
#
#
# def astra_csrtrack_track(scaling=3, sampling=1):
#     # This time we will use CSRTrack for the VBC
#     framework = fw.Framework(
#         directory="./example_ASTRA_CSRTrack",
#         clean=False,
#         verbose=True,
#         generator_defaults="clara.yaml",
#     )
#     framework.loadSettings("Lattices/clara400_v13.def")
#     framework.generator.number_of_particles = 2 ** (3 * scaling)
#     framework.change_Lattice_Code("All", "ASTRA", exclude=["injector400", "VBC"])
#     framework.change_Lattice_Code("VBC", "csrtrack")
#     # We want to start from the VBC so we don't run the injector again.
#     # Here we tell SimFrame where to look for the starting files (the "prefix" parameter) for the lattice we want to run
#     framework.set_lattice_prefix("VBC", "../example_ASTRA/")
#     # Here we ony run from the VBC
#     print("###########   Performing ASTRA + CSRTrack Track   ###########")
#     framework["bunch_compressor"].set_angle(0.12)
#     framework.track(startfile='VBC',endfile='S07')
#
# def elegant_track(scaling=3, sampling=1):
#     # This time we will use Elegant for everything except the injector
#     framework = fw.Framework(
#         directory="./example_Elegant",
#         clean=False,
#         verbose=True,
#         generator_defaults="clara.yaml",
#     )
#     framework.loadSettings("Lattices/clara400_v13.def")
#     framework.generator.number_of_particles = 2 ** (3 * scaling)
#     framework.change_Lattice_Code("All", "elegant", exclude=["injector400"])
#     # Set the prefix for S02 (where we will start)
#     framework.set_lattice_prefix("S02", "../example_ASTRA/")
#     # Run from S02 onwards
#     print("###########   Performing Elegant Track   ###########")
#     framework["bunch_compressor"].set_angle(0.12)
#     framework.track(startfile="S02", endfile="S07")
#
# def ocelot_track(scaling=3, sampling=1):
#     # This time we will use OCELOT for everything except the injector
#     framework = fw.Framework(
#         directory="./example_Ocelot",
#         clean=False,
#         verbose=True,
#         generator_defaults="clara.yaml",
#     )
#     framework.loadSettings("Lattices/clara400_v13.def")
#     framework.generator.number_of_particles = 2 ** (3 * scaling)
#     framework.change_Lattice_Code("All", "ocelot", exclude=["injector400"])
#     # Set the prefix for S02 (where we will start)
#     framework.set_lattice_prefix("S02", "../example_ASTRA/")
#     # Run from S02 onwards
#     print("###########   Performing Ocelot Track   ###########")
#     framework["bunch_compressor"].set_angle(0.12)
#     framework.track(startfile="S02", endfile="S07")
#
# def opal_track(scaling=3, sampling=1):
#     # This time we will use OCELOT for everything except the injector
#     framework = fw.Framework(
#         directory="./example_Opal",
#         clean=True,
#         verbose=True,
#         generator_defaults="clara.yaml",
#     )
#     framework.loadSettings("Lattices/clara400_v13.def")
#     framework.generator.number_of_particles = 2 ** (3 * scaling)
#     framework.change_Lattice_Code("All", "opal", exclude=["injector400"])
#     # Set the prefix for S02 (where we will start)
#     framework.set_lattice_prefix("S02", "../example_ASTRA/")
#     # Run from S02 onwards
#     print("###########   Performing Opal Track   ###########")
#     framework["bunch_compressor"].set_angle(0.12)
#     framework.track(startfile="S02", endfile="S07")
#
# def cheetah_track(scaling=3, sampling=1):
#     # This time we will use OCELOT for everything except the injector
#     framework = fw.Framework(directory="./example_Cheetah", clean=False, verbose=True)
#     framework.loadSettings("Lattices/clara400_v13.def")
#     framework.generator.number_of_particles = 2 ** (3 * scaling)
#     framework.change_Lattice_Code("All", "cheetah", exclude=["injector400"])
#     # Set the prefix for S02 (where we will start)
#     framework.set_lattice_prefix("S02", "../example_ASTRA/")
#     # Run from S02 onwards
#     print("###########   Performing Cheetah Track   ###########")
#     framework.track(startfile="S02", endfile="S07")
#
# def xsuite_track(scaling=3, sampling=1):
#     # This time we will use OCELOT for everything except the injector
#     framework = fw.Framework(directory="./example_Xsuite", clean=True, verbose=True)
#     framework.loadSettings("Lattices/clara400_v13_combined.def")
#     framework.generator.number_of_particles = 2 ** (3 * scaling)
#     framework.change_Lattice_Code("All", "xsuite", exclude=["injector400"])
#     # Set the prefix for S02 (where we will start)
#     framework.set_lattice_prefix("Linac", "../example_ASTRA/")
#     # Run from S02 onwards
#     print("###########   Performing Xsuite Track   ###########")
#     framework["bunch_compressor"].set_angle(0.12)
#     framework.track(startfile="Linac", endfile="Linac")
#
#
# sub_element_test()
# astra_track(endfile="S07")
# astra_csrtrack_track()
# elegant_track()
# ocelot_track()
# cheetah_track()
# xsuite_track()
# shutil.rmtree("./subelement_test")
# shutil.rmtree("./example_ASTRA_CSRTrack")
# shutil.rmtree("./example_Elegant")
# shutil.rmtree("./example_ASTRA")
# shutil.rmtree("./example_Ocelot")
# shutil.rmtree("./example_Cheetah")
# shutil.rmtree("./example_Xsuite")
# # exit()
# #
# # def set_crests(framework, crests):
# #     for cavity, crestdict in crests.items():
# #         if crestdict["crested"]:
# #             framework[cavity].crest = crestdict["phase"]
# #         # print(cavity, "after", framework[cavity].crest)
# #
# #
# # def gpt_track(startfile="generator", endfile="S07", scaling=3, sampling=1, crests={}):
# #     # Unless you have GPT installed, don't run this.
# #     framework = fw.Framework("example_GPT", clean=False, verbose=True)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     framework.change_generator("GPT")
# #     framework.generator.load_defaults("clara_400_2ps_Gaussian")
# #     framework.generator.number_of_particles = 2 ** (3 * scaling)
# #     framework.change_Lattice_Code("VBC", "elegant")
# #
# #     framework["injector400"].space_charge_mode = None
# #     set_crests(framework, crests)
# #
# #     print("###########   Performing GPT Track   ###########")
# #     if startfile == "generator":
# #         framework.generator.number_of_particles = 2 ** (3 * scaling)
# #     else:
# #         framework[startfile].sample_interval = 2 ** (3 * sampling)
# #     framework.track(startfile=startfile, endfile=endfile)
# #
# #
# # def gpt_crest_Gun(phase, crests):
# #     print("###########   Gun Phase = ", phase, "   ###########")
# #     framework = fw.Framework("example_GPT", clean=True, verbose=False)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.generator.number_of_particles = 512
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     framework["injector400"].space_charge_mode = None
# #     set_crests(framework, crests)
# #     framework["CLA-HRG1-GUN-CAV-01"].phase = 0
# #     framework["CLA-HRG1-GUN-CAV-01"].crest = phase
# #     framework["CLA-L01-LIN-CAV-01"].field_amplitude = 0
# #     framework.track(endfile="injector400")
# #     twiss = rtf.twiss()
# #     twiss.read_GPT_twiss_files(framework.subdirectory + "/injector400_emit.gdf")
# #     twiss.sort()
# #     print(phase, float(max(twiss.cp_eV.val) / 1e6))
# #     return float(max(twiss.cp_eV.val) / 1e6)
# #
# #
# # def gpt_crest_L01(phase, crests, amp=None):
# #     print("###########   L01 Phase = ", phase, "   ###########")
# #     framework = fw.Framework("example_GPT", clean=False, verbose=False)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     framework["injector400"].space_charge_mode = None
# #     set_crests(framework, crests)
# #     if amp is not None:
# #         framework["CLA-L01-LIN-CAV-01"].field_amplitude = amp
# #     framework["CLA-L01-LIN-CAV-01"].phase = 0
# #     framework["CLA-L01-LIN-CAV-01"].crest = phase
# #     framework["CLA-L01-LIN-CAV-01"].wakefield_definition = None
# #     framework.track(endfile="injector400")
# #     twiss = rtf.twiss()
# #     twiss.read_GPT_twiss_files(framework.subdirectory + "/injector400_emit.gdf")
# #     twiss.sort()
# #     print(phase, float(max(twiss.cp_eV.val) / 1e6))
# #     return float(max(twiss.cp_eV.val) / 1e6)
# #
# #
# # def gpt_crest_L02(phase, crests, amp=None):
# #     print("###########   L02 Phase = ", phase, "   ###########")
# #     framework = fw.Framework("example_GPT", clean=False, verbose=False)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     [setattr(framework[latt], "space_charge_mode", None) for latt in framework.lattices]
# #     set_crests(framework, crests)
# #     if amp is not None:
# #         framework["CLA-L02-LIN-CAV-01"].field_amplitude = amp
# #     framework["CLA-L02-LIN-CAV-01"].phase = 0
# #     framework["CLA-L02-LIN-CAV-01"].crest = phase
# #     framework["CLA-L02-LIN-CAV-01"].wakefield_definition = None
# #     framework.track(startfile="S02", endfile="L02")
# #     twiss = rtf.twiss()
# #     twiss.read_GPT_twiss_files(framework.subdirectory + "/L02_emit.gdf")
# #     twiss.sort()
# #     print(phase, float(max(twiss.cp_eV.val) / 1e6))
# #     return float(max(twiss.cp_eV.val) / 1e6)
# #
# #
# # def gpt_crest_L03(phase, crests, amp=None):
# #     print("###########   L03 Phase = ", phase, "   ###########")
# #     framework = fw.Framework("example_GPT", clean=False, verbose=False)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     [setattr(framework[latt], "space_charge_mode", None) for latt in framework.lattices]
# #     set_crests(framework, crests)
# #     if amp is not None:
# #         framework["CLA-L03-LIN-CAV-01"].field_amplitude = amp
# #     framework["CLA-L03-LIN-CAV-01"].phase = 0
# #     framework["CLA-L03-LIN-CAV-01"].crest = phase
# #     framework["CLA-L03-LIN-CAV-01"].wakefield_definition = None
# #     framework.track(startfile="S03", endfile="L03")
# #     twiss = rtf.twiss()
# #     twiss.read_GPT_twiss_files(framework.subdirectory + "/L03_emit.gdf")
# #     twiss.sort()
# #     print(phase, float(max(twiss.cp_eV.val) / 1e6))
# #     return float(max(twiss.cp_eV.val) / 1e6)
# #
# #
# # def gpt_crest_L4H(phase, crests, amp=None):
# #     print("###########   L4H Phase = ", phase, "   ###########")
# #     framework = fw.Framework("example_GPT", clean=False, verbose=False)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     [setattr(framework[latt], "space_charge_mode", None) for latt in framework.lattices]
# #     set_crests(framework, crests)
# #     if amp is not None:
# #         framework["CLA-L4H-LIN-CAV-01"].field_amplitude = amp
# #     framework["CLA-L4H-LIN-CAV-01"].phase = 0
# #     framework["CLA-L4H-LIN-CAV-01"].crest = phase
# #     framework["CLA-L4H-LIN-CAV-01"].wakefield_definition = None
# #     framework.track(startfile="S04", endfile="L4H")
# #     twiss = rtf.twiss()
# #     twiss.read_GPT_twiss_files(framework.subdirectory + "/L4H_emit.gdf")
# #     twiss.sort()
# #     print(phase, float(max(twiss.cp_eV.val) / 1e6))
# #     return float(max(twiss.cp_eV.val) / 1e6)
# #
# #
# # def gpt_crest_L04(phase, crests, amp=10e6):
# #     print("###########   L04 Phase = ", phase, "   ###########")
# #     framework = fw.Framework("example_GPT", clean=False, verbose=False)
# #     framework.loadSettings("Lattices/clara400_v13.def")
# #     framework.change_Lattice_Code("All", "GPT", exclude=[])
# #     [setattr(framework[latt], "space_charge_mode", None) for latt in framework.lattices]
# #     set_crests(framework, crests)
# #     if amp is not None:
# #         framework["CLA-L04-LIN-CAV-01"].field_amplitude = amp
# #     framework["CLA-L04-LIN-CAV-01"].phase = 0
# #     framework["CLA-L04-LIN-CAV-01"].crest = phase
# #     framework["CLA-L04-LIN-CAV-01"].wakefield_definition = None
# #     framework.track(startfile="S05", endfile="L04")
# #     twiss = rtf.twiss()
# #     twiss.read_GPT_twiss_files(framework.subdirectory + "/L04_emit.gdf")
# #     twiss.sort()
# #     print(phase, float(max(twiss.cp_eV.val) / 1e6))
# #     return float(max(twiss.cp_eV.val) / 1e6)
# #
# #
# # def fitting_equation_Fine(x, a, b, crest):
# #     return a + b * np.cos((crest - x) * np.pi / 180)
# #
# #
# # def fit_crest(data, approxcrest):
# #     x, y = zip(*data)
# #     popt, pcov = curve_fit(
# #         fitting_equation_Fine,
# #         x,
# #         y,
# #         p0=[1, max(y), approxcrest],
# #         bounds=[[-np.inf, 0, -180], [np.inf, np.inf, 360]],
# #     )
# #     return popt, pcov
# #
# #
# # def elegant_track(startfile="S02", endfile="S07_SP3"):
# #     framework = fw.Framework("example_elegant_sp3", clean=False, verbose=True)
# #     framework.loadSettings("Lattices/clara400_v13_SP3.def")
# #     framework.change_Lattice_Code(
# #         "All", "elegant", exclude=["generator", "injector400"]
# #     )
# #
# #     print("###########   Performing Elegant Track   ###########")
# #     # framework.set_lattice_prefix('S02', '../FEBE/StandardSetups/example_GPT_256k/')
# #     framework.set_lattice_prefix("S02", "../example_ASTRA/")
# #     framework.track(startfile=startfile, endfile=endfile)
# #
# #
# # def extract_ASTRA_phases(logfile, substitutions={}):
# #     with open(logfile, "r") as file:
# #         # Read the contents of the file
# #         content = file.read()
# #
# #         # Extract all the numbers from the content
# #         numbers = re.findall(r"(\d)\s+([\d\.]+)\s+([\d\.]+)", content)
# #
# #         # Convert the extracted numbers to int and sum them
# #         def sub(label):
# #             if label in substitutions:
# #                 return substitutions[label]
# #             else:
# #                 return label
# #
# #         result = {sub(a): 360 - float(b) - 90 for a, _, b in numbers[1:]}
# #
# #         return result
# #
# #
# # cavity_substitutions = {
# #     "injector400": {
# #         "1": "CLA-HRG1-GUN-CAV-01",
# #         "2": "CLA-L01-LIN-CAV-01",
# #     }
# # }
# #
# # RFcrests = {
# #     "CLA-HRG1-GUN-CAV-01": {
# #         "phase": 55.89917705,
# #         "method": gpt_crest_Gun,
# #         "crested": True,
# #         "amplitude": 50e6,
# #     },
# #     "CLA-L01-LIN-CAV-01": {
# #         "phase": 244.32580792776665,
# #         "method": gpt_crest_L01,
# #         "crested": True,
# #         "amplitude": 5e6,
# #     },
# #     "CLA-L02-LIN-CAV-01": {
# #         "phase": 94.23276465116447,
# #         "method": gpt_crest_L02,
# #         "crested": True,
# #         "amplitude": 10e6,
# #     },
# #     "CLA-L03-LIN-CAV-01": {
# #         "phase": 121.57361517000028,
# #         "method": gpt_crest_L03,
# #         "crested": True,
# #         "amplitude": 20e6,
# #     },
# #     "CLA-L4H-LIN-CAV-01": {
# #         "phase": 163.29534223881961,
# #         "method": gpt_crest_L4H,
# #         "crested": True,
# #         "amplitude": 25e6,
# #     },
# #     "CLA-L04-LIN-CAV-01": {
# #         "phase": 180,
# #         "method": gpt_crest_L04,
# #         "crested": False,
# #         "amplitude": 25e6,
# #     },
# # }
# #
# # if __name__ == "__main__":
# #     global scaling
# #     # master_lattice = '../MasterLattice/MasterLattice'
# #     scaling = 6
# #     sampling = 3
# #     # astra_track(startfile='S02', endfile='L02', scaling=scaling, sampling=sampling)
# #     # fwdir = load_directory('example_ASTRA', beams=False)
# #     # plt1, fig1, ax1 = fwdir.plot(include_layout=True, include_particles=False, ykeys=['sigma_x', 'sigma_y'], ykeys2=['sigma_z'])
# #     # plt1.show()
# #     # injectorcrests = extract_ASTRA_phases('example_ASTRA/injector400.log', cavity_substitutions['injector400'])
# #     # gpt_track(startfile='injector400', endfile='L02', scaling=scaling, sampling=sampling, crests=injectorcrests)
# #     # elegant_track(startfile="S02", endfile="L02")
# #     # for cav, cavdict in list(RFcrests.items())[:]:
# #     #     if not cavdict['crested']:
# #     #         guess = int(RFcrests[cav]['phase'])
# #     #         crestdata = []
# #     #         for c in range(guess-30, guess+30+10, 10):
# #     #             crestdata.append([c, cavdict['method'](c, RFcrests, RFcrests[cav]['amplitude'])])
# #     #         xpoints, ypoints = zip(*crestdata)
# #     #         plt.plot(xpoints, ypoints)
# #     #         plt.show()
# #     #         popt, pcov = fit_crest(crestdata, guess)
# #     #         RFcrests[cav]['phase'] = popt[2]
# #     #     print(RFcrests[cav]['method'](RFcrests[cav]['phase'], RFcrests))
# #     #     print(RFcrests)
