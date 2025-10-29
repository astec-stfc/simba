import sys
from argparse import Namespace

sys.path.append("../../")
from SimulationFramework.Framework import load_directory  # noqa E402
from src.FEBE_Simple_NM import FEBE_Mode_1
from SimulationFramework.Modules import constants  # noqa E402


if __name__ == "__main__":

    # opt = FEBE_Mode_1(argparse=Namespace(sample=1, charge=250, subdir="1"), charge=250)
    # opt.base_files = (
    #     "../../Basefiles/Base_" + str(250) + "pC/"
    # )  # This is where to look for the input files (in this case CLA-S02-APER-01.hdf5)
    # opt.deleteFolders = False
    # opt.sample_interval = 2 ** (3 * 2)
    # opt.set_start_file("FEBE")
    # opt.verbose = True
    # opt.Example(dir="Setups/Setup_" + str(1) + "_" + str(250) + "pC/")
    # print(opt.framework["FEBE"].global_parameters["beam"].twiss.normal)

    fwdir = load_directory(directory='Setups/Setup_1_250pC/', beams=False, rest_mass=constants.m_e)
    plt1, fig1, ax1 = fwdir.plot(include_layout=True, include_particles=False, ykeys=['sigma_x', 'sigma_y'], ykeys2=['sigma_z'])
    plt2, fig2, ax2 = fwdir.plot(include_layout=True, include_particles=False, ykeys=['mean_cp'], ykeys2=['sigma_cp'])
    plt1.show()
    plt2.show()
