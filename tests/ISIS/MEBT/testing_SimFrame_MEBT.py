import sys
import os

sys.path.append("c:/Users/jkj62/Documents/GitHub/simframe")
from SimulationFramework.Framework import load_directory
import SimulationFramework.Framework as fw
from SimulationFramework.Modules import constants


class ISIS:

    def __init__(self, dir: str = "ASTRA"):
        super().__init__()
        self.framework = fw.Framework(
            dir, verbose=True, clean=False, master_lattice="./input/"
        )
        self.settings_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "ISIS_MEBT.def")
        )
        self.framework.loadSettings(self.settings_file)


def export_HDF5_Field_Files():
    isis = ISIS("output/SC/ASTRA")
    for elemstr in isis.framework.elements:
        elem = isis.framework.getElement(elemstr)
        elem.update_field_definition()
        if hasattr(elem, "field_definition") and elem.field_definition is not None:
            elem.field_definition.write_field_file(code="hdf5")


if __name__ == "__main__":

    # export_HDF5_Field_Files()
    # exit()

    isis = ISIS("output/SC/ASTRA")
    isis.framework.track(endfile="MEBT")
    isis.framework.change_Lattice_Code("MEBT", "GPT")
    isis.framework["MEBT"].screen_step_size = 0.001
    isis.framework.change_subdirectory("output/SC/GPT")
    isis.framework.track(endfile="MEBT")

    # # Run without SC
    isis = ISIS("output/NoSC/ASTRA")
    isis.framework["MEBT"].space_charge_mode = False
    isis.framework.track()
    isis.framework.change_Lattice_Code("MEBT", "GPT")
    isis.framework["MEBT"].space_charge_mode = False
    isis.framework["MEBT"].screen_step_size = 0.001
    isis.framework.change_subdirectory("output/NoSC/GPT")
    isis.framework.track()

    fwdir = load_directory(
        "output/SC/ASTRA",
        beams=False,
        rest_mass=constants.m_p,
        master_lattice="./input/",
    )
    plt1, fig1, ax1 = fwdir.plot(
        include_layout=True,
        include_particles=False,
        ykeys=["sigma_x", "sigma_y"],
        ykeys2=["sigma_z"],
    )
    plt2, fig2, ax2 = fwdir.plot(
        include_layout=True,
        include_particles=False,
        ykeys=["mean_cp"],
        ykeys2=["sigma_cp"],
    )
    plt1.show()
    plt2.show()
