import os
import sys

sys.path.append("c:/Users/jkj62/Documents/GitHub/simframe")
from SimulationFramework.Framework import load_directory  # noqa E402
import SimulationFramework.Framework as fw  # noqa E402
from SimulationFramework.Modules import constants  # noqa E402


class ISIS:

    def __init__(self, dir: str = "GPT"):
        super().__init__()
        self.framework = fw.Framework(
            dir, verbose=True, clean=False, master_lattice="./input/"
        )
        self.settings_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "ISIS_LEBT.def")
        )
        self.framework.loadSettings(self.settings_file)
        # for elem in self.framework.elements:
        #     self.framework[elem].centre[2] += 1.142


def export_HDF5_Field_Files():
    isis = ISIS("output/SC/GPT")
    for elemstr in isis.framework.elements:
        elem = isis.framework.getElement(elemstr)
        elem.update_field_definition()
        if hasattr(elem, "field_definition") and elem.field_definition is not None:
            elem.field_definition.write_field_file(code="hdf5")


if __name__ == "__main__":

    # export_HDF5_Field_Files()
    # exit()

    isis = ISIS("output/SC/GPT")
    isis.framework["LEBT"].screen_step_size = 0.001
    isis.framework.track(startfile="LEBT", endfile="LEBT")
    isis.framework.change_Lattice_Code("LEBT", "ASTRA")
    isis.framework.change_subdirectory("output/SC/ASTRA")
    isis.framework.track(startfile="LEBT", endfile="LEBT")

    # #  # Run without SC
    isis = ISIS("output/NoSC/GPT")
    isis.framework["LEBT"].space_charge_mode = False
    isis.framework["LEBT"].screen_step_size = 0.001
    isis.framework.track()
    isis.framework.change_Lattice_Code("LEBT", "ASTRA")
    isis.framework.change_subdirectory("output/NoSC/ASTRA")
    isis.framework["LEBT"].space_charge_mode = False
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
