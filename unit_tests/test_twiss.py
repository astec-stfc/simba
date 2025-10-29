import SimulationFramework.Modules.Beams as rbf  # noqa E402
import SimulationFramework.Modules.Twiss as rtf
from SimulationFramework.Framework_lattices import (
    elegantLattice,
    astraLattice,
    ocelotLattice,
    gptLattice,
    cheetahLattice,
)
from test_beam import simple_beam
from test_track import prepare_lattice, test_fodo_elements
import pytest
import shutil

@pytest.mark.parametrize("code,lattice_class", [
    ("elegant", elegantLattice),
    ("astra", astraLattice),
    ("ocelot", ocelotLattice),
    # ("gpt", gptLattice),
    ("cheetah", cheetahLattice),
])
def test_twiss(simple_beam, test_fodo_elements, code, lattice_class):
    lattice = prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=False)
    lattice.preProcess()
    lattice.write()
    lattice.run()
    lattice.postProcess()
    twsobj = rtf.twiss()
    twsobj.load_directory(lattice.global_parameters["master_subdir"])
    rtf.plot.plot(twsobj)
    shutil.rmtree(f'./fodo/{code}')