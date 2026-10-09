"""
SIMBA Wake-T module: builds and tracks a Wake-T beamline. See `Wake-T github`_.

    .. _Wake-T github: https://github.com/AngelFP/Wake-T
"""

from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules.Beams.wake_t import (
    particle_bunch_to_beam,
    beam_to_particle_bunch,
)
from copy import deepcopy
from typing import List, Any
from numpy import mean

def all_subclasses(cls):
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses


class waketLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` built and tracked as a Wake-T beamline."""

    code: str = "waket"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    allow_negative_drifts: bool = True
    """Allow drifts to be of negative length (could be necessary for plasma injection)"""

    beamline: Any = None
    """Wake-T `Beamline`_

    .. _Beamline: https://github.com/AngelFP/Wake-T/blob/dev/wake_t/beamline_elements/beamline.py"""

    pin: Any = None
    """Input Wake-T `ParticleBunch`_

    .. _ParticleBunch: https://github.com/AngelFP/Wake-T/blob/dev/wake_t/particles/particle_bunch.py"""

    bunch_list: List[Any] | None = None
    """Wake-T `ParticleBunch`_ objects produced by tracking"""

    particle_definition: str = None
    """Name of the input particle distribution"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.input_particle_definition

    def write(self) -> None:
        """Build the beamline via :meth:`writeElements`; Wake-T cannot write a lattice file."""
        self.writeElements()

    def writeElements(self) -> None:
        """Build :attr:`beamline` from the section."""
        self.beamline = self.section.to_wake_t()

    def preProcess(self) -> None:
        """Load the input beam from ``file_block['input']['prefix']`` and set :attr:`pin`."""
        super().preProcess()
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.hdf5_to_particle_bunch(prefix)

    def hdf5_to_particle_bunch(self, prefix="", write=True) -> None:
        """
        Load the input beam and convert it to a Wake-T bunch in :attr:`pin`.

        Parameters
        ----------
        prefix: str
            Prefix for the input beam file
        write: bool
            Unused
        """
        self.load_input_beam(prefix, self.particle_definition)
        self.pin = beam_to_particle_bunch(
            self.global_parameters["beam"],
            zstart=mean(self.global_parameters["beam"].z.val),
        )

    def run(self) -> None:
        """Track :attr:`pin` and set :attr:`bunch_list`."""
        pin = deepcopy(self.pin)
        self.bunch_list = self.beamline.track(
            pin,
            show_progress_bar=False,
        )

    def postProcess(self) -> None:
        """Write the final bunch as openPMD to `master_subdir`."""
        super().postProcess()
        outbeamname = (
            f'{self.global_parameters["master_subdir"]}/'
            f'{self.output_basename(self.end)}.openpmd.hdf5'
        )
        particle_bunch_to_beam(
            self.global_parameters["beam"],
            self.bunch_list[-1],
            zpos=self.endObject.physical.end.z,
        )
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            outbeamname,
        )