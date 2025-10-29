"""
Simframe Wake-T Module

Various objects and functions to handle Wake-T lattices and commands. See `Wake-T github`_ for more details.

    .. _Wake-T github: https://github.com/AngelFP/Wake-T

Classes:
    - :class:`~SimulationFramework.Codes.Wake_T.Wake_T.waketLattice`: The Wake-T lattice object, used for
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject` s defined in the
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a Wake-T lattice object,
    and for tracking through it.

"""

from ...Framework_objects import (
    frameworkLattice,
    elementkeywords,
)
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
    """
        Class for defining the Wake-T lattice object, used for
        converting the :class:`~SimulationFramework.Framework_elements.frameworkObject`s defined in the
        :class:`~SimulationFramework.Framework_elements.frameworkLattice` into an Wake-T lattice object,
        and for tracking through it.
        """

    code: str = "waket"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    allow_negative_drifts: bool = True
    """Allow drifts to be of negative length (could be necessary for plasma injection)"""

    beamline: Any = None
    """Wake-T `Beamline`_ object
    
    .. _Beamline: https://github.com/AngelFP/Wake-T/blob/dev/wake_t/beamline_elements/beamline.py"""

    pin: Any = None
    """Wake-T `ParticleBunch`_ object

        .. _ParticleBunch: https://github.com/AngelFP/Wake-T/blob/dev/wake_t/particles/particle_bunch.py"""

    bunch_list: List[Any] | None = None
    """List of Wake-T `ParticleBunch`_ object produced by tracking"""

    particle_definition: str = None
    """Name of first object in lattice"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if (
            "input" in self.file_block
            and "particle_definition" in self.file_block["input"]
        ):
            if (
                self.file_block["input"]["particle_definition"]
                == "initial_distribution"
            ):
                self.particle_definition = "laser"
            else:
                self.particle_definition = self.file_block["input"][
                    "particle_definition"
                ]
        else:
            self.particle_definition = self.start

    def write(self) -> None:
        """
        Create the beamline object via :func:`~writeElements`;
        not that Wake-T appears not to support the writing of a lattice to a file.
        """
        self.writeElements()
        # if self.verbose:
        #     warn("Wake-T does not support writing of a lattice to a file.")

    def writeElements(self) -> None:
        """
        Create Wake-T objects for all the elements in the lattice and set the
        :attr:`~SimulationFramework.Codes.Wake_T.Wake_T.waketLattice.beamline`.
        """
        self.beamline = self.section.to_wake_t()

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
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
        Convert the initial HDF5 particle distribution to Wake-T format and set
        :attr:`~pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        self.read_input_file(prefix, self.particle_definition)
        self.global_parameters["beam"].beam.rematchXPlane(**self.initial_twiss["horizontal"])
        self.global_parameters["beam"].beam.rematchYPlane(**self.initial_twiss["vertical"])
        self.pin = beam_to_particle_bunch(
            self.global_parameters["beam"],
            zstart=mean(self.global_parameters["beam"].z.val),
        )

    def run(self) -> None:
        """
        Run the code, and set :attr:`~bunch_list`
        """
        pin = deepcopy(self.pin)
        self.bunch_list = self.beamline.track(
            pin,
            # opmd_diag=True,
            # diag_dir=self.global_parameters["master_subdir"],
            show_progress_bar=False,
        )

    def postProcess(self) -> None:
        """
        Convert the outputs from Wake-T to a `beam` object and save them to `master_subdir`.
        """
        super().postProcess()
        outbeamname = f'{self.global_parameters["master_subdir"]}/{self.end}.openpmd.hdf5'
        particle_bunch_to_beam(
            self.global_parameters["beam"],
            self.bunch_list[-1],
            zpos=self.endObject.physical.end.z,
        )
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            outbeamname,
        )