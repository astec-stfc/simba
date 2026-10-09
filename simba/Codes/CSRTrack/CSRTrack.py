"""
SIMBA CSRTrack module: writes CSRTrack input files and converts their output. See `CSRTrack manual`_.

    .. _CSRTrack manual: https://www.desy.de/xfel-beam/csrtrack/files/CSRtrack_User_Guide_(actual).pdf
"""

from pydantic import Field
from ...Framework_objects import frameworkLattice
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from typing import Dict, List, Any
from laura.translator.converters.codes.csrtrack import (
    CsrTrackParticles,
    CsrTrackForces,
    CsrTrackTrackStep,
    CsrTrackTracker,
    CsrTrackMonitor,
)


class csrtrackLattice(frameworkLattice):
    """A :class:`~simba.Framework_objects.frameworkLattice` written as a CSRTrack input file."""

    code: str = "csrtrack"
    """String indicating the lattice object type"""

    particle_definition: str = ""
    """String representing the initial particle distribution"""

    CSRTrackelementObjects: Dict = {}
    """Dictionary representing all CSRTrack object namelists"""

    csrtrack_headers: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.set_particles_filename()

    def set_particles_filename(self) -> None:
        """Set the ``particles`` header to read ``<start>.astra``."""
        self.particle_definition = self.input_particle_definition
        self.csrtrack_headers["particles"] = CsrTrackParticles(
            particle_definition=self.start,
            global_parameters=self.global_parameters,
            format="astra",
        )
        self.csrtrack_headers["particles"].array = (
            "#file{\nname="
            + self.start
            + ".astra"
            + "\n}"
        )

    @property
    def dipoles_screens_and_bpms(self) -> List:
        """
        Dipoles, screens and BPMs sorted by end position.

        Returns
        -------
        List
        """
        return sorted(
            self.getElementType("dipole")
            + self.getElementType("screen")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.position_end[2],
        )

    def setCSRMode(self) -> None:
        """Set the ``forces`` header from ``csr: csr_mode``: "3D" is ``csr_g_to_p``, "1D" is ``projected``."""
        if "csr" in self.file_block and "csr_mode" in self.file_block["csr"]:
            if self.file_block["csr"]["csr_mode"] == "3D":
                self.csrtrack_headers["forces"] = CsrTrackForces(type="csr_g_to_p")
            elif self.file_block["csr"]["csr_mode"] == "1D":
                self.csrtrack_headers["forces"] = CsrTrackForces(type="projected")
        else:
            self.csrtrack_headers["forces"] = CsrTrackForces()

    def writeElements(self) -> str:
        """
        Build the CSRTrack headers and render the section as CSRTrack input.

        Returns
        -------
        str
            CSRTrack input file text
        """
        self.set_particles_filename()
        self.setCSRMode()
        self.csrtrack_headers["track_step"] = CsrTrackTrackStep()
        self.csrtrack_headers["tracker"] = CsrTrackTracker(
            end_time_marker="screen"
            + str(len(self.screens))
            + "a"
        )
        self.csrtrack_headers["monitor"] = CsrTrackMonitor(
            name=self.end + ".fmt2", global_parameters=self.global_parameters
        )
        self.section.csrtrack_headers = self.csrtrack_headers
        return self.section.to_csrtrack()

    def write(self) -> str:
        """Write :meth:`writeElements` to ``<master_subdir>/csrtrk.in``."""
        code_file = self.global_parameters["master_subdir"] + "/csrtrk.in"
        saveFile(code_file, self.writeElements())

    def preProcess(self) -> None:
        """Load the input beam and write it as ASTRA via :meth:`hdf5_to_astra`."""
        super().preProcess()
        prefix = self.get_prefix()
        self.load_input_beam(prefix, self.particle_definition)
        astrabeamfilename = self.hdf5_to_astra()
        self.files.append(self.global_parameters["master_subdir"] + "/" + astrabeamfilename)

    def hdf5_to_astra(self) -> None:
        """
        Write the beam in ASTRA format, which CSRTrack reads.

        Returns
        -------
        str
            ASTRA beam filename
        """
        astrabeamfilename = self.csrtrack_headers["particles"].particle_definition + ".astra"
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        return astrabeamfilename

    def postProcess(self) -> None:
        """Convert the CSRTrack output via :meth:`csrtrack_to_hdf5`."""
        super().postProcess()
        self.csrtrack_to_hdf5()

    def csrtrack_to_hdf5(self) -> None:
        """Convert the CSRTrack monitor output (via ASTRA format) to openPMD in `master_subdir`."""
        csrtrackbeamfilename = self.csrtrack_headers["monitor"].name
        astrabeamfilename = csrtrackbeamfilename.replace(".fmt2", ".astra")
        rbf.astra.convert_csrtrackfile_to_astrafile(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + csrtrackbeamfilename,
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
        )
        rbf.astra.read_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        HDF5filename = self.output_basename(self.end) + ".openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )
