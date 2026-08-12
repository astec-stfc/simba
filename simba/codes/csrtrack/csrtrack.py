"""
SIMBA CSRTrack Module

Various objects and functions to handle CSRTrack lattices and commands. See `CSRTrack manual`_ for more details.

    .. _CSRTrack manual: https://www.desy.de/xfel-beam/csrtrack/files/CSRtrack_User_Guide_(actual).pdf

Classes:
    - :class:`~simba.Codes.CSRTrack.CSRTrack.CsrTrackLattice`: The CSRTrack lattice object, used for
    converting the :class:`~simba.Framework_objects.FrameworkLattice` into a string representation of
    the lattice suitable for a CSRTrack input file.
"""

from pydantic import Field
from ...framework_objects import FrameworkLattice
from ...framework_helper_functions import save_file
from ...modules import beams as rbf
from typing import Dict, List, Any
from laura.translator.converters.codes.csrtrack import (
    CsrTrackParticles,
    CsrTrackForces,
    CsrTrackTrackStep,
    CsrTrackTracker,
    CsrTrackMonitor,
)


class CsrTrackLattice(FrameworkLattice):
    """
    Class for defining the CSRTrack lattice object, used for
    converting the :class:`~simba.Framework_objects.FrameworkObject`s defined in the
    :class:`~simba.Framework_objects.FrameworkLattice` into a string representation of
    the lattice suitable for a CSRTrack input file.
    """

    _DEPRECATED_METHOD_ALIASES = {
        "postProcess": "post_process",
        "preProcess": "pre_process",
        "setCSRMode": "set_csr_mode",
        "writeElements": "write_elements",
    }

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
        """
        Set up the `CSRTrackelementObjects namelist for the initial particle distribution,
        based on the `particle_definition` and the `global_parameters` of the lattice.
        """
        self.csrtrack_headers["particles"] = CsrTrackParticles(
            particle_definition=self.particle_definition,
            global_parameters=self.global_parameters,
            format="astra",
        )
        if self.particle_definition == "initial_distribution":
            self.csrtrack_headers["particles"].particle_definition = "laser.astra"
            self.csrtrack_headers["particles"].array = "#file{name=laser.astra}"
        else:
            self.particle_definition = self.start
            self.csrtrack_headers["particles"].particle_definition = self.start
            self.csrtrack_headers["particles"].array = (
                "#file{name="
                + self.start
                + ".astra"
                + "}"
            )

    @property
    def dipoles_screens_and_bpms(self) -> List:
        """
        Get a list of the dipoles, screens and BPMs sorted by their position in the lattice

        Returns
        -------
        List
            A sorted list of dipoles, screens and BPMs.
        """
        return sorted(
            self.get_element_type("dipole")
            + self.get_element_type("screen")
            + self.get_element_type("beam_position_monitor"),
            key=lambda x: x.position_end[2],
        )

    def set_csr_mode(self) -> None:
        """
        Set up the `forces` key in `CSRTrackelementObjects based on the `csr_mode` defined in the settings
        file for this lattice section. `csr_mode` can be either ["csr_g_to_p" (2D) or "projected" (1D)]
        """
        if "csr" in self.file_block and "csr_mode" in self.file_block["csr"]:
            if self.file_block["csr"]["csr_mode"] == "3D":
                self.csrtrack_headers["forces"] = CsrTrackForces(type="csr_g_to_p")
            elif self.file_block["csr"]["csr_mode"] == "1D":
                self.csrtrack_headers["forces"] = CsrTrackForces(type="projected")
        else:
            self.CSRTrackelementObjects["forces"] = CsrTrackForces()

    def write_elements(self) -> str:
        """
        Write the lattice elements defined in this object into a CSRTrack-compatible format; see
        :attr:`~simba.Framework_objects.frameworkLattice.element_objects`.

        The appropriate headers required for ASTRA are written at the top of the file, see the `_write_CSRTrack`
        function in :class:`~simba.Codes.CSRTrack.csrtrack_element`.

        Returns
        -------
        str
            The lattice represented as a string compatible with CSRTrack
        """
        self.set_particles_filename()
        self.set_csr_mode()
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
        """
        Writes the CSRTrack input file from :func:`~simba.Codes.CSRTrack.csrtrackLattice.write_elements`
        to <master_subdir>/csrtrk.in.
        """
        code_file = self.global_parameters["master_subdir"] + "/csrtrk.in"
        save_file(code_file, self.write_elements())

    def pre_process(self) -> None:
        """
        Convert the beam file from the previous lattice section into CSRTrack format and set the number of
        particles based on the input distribution, see
        :func:`~simba.Codes.CSRTrack.csrtrack_particles.hdf5_to_astra`.
        """
        super().pre_process()
        prefix = self.get_prefix()
        self.read_input_file(
            prefix,
            self.csrtrack_headers["particles"].particle_definition.replace(".astra", ""),
        )
        self.hdf5_to_astra()
        self.files.append(self.csrtrack_headers["particles"].particle_definition)

    def hdf5_to_astra(self) -> None:
        """
        Convert HDF5 particle distribution to ASTRA format, suitable for inputting to CSRTrack.

        Parameters
        ----------
        prefix: str
            Prefix for filename
        """
        astrabeamfilename = self.csrtrack_headers["particles"].particle_definition + ".astra"
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        return astrabeamfilename

    def post_process(self) -> None:
        """
        Convert the beam file from the CSRTrack output into HDF5 format, see
        :func:`~simba.Codes.CSRTrack.csrtrack_monitor.csrtrack_to_hdf5`.
        """
        super().post_process()
        self.csrtrack_to_hdf5()

    def csrtrack_to_hdf5(self) -> None:
        """
        Convert the particle distribution from a CSRTrack monitor into HDF5 format,
        and write it to `master_subdir`.
        """
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
        HDF5filename = csrtrackbeamfilename.replace(".fmt2", ".openpmd.hdf5")
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )


from simba._compat import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(
    __name__,
    globals(),
    {
        "csrtrackLattice": "CsrTrackLattice",
    },
)
