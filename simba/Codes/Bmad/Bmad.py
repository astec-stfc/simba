"""
SIMBA Bmad Module

Various objects and functions to handle Bmad lattices and commands. See `Bmad manual`_
and `Tao manual`_ for more details.

    .. _Bmad manual: https://www.classe.cornell.edu/bmad/manual.html

    .. _Tao manual: https://www.classe.cornell.edu/bmad/tao.html

Classes:
    - :class:`~simba.Codes.Bmad.Bmad.bmadLattice`: The Bmad lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a Bmad lattice,
    and for tracking through it using PyTao.

"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from laura.models.simulation import TwissMatchSimulationElement
from laura.translator.utils.functions import sanitize_string

from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
from ...Modules import constants


class bmadLattice(frameworkLattice):
    """
    Class for defining the Bmad lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a Bmad lattice,
    and for tracking through it using PyTao.
    """

    code: str = "bmad"
    """String indicating the lattice type"""

    particle_definition: str | None = None
    """Initial particle distribution as a string"""

    input_beam_file: str | None = None
    """Input beam file name"""

    lattice_file: str | None = None
    """Lattice file name"""

    tao_init_file: str | None = None
    """Tao initialization file name"""

    tao: Any | None = None
    """PyTao instance"""

    ref_idx: int | None = None
    """Reference particle index"""

    space_charge_n_bin: int | None = None
    """Number of space-charge bins"""

    _SAVED_AT_LIMIT: int = 200
    """Used to avoid truncation of elements using beam_saved_at"""

    _ALIVE: int = 1
    """Value of the Bmad/openPMD particle status flag for a live particle"""

    libtao: str | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        particle_definition = self.file_block["input"].get(
            "particle_definition", self.start
        )
        self.particle_definition = (
            "laser"
            if particle_definition == "initial_distribution"
            else particle_definition
        )
        if self.libtao is None:
            self.libtao = self.executables["tao"][0]

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        space_charge_n_bin = self.csr_bins or self.lsc_bins
        super().preProcess()
        self.space_charge_n_bin = space_charge_n_bin
        self.read_input_file(self.get_prefix(), self.particle_definition)
        beam = self.global_parameters["beam"]
        self.ref_idx = beam.reference_particle_index
        self.input_beam_file = str(
            Path(self.global_parameters["master_subdir"])
            / f"{self.objectname}.bmad.beam"
        )
        self._write_bmad_beam_file()

    def _reference_value(self, values) -> float:
        values = np.asarray(values)
        if not values.size:
            raise ValueError("Cannot create a Bmad lattice without beam particles")
        if self.ref_idx is not None and 0 <= self.ref_idx < len(values):
            return float(values[self.ref_idx])
        return float(np.mean(values))

    def _write_bmad_beam_file(self) -> None:
        """
        Write the beam distribution to a text file.
        """
        beam = self.global_parameters["beam"]
        p0c = self._reference_value(beam.cp.val)
        z = beam.z.val - self._reference_value(beam.z.val)
        particles = np.column_stack(
            (
                beam.x.val,
                beam.cpx.val / p0c,
                beam.y.val,
                beam.cpy.val / p0c,
                z,
                beam.cp.val / p0c - 1,
                np.abs(beam.charge.val),
                beam.t.val,
            )
        )
        np.savetxt(
            self.input_beam_file,
            particles,
            header=(
                f"# species = {beam.species}\n"
                "# state = alive\n"
                f"# p0c = {p0c}\n"
                f"# charge_tot = {abs(float(beam.Q.val))}\n"
                "#! x px y py z pz charge time"
            ),
            comments="",
        )

    def _reference_energy(self) -> float:
        """
        Get energy of the reference particle.

        Returns
        -------
        float
            The energy of the reference particle in eV.
        """
        return self._reference_value(self.global_parameters["beam"].energy.val)

    def _bmad_initial_twiss(self) -> TwissMatchSimulationElement:
        """
        Get the initial Twiss, either from the simulation settings or the incoming beam

        Returns
        -------
        TwissMatchSimulationElement
            Section initial twiss object
        """
        if self.initial_twiss["horizontal"]["beta"] and self.initial_twiss["vertical"]["beta"]:
            return TwissMatchSimulationElement(
                beta_x=self.initial_twiss["horizontal"]["beta"],
                alpha_x=self.initial_twiss["horizontal"]["alpha"],
                beta_y=self.initial_twiss["vertical"]["beta"],
                alpha_y=self.initial_twiss["vertical"]["alpha"],
            )
        twiss = self.global_parameters["beam"].twiss
        return TwissMatchSimulationElement(
            beta_x=float(twiss.beta_x.val),
            alpha_x=float(twiss.alpha_x.val),
            beta_y=float(twiss.beta_y.val),
            alpha_y=float(twiss.alpha_y.val),
        )

    def write(self) -> None:
        """
        Create the lattice file using the LAURA ``SectionLatticeTranslator``
        and save it to `master_subdir`.
        """
        section = self.section
        section.reference_energy = self._reference_energy()
        lattice = section.to_bmad(
            particle=self.global_parameters["beam"].species,
            space_charge_n_bin=self.space_charge_n_bin,
            initial_twiss=self._bmad_initial_twiss(),
        )
        path = Path(self.global_parameters["master_subdir"]) / f"{self.objectname}.bmad"
        path.write_text(lattice, encoding="utf-8")
        self.lattice_file = str(path)
        tao_path = path.with_suffix(".tao.init")
        tao_path.write_text(
            f'&tao_beam_init\n  beam_saved_at = "{self._saved_at(lattice)}"\n/\n',
            encoding="utf-8",
        )
        self.tao_init_file = str(tao_path)

    @property
    def _saved_elements(self) -> list[str]:
        return list(
            dict.fromkeys(
                sanitize_string(element.name)
                for element in self.screens_and_markers_and_bpms
            )
        )

    def _saved_at(self, lattice: str) -> str:
        """
        Build the Tao ``beam_saved_at`` list covering every output element.

        Parameters
        ----------
        lattice: str
            The Bmad lattice text, used to look up the class of each element.

        Returns
        -------
        str
            A comma-separated list of Bmad element classes, or ``*`` if even
            that does not fit within Tao's 200 character limit.
        """
        classes = {}
        for line in lattice.splitlines():
            name, _, remainder = line.partition(":")
            element_class = remainder.strip().partition(",")[0].strip().lower()
            if element_class:
                classes[name.strip()] = element_class
        saved_at = ", ".join(
            dict.fromkeys(
                [
                    f"{classes[name]}::*"
                    for name in self._saved_elements
                    if name in classes
                ]
                + ["END"]
            )
        )
        return saved_at if len(saved_at) <= self._SAVED_AT_LIMIT else "*"

    def run(self) -> None:
        """
        Run the code via PyTao.
        """
        if (
            self.lattice_file is None
            or self.input_beam_file is None
            or self.tao_init_file is None
        ):
            raise RuntimeError(
                "Bmad lattice and input beam must be written before tracking"
            )
        from pytao import Tao

        working_directory = Path(self.lattice_file).parent
        previous_directory = Path.cwd()
        try:
            os.chdir(working_directory)
            self.tao = Tao(
                init_file=Path(self.tao_init_file).name,
                lattice_file=Path(self.lattice_file).name,
                beam_init_position_file=Path(self.input_beam_file).name,
                so_lib=self.libtao,
                noplot=True,
            )
            self.tao.track_beam("BEGINNING", "END", use_progress_bar=False)
        finally:
            os.chdir(previous_directory)

    def _particles_at(self, element: str) -> tuple:
        """
        Get the particle distribution at a given element.

        Bmad keeps particles lost upstream in the bunch, frozen at the
        coordinates they had when they died, so they are dropped here; every
        other code reports only the particles that survived to the element.

        Parameters
        ----------
        element: str
            The name of the element to query.

        Returns
        -------
        tuple[ParticleGroup, int | None]
            openPMD particle group object, and the index of the reference
            particle within it, or None if the reference particle is not alive.
        """
        data = self.tao.bunch_data(element)
        alive = np.asarray(data["status"]) == self._ALIVE
        ref_idx = (
            int(np.count_nonzero(alive[: self.ref_idx]))
            if self.ref_idx is not None
            and 0 <= self.ref_idx < len(alive)
            and alive[self.ref_idx]
            else None
        )
        particles = rbf.openpmd.ParticleGroup(
            data={
                key: value[alive] if np.shape(value) == alive.shape else value
                for key, value in data.items()
            }
        )
        reference_time = (
            particles.t[ref_idx] if ref_idx is not None else np.mean(particles.t)
        )
        particles.z = (
            -particles.beta_z
            * constants.speed_of_light
            * (particles.t - reference_time)
        )
        return particles, ref_idx

    def postProcess(self) -> None:
        """
        Retrieve the outputs from Bmad and save them to `master_subdir` in openPMD format.
        """
        super().postProcess()
        if self.tao is None:
            raise RuntimeError("Bmad tracking must finish before post-processing")
        source_beam = self.global_parameters["beam"]
        s_values = self.getSValues(as_dict=True)
        outputs = {
            element.name: sanitize_string(element.name)
            for element in self.screens_and_markers_and_bpms
        }
        outputs[self.end] = "END"
        final_beam = None
        for output_name, tao_element in outputs.items():
            beam = deepcopy(source_beam)
            particles, ref_idx = self._particles_at(tao_element)
            rbf.openpmd.read_particle_group(
                beam,
                particles,
                s=s_values[output_name],
                reference_particle_index=ref_idx,
            )
            rbf.openpmd.write_openpmd_beam_file(
                beam,
                str(
                    Path(self.global_parameters["master_subdir"])
                    / f"{output_name}.openpmd.hdf5"
                ),
            )
            if output_name == self.end:
                final_beam = beam
        self.global_parameters["beam"] = final_beam
