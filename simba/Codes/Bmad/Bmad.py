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
from ...Modules.Twiss.bmad import save_bmad_twiss_hdf

LATTICE_TWISS = {
    "s": "ele.s",
    "e_tot": "ele.e_tot",
    "p0c": "ele.p0c",
    "design_beta_x": "ele.a.beta",
    "design_alpha_x": "ele.a.alpha",
    "design_gamma_x": "ele.a.gamma",
    "design_beta_y": "ele.b.beta",
    "design_alpha_y": "ele.b.alpha",
    "design_gamma_y": "ele.b.gamma",
    "design_eta_x": "ele.x.eta",
    "design_etap_x": "ele.x.etap",
    "design_eta_y": "ele.y.eta",
    "design_etap_y": "ele.y.etap",
    "mu_x": "ele.a.phi",
    "mu_y": "ele.b.phi",
}
"""Lattice functions to extract from Tao, keyed by their name in the twiss file"""

BEAM_TWISS = {
    "beam_charge": "charge_live",
    "beam_n_particle": "n_particle_live",
    "beam_t": "centroid_t",
    "beam_p0c": "centroid_p0c",
    "beam_x": "centroid_vec_1",
    "beam_y": "centroid_vec_3",
    "beam_delta": "centroid_vec_6",
    "beam_sigma_x": "twiss_sigma_x",
    "beam_sigma_xp": "twiss_sigma_p_x",
    "beam_sigma_y": "twiss_sigma_y",
    "beam_sigma_yp": "twiss_sigma_p_y",
    "beam_sigma_z": "twiss_sigma_z",
    "beam_sigma_delta": "twiss_sigma_p_z",
    "beam_sigma_t": "sigma_t",
    "beam_emit_x": "twiss_emit_x",
    "beam_emit_y": "twiss_emit_y",
    "beam_emit_z": "twiss_emit_z",
    "beam_norm_emit_x": "twiss_norm_emit_x",
    "beam_norm_emit_y": "twiss_norm_emit_y",
    "beam_norm_emit_z": "twiss_norm_emit_z",
    "beam_norm_emit_a": "twiss_norm_emit_a",
    "beam_norm_emit_b": "twiss_norm_emit_b",
    "beam_beta_x": "twiss_beta_x",
    "beam_alpha_x": "twiss_alpha_x",
    "beam_gamma_x": "twiss_gamma_x",
    "beam_beta_y": "twiss_beta_y",
    "beam_alpha_y": "twiss_alpha_y",
    "beam_gamma_y": "twiss_gamma_y",
    "beam_beta_a": "twiss_beta_a",
    "beam_alpha_a": "twiss_alpha_a",
    "beam_beta_b": "twiss_beta_b",
    "beam_alpha_b": "twiss_alpha_b",
    "beam_beta_z": "twiss_beta_z",
    "beam_alpha_z": "twiss_alpha_z",
    "beam_gamma_z": "twiss_gamma_z",
    "beam_eta_x": "twiss_eta_x",
    "beam_etap_x": "twiss_etap_x",
    "beam_eta_y": "twiss_eta_y",
    "beam_etap_y": "twiss_etap_y",
}
"""Bunch parameters to extract from Tao, keyed by their name in the twiss file"""


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

    ref_s: float | None = None
    """S position at the start of the lattice"""

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
        self.ref_s = beam.s
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

    def _particles_at(self, element: str, zstart: float = 0) -> tuple:
        """
        Get the particle distribution at a given element.

        Bmad keeps particles lost upstream in the bunch, frozen at the
        coordinates they had when they died, so they are dropped here; every
        other code reports only the particles that survived to the element.

        Parameters
        ----------
        element: str
            The name of the element to query.
        zstart: float
            Position of the element along the machine.

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
        particles.z = zstart + (
            -particles.beta_z
            * constants.speed_of_light
            * (particles.t - reference_time)
        )
        return particles, ref_idx

    def _twiss_data(self) -> dict:
        """
        Extract the twiss parameters along the lattice from Tao.

        Returns
        -------
        dict
            Twiss data, ready to be written by
            :func:`~simba.Modules.Twiss.bmad.save_bmad_twiss_hdf`.
        """
        indices = self.tao.lat_list("*", "ele.ix_ele", flags="-array_out -track_only")
        twiss = {
            name: self.tao.lat_list("*", command, flags="-array_out -track_only")
            for name, command in LATTICE_TWISS.items()
        }
        twiss["element_name"] = np.asarray(
            self.tao.lat_list("*", "ele.name", flags="-track_only")
        )
        bunch_params = [self.tao.bunch_params(int(index)) for index in indices]
        twiss.update(
            {
                name: np.array([params[key] for params in bunch_params])
                for name, key in BEAM_TWISS.items()
            }
        )
        ref_s = self.ref_s if self.ref_s is not None else self.startObject.physical.start.z
        twiss["s"] = twiss["s"] + ref_s
        s_values = np.array(self.getSValues(at_entrance=False)) + ref_s
        z_values = [z[-1] for z in self.getZValues()]
        twiss["z"] = np.interp(twiss["s"], s_values, z_values)
        return twiss

    def postProcess(self) -> None:
        """
        Retrieve the outputs from Bmad and save them to `master_subdir`,
        and the twiss parameters along the lattice in HDF5 format.
        """
        super().postProcess()
        if self.tao is None:
            raise RuntimeError("Bmad tracking must finish before post-processing")
        source_beam = self.global_parameters["beam"]
        ref_s = self.ref_s if self.ref_s is not None else self.startObject.physical.start.z
        s_values = {
            name: value + ref_s
            for name, value in self.getSValues(as_dict=True).items()
        }
        outputs = {
            element.name: (sanitize_string(element.name), element.physical.start.z)
            for element in self.screens_and_markers_and_bpms
        }
        outputs[self.end] = ("END", self.endObject.physical.end.z)
        final_beam = None
        for output_name, (tao_element, zstart) in outputs.items():
            beam = deepcopy(source_beam)
            particles, ref_idx = self._particles_at(tao_element, zstart=zstart)
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
        save_bmad_twiss_hdf(
            filename=str(
                Path(self.global_parameters["master_subdir"])
                / f"{self.objectname}_twiss.bmad.hdf5"
            ),
            twiss=self._twiss_data(),
        )
        self.global_parameters["beam"] = final_beam
