"""GPT beam generator."""
import os
import numpy as np
import subprocess
from ...FrameworkHelperFunctions import (
    saveFile,
    copylink,
    expand_substitution,
)
from .Generators import (
    frameworkGenerator,
    aliases,
)
from typing import Any
from ...Modules import constants
from easygdf import load
from ...Modules import Beams as rbf
from ...Modules.units import UnitValue

mass_index = {
        "electron": "me",  # electron
        "positron": "me",  # positron
        "proton": "mp",    # proton
        "hydrogen": "mp",  # hydrogen ion
        "electrons": "me",  # electron
        "positrons": "me",  # positron
        "protons": "mp",    # proton
    }
charge_sign_index = {
    "electron": -1,
    "positron": 1,
    "proton": 1,
    "hydrogen": -1,
    "electrons": -1,
    "positrons": 1,
    "protons": 1,
}


class GPTGenerator(frameworkGenerator):
    """
    Generates a cathode beam with GPT.
    """

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.apply_alias_and_multiplier(aliases, "gpt")
        self.code = "gpt"

    def run(self):
        """
        Run the GPT generator to create the beam file.
        """
        command = (
            self.executables[self.code]
            + ["-o", "generator.gdf"]
            + [f"GPTLICENSE={self.global_parameters['GPTLICENSE']}"]
            + [f"{self.objectname}.in"]
        )
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = (
            f"{my_env.get('LD_LIBRARY_PATH', '')}:/opt/GPT3.3.6/lib/"
        )
        my_env["OMP_WAIT_POLICY"] = "PASSIVE"

        log_path = os.path.join(
            self.global_parameters["master_subdir"], f"{self.objectname}.log"
        )
        with open(os.path.abspath(log_path), "w") as log_file:
            subprocess.call(
                command,
                stdout=log_file,
                cwd=self.global_parameters["master_subdir"],
                env=my_env,
            )

    def load_longitudinal_profile(self, v: str) -> None:
        if len(v) > 0:
            if ".gdf" not in v:
                raise NotImplementedError("Longitudinal profiles only defined for GPT; fields must be GDF format")
            fi = load(v)
            self.longitudinal_profile = v
            self.longitudinal_fields = [p["name"] for p in fi["blocks"]]
            self.distribution_type_z = "f"

    def generate_particles(self):
        """
        GPT input defining the particles: thermal energy, charge and number.
        """
        return (
        f"""#--Basic beam parameters--
        E0 = {self.thermal_kinetic_energy};
        G = 1-(qe)*E0/({mass_index[self.species]} * c * c);
        GB = sqrt(G^2 - 1);
        Qtot = {charge_sign_index[self.species]}*{str(abs(1e12 * self.charge))}e-12;
        npart = {self.number_of_particles};
        setparticles( "beam", npart, {mass_index[self.species]}, {-charge_sign_index[self.species]}*qe, Qtot ) ;
        """
        )

    def check_xy_parameters(
            self,
            x: str,
            y: str,
            default: str
    ) -> None:
        """
        Fill whichever of attributes `x` and `y` is None from the other, or both with `default`.

        :param x: first attribute name
        :param y: second attribute name
        :param default: value if both are None
        """
        x_val, y_val = getattr(self, x, None), getattr(self, y, None)
        if x_val is None and y_val is not None:
            setattr(self, x, y_val)
        elif x_val is not None and y_val is None:
            setattr(self, y, x_val)
        elif x_val is None and y_val is None:
            setattr(self, x, default)
            setattr(self, y, default)

    def _uniform_distribution(
            self,
            distname: str,
            variable: str,
            left_cutoff: float = 0,
            right_cutoff: float = 0,
            left_multiplier: float = 1,
            right_multiplier: float = 2,
    ) -> str:
        """
        GPT uniform-distribution command.

        :param distname: GPT command, e.g. ``settdist``
        :param variable: GPT variable to spread
        :param left_cutoff: unused
        :param right_cutoff: unused
        :param left_multiplier: lower bound, in units of `variable`
        :param right_multiplier: upper bound, in units of `variable`
        """
        return f'{distname}( "beam", "u", {left_multiplier}*{variable}, {right_multiplier}*{variable} ) ;'

    def _gaussian_distribution(
            self,
            distname: str,
            variable: str,
            left_multiplier: float = 3,
            right_multiplier: float = 3,
            left_cutoff: float = 3,
            right_cutoff: float = 3,
    ) -> str:
        """
        GPT Gaussian-distribution command.

        :param distname: GPT command, e.g. ``settdist``
        :param variable: GPT variable holding sigma
        :param left_multiplier: unused
        :param right_multiplier: unused
        :param left_cutoff: lower cutoff [sigma]
        :param right_cutoff: upper cutoff [sigma]
        """
        return f'{distname}( "beam", "g", 0, {variable}, {left_cutoff}, {right_cutoff} ) ;'

    def _file_distribution(
            self,
            distname: str,
            variable: str,
            column1: str,
            column2: str,
            scaling: float = 1.0,
            offset: float = 0.0,
    ) -> str:
        """
        GPT from-file distribution command.

        :param distname: GPT command, e.g. ``settdist``
        :param variable: distribution filename
        :param column1: time column in the file
        :param column2: probability column in the file
        :param scaling: scaling of the file (should be normalised to 1)
        :param offset: offset of the file
        """
        return f'{distname}( "beam", "F", "{variable}", "{column1}", "{column2}", {scaling}, {offset} ) ;'

    def _distribution(
            self,
            param: str,
            distname: str,
            variable: str,
            **kwargs
    ) -> str:
        """
        GPT distribution command (Gaussian, uniform or from file) of the type in attribute `param`.

        :param param: attribute naming the distribution type
        :param distname: GPT command, e.g. ``settdist``
        :param variable: GPT variable, or filename
        :param kwargs: passed to the distribution's writer
        """
        param_value = getattr(self, param, "").lower()
        if param_value in ["g", "gaussian", "2dgaussian", "radial", "r"]:
            return self._gaussian_distribution(distname, variable, **kwargs)
        elif param_value in ["u", "uniform", "p", "plateau"]:
            return self._uniform_distribution(distname, variable, **kwargs)
        elif param_value in ["F", "f", "file"]:
            return self._file_distribution(distname, variable, **kwargs)
        else:
            raise NotImplementedError("Only uniform, gaussian and from-file distributions are supported")

    def generate_image_name(self, param: str) -> str:
        """
        Link or copy a file (after substitution) into `master_subdir`.

        :param param: the filename
        :return: its basename
        """
        basename = os.path.basename(param).replace('"', "").replace("'", "")
        location = os.path.abspath(
            expand_substitution(self, param).replace("\\", "/").replace('"', "").replace("'", "")
        )
        efield_basename = os.path.join(
            self.global_parameters["master_subdir"], basename
        ).replace("\\", "/")
        copylink(location, efield_basename)
        return basename

    def generate_radial_distribution(self):
        """
        GPT transverse distribution: from an image, an ellipse, or a circle; empty if
        x and y have different distribution types.
        """
        if self.distribution_type_x == "image" or self.distribution_type_y == "image":
            image_filename = os.path.abspath(self.image_filename)
            image_calibration_x = (
                self.image_calibration_x
                if isinstance(self.image_calibration_x, int)
                   and self.image_calibration_x > 0
                else 1000 * 1e3
            )
            image_calibration_y = (
                self.image_calibration_y
                if isinstance(self.image_calibration_y, int)
                   and self.image_calibration_y > 0
                else 1000 * 1e3
            )
            image_filename = self.generate_image_name(image_filename)
            return f'setxydistbmp("beam", "{image_filename}", {image_calibration_x}, {image_calibration_y}) ;\n'
        elif self.sigma_x != self.sigma_y and self.distribution_type_x == self.distribution_type_y:
            return (
                f"radius_x = {self.sigma_x};\n"
                f"radius_y = {self.sigma_y};\n"
                'setellipse("beam", 2.0*radius_x, 2.0*radius_y, 1e-12);\n'
            )
        elif self.sigma_x == self.sigma_y and self.distribution_type_x == self.distribution_type_y:
            return (
                    f"radius = {self.sigma_x};\n"
                    + self._distribution(
                "distribution_type_x",
                "setrxydist",
                "radius",
                left_cutoff=0,
                right_cutoff=self.gaussian_cutoff_x,
            )
                    + '\nsetphidist("beam", "u", 0, 2*pi) ;\n'
            )
        return ""

    def generate_phase_space_distribution(self):
        """
        GPT initial momentum distribution: fixed uniform GBz, theta and phi.
        """
        return """#--Initial Phase-Space--
setGBzdist( "beam", "u", GB, 0 ) ;
setGBthetadist("beam","u", pi/4, pi/2);
setGBphidist("beam","u", 0, 2*pi);
"""

    def generate_correlated_divergences(self) -> str:
        """GPT correlated x and y divergences."""
        output = ""
        if self.correlation_px is not None:
            xc = self.offset_x if self.offset_x is not None else 0
            output += f"""addxdiv("beam" , {xc}, {self.correlation_px});\n"""
        if self.correlation_py is not None:
            yc = self.offset_y if self.offset_y is not None else 0
            output += f"""addydiv("beam" , {yc}, {self.correlation_py});\n"""
        return output

    def generate_thermal_emittance(self):
        """
        GPT emittance commands, normalised plus thermal; none for an image distribution.
        """
        normalized_emittance_x = (
            float(self.normalized_horizontal_emittance)
            if self.normalized_horizontal_emittance is not None
            else 0
        )
        normalized_emittance_y = (
            float(self.normalized_vertical_emittance)
            if self.normalized_vertical_emittance is not None
            else 0
        )
        if self.distribution_type_x == "image" or self.distribution_type_y == "image":
            return "\n"
        elif self.sigma_x != self.sigma_y:
            thermal_emittance = (
                float(self.thermal_emittance)
                if self.thermal_emittance is not None
                else 0
            )
            return (
                f"""setGBxemittance("beam", {normalized_emittance_x}/2 + ({thermal_emittance}*radius_x)) ;
        setGByemittance("beam", {normalized_emittance_y}/2 + ({thermal_emittance}*radius_y)) ;
        """
            )
        else:
            thermal_emittance = (
                float(self.thermal_emittance)
                if self.thermal_emittance is not None
                else 0
            )
            return (
                f"""setGBxemittance("beam", {normalized_emittance_x} + ({thermal_emittance}*radius)) ;
        setGByemittance("beam", {normalized_emittance_y} + ({thermal_emittance}*radius)) ;
        """
            )

    def generate_longitudinal_distribution(self):
        """
        GPT time distribution: from :attr:`longitudinal_profile`, Gaussian, or plateau.
        """
        output = ""
        if (self.distribution_type_z.lower() in ["f", "file"]) and len(self.longitudinal_profile) > 0:
            profile_name = os.path.abspath(self.longitudinal_profile)
            profile_name = self.generate_image_name(profile_name)
            if len(self.longitudinal_fields) != 2:
                raise ValueError(f"length of longitudinal_fields is not correct; check {self.longitudinal_profile}")
            variable = profile_name
            dist_params = {
                "column1": self.longitudinal_fields[0],
                "column2": self.longitudinal_fields[1],
            }
        else:
            dist_params = {
                "left_cutoff": self.gaussian_cutoff_z,
                "right_cutoff": self.gaussian_cutoff_z,
                "left_multiplier": 0,
                "right_multiplier": 1,
            }
            variable = "tlen"
            if self.distribution_type_z.lower() in ["g", "gaussian"]:
                sigma_t = self.sigma_t or self.sigma_z / constants.speed_of_light
                output += f"""tlen = {1e12 * sigma_t}e-12;\n"""
            else:
                output += f"""tlen = {1e12 * self.plateau_bunch_length}e-12;\n"""
        output += (
                self._distribution(
                    "distribution_type_z",
                    "settdist",
                    variable,
                    **dist_params
                )
                + "\n"
        )
        return output

    def generate_output(self):
        """
        GPT screen at z=0.
        """
        return """screen( "wcs", "I", 0) ;
"""

    def generate_offset_transform(self):
        """
        GPT ``settransform`` offsetting the beam by ``offset_x``, ``offset_y``.
        """
        return (
            f'settransform("wcs", {self.offset_x}, {self.offset_y}, 0, 1, 0, 0, 0, 1, 0, "beam");\n'
        )

    def write(self):
        """
        Write the GPT input file to `master_subdir`; cathode beams only.
        """
        if not self.cathode:
            raise NotImplementedError("Only cathode beams are currently supported in GPT generator")
        output = ""
        output += self.generate_particles()
        output += self.generate_radial_distribution()
        output += self.generate_phase_space_distribution()
        output += self.generate_thermal_emittance()
        output += self.generate_longitudinal_distribution()
        output += self.generate_offset_transform()
        output += self.generate_output()
        saveFile(
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in",
            output,
        )

    def postProcess(self):
        """
        Convert ``generator.gdf`` to ``laser.openpmd.hdf5`` (names hardcoded), with z set to zero.
        """
        gptbeamfilename = "generator.gdf"
        self.global_parameters["beam"] = rbf.beam()
        rbf.gdf.read_gdf_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + gptbeamfilename,
            position=0,
            longitudinal_reference="t",
        )
        self.global_parameters["beam"].z = UnitValue(np.full(len(self.global_parameters["beam"].z), 0), units="m")
        HDF5filename = "laser.openpmd.hdf5" if self.cathode else self.filename
        self.global_parameters["beam"].set_species(self.species)
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )
