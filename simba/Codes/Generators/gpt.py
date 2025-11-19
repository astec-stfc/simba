"""
This file generates a GPT beam file from the provided parameters.

It includes methods to run the GPT generator, write the input file, and post-process the generated beam data.
"""
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
    A class to generate a GPT beam file from the provided parameters.

    :param executables: Dictionary containing the paths to the executables.
    :param global_parameters: Dictionary containing global parameters for the simulation.
    :param generator_keywords: Dictionary containing keywords for the generator.
    :param kwargs: Additional keyword arguments for the generator.
    :ivar filename: Name of the output file.
    :ivar code: Code identifier for the generator.
    :ivar allowedKeyWords: List of allowed keywords for the generator.
    :ivar generator_keywords: Keywords specific to the GPT generator.
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
            self.distribution_type_z = "F"

    def generate_particles(self):
        """
        Generate the basic beam parameters for the GPT input file.
        These include the thermal kinetic energy, charge, and number of particles.
        :return: A string representation of the basic beam parameters.
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
        Check if the parameters x and y are set correctly.
        If one of them is None and the other is not, set the None parameter to the value of the other.
        If both are None, set both to the default value.
        :param x: The first parameter to check.
        :param y: The second parameter to check.
        :param default: The default value to set if both parameters are None.
        #TODO This appears not to be used in the code, consider removing it.
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
        Generate a uniform distribution string for the GPT input file.

        :param distname: The name of the distribution.
        :param variable: The variable to apply the distribution to.
        :param left_multiplier: The multiplier for the left side of the distribution.
        :param right_multiplier: The multiplier for the right side of the distribution.
        :return: A string representation of the uniform distribution.
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
        Generate a Gaussian distribution string for the GPT input file.

        :param distname: The name of the distribution.
        :param variable: The variable to apply the distribution to.
        :param left_cutoff: The cutoff value for the left side of the distribution.
        :param right_cutoff: The cutoff value for the right side of the distribution.
        :return: A string representation of the Gaussian distribution.
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
        Generate a Gaussian distribution string for the GPT input file.

        :param distname: The name of the distribution.
        :param variable: The filename for the distribution
        :param column1: Time parameter in distribution file
        :param column2: Probability parameter in distribution file
        :param scaling: Scaling for distribution file (should be normalised to 1)
        :param offset: Offset in distribution file
        :return: A string representation of the Gaussian distribution.
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
        Generate a distribution string based on the type of distribution specified in the parameter.
        Only supports Gaussian and Uniform distributions.

        :param param: The parameter that specifies the type of distribution.
        :param distname: The name of the distribution to generate.
        :param variable: The variable to apply the distribution to.
        :param kwargs: Additional keyword arguments for the distribution.
        :return: A string representation of the distribution.
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
        Generate a name for the image file based on the provided parameter.
        This function expands the parameter substitution, checks if the file exists,
        and creates a symbolic link or copies the file to the master subdirectory.
        This is used for image-based distributions in the beam generation in :function:`generate_radial_distribution`.

        :param param: The parameter containing the image filename.
        :return: The basename of the image file.
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
        Generate the radial distribution for the beam based on the parameters provided.

        This function checks the distribution type and sigma values for both x and y dimensions.
        If the distribution type is 'image', it generates a command to set the beam distribution from an image file.
        If the sigma values for x and y are equal and the distribution types are the same, it generates a command
        to set a circular distribution.

        If the sigma values are different but the distribution types are the same, it generates a command
        to set an elliptical distribution. If the distribution types are different, it returns an empty string.

        :return: A string representation of the radial distribution command for the GPT input file.
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
        Generate the initial phase-space distribution for the beam.

        A uniform distribution is set for z, theta, and phi.
        #TODO Should these be configurable parameters?

        :return: A string representation of the initial phase-space distribution command for the GPT input file.
        """
        return """#--Initial Phase-Space--
setGBzdist( "beam", "u", GB, 0 ) ;
setGBthetadist("beam","u", pi/4, pi/2);
setGBphidist("beam","u", 0, 2*pi);
"""

    def generate_correlated_divergences(self) -> str:
        """Generate correlated divergences.

        :return: String representing x-px and y-py divergences to be applied
        """
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
        Generate the thermal emittance for the beam based on the distribution type and sigma values.

        If the distribution type is 'image', it returns an empty string.
        If the sigma values for x and y are different, it generates separate commands for x and y emittance.
        If the sigma values are equal, it generates a command for circular emittance.

        :return: A string representation of the thermal emittance command for the GPT input file.
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
        if self.distribution_type_x == "image" or self.distribution_type_x == "image":
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
        Generate the longitudinal distribution for the beam based on the distribution type and parameters.

        If the distribution type is 'gaussian', it generates a command for a Gaussian distribution with the
        specified sigma.
        If the distribution type is not 'gaussian', it generates a command for a plateau bunch length distribution.

        :return: A string representation of the longitudinal distribution command for the GPT input file.
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
                if self.sigma_t is None:
                    self.sigma_t = self.sigma_z / constants.speed_of_light
                output += f"""tlen = {1e12 * self.sigma_t}e-12;\n"""
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
        Generate the output command for the GPT input file at z=0.
        This command sets the screen for the beam output, specifying the type and position.

        :return: A string representation of the output command for the GPT input file.
        """
        return """screen( "wcs", "I", 0) ;
"""

    def generate_offset_transform(self):
        """
        Generate the offset transform command for the beam.

        This command sets the transformation of the beam in the WCS (World Coordinate System)
        with the specified offsets in x and y (i.e. if the beam is offset from the nominal axis).

        :return: A string representation of the offset transform command for the GPT input file.
        """
        return (
            f'settransform("wcs", {self.offset_x}, {self.offset_y}, 0, 1, 0, 0, 0, 1, 0, "beam");\n'
        )

    def write(self):
        """
        Write the GPT input file with the parameters defined in the class.

        Base attributes of :class:`~simba.Codes.Generators.frameworkGenerator`
        are used to generate the input file, with the appropriate aliases and multipliers applied for the GPT code.

        :return: None
        #TODO Filenames are hardcoded for simplicity and they shouldn't be.
        """
        # try:
        #     npart = eval(self.number_of_particles)
        # except:
        #     npart = self.number_of_particles
        if not self.cathode:
            raise NotImplementedError("Only cathode beams are currently supported in GPT generator")
        output = ""
        output += self.generate_particles()
        output += self.generate_radial_distribution()
        output += self.generate_phase_space_distribution()
        output += self.generate_thermal_emittance()
        output += self.generate_longitudinal_distribution()
        # output += self.generate_correlated_divergences()
        output += self.generate_offset_transform()
        output += self.generate_output()
        saveFile(
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in",
            output,
        )

    def postProcess(self):
        """
        Post-process the generated beam data to create an HDF5 file.

        This method reads the GPT beam file and writes it to an HDF5 format.
        The Z component of the beam is set to zero, and the status of the particles is initialized to -1.
        The output HDF5 file is named 'laser.hdf5' and is saved in the master subdirectory.
        #TODO Filenames are hardcoded for simplicity and they shouldn't be.
        """
        gptbeamfilename = "generator.gdf"
        self.global_parameters["beam"] = rbf.beam()
        rbf.gdf.read_gdf_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + gptbeamfilename,
            position=0,
            longitudinal_reference="t",
        )
        # Set the Z component to be zero
        self.global_parameters["beam"].z = UnitValue(np.full(len(self.global_parameters["beam"].z), 0), units="m")
        if self.cathode:
            HDF5filename = "laser.openpmd.hdf5"
        else:
            HDF5filename = self.filename
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )
