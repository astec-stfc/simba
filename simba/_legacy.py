"""
Legacy module path aliasing for previous non-PEP8-compliant code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

__all__ = ["LEGACY_MODULES", "install"]

LEGACY_MODULES: dict[str, str] = {
    # --- simba/Codes ----------------------------------------------------
    "simba.Codes": "simba.codes",
    "simba.Codes.ASTRA": "simba.codes.astra",
    "simba.Codes.ASTRA.ASTRA": "simba.codes.astra.astra",
    "simba.Codes.ASTRA.ASTRARules": "simba.codes.astra.astra_rules",
    "simba.Codes.CSRTrack": "simba.codes.csrtrack",
    "simba.Codes.CSRTrack.CSRTrack": "simba.codes.csrtrack.csrtrack",
    "simba.Codes.Cheetah": "simba.codes.cheetah",
    "simba.Codes.Cheetah.Cheetah": "simba.codes.cheetah.cheetah",
    "simba.Codes.Elegant": "simba.codes.elegant",
    "simba.Codes.Elegant.Elegant": "simba.codes.elegant.elegant",
    "simba.Codes.Executables": "simba.codes.executables",
    "simba.Codes.GPT": "simba.codes.gpt",
    "simba.Codes.GPT.GPT": "simba.codes.gpt.gpt",
    "simba.Codes.Generators": "simba.codes.generators",
    "simba.Codes.Generators.Generators": "simba.codes.generators.generators",
    "simba.Codes.Generators.astra": "simba.codes.generators.astra",
    "simba.Codes.Generators.gpt": "simba.codes.generators.gpt",
    "simba.Codes.Generators.opal": "simba.codes.generators.opal",
    "simba.Codes.Genesis": "simba.codes.genesis",
    "simba.Codes.Genesis.Genesis": "simba.codes.genesis.genesis",
    "simba.Codes.OPAL": "simba.codes.opal",
    "simba.Codes.OPAL.OPAL": "simba.codes.opal.opal",
    "simba.Codes.Ocelot": "simba.codes.ocelot",
    "simba.Codes.Ocelot.Ocelot": "simba.codes.ocelot.ocelot",
    "simba.Codes.Ocelot.mbi": "simba.codes.ocelot.mbi",
    "simba.Codes.Ocelot.savebeamopenpmd": "simba.codes.ocelot.savebeamopenpmd",
    "simba.Codes.Wake_T": "simba.codes.wake_t",
    "simba.Codes.Wake_T.Wake_T": "simba.codes.wake_t.wake_t",
    "simba.Codes.Xsuite": "simba.codes.xsuite",
    "simba.Codes.Xsuite.Xsuite": "simba.codes.xsuite.xsuite",
    # --- simba top-level modules ------------------------------------------
    "simba.Framework": "simba.framework",
    "simba.FrameworkHelperFunctions": "simba.framework_helper_functions",
    "simba.Framework_Settings": "simba.framework_settings",
    "simba.Framework_elements": "simba.framework_elements",
    "simba.Framework_lattices": "simba.framework_lattices",
    "simba.Framework_objects": "simba.framework_objects",
    # --- simba/Modules ------------------------------------------------------
    "simba.Modules": "simba.modules",
    "simba.Modules.Beams": "simba.modules.beams",
    "simba.Modules.Beams.Particles": "simba.modules.beams.particles",
    "simba.Modules.Beams.Particles.centroids": "simba.modules.beams.particles.centroids",
    "simba.Modules.Beams.Particles.emittance": "simba.modules.beams.particles.emittance",
    "simba.Modules.Beams.Particles.kde": "simba.modules.beams.particles.kde",
    "simba.Modules.Beams.Particles.minimumVolumeEllipse":
        "simba.modules.beams.particles.minimum_volume_ellipse",
    "simba.Modules.Beams.Particles.mve": "simba.modules.beams.particles.mve",
    "simba.Modules.Beams.Particles.sigmas": "simba.modules.beams.particles.sigmas",
    "simba.Modules.Beams.Particles.slice": "simba.modules.beams.particles.slice",
    "simba.Modules.Beams.Particles.twiss": "simba.modules.beams.particles.twiss",
    "simba.Modules.Beams.astra": "simba.modules.beams.astra",
    "simba.Modules.Beams.cheetah": "simba.modules.beams.cheetah",
    "simba.Modules.Beams.gdf": "simba.modules.beams.gdf",
    "simba.Modules.Beams.genesis": "simba.modules.beams.genesis",
    "simba.Modules.Beams.hdf5": "simba.modules.beams.hdf5",
    "simba.Modules.Beams.mad8": "simba.modules.beams.mad8",
    "simba.Modules.Beams.ocelot": "simba.modules.beams.ocelot",
    "simba.Modules.Beams.opal": "simba.modules.beams.opal",
    "simba.Modules.Beams.openpmd": "simba.modules.beams.openpmd",
    "simba.Modules.Beams.plot": "simba.modules.beams.plot",
    "simba.Modules.Beams.sdds": "simba.modules.beams.sdds",
    "simba.Modules.Beams.vsim": "simba.modules.beams.vsim",
    "simba.Modules.Beams.wake_t": "simba.modules.beams.wake_t",
    "simba.Modules.Beams.xsuite": "simba.modules.beams.xsuite",
    "simba.Modules.Fields": "simba.modules.fields",
    "simba.Modules.Fields.FieldParameter": "simba.modules.fields.field_parameter",
    "simba.Modules.Fields.astra": "simba.modules.fields.astra",
    "simba.Modules.Fields.gdf": "simba.modules.fields.gdf",
    "simba.Modules.Fields.hdf5": "simba.modules.fields.hdf5",
    "simba.Modules.Fields.opal": "simba.modules.fields.opal",
    "simba.Modules.Fields.sdds": "simba.modules.fields.sdds",
    "simba.Modules.MathParser": "simba.modules.math_parser",
    "simba.Modules.Matrices": "simba.modules.matrices",
    "simba.Modules.Matrices.elegant": "simba.modules.matrices.elegant",
    "simba.Modules.Matrices.hdf5": "simba.modules.matrices.hdf5",
    "simba.Modules.SDDSFile": "simba.modules.sdds_file",
    "simba.Modules.twiss": "simba.modules.twiss",
    "simba.Modules.twiss.astra": "simba.modules.twiss.astra",
    "simba.Modules.twiss.cheetah": "simba.modules.twiss.cheetah",
    "simba.Modules.twiss.elegant": "simba.modules.twiss.elegant",
    "simba.Modules.twiss.genesis": "simba.modules.twiss.genesis",
    "simba.Modules.twiss.gpt": "simba.modules.twiss.gpt",
    "simba.Modules.twiss.hdf5": "simba.modules.twiss.hdf5",
    "simba.Modules.twiss.ocelot": "simba.modules.twiss.ocelot",
    "simba.Modules.twiss.opal": "simba.modules.twiss.opal",
    "simba.Modules.twiss.plot": "simba.modules.twiss.plot",
    "simba.Modules.twiss.xsuite": "simba.modules.twiss.xsuite",
    "simba.Modules.constants": "simba.modules.constants",
    "simba.Modules.gdf_beam": "simba.modules.gdf_beam",
    "simba.Modules.gdf_emit": "simba.modules.gdf_emit",
    "simba.Modules.id_number": "simba.modules.id_number",
    "simba.Modules.id_number_server": "simba.modules.id_number_server",
    "simba.Modules.merge_two_dicts": "simba.modules.merge_two_dicts",
    "simba.Modules.optimisation": "simba.modules.optimisation",
    "simba.Modules.optimisation.constraints": "simba.modules.optimisation.constraints",
    "simba.Modules.optimisation.nelder_mead": "simba.modules.optimisation.nelder_mead",
    "simba.Modules.optimisation.optimiser": "simba.modules.optimisation.optimiser",
    "simba.Modules.optimisation.xopt": "simba.modules.optimisation.xopt",
    "simba.Modules.plotting": "simba.modules.plotting",
    "simba.Modules.plotting.latticeDraw": "simba.modules.plotting.lattice_draw",
    "simba.Modules.plotting.multiAxisPlot": "simba.modules.plotting.multi_axis_plot",
    "simba.Modules.plotting.multiPlot": "simba.modules.plotting.multi_plot",
    "simba.Modules.plotting.plotting": "simba.modules.plotting.plotting",
    "simba.Modules.pmd_units": "simba.modules.pmd_units",
    "simba.Modules.symmlinks": "simba.modules.symmlinks",
    "simba.Modules.units": "simba.modules.units",
    # --- simba/Support_Files -------------------------------------------------
    "simba.Support_Files": "simba.support_files",
    "simba.Support_Files.Check_YAML_with_Center_Datums":
        "simba.support_files.check_yaml_with_center_datums",
    "simba.Support_Files.Convert_YAML_to_DB": "simba.support_files.convert_yaml_to_db",
    "simba.Support_Files.Update_YAML_to_Center_Datums":
        "simba.support_files.update_yaml_to_center_datums",
    "simba.Support_Files.convert_txt_to_hdf5_field":
        "simba.support_files.convert_txt_to_hdf5_field",
    "simba.Support_Files.elegant_to_YAML": "simba.support_files.elegant_to_yaml",
    "simba.Support_Files.gdf_to_hdf5": "simba.support_files.gdf_to_hdf5",
    "simba.Support_Files.gdf_to_sdds": "simba.support_files.gdf_to_sdds",
    "simba.Support_Files.sdds_to_HDF5": "simba.support_files.sdds_to_hdf5",
    "simba.Support_Files.sdds_to_gdf": "simba.support_files.sdds_to_gdf",
    "simba.Support_Files.tempdir": "simba.support_files.tempdir",
}


class _LegacyLoader(importlib.abc.Loader):
    """Loader that returns the renamed module in place of the legacy one."""

    def __init__(self, legacy: str, current: str) -> None:
        self.legacy = legacy
        self.current = current

    def create_module(self, spec):
        warnings.warn(
            f"{self.legacy} was renamed to {self.current} for PEP 8 compliance. "
            f"The old import path still works but will be removed in a future "
            f"release; import from {self.current} instead.",
            FutureWarning,
            stacklevel=2,
        )
        return importlib.import_module(self.current)

    def exec_module(self, module) -> None:
        """No-op: the returned module was already executed under its real name."""


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        current = LEGACY_MODULES.get(fullname)
        if current is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _LegacyLoader(fullname, current)
        )


def _check_table() -> None:
    """A self-mapping entry would make the loader import itself forever."""
    for legacy, current in LEGACY_MODULES.items():
        if legacy == current:
            raise RuntimeError(
                f"LEGACY_MODULES maps {legacy!r} to itself, which would recurse."
            )


def install() -> None:
    """Register the finder. Idempotent."""
    _check_table()
    if not any(isinstance(f, _LegacyFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
