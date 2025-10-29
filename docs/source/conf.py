# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
#
# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
#
# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#
# Options for Pydantic models
# https://autodoc-pydantic.readthedocs.io/en/stable/users/configuration.html

import os
import sys
import subprocess


def get_git_revision_hash(short: bool = True) -> str:
    """Returns the hash representing the current git revision

    :param bool short: Return the short version of the hash
    :returns: Hash representing the current local commit
    """
    command = "git rev-parse --short HEAD" if short else "git rev-parse HEAD"
    revision = subprocess.check_output(command.split())
    return revision.decode("ascii").strip()


def get_git_revision_tags() -> str:
    """Returns tags associated with the current git revision

    :returns: Tags assigned the the current local commit
    """
    command = "git tag --points-at HEAD"
    tags = subprocess.check_output(command.split())
    if len(tags) != 0:
        return tags.decode("ascii").strip()
    else:
        return None


# # use custom theme to set html width
# def setup(app):
#     app.add_css_file("my_theme.css")


# set relative path to the documented project
sys.path.insert(0, os.path.abspath("../../"))

# set project name, author and copyright
author = "James Jones"
copyright = "2024, STFC ASTeC"
project = "SIMBA"

# fetch version (commit hash or release tags) from git
_git_tags = get_git_revision_tags()
version = get_git_revision_hash(short=True) if _git_tags is None else _git_tags

# set sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinxcontrib.bibtex",
]


# set sphinx options
source_suffix = ".rst"  # use reStructedText files for sphinx pages
master_doc = "index"  # name for the root document
exclude_patterns = ["_build"]  # patterns to exclude when looking for source files
templates_path = ["_templates"]  # list of paths that contain extra templates
add_function_parentheses = True  # display function and method names with parentheses
add_module_names = False  # don't include module names before object names
pygments_style = "sphinx"  # style for highlighting of source code
bibtex_bibfiles = ["references.bib"]

# set automodapi options
automodapi_toctreedirnm = (
    "autodoc"  # relative path to automodapi generated documentation
)
automodapi_writereprocessed = False  # special flag for automodapi debugging
typehints_fully_qualified = False  # don't include module names before object names
always_document_param_types = True
typehints_document_rtype = True  # adds :rtype: directives for autodoc type hints

automodsumm_inherited_members = (
    False  # don't include inherited members in documentations for classes
)

# set options for pydantic models
autodoc_pydantic_model_show_json = (
    False  # don't include JSON schema for pydantic models
)
autodoc_pydantic_model_show_field_summary = (
    False  # don't include a bullet-point list of model fields
)
autodoc_pydantic_model_show_config_summary = (
    False  # don't include model configurations for pydantic models
)
autodoc_pydantic_field_list_validators = (
    False  # don't list validators for pydantic model fields
)
autodoc_pydantic_field_show_constraints = (
    False  # don't list constraints for pydantic model fields
)
autodoc_pydantic_model_show_validator_summary = (
    False  # dont' include validator methods for pydantic models
)
autodoc_pydantic_model_show_validator_members = (
    False  # don't include documentation for validator methods
)

autodoc_mock_imports = [
    "pydantic",
    "ocelot",
    "cheetah",
    ]

autodoc_default_options = {
    "exclude-members": ",".join([
        "model_post_init",
        "model_fields",
        "model_config",
        "model_computed_fields",
        "model_dump",
        "model_dump_json",
        "model_validate",
        "model_validate_json",
        "model_copy",
        "model_json_schema",
        "construct",
        "schema",
        "schema_json",
        "dict",
        "json",
    ]),
    "undoc-members": False,
    "inherited-members": False,
}


autodoc_pydantic_settings_show_config_summary = (
    False
)

autodoc_pydantic_field_list_validators = False
autodoc_pydantic_model_show_validator_members = False

nb_execution_mode = "off"  # options: "off", "auto", "force"


# mapping to other projects' documentation pages
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# set HTML output options
html_theme = "sphinx_rtd_theme"  # set HTML themse (read the docs theme)
html_logo = "icon.png"  # set logo for top-left corner of HTML pages
html_favicon = "favicon.ico"  # set HTML favicon
html_static_path = ["_static"]  # path to custom static files

# set mathjax equation rendering options
mathjax3_config = {"chtml": {"displayAlign": "left", "displayIndent": "2em"}}

# set date and time format string
today_fmt = "%A %B %d %Y (%H:%M:%S)"
