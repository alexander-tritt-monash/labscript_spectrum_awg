# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("../../"))
# sys.path.insert(0, os.path.abspath('../../'))
# sys.path.append(os.path.abspath("../../../"))


# -- Project information -----------------------------------------------------

import sphinx_rtd_theme

project = 'Spectrum AWG'
copyright = '2023, Alex Tritt'
author = 'Alex Tritt'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	"sphinx.ext.autodoc",
  "sphinx_rtd_theme",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
]

# # Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
add_module_names = False
autodoc_mock_imports = ["labscript", "labscript_devices", "blacs.tab_base_classes", "blacs.device_base_class", "labscript_utils.h5_lock", "blacs", "labscript_utils"]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intersphinx_mapping = {
    "python"        : ("https://docs.python.org/3",                       None),
    "numpy"         : ("https://numpy.org/doc/stable/",                   None),
    "labscript"     : ("https://docs.labscriptsuite.org/en/latest/",      None),
    "h5py"          : ("https://docs.h5py.org/en/stable/",                None),
    "spectrum_card" : ("https://spectrum-card.readthedocs.io/en/latest/", None)
}