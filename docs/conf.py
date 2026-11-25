"""Sphinx configuration for the complexus documentation."""

import os
import sys

# The project root (the folder that contains ``measures/`` and ``utils/``)
# must be importable so that autodoc can find the modules.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "complexus"
author = "Marco Antonio Reséndiz Díaz"
copyright = "2026, Marco Antonio Reséndiz Díaz"
release = "dev"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # reads the docstrings
    "sphinx.ext.autosummary",  # summary tables of the API
    "numpydoc",  # understands the NumPy docstring style
    "sphinx.ext.mathjax",  # renders the formulas
    "sphinx.ext.viewcode",  # [source] links next to each object
    "sphinx.ext.intersphinx",  # links to NumPy, Python, scikit-learn docs
]

autosummary_generate = True
autodoc_typehints = "none"  # types are documented in the numpydoc sections
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "complexus"
html_theme_options = {
    "navigation_with_keys": False,
}
