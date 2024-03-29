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

from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels.alpha import LabelStyle as AlphaLabelStyle

sys.path.insert(0, os.path.abspath("../.."))

# Set variable so that todos are shown in local build
on_rtd = os.environ.get("READTHEDOCS") == "True"
if not on_rtd:
    todo_include_todos = True

# -- Project information -----------------------------------------------------

project = "temfpy"
copyright = "2020, dev-team temfpy"
author = "dev-team temfpy"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.doctest",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# Configuration for numpydoc
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"type", "optional", "default"}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for nbsphinx  ----------------------------------------
nbsphinx_execute = "auto"

nbsphinx_prolog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}
.. |binder| image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/OpenSourceEconomics/temfpy/master?filepath={{ docname|e }}

.. only:: html

    .. nbinfo::

        Download the notebook :download:`here <https://nbviewer.jupyter.org/github/OpenSourceEconomics/temfpy/blob/master/{{ docname }}>`!
        Interactive online version: |binder|
"""
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# We want to remove module-wide dosctrings.
# See https://stackoverflow.com/questions/17927741/exclude-module-docstring-in-autodoc
def remove_module_docstring(app, what, name, obj, options, lines):
    if what == "module":
        del lines[:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_module_docstring)


# -- Options for bibliography style -------------------------------------------------

# We want the top level bibliography to look like in-line bibliography.
# Source: https://stackoverflow.com/a/56030812/10668706


class KeyLabelStyle(AlphaLabelStyle):
    def format_label(self, entry):
        label = entry.key
        return label


class CustomStyle(UnsrtStyle):
    default_sorting_style = "author_year_title"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_style = KeyLabelStyle()
        self.format_labels = self.label_style.format_labels


register_plugin("pybtex.style.formatting", "custom", CustomStyle)

# Set bib file for sphinxcontrib-bibtex
bibtex_bibfiles = ["ref.bib"]
