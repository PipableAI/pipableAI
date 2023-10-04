# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# Set the theme to 'sphinx_rtd_theme'
html_theme = "sphinx_rtd_theme"

project = "Pipable"
copyright = "2023, Pipable"
author = "Pipable"
release = "1.0.0"

# Configure autodoc_default_options as needed
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme", "sphinx.ext.githubpages"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
