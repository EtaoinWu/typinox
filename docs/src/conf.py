# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Typinox"
copyright = "2025, Etaoin Wu"
author = "Etaoin Wu"
release = "0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    # 'autoapi.extension',
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]


# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "beartype": ("https://beartype.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
