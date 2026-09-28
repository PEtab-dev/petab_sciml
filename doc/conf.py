# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import re

# Copy CONTRIBUTING file here to avoid access issues.
import shutil
from pathlib import Path

import sphinx

doc_path = Path(__file__).resolve().parent
tmp_path = doc_path / "_tmp"
tmp_path.mkdir(exist_ok=True, parents=True)
shutil.copy(doc_path.parent / "CONTRIBUTING.md", tmp_path / "CONTRIBUTING.md")

_ADMONITION_RE = re.compile(r"^!!!\s+\S+")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PEtab SciML"
copyright = "2025, The PEtab SciML developers"
author = "The PEtab SciML developers"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_autodoc_typehints",
    # markdown files
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


intersphinx_mapping = {
    "petab": (
        "https://petab.readthedocs.io/projects/libpetab-python/en/latest/",
        None,
    ),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "python": ("https://docs.python.org/3", None),
}

autosummary_generate = True
autodoc_default_options = {
    "special-members": "__init__",
    "inherited-members": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["standard"]
html_title = "PEtab SciML"
# html_logo = "logo/logo-wide.svg"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PEtab-dev/petab_sciml",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "image_light": "logo/logo_light.png",
        "image_dark": "logo/logo_dark.png",
        "alt_text": "PEtab SciML",
    },
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 1,
    "navigation_depth": 4,
    "show_toc_level": 3,
}

html_context = {
    "default_mode": "light",
}


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Exclude some objects from the documentation."""
    if inspect.isbuiltin(obj):
        return True

    # Skip inherited members from builtins
    #  (skips, for example, all the int/str-derived methods of enums
    if (
        objclass := getattr(obj, "__objclass__", None)
    ) and objclass.__module__ == "builtins":
        return True

    return None

def strip_mkdocs_admonitions(app, what, name, obj, options, lines):
    cleaned = []
    in_admonition = False
    for line in lines:
        if _ADMONITION_RE.match(line.strip()):
            in_admonition = True
            continue
        if in_admonition:
            if line.strip() == "" or line.startswith((" ", "\t")):
                continue
            in_admonition = False
        cleaned.append(line)
    lines[:] = cleaned


def setup(app: sphinx.application.Sphinx):
    app.connect("autodoc-skip-member", autodoc_skip_member, priority=0)
    app.connect("autodoc-process-docstring", strip_mkdocs_admonitions, priority=0)