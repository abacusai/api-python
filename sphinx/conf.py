import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'AbacusAI'
copyright = '2022, Abacus.ai'
author = 'Abacus.ai'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

autoapi_type = 'python'
autoapi_dirs = [
    '../abacusai',
]
autoapi_ignore = [
    '*tests/*',
    '*experimental/*',
    '*notebook*',
    '*jupyter*',
    '*customers*',
]
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'autoapi.extension'
]
autodoc_typehints = 'description'
autoapi_add_toctree_entry = False
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}
pdf_documents = [('index', u'rst2pdf', u'pydocs.pdf', u'AbacusAi')]
latex_documents = [('index', u'doc.tex', u'pydocs.tex', u'AbacusAi', 'manual')]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
