# conf.py

# -- Project information -----------------------------------------------------
project = 'Mi Proyecto Python'
copyright = '2025, Mi Nombre'
author = 'Mi Nombre'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',  # Usa la guía de estilo de Google
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'  # El tema no importa mucho si GitHub Pages usa otro
html_static_path = ['_static']

# Directorio de salida
output_dir = '_build'

# Configura Napoleon para Google Style
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = False
napoleon_use_rtype = True
napoleon_use_param = True