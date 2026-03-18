Read the Docs Publishing
========================

What to commit
--------------

Commit the docs source only:

* ``docs/``
* ``docs/conf.py``
* ``docs/requirements.txt``
* ``.readthedocs.yaml``

Do not commit generated HTML:

* ``docs/_build/``

The repository is configured to ignore ``docs/_build/`` so local Sphinx builds
do not pollute the Git tree.

How Read the Docs works here
----------------------------

This repository is already prepared for a standard Read the Docs build:

* ``.readthedocs.yaml`` points Read the Docs to ``docs/conf.py``
* ``docs/requirements.txt`` installs the documentation dependencies
* ``docs/index.rst`` is the Sphinx entry point

What you need to do on GitHub
-----------------------------

1. Push the repository changes to GitHub.
2. Make sure the default branch contains:

   * ``docs/``
   * ``.readthedocs.yaml``

3. Do not push ``docs/_build/html``.

What you need to do on Read the Docs
------------------------------------

1. Sign in to Read the Docs with your Git provider.
2. Import this GitHub repository.
3. Confirm that the project uses the repository-root ``.readthedocs.yaml``.
4. Trigger the first build.
5. After the first successful build, Read the Docs will host the generated HTML.

Expected result
---------------

After import, the hosted docs site is built by Read the Docs directly from the
repository source. You do not need to commit generated HTML files.

Local verification
------------------

If you want to verify the docs before pushing:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

Then open the generated local entry file in a browser:

* ``docs/_build/html/index.html``

Operational notes
-----------------

* Read the Docs is the recommended hosted docs path for this project.
* GitHub Pages is still possible, but that would be a separate deployment
  path and a different hosted URL.
