Development
===========

To run the tests, you will need `pytest`, `pytest-cov` and `chex`.
They can be installed with the `dev` option:

.. code-block:: bash

    poetry install --with dev

To build the documentation, you will need `sphinx`. It is included in the `dev` option.
The documentation can be built with:

.. code-block:: bash

    poetry run make -C docs html

With `sphinx-autobuild`, you can also build the documentation and serve it locally with:

.. code-block:: bash

    poetry run make -C docs serve
