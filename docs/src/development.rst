Development
===========

To run the tests, you will need ``pytest``, ``pytest-cov`` and ``chex``.
They can be installed with the ``dev`` option:

.. _install-dev:

.. code-block:: bash

    poetry install --with dev

And the tests can be run with:

.. code-block:: bash

    poetry run pytest ./test

Linting
-------

To lint the code, you will need ``ruff``, ``pyright`` and ``mypy``.
They can also be :ref:`installed <install-dev>` with the ``dev`` option.
Usage:

.. code-block:: bash

    poetry run ruff check
    poetry run ruff format --check
    poetry run pyright .
    poetry run mypy .

Documentation
-------------

To build the documentation, you will need ``sphinx``. It is :ref:`included <install-dev>` in the ``dev`` option.
The documentation can be built with:

.. code-block:: bash

    poetry run make -C docs html

With ``sphinx-autobuild``, you can also build the documentation and serve it locally with:

.. code-block:: bash

    poetry run make -C docs serve
