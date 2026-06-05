Development
===========

To run the tests, you will need ``pytest``, ``pytest-cov`` and ``chex``.
They can be installed with the ``dev`` option:

.. _install-dev:

.. code-block:: bash

    uv sync --group dev

And the tests can be run with:

.. code-block:: bash

    uv run pytest ./test

Linting
-------

To lint the code, you will need ``ruff``, ``pyright`` and ``mypy``.
They can also be :ref:`installed <install-dev>` with the ``dev`` option.
Usage:

.. code-block:: bash

    uv run ruff check
    uv run ruff format --check
    uv run pyright .
    uv run mypy .

Documentation
-------------

To build the documentation, you will need ``sphinx``. It is :ref:`included <install-dev>` in the ``dev`` option.
The documentation can be built with:

.. code-block:: bash

    uv run make -C docs html

With ``sphinx-autobuild``, you can also build the documentation and serve it locally with:

.. code-block:: bash

    uv run make -C docs serve
