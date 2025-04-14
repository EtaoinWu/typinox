.. meta::
    :description lang=en:
        Typinox is an open-source Python library for enhancing run-time type-checking of
        jax arrays and Equinox modules.

Typinox
=======

**Typinox** (TAI-pee-nox) is a Python library for enhancing run-time type-checking of
``jaxtyping``-annotated arrays and :class:`equinox.Module` s.

.. note::

    Typinox is currently in very early stages and is not yet ready for general use.
    The documentation is also a work in progress.

Installation
------------

To use Typinox, first install it using pip:

.. code-block:: console

    $ pip install typinox

Python 3.12 or later is required.

Basic usage
-----------

Typinox has two main components: :mod:`typinox.vmapped`,
providing a :class:`Vmapped <typinox.VmappedT>` annotation
for :func:`vmap <jax.vmap>`-compatible functions,
and :mod:`typinox.module`,
providing a :class:`TypedModule <typinox.TypedModule>` class for run-time type-checking of Equinox modules.

With ``Vmapped[T, "dims"]`` you can annotate variables that are valid arguments to a
:func:`vmap <jax.vmap>`-ed function. For example, if

.. code-block:: python

    T = tuple[Float[Array, "3"], Float[Array, "2"]]
    (jnp.zeros(3), jnp.ones(2)) : T

Then ``Vmapped[T, "n"]`` is equivalent to ``tuple[Float[Array, "n 3"], Float[Array, "n 2"]]``:

.. code-block:: python

    assert isinstance((jnp.zeros((4, 3)), jnp.ones((4, 2))), Vmapped[T, "n"])

And you can use it with :func:`vmap() <jax.vmap>`:

.. code-block:: python

    def my_function(_) -> T:
        return (jnp.zeros(3), jnp.ones(2))

    a = jax.vmap(my_function)(jnp.arange(4))
    assert isinstance(a, Vmapped[T, "n"])

:class:`TypedModule <typinox.TypedModule>` is an extension of :class:`equinox.Module`
with automatic type-checking. It uses :func:`jaxtyped() <jaxtyping.jaxtyped>`
and :func:`beartype() <beartype.beartype>` to check method calls and return values.

.. code-block:: python

    class AffineMap(TypedModule):
        k: Float[Array, "n m"]
        b: Float[Array, "n"]

        def __call__(self: Self, x: Float[Array, "m"]) -> Float[Array, "n"]:
            return jnp.dot(self.k, x) + self.b

    f = AffineMap(k=jnp.ones((3, 2)).astype(float), b=jnp.zeros(3))

:class:`TypedModule <typinox.TypedModule>`\s are designed to
work with :func:`Vmapped <typinox.VmappedT>` perfectly. If ``f()`` returns
``AffineMap``, then ``jax.vmap(f)`` returns ``Vmapped[AffineMap, "n"]``.

Check out the :doc:`usage/index` section for further information.

Dependencies
------------

Typinox aggressively tracks the latest versions of its dependencies.
It currently depends on:

- Python 3.12 (for `PEP 695`_ syntax)
- ``beartype`` 0.20.0
- ``jaxtyping`` 0.2.38
- ``equinox`` 0.11.12

Typinox may drop support for older versions of these dependencies
if newer ones provide any benefits that Typinox can leverage.

.. _PEP 695: https://peps.python.org/pep-0695/

.. toctree::
    :maxdepth: 2

    self
    Using Typinox <usage/index>
    annotation_best_practice
    API Reference <api/index>
    Development <development>
