.. meta::
    :description lang=en:
        Typinox is an open-source Python library for enhancing run-time type-checking of
        jax arrays and Equinox modules.

Typinox
=======

**Typinox** (TAI-pee-nox) is a Python library for enhancing run-time type-checking of
``jaxtyping``-annotated arrays and :py:class:`equinox.Module` s. 

.. note::

    Typinox is currently in very early stages and is not yet ready for general use.
    The documentation is also a work in progress.

Installation
------------

To use Typinox, first install it using pip:

.. code-block:: console

    $ pip install git+https://github.com/EtaoinWu/typinox.git

Python 3.12 or later is required.

Basic usage
-----------

Typinox has two main components: :py:mod:`typinox.vmapped`, providing a :py:class:`VmappedT`
annotation for :py:func:`jax.vmap`-compatible functions, and :py:mod:`typinox.module`, providing
a :py:class:`typinox.TypedModule` class for run-time type-checking of Equinox modules.

To use , simply import it and use it as an annotation for your function:

.. code-block:: python

    import jax
    from jax import numpy as jnp
    from jaxtyping import Float, Array, jaxtyped
    from beartype import beartype
    from typinox.vmapped import VmappedT

    ArrayOfTwo = Float[Array, " 2"]

    @jaxtyped(typechecker=beartype)
    def my_function(x: ArrayOfTwo, y: ArrayOfTwo) -> ArrayOfTwo:
        return x + y

    @jaxtyped(typechecker=beartype)
    def my_vmapped(x: VmappedT[ArrayOfTwo, " n"],
                   y: VmappedT[ArrayOfTwo, " n"]
                 ) -> VmappedT[ArrayOfTwo, " n"]:
        return jax.vmap(my_function)(x, y)

    print(my_vmapped(jnp.ones((3, 2)), jnp.ones((3, 2))))

    my_vmapped(jnp.ones((3, 2)), jnp.ones((4, 2)))  # raises a TypeError

To use :py:class:`TypedModule`, subclass it and use it in place of :py:class:`equinox.Module`.
You will then automatically get runtime type-checking via :py:func:`jaxtyping.jaxtyped`
and :py:func:`beartype.beartype`.

.. code-block:: python

    from jax import numpy as jnp
    from jaxtyping import Float, Array, jaxtyped
    from beartype import beartype
    from equinox import Module
    from typing import Self
    from typinox.module import TypedModule

    class AffineMap(TypedModule): # also known as linear layer
        k: Float[Array, "n m"]
        b: Float[Array, "n"]

        def __call__(self: Self, x: Float[Array, "m"]) -> Float[Array, "n"]:
            return jnp.dot(self.k, x) + self.b

        def compose(self, other: Self) -> Self:  # Self annotation is supported!
            return self.__class__(k=jnp.dot(self.k, other.k), b=self.b + jnp.dot(self.k, other.b))

    f1 = AffineMap(k=jnp.arange(6).reshape((3, 2)).astype(float), b=jnp.ones(3))
    f2 = AffineMap(k=jnp.ones((5, 3)) / 18, b=jnp.ones(5))
    
    print(f1(jnp.ones(2)))
    print(f2.compose(f1)(jnp.ones(2)))

    f1(jnp.ones(3))  # raises a TypeError

Check out the :doc:`usage` section for further information.

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
    usage
    annotation_best_practice
    api/index
    development

