Basic Usage
===========

:class:`Vmapped` has a similar notation to jaxtyping's ``Shaped[T, "dims"]``,
but the first argument is not restricted to an array type and can be any PyTree.

.. code-block:: python

    import jax
    from jax import numpy as jnp
    from jaxtyping import Float, Array, jaxtyped
    from beartype import beartype
    from typinox.vmapped import Vmapped

    ArrayOfTwo = Float[Array, " 2"]

    @jaxtyped(typechecker=beartype)
    def my_function(x: ArrayOfTwo, y: ArrayOfTwo) -> ArrayOfTwo:
        return x + y

    @jaxtyped(typechecker=beartype)
    def my_vmapped(x: Vmapped[ArrayOfTwo, " n"],
                   y: Vmapped[ArrayOfTwo, " n"]
                 ) -> Vmapped[ArrayOfTwo, " n"]:
        return jax.vmap(my_function)(x, y)

And you can use them like this:

.. code-block:: python

    print(my_vmapped(jnp.ones((3, 2)), jnp.ones((3, 2))))

    my_vmapped(jnp.ones((3, 2)), jnp.ones((4, 2)))  # raises a TypeError

To use :class:`TypedModule`, subclass it and use it in place of :class:`equinox.Module`.
You will then automatically get runtime type-checking via :func:`jaxtyped <jaxtyping.jaxtyped>`
and :func:`beartype <beartype.beartype>`.

.. code-block:: python

    from typing import Self
    from typinox.module import TypedModule

    class AffineMap(TypedModule): # also known as linear layer
        k: Float[Array, "n m"]
        b: Float[Array, "n"]

        def __call__(self: Self, x: Float[Array, "m"]) -> Float[Array, "n"]:
            return jnp.dot(self.k, x) + self.b

        # Self annotation is supported!
        def compose(self, other: Self) -> Self:
            return self.__class__(k=jnp.dot(self.k, other.k),
                                  b=self.b + jnp.dot(self.k, other.b))

Method calls and return values are automatically checked:

.. code-block:: python

    f1 = AffineMap(k=jnp.arange(6).reshape((3, 2)).astype(float), b=jnp.ones(3))
    f2 = AffineMap(k=jnp.ones((5, 3)) / 18, b=jnp.ones(5))

    print(f1(jnp.ones(2)))
    print(f2.compose(f1)(jnp.ones(2)))

    f1(jnp.ones(3))  # raises a TypeError

.. toctree::
    :hidden:
    :maxdepth: 2

    self
    custom_validator
