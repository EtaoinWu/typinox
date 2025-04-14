Custom Type Validator
=====================

Typinox supports custom type validators. It is useful to check the internal consistency
of an object.

They are used in a similar way to
``__check_init__()`` methods in :class:`equinox.Module`\s.
Since PyTree operations does not call the ``__check_init__`` method,
the result of an operation like :func:`jax.vmap` or :func:`equinox.filter` may
be invalid, and it is dangerous to call methods on it.

For a :class:`TypedModule`,
the ``__validate__()`` method is called when another method requires to validate
the module itself. For example, with the following module:

.. code-block:: python

    from typing import Self
    import jax.numpy as jnp
    from jaxtyping import Float, Array

    import typinox as tpx
    from typinox import ValidationFailed

    class SquareMat(tpx.TypedModule):
        mat: Float[Array, "n n"]
        n: int = tpx.field(static=True)

        def __validate__(self):
            if self.n != self.mat.shape[0]:
                raise ValidationFailed(f"n={self.n} does not match mat.shape[0]={self.mat.shape[0]}")

        def diagonal_plus_one(self: Self):
            return jnp.diagonal(self.mat) + 1

With the explicit ``Self`` annotation,
the ``diagonal_plus_one`` method will validate ``self`` before executing.

.. code-block:: python

    f = SquareMat(mat=jnp.eye(3), n=3)
    f.diagonal_plus_one()  # works

    _ = SquareMat(mat=jnp.eye(3), n=4)  # fails

    f = jax.tree.map(lambda x: x.reshape((9, 1)), f)
    # here f.mat has shape (9, 1) and f.n is 3
    f.diagonal_plus_one()  # fails

There are three ways to define a custom validator:

1. method named ``__validate__()`` returning ``bool``.
    If it returns ``False``, the validation fails.
2. method named ``__validate__()`` returning ``None``.
    It raise a :class:`ValidationFailed <typinox.error.ValidationFailed>` exception if the validation fails.
3. method named ``__validate_str__()`` returning ``str``.
    If it returns an empty string, the validation passes.
    The returned non-empty string will be used as the error message.

.. note::
    When both methods are present, both needs to pass for the validation to succeed.
    This is not recommended due to the potential for confusion.

When type-checking an object, each of its base classes are checked in order.
All of them need to pass for the object to be considered valid.

Validating outside the class scope
----------------------------------

If you want to validate an object outside the class scope,
you can use the :func:`ValidateT <typinox.ValidateT>` type annotation.
In the previous example, the value of ``f`` is invalid after the reshape operation.
Therefore,

.. code-block:: python

    from beartype.door import is_bearable

    is_bearable(f, SquareMat)  # True
    is_bearable(f, ValidateT[SquareMat])  # False
