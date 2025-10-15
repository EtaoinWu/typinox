import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Bool, Float, Scalar, jaxtyped

from typinox import (
    TypedModule,
    set_debug_mode,
)

set_debug_mode(True)


class Foo(TypedModule):
    x: Float[Array, "n m"]

    def equal(self, y: Float[Array, "n m"]) -> Bool[Scalar, ""]:
        return jnp.array_equal(self.x, y)

    def equal_val(self, y: Float[Array, "n m"]) -> Bool[Scalar, ""]:
        self._validate()
        return jnp.array_equal(self.x, y)


@jaxtyped(typechecker=beartype)
def foo_equal(foo: Foo, y: Float[Array, "n m"]) -> Bool[Scalar, ""]:
    return jnp.array_equal(foo.x, y)


def test_module_constraint_scope():
    foo = Foo(x=jnp.zeros((3, 4)))
    assert foo_equal(foo, jnp.zeros((3, 4)))
    assert foo.equal(jnp.zeros((3, 4)))
    assert foo.equal_val(jnp.zeros((3, 4)))

    assert not foo_equal(foo, jnp.ones((3, 4)))
    assert not foo.equal(jnp.ones((3, 4)))
    assert not foo.equal_val(jnp.ones((3, 4)))

    assert not foo_equal(foo, jnp.zeros((3, 5)))
    assert not foo.equal(jnp.zeros((3, 5)))
    with pytest.raises(TypeError, match=r"Float\[Array, 'n m'\]"):
        foo.equal_val(jnp.zeros((3, 5)))
