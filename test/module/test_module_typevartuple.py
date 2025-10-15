import chex
from beartype.typing import Self
from jax import (
    numpy as jnp,
)

from typinox import (
    TypedModule,
    set_debug_mode,
)

set_debug_mode(True)


class MyTuple[*Ts](TypedModule):
    values: tuple[*Ts]

    def set(self, *args: *Ts) -> Self:
        return self.__class__(values=args)


def test_my_tuple():
    t = MyTuple(values=(jnp.array([1.0, 2.0]), jnp.array([[1.0], [2.0]])))
    assert isinstance(t.values, tuple)
    assert len(t.values) == 2
    chex.assert_trees_all_close(t.values[0], jnp.array([1.0, 2.0]))
    chex.assert_trees_all_close(t.values[1], jnp.array([[1.0], [2.0]]))

    t1 = t.set(jnp.array([3.0, 4.0]), jnp.array([[3.0], [4.0]]))
    assert isinstance(t1.values, tuple)
    assert len(t1.values) == 2
    chex.assert_trees_all_close(t1.values[0], jnp.array([3.0, 4.0]))
    chex.assert_trees_all_close(t1.values[1], jnp.array([[3.0], [4.0]]))

    # The original instance is unchanged.
    assert isinstance(t.values, tuple)
    assert len(t.values) == 2
    chex.assert_trees_all_close(t.values[0], jnp.array([1.0, 2.0]))
    chex.assert_trees_all_close(t.values[1], jnp.array([[1.0], [2.0]]))
