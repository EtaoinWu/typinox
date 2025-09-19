from typing import Any

import chex
import jax
from beartype.door import is_bearable
from jax import numpy as jnp
from jaxtyping import Array, Shaped

import typinox as tpx
from typinox import Vmapped


def my_is_bearable(x: Any, T) -> bool:  # noqa: F821
    """To make pyright happy."""
    return is_bearable(x, T)


def test_unstack_simple():
    t = jnp.arange(6).reshape(3, 2)
    u = jnp.arange(45).reshape(3, 5, 3)
    l = (t, u)
    s = list(tpx.tree.unstack(l))
    assert len(s) == 3
    assert my_is_bearable(
        s, list[tuple[Shaped[Array, " 2"], Shaped[Array, " 5 3"]]]
    )


def test_stack_simple():
    l = [
        (
            jnp.arange(4).reshape(2, 2),
            {"a": jnp.arange(3).reshape(1, 1, 3, 1, 1)},
        )
        for _ in range(5)
    ]
    s = tpx.tree.stack(l)
    assert s[0].shape == (5, 2, 2)
    assert s[1]["a"].shape == (5, 1, 1, 3, 1, 1)
    assert my_is_bearable(
        s,
        Vmapped[
            tuple[
                Shaped[Array, " 2 2"], dict[str, Shaped[Array, " 1 1 3 1 1"]]
            ],
            "n",
        ],
    )


def test_stack_unstack():
    def f(x):
        return {
            "a": x,
            "b": [
                jnp.arange(3).reshape(1, 1, 3, 1, 1) + x,
                (jnp.arange(12).reshape(2, 3, 1, 2) - x).astype(jnp.float16),
                (
                    (jnp.arange(6) - x).reshape(2, 3),
                    jnp.sin(jnp.arange(10)) * x,
                ),
            ],
        }

    xs = jnp.arange(10)
    xl = [float(x) for x in range(10)]

    ys = jax.vmap(f)(xs)
    yl = [f(x) for x in xl]

    chex.assert_trees_all_equal(ys, tpx.tree.stack(tpx.tree.unstack(ys)))
    chex.assert_trees_all_equal(yl, list(tpx.tree.unstack(tpx.tree.stack(yl))))

    chex.assert_trees_all_equal(ys, tpx.tree.stack(yl))
    chex.assert_trees_all_equal(yl, list(tpx.tree.unstack(ys)))
