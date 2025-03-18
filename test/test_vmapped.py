import chex
import equinox as eqx
import jax
import numpy as np
import pytest
from beartype import beartype as typechecker
from beartype.door import is_bearable
from jax import (
    numpy as jnp,
)
from jaxtyping import Array, Float, Integer, Scalar, Shaped, jaxtyped
from numpy import ndarray

from typinox.vmapped import VmappedI, VmappedT


def t(a, b):
    return is_bearable(a, b)


@pytest.mark.parametrize("Vmapped", [VmappedI, VmappedT])
def test_vmapped_basic(Vmapped):
    """Test the basic shape handling of Vmapped."""
    arr3 = jnp.array([1, 2, 3])
    arr34 = jnp.arange(12).reshape(3, 4)
    narr34 = np.arange(12).reshape(3, 4)
    arr345 = jnp.arange(60).reshape(3, 4, 5)

    assert t(arr3, Integer[Array, " 3"])
    assert t(arr34, Integer[Array, "3 4"])
    assert t(arr345, Integer[Array, "3 4 5"])

    assert t(arr3, Vmapped[Integer[Scalar, ""], " 3"])
    assert t(arr3, Vmapped[Integer[Array, " 3"], ""])

    assert not t(arr3, Vmapped[Integer[Scalar, ""], " 4"])
    assert not t(arr3, Vmapped[Integer[Array, " 4"], ""])

    assert not t(arr3, Vmapped[Float[Scalar, ""], " 3"])
    assert not t(arr3, Vmapped[Float[Scalar, " 3"], ""])

    assert not t(arr3, Shaped[Array, "3 3"])
    assert not t(arr34, Shaped[Array, "3 3"])
    assert not t(arr3, Shaped[Array, "n n"])
    assert not t(arr34, Shaped[Array, "n n"])

    assert not t(arr3, Vmapped[Shaped[Array, " 3"], " 3"])
    assert not t(arr34, Vmapped[Shaped[Array, " 3"], " 3"])

    assert not t(arr3, Vmapped[Shaped[Array, "n"], " 3"])
    assert not t(arr34, Vmapped[Shaped[Array, "n"], " 4"])
    assert not t(arr34, Vmapped[Shaped[Array, "n"], "m q"])
    assert not t(arr34, Vmapped[Shaped[Array, "n m"], "q"])

    assert t(arr345, Vmapped[Shaped[Array, "n"], "m q"])
    assert t(arr345, Vmapped[Shaped[Array, "n m"], "q"])
    assert not t(arr345, Vmapped[Shaped[Array, "n"], "m"])
    assert not t(arr345, Vmapped[Shaped[Array, ""], "n m q r"])
    assert not t(arr345, Vmapped[Shaped[Array, " n"], "m q r"])
    assert not t(arr345, Vmapped[Shaped[Array, "n m"], "q r"])
    assert not t(arr345, Vmapped[Shaped[Array, "n m q"], " r"])
    assert not t(arr345, Vmapped[Shaped[Array, "n m q r"], ""])

    with jaxtyped("context"):  # type: ignore
        assert not t(arr3, Vmapped[Shaped[Array, " n"], " n"])

    with jaxtyped("context"):  # type: ignore
        assert not t(arr34, Vmapped[Shaped[Array, " n"], " n"])

    with jaxtyped("context"):  # type: ignore
        # n=3, m=4
        assert t(arr3, Integer[Array, " n"])

        assert t(arr3, Vmapped[Integer[Scalar, ""], " n"])
        assert t(arr3, Vmapped[Integer[Array, "n"], ""])

        assert t(arr34, Integer[Array, "n m"])
        assert t(arr34, Vmapped[Integer[Array, "m"], "n"])
        assert t(arr34, Vmapped[Integer[Array, "m"], "n $"])
        assert t(arr34, Vmapped[Integer[Array, "n"], "$ m"])
        assert t(arr34, Vmapped[Integer[Scalar, ""], "n $ m"])

        assert t(arr34, Vmapped[Vmapped[Integer[Scalar, ""], "n $"], "$ m"])

        assert t(narr34, Integer[ndarray, "n m"])
        assert t(narr34, Vmapped[Integer[ndarray, "m"], "n"])
        assert t(narr34, Vmapped[Integer[ndarray, "m"], "n $"])
        assert t(narr34, Vmapped[Integer[ndarray, "n"], "$ m"])


@pytest.mark.parametrize("Vmapped", [VmappedI, VmappedT])
def test_vmapped_large(Vmapped):
    """Test the basic shape handling of Vmapped with more dimensions."""
    arr3345 = jnp.arange(180).reshape(3, 3, 4, 5)
    arr4335 = jnp.einsum("abcd->cbad", arr3345)

    assert t(arr3345, Integer[Array, "3 3 4 5"])
    assert t(arr4335, Integer[Array, "4 3 3 5"])

    assert t(arr3345, Vmapped[Integer[Array, "3 3"], "$ 4 5"])
    assert t(arr4335, Vmapped[Integer[Array, "3 3"], "4 $ 5"])
    assert not t(arr4335, Vmapped[Integer[Array, "3 3"], "$ 4 5"])
    assert not t(arr3345, Vmapped[Integer[Array, "3 3"], "4 $ 5"])

    assert t(arr3345, Vmapped[Integer[Array, "n n"], "$ 4 5"])
    assert t(arr4335, Vmapped[Integer[Array, "n n"], "4 $ 5"])
    assert not t(arr4335, Vmapped[Integer[Array, "n n"], "$ 4 5"])
    assert not t(arr3345, Vmapped[Integer[Array, "n n"], "4 $ 5"])

    assert t(arr3345, Vmapped[Integer[Array, "m q"], "n n $"])
    assert not t(arr4335, Vmapped[Integer[Array, "m q"], "n n $"])

    with jaxtyped("context"):  # type: ignore
        assert t(arr4335, Vmapped[Integer[Array, "n n"], "m $ q"])
    with jaxtyped("context"):  # type: ignore
        assert not t(arr3345, Vmapped[Integer[Array, "n n"], "m $ q"])

    with jaxtyped("context"):  # type: ignore
        assert t(arr4335, Vmapped[Integer[Array, "n q"], "m n $"])
    with jaxtyped("context"):  # type: ignore
        assert not t(arr3345, Vmapped[Integer[Array, "n q"], "m n $"])

    with jaxtyped("context"):  # type: ignore
        assert t(arr4335, Vmapped[Integer[Array, "n q"], "m n"])
    with jaxtyped("context"):  # type: ignore
        assert not t(arr3345, Vmapped[Integer[Array, "n q"], "m n"])

    with jaxtyped("context"):  # type: ignore
        assert t(arr4335, Vmapped[Integer[Array, "m n"], "$ n q"])
    with jaxtyped("context"):  # type: ignore
        assert not t(arr3345, Vmapped[Integer[Array, "m n"], "$ n q"])


@pytest.mark.parametrize("Vmapped", [VmappedT])
def test_vmapped_vmap_simple(Vmapped):
    """Test Vmapped with vmap using a basic function."""

    @eqx.filter_jit
    def inner(x: Float[Scalar, ""]) -> tuple[str, Float[Array, "3"]]:
        y = x * jnp.ones(3)
        assert t(y, Float[Array, "3"])
        return "aaa", y

    xs = jnp.sin(jnp.arange(5))
    ys = eqx.filter_vmap(inner)(xs)
    with jaxtyped("context"):  # type: ignore
        assert t(xs, Vmapped[inner.__annotations__["x"], "n"])
        assert t(ys, Vmapped[inner.__annotations__["return"], "n"])

    xss = jnp.sin(jnp.arange(6).reshape(2, 3))
    yss = eqx.filter_vmap(eqx.filter_vmap(inner))(xss)
    with jaxtyped("context"):  # type: ignore
        assert t(xss, Vmapped[inner.__annotations__["x"], "m n"])
        assert t(yss, Vmapped[inner.__annotations__["return"], "m n"])
    with jaxtyped("context"):  # type: ignore
        assert not t(xss, Vmapped[inner.__annotations__["x"], "n n"])
        assert not t(yss, Vmapped[inner.__annotations__["return"], "n n"])

    xss = jnp.sin(jnp.arange(9).reshape(3, 3))
    yss = eqx.filter_vmap(eqx.filter_vmap(inner))(xss)
    with jaxtyped("context"):  # type: ignore
        assert t(xss, Vmapped[inner.__annotations__["x"], "n n"])
        assert t(yss, Vmapped[inner.__annotations__["return"], "n n"])


@pytest.mark.parametrize("Vmapped", [VmappedI, VmappedT])
def test_vmapped_vmap_axes(Vmapped):
    """Test Vmapped with vmap using a function with in_axes and out_axes."""

    @jax.jit
    def inner(
        x: Float[Scalar, ""], y: Float[Array, " n"]
    ) -> Float[Array, "n n"]:
        z = (y[:, None] + x) * y[None, :]
        return jnp.sin(z) / x

    x = jnp.array(2.0)
    ys = jnp.arange(12).astype(float).reshape(3, 4)

    zs = jax.vmap(inner, in_axes=(None, 0))(x, ys)
    with jaxtyped("context"):  # type: ignore
        assert t(x, Vmapped[inner.__annotations__["x"], ""])
        assert t(ys, Vmapped[inner.__annotations__["y"], "m"])
        assert t(zs, Vmapped[inner.__annotations__["return"], "m"])

    zs = jax.vmap(inner, in_axes=(None, 1), out_axes=2)(x, ys)
    with jaxtyped("context"):  # type: ignore
        assert t(x, Vmapped[inner.__annotations__["x"], ""])
        assert t(ys, Vmapped[inner.__annotations__["y"], "$ m"])
        assert t(zs, Vmapped[inner.__annotations__["return"], "$ m"])

    zs = jax.vmap(inner, in_axes=(None, -1), out_axes=-1)(x, ys)
    with jaxtyped("context"):  # type: ignore
        assert t(x, Vmapped[inner.__annotations__["x"], ""])
        assert t(ys, Vmapped[inner.__annotations__["y"], "$ m"])
        assert t(zs, Vmapped[inner.__annotations__["return"], "$ m"])

    xs = jnp.arange(5).astype(float)
    ys = jnp.arange(4).astype(float)
    zs = jax.vmap(inner, in_axes=(0, None), out_axes=0)(xs, ys)
    with jaxtyped("context"):  # type: ignore
        assert t(xs, Vmapped[inner.__annotations__["x"], "m"])
        assert t(ys, Vmapped[inner.__annotations__["y"], ""])
        assert t(zs, Vmapped[inner.__annotations__["return"], "m"])


ArrayOfTwo = Float[Array, " 2"]


def test_doc_example():
    @jaxtyped(typechecker=typechecker)
    def my_function(x: ArrayOfTwo, y: ArrayOfTwo) -> ArrayOfTwo:
        return x + y

    @jaxtyped(typechecker=typechecker)
    def my_vmapped(
        x: VmappedT[ArrayOfTwo, " n"], y: VmappedT[ArrayOfTwo, " n"]
    ) -> VmappedT[ArrayOfTwo, " n"]:
        return jax.vmap(my_function)(x, y)

    chex.assert_trees_all_close(
        my_vmapped(jnp.ones((3, 2)), jnp.ones((3, 2))), jnp.ones((3, 2)) * 2
    )

    with pytest.raises(TypeError):
        my_vmapped(jnp.ones((3, 2)), jnp.ones((4, 2)))
