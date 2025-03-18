import jax
import pytest
from beartype.door import is_bearable
from jax import (
    numpy as jnp,
    tree as jt,
)
from jaxtyping import Array, Float, Scalar

import typinox as tpx
from typinox import (
    TypedModule,
    ValidatedT,
    ensure_shape,
    ensure_shape_equal,
    set_debug_mode,
)

set_debug_mode(True)


class MatMul(TypedModule):
    a: Float[Array, "m n"]
    b: Float[Array, "n p"]
    c: Float[Array, "m p"]

    def __validate__(self):
        return self.a.shape[0] % 2 == 0

    def residue(self) -> Float[Array, "m p"]:
        return self.a @ self.b - self.c

    def total_residue(self) -> Float[Scalar, ""]:
        return jnp.sum(self.residue() ** 2)


MatMulT = ValidatedT[MatMul]


def test_init_consistent():
    a = jnp.arange(6).reshape(2, 3) / 3 + 1.8
    b = jnp.arange(15).reshape(3, 5) / 7 - 2.4
    c = a @ b
    m = MatMul(a=a, b=b, c=c)
    assert m.total_residue() <= 0.01

    with pytest.raises(TypeError, match="does not match type hint"):
        m = MatMul(a=a, b=b, c=c.T)

    with pytest.raises(TypeError, match="failed its custom validation"):
        m = MatMul(a=b.T, b=a.T, c=c.T)


class MatMulChain(TypedModule):
    a: int = tpx.field(static=True)
    d: int = tpx.field(static=True)
    left: MatMulT
    right: MatMulT
    final: Float[Array, "a d"]

    def __validate__(self):
        ensure_shape_equal(".left.c", self.left.c, ".right.a", self.right.a)
        ensure_shape(".left.a", self.left.a, "a _")
        ensure_shape(".right.b", self.right.b, "_ d")
        ensure_shape(".a", self.a, "a")
        ensure_shape(".d", self.d, "d")

    def total_residue(self) -> Float[Scalar, ""]:
        return (
            self.left.total_residue()
            + self.right.total_residue()
            + jnp.sum((self.left.c - self.right.a) ** 2)
            + jnp.sum((self.final - self.right.c) ** 2)
        )


def meaningless_array(shape):
    size = int(jnp.prod(jnp.array(shape)))
    return jnp.arange(size).reshape(shape) / size - 0.2


def test_fields():
    a = meaningless_array((2, 5))
    b = meaningless_array((5, 4))
    c = a @ b - 1e-5 * meaningless_array((2, 4))
    d = c + 2.3e-6 * meaningless_array((4, 2)).T
    e = meaningless_array((4, 3))
    f = d @ e + 1.7e-5 * meaningless_array((2, 3))
    good_obj = MatMulChain(
        a=2,
        d=3,
        left=MatMul(a=a, b=b, c=c),
        right=MatMul(a=d, b=e, c=f),
        final=f - 1.2e-6 * meaningless_array((3, 2)).T,
    )
    assert good_obj.total_residue() <= 0.01

    good_matmul = good_obj.left
    bad_matmul = jt.map(lambda x: x.T, good_matmul)
    with pytest.raises(TypeError, match="does not match"):
        _ = MatMulChain(
            a=2,
            d=3,
            left=good_matmul,
            right=bad_matmul,
            final=good_obj.final,
        )

    with pytest.raises(TypeError, match="does not match"):
        _ = MatMulChain(
            a=2,
            d=3,
            left=good_matmul,
            right=good_matmul,
            final=good_obj.final,
        )

    bad_f = meaningless_array((2, 4))
    with pytest.raises(TypeError, match="does not match the named dimensions"):
        _ = MatMulChain(
            a=2,
            d=3,
            left=good_obj.left,
            right=good_obj.right,
            final=bad_f,
        )

    with pytest.raises(TypeError, match="does not match the named dimensions"):
        _ = MatMulChain(
            a=2,
            d=4,
            left=good_obj.left,
            right=good_obj.right,
            final=f,
        )


def test_doc_example():
    from typing import Self

    from typinox import ValidationFailed
    from typinox.error import TypinoxTypeViolation

    class SquareMat(TypedModule):
        mat: Float[Array, "n n"]
        n: int = tpx.field(static=True)

        def __validate__(self):
            if self.n != self.mat.shape[0]:
                raise ValidationFailed(
                    f"n={self.n} does not match mat.shape[0]={self.mat.shape[0]}"
                )

        def diagonal_plus_one(self: Self):
            return jnp.diagonal(self.mat) + 1

    f = SquareMat(mat=jnp.eye(3), n=3)
    f.diagonal_plus_one()  # works

    with pytest.raises(TypinoxTypeViolation):
        _ = SquareMat(mat=jnp.eye(3), n=4)  # fails

    f = jax.tree.map(lambda x: x.reshape((9, 1)), f)
    # here f.mat has shape (9, 1) and f.n is 3
    with pytest.raises(TypeError):
        f.diagonal_plus_one()  # fails

    assert isinstance(f, SquareMat)
    assert not is_bearable(f, ValidatedT[SquareMat])  # type: ignore
