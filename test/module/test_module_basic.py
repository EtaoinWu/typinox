from typing import Self

import chex
import numpy as np
import pytest
from beartype.door import die_if_unbearable
from jax import (
    numpy as jnp,
    tree as jt,
)
from jaxtyping import Array, Float, Scalar

import typinox as tpx
from typinox import (
    TypedModule,
    ValidatedT,
    set_debug_mode,
)

set_debug_mode(True)


class Point2D(TypedModule):
    x: Float[Scalar, ""]
    y: Float[Scalar, ""]

    def norm_correct_1(self) -> Float[Scalar, ""]:
        return (self.x**2 + self.y**2) ** 0.5

    norm_correct_1_1 = norm_correct_1

    def norm_correct_2(self) -> Float[Array, "..."]:
        return (self.x**2 + self.y**2) ** 0.5

    def norm_correct_3(self):
        return (self.x**2 + self.y**2) ** 0.5

    def norm_incorrect_1(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5  # type: ignore

    def norm_incorrect_2(self) -> Float[Array, " n"]:
        return (self.x**2 + self.y**2) ** 0.5

    def swap_correct(self) -> Self:
        return self.__class__(x=self.y, y=self.x)

    def swap_incorrect(self) -> Float[Array, " 2"]:
        return self.__class__(x=self.y, y=self.x)  # type: ignore


Point2D_T = ValidatedT[Point2D]


def test_point2d():
    with pytest.raises(TypeError):
        Point2D_T(x=3.0, y=4.0)  # type: ignore

    p = Point2D_T(x=jnp.float32(3.0), y=jnp.float32(4.0))
    assert 4.99 < p.norm_correct_1() < 5.01
    assert 4.99 < p.norm_correct_1_1() < 5.01
    assert 4.99 < p.norm_correct_2() < 5.01
    assert 4.99 < p.norm_correct_3() < 5.01

    with pytest.raises(TypeError):
        p.norm_incorrect_1()

    with pytest.raises(TypeError):
        p.norm_incorrect_2()

    assert p.swap_correct().x == 4.0
    assert p.swap_correct().y == 3.0

    with pytest.raises(TypeError):
        p.swap_incorrect()


class TerriblePoint2D(TypedModule):
    # Why, oh why
    x: Float[Scalar, ""] = tpx.field(typecheck=False)
    y: Float[Scalar, ""] = tpx.field(typecheck=False)

    def norm_whatever(self):
        return (self.x**2 + self.y**2) ** 0.5


TerriblePoint2D_T = ValidatedT[TerriblePoint2D]


def test_field_untypecheck():
    p = TerriblePoint2D(x=jnp.float32(3.0), y=jnp.float32(4.0))
    assert 4.99 < p.norm_whatever() < 5.01
    die_if_unbearable(p, TerriblePoint2D_T)

    p = TerriblePoint2D(x=3, y=4)  # type: ignore
    assert 4.99 < p.norm_whatever() < 5.01
    die_if_unbearable(p, TerriblePoint2D_T)

    p = TerriblePoint2D(x=np.float16(3), y=np.int64(4))  # type: ignore
    assert 4.99 < p.norm_whatever() < 5.01
    die_if_unbearable(p, TerriblePoint2D_T)


class Point3D(Point2D):
    z: Float[Scalar, ""]

    def norm_correct_1(self) -> Float[Scalar, ""]:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def norm_incorrect_1(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5  # type: ignore

    def cycle_correct_1(self) -> Self:
        return self.__class__(x=self.y, y=self.z, z=self.x)

    def cycle_correct_2(self) -> Self:
        leaves, treedef = jt.flatten(self)
        return jt.unflatten(treedef, leaves[1:] + leaves[:1])

    def cycle_incorrect(self) -> Self:
        leaves, treedef = jt.flatten(self)
        return jt.unflatten(treedef, [float(leaf) for leaf in leaves])


def test_subclassing():
    p = Point3D(
        x=jnp.float32(3.0 * 4), y=jnp.float32(4.0 * 4), z=jnp.float32(5.0 * 3)
    )
    assert 24.99 < p.norm_correct_1() < 25.01

    with pytest.raises(TypeError):
        p.norm_incorrect_1()

    assert p.cycle_correct_1().x == 4.0 * 4
    assert p.cycle_correct_2().x == 4.0 * 4


class SquareMat(TypedModule):
    data: Float[Array, "n n"]

    def transpose_correct(self) -> Self:
        return self.__class__(data=self.data.T)

    def transpose_danger_but_correct(self):
        return jt.map(lambda x: x[0], self)

    def transpose_incorrect_1(self) -> Self:
        return self.transpose_danger_but_correct()

    def transpose_incorrect_selfparam(self: Self):
        return jt.map(lambda x: x[0], self.data)


def test_self_check():
    p = Point3D(
        x=jnp.float32(3.0 * 4), y=jnp.float32(4.0 * 4), z=jnp.float32(5.0 * 3)
    )
    with pytest.raises(TypeError):
        p.cycle_incorrect()

    m = SquareMat(data=jnp.eye(3))
    assert m.transpose_correct().data.shape == (3, 3)
    assert m.transpose_danger_but_correct().data.shape == (3,)

    with pytest.raises(TypeError):
        m.transpose_incorrect_1()

    bad_m = m.transpose_danger_but_correct()
    with pytest.raises(TypeError):
        bad_m.transpose_incorrect_selfparam()


def test_doc_example():
    class AffineMap(TypedModule):  # also known as linear layer
        k: Float[Array, "n m"]
        b: Float[Array, " n"]

        def __call__(self, x: Float[Array, " m"]) -> Float[Array, " n"]:
            self._validate()  # populates jaxtyping shape storage
            return jnp.dot(self.k, x) + self.b

        def compose(self, other: Self) -> Self:  # Self annotation is supported!
            return self.__class__(
                k=jnp.dot(self.k, other.k), b=self.b + jnp.dot(self.k, other.b)
            )

    f1 = AffineMap(k=jnp.arange(6).reshape((3, 2)).astype(float), b=jnp.ones(3))
    f2 = AffineMap(k=jnp.ones((5, 3)) / 18, b=jnp.ones(5))

    chex.assert_trees_all_close(f1(jnp.ones(2)), jnp.array([2.0, 6.0, 10.0]))

    with pytest.raises(TypeError):
        f1(jnp.ones(3))

    f3 = f2.compose(f1)
    chex.assert_trees_all_close(f3(jnp.ones(2)), jnp.ones(5) * 2)
