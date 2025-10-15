from typing import Any, Self

import jax
import pytest
from beartype.door import is_bearable
from jax import (
    numpy as jnp,
    tree as jt,
)
from jaxtyping import Float, Scalar, jaxtyped

import typinox as tpx
from typinox import (
    TypedModule,
    ValidatedT,
    Vmapped,
    set_debug_mode,
)
from typinox.tree import unstack

set_debug_mode(True)


def my_is_bearable(x: Any, T) -> bool:
    """To make pyright happy."""
    return is_bearable(x, T)


class SinWave(TypedModule):
    freq: Float[Scalar, ""]
    phase: Float[Scalar, ""]
    amp: Float[Scalar, ""]

    def __call__(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return self.amp * jnp.sin(self.freq * x + self.phase)

    def rotate_90_degs_bad(self) -> Vmapped[Self, "4"]:
        return self

    def actual_rotate_90_degs(self):
        def rotate(x):
            return self.__class__(
                freq=self.freq,
                phase=self.phase + jnp.pi / 4 * x,
                amp=self.amp,
            )

        return jax.vmap(rotate)(jnp.arange(4))

    def rotate_90_degs_bad_sig(self) -> Self:
        return self.actual_rotate_90_degs()

    def rotate_90_degs_good(self) -> Vmapped[Self, "4"]:
        return self.actual_rotate_90_degs()


SinWaveT = ValidatedT[SinWave]


def test_simple_vmap():
    freqs = jnp.array([1.0, 2.0, 3.0])
    phases = jnp.array([0.0, 1.0, 2.0]) / jnp.pi
    amps = jnp.array([0.5, 1.0, 1.5])
    waves = jax.vmap(SinWave)(freqs, phases, amps)
    assert isinstance(waves, SinWave)
    assert not my_is_bearable(waves, SinWaveT)
    assert my_is_bearable(waves, Vmapped[SinWaveT, "3"])

    nums = jnp.array([1.0, 2.0, 3.0])

    def call(wave, num):
        return wave(num)

    _ = jax.vmap(call)(waves, nums)
    _ = jax.vmap(call, in_axes=(0, None))(waves, nums[0])


def test_vmapped_self():
    sw = SinWave(jnp.array(2.0), jnp.array(4.0), jnp.array(6.0))
    with pytest.raises(TypeError):
        sw.rotate_90_degs_bad()

    with pytest.raises(TypeError):
        sw.rotate_90_degs_bad_sig()

    sws = sw.rotate_90_degs_good()

    swss = jax.vmap(lambda x: x.rotate_90_degs_good())(sws)
    assert swss.freq.shape == (4, 4)


class MultiWaves(TypedModule):
    n: int = tpx.field(static=True)
    waves: tuple[SinWaveT, ...]
    shifts: tuple[Float[Scalar, ""], ...]

    def __validate__(self) -> bool:
        return len(self.waves) == self.n and len(self.shifts) == self.n

    def __call__(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        for wave, shift in zip(self.waves, self.shifts):
            x = wave(x) + shift
        return x


MultiWavesT = ValidatedT[MultiWaves]


def test_multi_waves():
    def create_multiwave(k):
        n = 4
        freqs = jnp.array([1.0, 2.0, 3.0, 4.0]) + k
        phases = jnp.array([0.0, 1.0, 2.0, 3.0]) / jnp.pi
        amps = jnp.array([0.5, 1.0, 1.5, 2.0 - k])
        waves = jax.vmap(SinWave)(freqs, phases, amps)
        shifts = jnp.array([0.0, 1.0, -1.0, 0.0]) * k
        return MultiWaves(n, tuple(unstack(waves)), tuple(shifts))

    mw = create_multiwave(jnp.array(3.0))
    assert my_is_bearable(mw, MultiWavesT)
    assert my_is_bearable(mw(jnp.array(4.0)), Float[Scalar, ""])

    leaves, treedef = jt.flatten(mw)
    leaves[-1] = jnp.array([4.0, 3.0])
    bad_mw = jt.unflatten(treedef, leaves)
    assert not all([my_is_bearable(bad_mw, MultiWavesT) for _ in range(30)])

    with pytest.raises(TypeError):
        mw(jnp.array([4.0, 3.0]))

    with pytest.raises(TypeError):
        _ = MultiWaves(3, mw.waves, mw.shifts)

    mws = jax.vmap(create_multiwave)(jnp.array([1.0, 2.0, 3.0]))
    assert isinstance(mws, MultiWaves)
    assert not my_is_bearable(mws, MultiWavesT)
    assert my_is_bearable(mws, Vmapped[MultiWavesT, "c"])


class SegmentTree(TypedModule):
    n: int = tpx.field(static=True)
    left: Self | None
    right: Self | None
    value: Float[Scalar, ""]
    total: Float[Scalar, ""]

    def __validate__(self) -> bool:
        if self.n == 1:
            return self.left is None and self.right is None
        left_size = self.left.n if self.left is not None else 0
        right_size = self.right.n if self.right is not None else 0
        return (
            self.n == left_size + right_size + 1
            and -1 <= left_size - right_size <= 1
        )

    @classmethod
    def create_leaf(cls, value) -> Self:
        return cls(1, None, None, value, value)

    @classmethod
    def create_node(cls, left, value, right):
        return cls(
            (left.n if left else 0) + 1 + (right.n if right else 0),
            left,
            right,
            value,
            (left.total if left else 0.0)
            + value
            + (right.total if right else 0.0),
        )

    @classmethod
    def build(cls, values) -> Self | None:
        n = len(values)
        if n == 0:
            return None
        if n == 1:
            return cls.create_leaf(values[0])
        mid = n // 2
        left = cls.build(values[:mid])
        right = cls.build(values[mid + 1 :])
        return cls.create_node(left, values[mid], right)

    def query(self, i, j):
        if i < 0:
            i = 0
        if j >= self.n:
            j = self.n
        if i >= j:
            return jnp.array(0.0)
        if i <= 0 <= self.n <= j:
            return self.total
        left_size = self.left.n if self.left is not None else 0
        left_value = (
            self.left.query(i, j) if self.left is not None else jnp.array(0.0)
        )
        right_value = (
            self.right.query(i - left_size - 1, j - left_size - 1)
            if self.right is not None
            else jnp.array(0.0)
        )
        return (
            left_value
            + right_value
            + (self.value if i <= left_size < j else 0.0)
        )


SegmentTreeT = ValidatedT[SegmentTree]


def test_segment_tree():
    arr = jnp.sin(jnp.arange(20))
    st = SegmentTree.build(arr)
    assert st is not None
    assert my_is_bearable(st, SegmentTreeT)

    for i, j in [(3, 7), (6, 16), (0, 15), (-3, 37)]:
        st_value = st.query(i, j)
        cons_arr = (jnp.arange(20) >= i) & (jnp.arange(20) < j)
        arr_value = jnp.sum(arr * cons_arr)
        assert jnp.allclose(st_value, arr_value)

    def create_st(k, z):
        arr = jnp.sin(jnp.arange(20) * k + z)
        return SegmentTree.build(arr)

    def elim_st(st):
        return tuple(
            [st.query(i, j) for i, j in [(3, 7), (6, 16), (0, 15), (-3, 37)]]
        )

    sts = jax.vmap(create_st)(
        jnp.array([1.0, 2.0, 3.0]), jnp.array([0.0, 1.0, 2.0])
    )
    assert isinstance(sts, SegmentTree)
    assert not my_is_bearable(sts, SegmentTreeT)
    sums = jax.vmap(elim_st)(sts)

    with jaxtyped("context"):  # type: ignore
        assert my_is_bearable(sts, Vmapped[SegmentTreeT, "c"])
        assert my_is_bearable(
            sums, Vmapped[tuple[(Float[Scalar, ""],) * 4], "c"]
        )

    with jaxtyped("context"):  # type: ignore
        assert my_is_bearable(sts, Vmapped[SegmentTreeT, "c"])
        assert not my_is_bearable(
            sums, Vmapped[tuple[(Float[Scalar, ""],) * 4], "c+1"]
        )
