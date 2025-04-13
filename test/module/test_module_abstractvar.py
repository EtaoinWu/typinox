import pytest

from typinox import (
    AbstractVar,
    TypedModule,
    set_debug_mode,
)

set_debug_mode(True)


class Base(TypedModule):
    a: int
    b: AbstractVar[int]


def test_abstract_var_basic():
    with pytest.raises(TypeError):
        _ = Base(a=1, b=1)


class SubWithValue(Base):
    b: int  # type: ignore


def test_abstract_var_subclass():
    obj = SubWithValue(a=1, b=3)
    assert obj.a == 1
    assert obj.b == 3


class SubWithProperty(Base):
    @property  # type: ignore
    def b(self) -> int:  # type: ignore
        return self.a + 2


def test_abstract_var_property():
    obj = SubWithProperty(a=1)  # type: ignore
    assert obj.a == 1
    assert obj.b == 3
