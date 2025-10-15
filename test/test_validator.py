from dataclasses import dataclass
from typing import Any

import beartype
import beartype.roar
import pytest
from beartype import beartype as typechecker
from beartype.door import is_bearable

from typinox import ValidatedT, ValidationFailed


def my_is_bearable(x: Any, T) -> bool:
    """To make pyright happy."""
    return is_bearable(x, T)


class ClassA:
    def __init__(self, x: int, y: int):
        if x > 0:
            self.x = x
        self.y = y

    def __validate__(self):
        if hasattr(self, "x"):
            return True
        else:
            if self.y > 0:
                return False
            else:
                raise ValidationFailed("the x attribute is missing")


class ClassBBase:
    def __init__(self, x: int, y: int):
        if x > 0:
            self.x = x
        self.y = y

    def __validate_str__(self):
        if hasattr(self, "x"):
            return ""
        else:
            return "this instance has no x attribute"


class ClassB(ClassBBase):
    pass


@dataclass
class ClassC:
    x: int
    y: int


def fn_a_vanilla(a: ClassA) -> int:
    return a.x + a.y


def fn_a_custom(a: ValidatedT[ClassA]) -> int:
    return a.x + a.y


fn_a_typechecked = typechecker(fn_a_vanilla)
fn_a_custom_typechecked = typechecker(fn_a_custom)

fn_a = (fn_a_vanilla, fn_a_custom, fn_a_typechecked, fn_a_custom_typechecked)


def fn_b_vanilla(b: ClassB) -> int:
    return b.x + b.y


def fn_b_custom(b: ValidatedT[ClassB]) -> int:
    return b.x + b.y


fn_b_typechecked = typechecker(fn_b_vanilla)
fn_b_custom_typechecked = typechecker(fn_b_custom)

fn_b = (fn_b_vanilla, fn_b_custom, fn_b_typechecked, fn_b_custom_typechecked)


@pytest.mark.parametrize("cls,fn", [(ClassA, fn_a), (ClassB, fn_b)])
def test_good_case(cls, fn):
    fn_vanilla, fn_custom, fn_typechecked, fn_custom_typechecked = fn
    assert fn_vanilla(cls(1, 2)) == 3
    assert fn_custom(cls(1, 2)) == 3
    assert fn_typechecked(cls(1, 2)) == 3
    assert fn_custom_typechecked(cls(1, 2)) == 3


@pytest.mark.parametrize(
    "cls,fn,errorstr",
    [
        (ClassA, fn_a, "custom validation failed"),
        (ClassB, fn_b, "this instance has no x attribute"),
    ],
)
def test_bad_attribute_read(cls, fn, errorstr):
    fn_vanilla, fn_custom, fn_typechecked, fn_custom_typechecked = fn
    with pytest.raises(AttributeError):
        fn_vanilla(cls(0, 2))

    with pytest.raises(AttributeError):
        fn_custom(cls(0, 2))

    with pytest.raises(AttributeError):
        fn_typechecked(cls(0, 2))

    with pytest.raises(
        beartype.roar.BeartypeCallHintParamViolation,
        # match=errorstr,
    ):
        fn_custom_typechecked(cls(0, 2))


@pytest.mark.parametrize(
    "cls,fn,errorstr",
    [
        (ClassA, fn_a, "the x attribute is missing"),
        (ClassB, fn_b, "this instance has no x attribute"),
    ],
)
def test_bad_attribute_read_pretty(cls, fn, errorstr):
    fn_vanilla, fn_custom, fn_typechecked, fn_custom_typechecked = fn
    with pytest.raises(AttributeError):
        fn_vanilla(cls(0, 0))

    with pytest.raises(AttributeError):
        fn_custom(cls(0, 0))

    with pytest.raises(AttributeError):
        fn_typechecked(cls(0, 0))

    with pytest.raises(
        beartype.roar.BeartypeCallHintParamViolation,
        # match=errorstr,
    ):
        fn_custom_typechecked(cls(0, 0))


@pytest.mark.parametrize("cls,fn", [(ClassA, fn_a), (ClassB, fn_b)])
def test_bad_type(cls, fn):
    fn_vanilla, fn_custom, fn_typechecked, fn_custom_typechecked = fn
    assert fn_vanilla(ClassC(1, 2)) == 3  # type: ignore
    assert fn_custom(ClassC(1, 2)) == 3  # type: ignore

    with pytest.raises(beartype.roar.BeartypeCallHintParamViolation):
        fn_typechecked(ClassC(1, 2))  # type: ignore

    with pytest.raises(beartype.roar.BeartypeCallHintParamViolation):
        fn_custom_typechecked(ClassC(1, 2))  # type: ignore


def test_validator_unrelated():
    assert my_is_bearable(1, ValidatedT[int])
    assert not my_is_bearable(1, ValidatedT[str])


def test_validator_union():
    assert not my_is_bearable(1, ValidatedT[ClassA | ClassB])
    assert my_is_bearable(ClassA(1, 2), ValidatedT[ClassA | ClassB])
    assert my_is_bearable(ClassB(2, 1), ValidatedT[ClassA | ClassB])
    assert not my_is_bearable(ClassA(-1, 2), ValidatedT[ClassA | ClassB])
    assert not my_is_bearable(ClassB(-2, 1), ValidatedT[ClassA | ClassB])
