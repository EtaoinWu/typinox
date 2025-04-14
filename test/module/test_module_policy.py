from typing import Self

import jax
import pytest

from typinox import (
    TypedModule,
    TypedPolicy,
)


def test_no_typecheck():
    from typing import no_type_check

    class Module(TypedModule):
        @no_type_check
        def f(self, x: int) -> int:
            return x

    obj = Module()
    obj.f("not an int")


def test_skip_method():
    policy = TypedPolicy(skip_methods=["f"])

    class Module(TypedModule, typed_policy=policy):
        def f(self, x: int) -> int:
            return x

    obj = Module()
    obj.f("not an int")  # type: ignore


def test_not_always_validated():
    policy = TypedPolicy(always_validated=False)

    class Module(TypedModule, typed_policy=policy):
        x: int

        def f(self, rhs: Self) -> None:
            pass

    obj = Module(2)
    rhs = jax.tree.map(lambda x: str(x), obj)
    obj.f(rhs)

    class ControlGroup(TypedModule):
        x: int

        def f(self, rhs: Self) -> None:
            pass

    obj = ControlGroup(2)
    rhs = jax.tree.map(lambda x: str(x), obj)
    with pytest.raises(TypeError):
        obj.f(rhs)


def test_no_typecheck_init_result():
    policy = TypedPolicy(typecheck_init_result=False)

    class Module(TypedModule, typed_policy=policy):
        x: int

        def __init__(self, x: int) -> None:
            super().__init__()
            self.x = str(x)  # type: ignore

    obj = Module(2)
    assert obj.x == "2"

    class ControlGroup(TypedModule):
        x: int

        def __init__(self, x: int) -> None:
            super().__init__()
            self.x = str(x)  # type: ignore

    with pytest.raises(TypeError):
        obj = ControlGroup(2)
