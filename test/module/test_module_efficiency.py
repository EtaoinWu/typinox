from typing import Self
from unittest.mock import Mock

from beartype.door import die_if_unbearable
from beartype.typing import Annotated, Any, Callable
from beartype.vale import Is

from typinox import (
    TypedModule,
    ValidatedT,
    set_debug_mode,
)

set_debug_mode(True)


def test_diamond_inheritance():
    quacks = [Mock(return_value=True) for _ in range(4)]
    happys = [Mock(return_value=True) for _ in range(4)]

    def happy_func(i: int) -> Callable[[Any], bool]:
        def z(x):
            happys[i](x)
            return True

        return z

    result_mock = Mock(return_value=True)

    def good_result(x):
        return result_mock(x)

    class A(TypedModule):
        a: Annotated[int, Is[happy_func(0)]]

        def __validate__(self):
            return self.a >= 0 and quacks[0]()

    class B(A):
        b: Annotated[int, Is[happy_func(1)]]

        def __validate__(self):
            return self.b >= 0 and quacks[1]()

    class C(A):
        c: Annotated[int, Is[happy_func(2)]]

        def __validate__(self):
            return self.c >= 0 and quacks[2]()

    class D(B, C):
        d: Annotated[int, Is[happy_func(3)]]

        def __validate__(self):
            return self.d >= 0 and quacks[3]()

        def value(self: Self) -> Annotated[int, Is[good_result]]:
            return self.a + self.b + self.c + self.d

    obj = D(a=9, b=12, c=233, d=7)

    for q in quacks:
        q.assert_called_once()
        q.reset_mock()
    for h, num in zip(happys, [9, 12, 233, 7]):
        h.assert_called_once_with(num)
        h.reset_mock()

    die_if_unbearable(obj, ValidatedT[D])

    for q in quacks:
        q.assert_called_once()
        q.reset_mock()
    for h, num in zip(happys, [9, 12, 233, 7]):
        h.assert_called_once_with(num)
        h.reset_mock()

    assert obj.value() == 9 + 12 + 233 + 7

    # Because jaxtyping likes to call every parameter twice
    for q in quacks:
        assert q.call_count == 2
        q.reset_mock()
    for h, num in zip(happys, [9, 12, 233, 7]):
        assert h.call_count == 2
        h.assert_called_with(num)
        h.reset_mock()
    result_mock.assert_called_once()
    result_mock.reset_mock()
