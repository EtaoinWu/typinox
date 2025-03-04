import dataclasses
from typing import Any, Protocol, overload


@dataclasses.dataclass(frozen=True)
class FakeArray:
    shape: tuple[int, ...]
    dtype: Any = dataclasses.field(default=None)


class NDArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


ShapeLike = int | tuple[int, ...] | NDArray


def shape_sanitize(shape: ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, tuple):
        return shape
    return shape.shape


@overload
def ensure_shape(shape: ShapeLike, dim_spec: str, /): ...


@overload
def ensure_shape(name: str, shape: ShapeLike, dim_spec: str, /): ...


def ensure_shape(*args):
    from jaxtyping import Shaped

    from .validator import ValidateFailed

    if len(args) == 2:
        shape, dim_spec = args
        name = ""
    elif len(args) == 3:
        name, shape, dim_spec = args
    else:
        raise ValueError(f"invalid number of arguments: {len(args)}")

    obj = FakeArray(shape_sanitize(shape))
    if not isinstance(obj, Shaped[FakeArray, dim_spec]):  # type: ignore
        if name:
            raise ValidateFailed(
                f'{name} has shape {shape} which does not match the named dimensions "{dim_spec}"'
            )
        else:
            raise ValidateFailed(
                f"shape {shape} does not match the named dimensions {dim_spec}"
            )


@overload
def ensure_shape_equal(shape1: ShapeLike, shape2: ShapeLike, /): ...


@overload
def ensure_shape_equal(name: str, shape1: ShapeLike, shape2: ShapeLike, /): ...


@overload
def ensure_shape_equal(
    name1: str, shape1: ShapeLike, name2: str, shape2: ShapeLike, /
): ...


def ensure_shape_equal(*args):
    from .validator import ValidateFailed

    if len(args) == 2:
        shape1, shape2 = args
        name1 = name2 = ""
    elif len(args) == 3:
        name, shape1, shape2 = args
        name1 = name2 = name
    elif len(args) == 4:
        name1, shape1, name2, shape2 = args
    else:
        raise ValueError(f"invalid number of arguments: {len(args)}")

    if shape_sanitize(shape1) != shape_sanitize(shape2):
        raise ValidateFailed(
            f"{name1} {shape1} does not match {name2} {shape2}"
        )
