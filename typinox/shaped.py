import dataclasses
from typing import Any, Protocol, overload


@dataclasses.dataclass(frozen=True)
class _FakeArray:
    shape: tuple[int, ...]
    dtype: Any = dataclasses.field(default=None)


class NDArray(Protocol):
    """
    Duck match for any object that behaves like a numpy array
    (has ``.shape`` and ``.dtype`` attributes).
    """

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
    """
    Ensure that the shape of an array matches :mod:`jaxtyping` named dimensions.

    Parameters
    ----------
    name: str, optional
        The name of the array.

    shape : ShapeLike
        The shape of the array.

    dim_spec : str
        The :mod:`jaxtyping` named dimensions to be matched.

    Returns
    -------
    None
        If the shape matches the named dimensions, return None.

    Raises
    ------
    ValidationFailed
        when the shape does not match the named dimensions
    """

    from jaxtyping import Shaped

    from .validator import ValidationFailed

    if len(args) == 2:
        shape, dim_spec = args
        name = ""
    elif len(args) == 3:
        name, shape, dim_spec = args
    else:
        raise ValueError(f"invalid number of arguments: {len(args)}")

    obj = _FakeArray(shape_sanitize(shape))
    if not isinstance(obj, Shaped[_FakeArray, dim_spec]):  # type: ignore
        if name:
            raise ValidationFailed(
                f'{name} has shape {shape} which does not match the named dimensions "{dim_spec}"'
            )
        else:
            raise ValidationFailed(
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
    """
    Ensure that the shapes of two arrays are equal.

    Parameters
    ----------
    name : str, optional
        The name of the arrays. Cannot be provided if name1 or name2 is provided.

    name1 : str, optional
        The name of the first array. Must be provided if name2 is provided.

    shape1 : int, or tuple[int, ...], or an object with ``.shape``
        The shape of the first array.

    name2 : str, optional
        The name of the second array. Must be provided if name1 is provided.

    shape2 : int, or tuple[int, ...], or an object with ``.shape``
        The shape of the second array.

    Returns
    -------
    None
        If the shapes are equal, return None.

    Raises
    ------
    ValidationFailed
        when the shapes are not equal.
    """

    from .validator import ValidationFailed

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
        raise ValidationFailed(
            f"{name1} {shape1} does not match {name2} {shape2}"
        )
