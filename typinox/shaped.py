import dataclasses
from typing import Protocol, cast, overload


@dataclasses.dataclass(frozen=True)
class _FakeArray:
    shape: tuple[int, ...]
    dtype: object | None = dataclasses.field(default=None)


class NDArray(Protocol):
    """
    Duck match for any object that behaves like a numpy array
    (has ``.shape`` and ``.dtype`` attributes).
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> object: ...


ShapeLike = int | tuple[int, ...] | NDArray


def shape_sanitize(shape: ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, tuple):
        return shape
    return shape.shape


@overload
def ensure_shape(shape: ShapeLike, dim_spec: str, /) -> None: ...


@overload
def ensure_shape(name: str, shape: ShapeLike, dim_spec: str, /) -> None: ...


def ensure_shape(*args: object) -> None:
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
        raw_name, shape, dim_spec = args
        if not isinstance(raw_name, str):
            raise TypeError("shape name must be a string")
        name = raw_name
    else:
        raise ValueError(f"invalid number of arguments: {len(args)}")
    if isinstance(shape, str):
        raise TypeError("shape must be a shape-like object")
    if not isinstance(dim_spec, str):
        raise TypeError("dimension spec must be a string")
    shape = cast(ShapeLike, shape)

    obj = _FakeArray(shape_sanitize(shape))
    shaped_hint = cast(type, cast(object, Shaped[_FakeArray, dim_spec]))
    if not isinstance(obj, shaped_hint):
        if name:
            raise ValidationFailed(
                f"{name} has shape {shape} which"
                + f' does not match the named dimensions "{dim_spec}"'
            )
        else:
            raise ValidationFailed(
                f"shape {shape} does not match the named dimensions {dim_spec}"
            )


@overload
def ensure_shape_equal(shape1: ShapeLike, shape2: ShapeLike, /) -> None: ...


@overload
def ensure_shape_equal(
    name: str, shape1: ShapeLike, shape2: ShapeLike, /
) -> None: ...


@overload
def ensure_shape_equal(
    name1: str, shape1: ShapeLike, name2: str, shape2: ShapeLike, /
) -> None: ...


def ensure_shape_equal(*args: object) -> None:
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
        if not isinstance(name, str):
            raise TypeError("shape name must be a string")
        name1 = name2 = name
    elif len(args) == 4:
        raw_name1, shape1, raw_name2, shape2 = args
        if not isinstance(raw_name1, str) or not isinstance(raw_name2, str):
            raise TypeError("shape names must be strings")
        name1 = raw_name1
        name2 = raw_name2
    else:
        raise ValueError(f"invalid number of arguments: {len(args)}")
    if isinstance(shape1, str) or isinstance(shape2, str):
        raise TypeError("shapes must be shape-like objects")
    shape1 = cast(ShapeLike, shape1)
    shape2 = cast(ShapeLike, shape2)

    if shape_sanitize(shape1) != shape_sanitize(shape2):
        raise ValidationFailed(
            f"{name1} {shape1} does not match {name2} {shape2}"
        )
