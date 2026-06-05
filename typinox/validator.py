from collections.abc import Callable
from typing import cast

from beartype.typing import TYPE_CHECKING, Annotated, TypeAliasType, Unpack
from beartype.vale import Is

from ._helper import func_to_bracket

UnpackType = type(Unpack[tuple[int, str]])


class ValidationFailed(ValueError):
    pass


class ValidateProto:
    def __validate__(self) -> bool:
        return True

    def __validate_str__(self) -> str:
        return ""


def validate_str_single(cls: type[object], obj: object) -> str:
    validate_str_fn = cls.__dict__.get("__validate_str__", None)
    if validate_str_fn is not None:
        validate_str_fn = cast(Callable[[object], str], validate_str_fn)
        validated = validate_str_fn(obj)
        if validated != "":
            return validated
    validate_fn = cls.__dict__.get("__validate__", None)
    if validate_fn is not None:
        validate_fn = cast(Callable[[object], bool], validate_fn)
        try:
            is_valid = validate_fn(obj)
        except ValidationFailed as e:
            return str(e)
        if is_valid is False:
            return "the custom validation failed"
    return ""


def validate_str(obj: object) -> str:
    cls = type(obj)
    for kls in cls.__mro__[
        :-1
    ]:  # skip the last class in the mro, which is object
        validated = validate_str_single(kls, obj)
        if validated != "":
            return validated
    return ""


def _validate(obj: object) -> bool:
    return validate_str(obj) == ""


TypinoxValid = Is[_validate]
if TYPE_CHECKING:
    type ValidatedT[_T] = Annotated[_T, TypinoxValid]
else:

    @func_to_bracket
    def ValidatedT[T](cls: type[T]) -> type[T]:
        from ._vmapped import AbstractVmapped

        if isinstance(cls, UnpackType):
            # workaround for weird Python typing restriction
            # see: https://github.com/beartype/beartype/issues/562#issuecomment-3404611632
            wrap_cls = TypeAliasType(cls.__name__, cls)
            return Annotated[wrap_cls, TypinoxValid]
        if not isinstance(cls, type):
            return Annotated[cls, TypinoxValid]
        if issubclass(cls, AbstractVmapped):
            return cls.replace_inner(ValidatedT[cls.inner])
        if (
            getattr(cls, "__validate__", None) is None
            and getattr(cls, "__validate_str__", None) is None
        ):
            return cls
        return Annotated[cls, TypinoxValid]
