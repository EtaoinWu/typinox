from beartype.typing import TYPE_CHECKING, Annotated, TypeVar
from beartype.vale import Is

from ._helper import func_to_bracket


class ValidateFailed(ValueError):
    pass


class ValidateProto:
    def __validate__(self, obj) -> bool:
        return True

    def __validate_str__(self, obj) -> str:
        return ""


def validate_str_single(cls, obj) -> str:
    if cls.__dict__.get("__validate_str__", None) is not None:
        validated = cls.__validate_str__(obj)
        if validated != "":
            return validated
    if cls.__dict__.get("__validate__", None) is not None:
        try:
            validated = cls.__validate__(obj)
        except ValidateFailed as e:
            return str(e)
        if validated is False:
            return "the custom validation failed"
    return ""


def validate_str(obj) -> str:
    cls = type(obj)
    for kls in cls.__mro__[
        :-1
    ]:  # skip the last class in the mro, which is object
        validated = validate_str_single(kls, obj)
        if validated != "":
            return validated
    return ""


def _validate(obj):
    return validate_str(obj) == ""


TypinoxValid = Is[_validate]
_T = TypeVar("_T")
if TYPE_CHECKING:
    ValidatedT = Annotated[_T, TypinoxValid]
else:

    @func_to_bracket
    def ValidatedT(cls: _T) -> _T:
        from ._vmapped import AbstractVmapped
        if not isinstance(cls, type):
            return Annotated[cls, TypinoxValid]
        if issubclass(cls, AbstractVmapped):
            return cls.replace_inner(ValidatedT[cls.inner])
        if (
            getattr(cls, "__validate__", None) is not None
            and getattr(cls, "__validate_str__", None) is not None
        ):
            return cls
        return Annotated[cls, TypinoxValid]
