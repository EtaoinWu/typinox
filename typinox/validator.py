from beartype.typing import TYPE_CHECKING, Annotated, Unpack
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


def validate_str_single(cls, obj) -> str:
    if cls.__dict__.get("__validate_str__", None) is not None:
        validated = cls.__validate_str__(obj)
        if validated != "":
            return validated
    if cls.__dict__.get("__validate__", None) is not None:
        try:
            validated = cls.__validate__(obj)
        except ValidationFailed as e:
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
if TYPE_CHECKING:
    type ValidatedT[_T] = Annotated[_T, TypinoxValid]
else:

    @func_to_bracket
    def ValidatedT[_T](cls: type[_T]) -> type[_T]:
        from ._vmapped import AbstractVmapped

        if isinstance(cls, UnpackType):
            # sadly beartype cannot do custom validation
            # on TypeVarTuple yet; see
            # https://github.com/beartype/beartype/issues/562
            return cls
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
