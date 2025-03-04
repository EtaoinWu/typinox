import dataclasses
from typing import TYPE_CHECKING, dataclass_transform

import equinox
import equinox._module

from ._module import (
    TypedPolicy as TypedPolicy,
    field as field,
)
from .error import TypinoxTypeViolation

if TYPE_CHECKING:

    @dataclass_transform(
        field_specifiers=(dataclasses.field, equinox.field, field)
    )
    class TypedModuleMeta(equinox._module._ModuleMeta):
        pass

    class TypedModule(equinox.Module, metaclass=TypedModuleMeta):
        def __validate_self_str__(self) -> str:
            return ""

        def _validate(self) -> None: ...

else:
    from ._module import RealTypedModuleMeta

    TypedModuleMeta = RealTypedModuleMeta

    class TypedModule(equinox.Module, metaclass=TypedModuleMeta):
        pass

    def _validate(self) -> None:
        __tracebackhide__ = True
        cls = type(self)
        for kls in cls.__mro__[-2::-1]:
            if hasattr(kls, "__validate_self_str__"):
                validated = kls.__validate_self_str__(self)
                if validated != "":
                    raise TypinoxTypeViolation(
                        f"the value ({self}) is not a {cls}, as {validated}"
                    )

    type.__setattr__(TypedModule, "_validate", _validate)
