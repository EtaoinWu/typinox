import dataclasses
from typing import TYPE_CHECKING, dataclass_transform

import equinox
import equinox._module

from ._module import (
    TypedPolicy as TypedPolicy,
    field as field,
)

if TYPE_CHECKING:

    @dataclass_transform(field_specifiers=(dataclasses.field, equinox.field, field))
    class TypedModuleMeta(equinox._module._ModuleMeta):
        pass
else:
    from ._module import RealTypedModuleMeta

    TypedModuleMeta = RealTypedModuleMeta


class TypedModule(equinox.Module, metaclass=TypedModuleMeta):
    pass
