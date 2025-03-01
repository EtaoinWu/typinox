import typing

if typing.TYPE_CHECKING:
    from typing import (
        Annotated as Vmapped,  # noqa: F401
        Annotated as VmappedT,  # noqa: F401
    )
else:
    from ._vmapped import (
        Vmapped as Vmapped,
        VmappedT as VmappedT,
    )
