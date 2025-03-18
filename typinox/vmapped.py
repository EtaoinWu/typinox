import typing

if typing.TYPE_CHECKING:
    from typing import (
        Annotated as Vmapped,  # noqa: F401
        Annotated as VmappedI,  # noqa: F401
        Annotated as VmappedT,  # noqa: F401
    )
else:
    from ._vmapped import (
        VmappedI as VmappedI,
        VmappedT as VmappedT,
    )

    # Vmapped is the convenient alias for VmappedT
    Vmapped = VmappedT
