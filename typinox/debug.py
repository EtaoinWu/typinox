import warnings
from typing import TextIO

debug_mode = False


class TypinoxUnknownTypeWarning(UserWarning):
    pass


class TypinoxUnknownFunctionWarning(UserWarning):
    pass


def set_debug_mode(value: bool = True) -> None:
    global debug_mode
    debug_mode = value


def debug_print(
    *args: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    if debug_mode:
        print(*args, sep=sep, end=end, file=file, flush=flush)


def debug_warn(
    message: Warning | str,
    category: type[Warning] | None = None,
    stacklevel: int = 1,
    source: object | None = None,
) -> None:
    if debug_mode:
        warnings.warn(
            message, category=category, stacklevel=stacklevel, source=source
        )


def debug_raise(err: type[Exception], *args: object) -> None:
    if debug_mode:
        raise err(*args)
