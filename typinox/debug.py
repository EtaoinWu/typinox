import warnings

debug_mode = False


class TypinoxUnknownTypeWarning(UserWarning):
    pass


class TypinoxUnknownFunctionWarning(UserWarning):
    pass


def set_debug_mode(value: bool = True):
    global debug_mode
    debug_mode = value


def debug_print(*args, **kwargs):
    if debug_mode:
        print(*args, **kwargs)


def debug_warn(*args, **kwargs):
    if debug_mode:
        warnings.warn(*args, **kwargs)


def debug_raise(err: type[Exception], *args, **kwargs):
    if debug_mode:
        raise err(*args, **kwargs)
