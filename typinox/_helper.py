from typing import Callable  # noqa: UP035

__all__: list[str] = []


def func_to_bracket(fn: Callable | None = None, name: str | None = None):
    if fn is None:
        return lambda fn: func_to_bracket(fn, name)
    if name is None:
        name = fn.__name__
    cls = type(name, (), {"__class_getitem__": lambda cls, item: fn(item)})
    cls.__module__ = fn.__module__
    return cls
