from typing import Any, Callable, cast  # noqa: UP035

__all__: list[str] = []


def func_to_bracket(
    fn: Callable[..., Any] | None = None, name: str | None = None
) -> type[Any] | Callable[[Callable[..., Any]], type[Any]]:
    if fn is None:

        def decorator(fn: Callable[..., Any]) -> type[Any]:
            return cast(type[Any], func_to_bracket(fn, name))

        return decorator
    if name is None:
        name = fn.__name__

    def __class_getitem__(cls: type[Any], item: Any) -> Any:
        return fn(item)

    cls = type(name, (), {"__class_getitem__": __class_getitem__})
    cls.__module__ = fn.__module__
    return cls
