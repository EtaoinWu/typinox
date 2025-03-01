import dataclasses
import inspect
import warnings
import weakref
from types import FunctionType

import beartype
from beartype.door import is_bearable
from beartype.typing import (
    Annotated,
    Any,
    Callable,
    Self,
    overload,
    cast,
)
from equinox import field as eqx_field
from equinox._module import (
    StrictConfig,
    _ModuleMeta as EqxModuleMeta,
    _wrap_method as EqxWrapMethod,
)
from jaxtyping import jaxtyped

from ._shaped import ensure_shape as ensure_shape
from ._vmapped import (
    AbstractVmapped,
    VmappedMeta,
    get_vmapped_origin_or_none,
    make_vmapped,
)
from .debug import (
    TypinoxUnknownFunctionWarning,
    debug_warn,
)
from .error import TypinoxTypeViolation
from .validator import ValidatedT, ValidateFailed, validate_str

AnnotatedAlias = type(Annotated[int, ">3"])
UnionType = type(int | float)
UnionGenericAlias = type(Self | None)


@overload
def field(): ...


@overload
def field(
    *,
    typecheck: bool = True,
    converter: Callable[[Any], Any] = ...,
    static: bool = False,
    default: Any = ...,
    default_factory: Callable[[], Any] | Any = ...,
    init: bool = True,
    hash: bool | None = None,
    metadata: dict[str, Any] | None = None,
    kw_only: bool = ...,
): ...


def field(
    *,
    typecheck: bool = True,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dataclasses.Field:
    if metadata is None:
        metadata = {}
    metadata["typecheck"] = typecheck
    return eqx_field(
        metadata=metadata,
        **kwargs,
    )


@dataclasses.dataclass(frozen=True)
class TypedPolicy:
    """Used to configure the typechecking behavior of a module."""

    always_validated: bool = dataclasses.field(default=True)
    typecheck_init: bool = dataclasses.field(default=True)
    typecheck_init_result: bool = dataclasses.field(default=True)
    typecheck_magic_methods: bool | frozenset[str] = dataclasses.field(
        default=frozenset(["__call__"])
    )
    typecheck_skip: frozenset[str] = dataclasses.field(
        default_factory=frozenset
    )


policy_for_type: weakref.WeakKeyDictionary[type, TypedPolicy] = (
    weakref.WeakKeyDictionary()
)


def mark_as_typed[T: Callable](fn: T) -> T:
    if getattr(fn, "__typinox_typed__", False):
        return fn
    setattr(fn, "__typinox_typed__", True)
    return fn


def marked_as_typed(fn: Callable) -> bool:
    return getattr(fn, "__typinox_typed__", False)


def decorate_function(fn: Callable) -> Callable:
    return jaxtyped(fn, typechecker=beartype.beartype)


def sanitize_annotation(annotation: Any, cls: type) -> Any:
    if annotation is Self:
        return cls
    if isinstance(annotation, UnionType | UnionGenericAlias):
        origin = annotation.__origin__  # type: ignore
        args = annotation.__args__  # type: ignore
        return origin[tuple(sanitize_annotation(arg, cls) for arg in args)]
    if isinstance(annotation, VmappedMeta):
        origin = get_vmapped_origin_or_none(annotation)
        if origin is Self:
            return cast(type[AbstractVmapped], annotation).replace_inner(cls)
    if isinstance(annotation, AnnotatedAlias):  # type: ignore
        origin = annotation.__origin__  # type: ignore
        if origin is Self:
            return cls
        return Annotated[
            sanitize_annotation(origin, cls), annotation.__metadata__  # type: ignore
        ]
    return annotation


def method_transform_annotations(
    fn: FunctionType, cls: type, policy: TypedPolicy
) -> FunctionType:
    annotations = fn.__annotations__
    for key, value in annotations.items():
        if isinstance(value, str):
            warnings.warn(
                f"Typinox: string annotations are not supported: `{value}` in {fn} of {cls}"
            )
            continue
        new_annotation = sanitize_annotation(value, cls)
        if policy.always_validated:
            new_annotation = ValidatedT[new_annotation]
        if new_annotation is not value:
            annotations[key] = new_annotation
    return fn


def is_magic(fn: Callable) -> bool:
    return fn.__name__.startswith("__") and fn.__name__.endswith("__")


def decorate_method(fn: Callable, cls: type, policy: TypedPolicy) -> Callable:
    if getattr(fn, "__name__", None) in policy.typecheck_skip:
        return fn
    if isinstance(fn, staticmethod):
        actual_method = fn.__func__
        return staticmethod(decorate_method(actual_method, cls, policy))
    if isinstance(fn, classmethod):
        actual_method = fn.__func__
        return classmethod(decorate_method(actual_method, cls, policy))
    if isinstance(fn, property):
        fget = (
            decorate_method(fn.fget, cls, policy)
            if fn.fget is not None
            else None
        )
        fset = (
            decorate_method(fn.fset, cls, policy)
            if fn.fset is not None
            else None
        )
        fdel = (
            decorate_method(fn.fdel, cls, policy)
            if fn.fdel is not None
            else None
        )
        return property(fget, fset, fdel)
    if isinstance(fn, EqxWrapMethod):
        return EqxWrapMethod(decorate_method(fn.method, cls, policy))
    if not callable(fn):
        return fn
    if not inspect.isfunction(fn):
        # We can only wrap Python-native functions.
        debug_warn(
            f"Typinox: attempting to perform typechecking decoration on unknown object: {fn}",
            TypinoxUnknownFunctionWarning,
        )
        return fn
    if marked_as_typed(fn):
        return fn
    if is_magic(fn):
        if isinstance(policy.typecheck_magic_methods, bool):
            if not policy.typecheck_magic_methods:
                return fn
        else:
            if fn.__name__ not in policy.typecheck_magic_methods:
                return fn
    # Main case: pure-python function.
    fn = method_transform_annotations(fn, cls, policy)
    fn = decorate_function(fn)
    fn = mark_as_typed(fn)
    return fn


class RealTypedModuleMeta(EqxModuleMeta):
    def __new__(
        mcs,
        name,
        bases,
        dict_,
        /,
        strict: bool | StrictConfig = False,
        typed_policy: TypedPolicy | None = None,
        **kwargs,
    ):
        # [Step 1] Create the Module as normal.
        cls = super().__new__(mcs, name, bases, dict_, strict=strict, **kwargs)
        # Assumption:
        # - Every non-magic normal method is wrapped by Equinox.
        # - A __init__ method is created, either by the user or by Equinox.

        # [Step 2] Wrap all methods with the typechecker.
        # [Step 2.0] Prepare the typechecking policy.
        if typed_policy is None:
            typed_policy = TypedPolicy()
        policy_for_type[cls] = typed_policy
        # [Step 2.1] Wrap the methods with the typechecker.
        for key, value in cls.__dict__.items():
            if key == "__init__":
                if not typed_policy.typecheck_init:
                    continue
            decorated_value = decorate_method(value, cls, typed_policy)
            if decorated_value is not value:
                setattr(cls, key, decorated_value)

        # [Step 3] Add the validator methods.
        old_validate = cls.__dict__.get("__validate__", None)
        old_validate_str = cls.__dict__.get("__validate_str__", None)

        # [Step 3.1] Recursively validate the fields.
        # [Step 3.1.0] Prepare the annotations to check.
        sanitized_annotations = {
            key: sanitize_annotation(value, cls)
            for key, value in cls.__annotations__.items()
        }
        # Exclude the fields that are marked as not typechecking.
        for field in dataclasses.fields(cls):
            if not field.metadata.get("typecheck", True):
                sanitized_annotations.pop(field.name, None)

        # [Step 3.1 cont'd] Actually validate the fields.
        def __validate_str__(self):
            with jaxtyped("context"):  # type: ignore
                for member, hint in sanitized_annotations.items():
                    if member not in self.__dict__:
                        continue
                    value = self.__dict__[member]
                    if not is_bearable(value, hint):
                        return f"its {member} does not match type hint {hint}, got {value}"
                if old_validate_str is not None:
                    result = old_validate_str(self)
                    if result:
                        return result
                if old_validate is not None:
                    try:
                        result = old_validate(self)
                    except ValidateFailed as e:
                        return str(e)
                    if result is False:
                        return "it failed its custom validation"
            return ""

        # Add the methods to the class.
        __validate_str__.__qualname__ = "__validate_str__"
        setattr(cls, "__validate_str__", __validate_str__)
        setattr(cls, "__validate__", None)
        return cls

    # Creating an instance with MyModule(...) will call this method.
    def __call__(cls, *args, **kwargs):
        # [Step 1] Create the instance as normal.
        instance = super().__call__(*args, **kwargs)
        # [Step 2] Typecheck the instance.
        policy = policy_for_type[cls]
        if policy.typecheck_init_result:
            check_result = validate_str(instance)
            if check_result:
                raise TypinoxTypeViolation(
                    f"The instance {instance} of {cls} has failed typechecking, as {check_result}"
                )
        return instance
