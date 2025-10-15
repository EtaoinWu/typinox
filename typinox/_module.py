import dataclasses
import inspect
import warnings
import weakref
from abc import ABCMeta
from types import FunctionType

import beartype
from beartype.door import is_bearable
from beartype.typing import (
    Annotated,
    Any,
    Callable,
    Iterable,
    Never,
    Self,
    Sequence,
    Unpack,
    cast,
    get_args,
    get_origin,
    overload,
)
from equinox import (
    AbstractClassVar,
    AbstractVar,
    field as eqx_field,
)
from equinox._module._module import (
    _has_dataclass_init,
    _ModuleMeta as EqxModuleMeta,
)
from equinox._module._prebuilt import BoundMethod as EqxWrapMethod
from jaxtyping import jaxtyped

from ._vmapped import (
    AbstractVmapped,
    VmappedMeta,
    get_vmapped_origin_or_none,
)
from .debug import (
    TypinoxUnknownFunctionWarning,
    debug_warn,
)
from .error import TypinoxNotImplementedError, TypinoxTypeViolation
from .shaped import ensure_shape as ensure_shape
from .validator import ValidatedT, ValidationFailed, validate_str

AnnotatedAlias = cast(type, type(Annotated[int, ">3"]))
CallableAliasType = type(Callable[[int], float])
GenericAliasType = type(tuple[int, str])
UnpackType = type(Unpack[tuple[int, str]])
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
    """Specify a field of a typinox typed module.

    Parameters
    ----------

    typecheck : bool, default True
        If specified as False, the field will be ignored during typechecking.

    converter : Callable[[Any], Any], optional
        Used by ``__init__`` to pre-process the value.
        See :func:`equinox.field`.

    static : bool, default False
        If specified as True, the field will be static. This will make it
        a part of the pytree structure, instead of a subtree or a leaf.
        See :func:`equinox.field`.

    default : Any, optional
        Default value used by ``__init__`` if this field is not provided.
        See :func:`dataclasses.field`.

    default_factory : Callable[[], Any] | Any, optional
        Used to generate a default value by ``__init__`` if this field is not provided.
        See :func:`dataclasses.field`.

    metadata : dict[str, Any] | None, optional
        Metadata to be attached to the field.
    """
    if metadata is None:
        metadata = {}
    metadata["typecheck"] = typecheck
    return eqx_field(
        metadata=metadata,
        **kwargs,
    )


@dataclasses.dataclass(frozen=True)
class TypedPolicy:
    """Used to configure the typechecking behavior of a module.

    As an example, if you want to disable typechecking for a specific method,
    you can use the following:

    .. code-block:: python

        class MyModule(TypedModule,
                    typed_policy=TypedPolicy(skip_methods={"wtf"})):
            a: int
            b: int = field(typecheck=False)

            def wtf(self, x: int):
                return x

        z = MyModule(1, "not an int")
        z.wtf("also not an int")

    .. hint::

        You can also use the :func:`typing.no_type_check` decorator
        to disable typechecking for a specific method.

    Parameters
    ----------

    always_validated : bool, default True
        If True, every argument of methods will be type checked with a
        custom validator if present.

    typecheck_init_result : bool, default True
        If False, type checking will not be performed on the result
        of ``__init__``.

    skip_methods : frozenset[str], default {}
        Specifies which methods should skip type checking.
    """

    always_validated: bool = dataclasses.field(default=True)
    typecheck_init_result: bool = dataclasses.field(default=True)
    skip_methods: frozenset[str] = dataclasses.field(default_factory=frozenset)

    def __init__(
        self,
        always_validated: bool = True,
        typecheck_init_result: bool = True,
        skip_methods: Iterable[str] = frozenset(),
    ):
        object.__setattr__(self, "always_validated", always_validated)
        object.__setattr__(self, "typecheck_init_result", typecheck_init_result)
        object.__setattr__(self, "skip_methods", frozenset(skip_methods))


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


def fold_or(args: Sequence[Any]) -> Any:
    if len(args) == 0:
        return Never
    result = args[0]
    for arg in args[1:]:
        result = result | arg
    return result


def sanitize_annotation(annotation: Any, cls: type) -> Any:
    """Recursively sanitize the annotation.

    Replaces ``Self`` with the class itself; recurses into ``Union`` and
    similar types."""
    if annotation is Self:
        return cls
    if isinstance(annotation, UnionType | UnionGenericAlias):
        args = get_args(annotation)
        return fold_or([sanitize_annotation(arg, cls) for arg in args])
    if isinstance(annotation, UnpackType):
        args = get_args(annotation)
        assert len(args) == 1
        inner = args[0]
        return Unpack[sanitize_annotation(inner, cls)]
    if isinstance(annotation, GenericAliasType):
        if isinstance(annotation, CallableAliasType):
            return annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is None:
            raise TypinoxNotImplementedError(
                f"Unsupported GenericAlias: {annotation}"
            )
        return origin[tuple(sanitize_annotation(arg, cls) for arg in args)]
    if isinstance(annotation, VmappedMeta):
        origin = get_vmapped_origin_or_none(annotation)
        if origin is Self:
            return cast(type[AbstractVmapped], annotation).replace_inner(cls)
    if isinstance(annotation, AnnotatedAlias):
        origin = getattr(annotation, "__origin__")
        if origin is Self:
            return cls
        return Annotated[
            sanitize_annotation(origin, cls),
            *getattr(annotation, "__metadata__", []),
        ]
    return annotation


def sanitize_member_annotation(annotation: Any, cls: type) -> Any:
    if get_origin(annotation) is AbstractVar:
        annotation_args = get_args(annotation)
        assert len(annotation_args) == 1
        inner_annotation = sanitize_annotation(annotation_args[0], cls)
        return inner_annotation | property
    if get_origin(annotation) is AbstractClassVar:
        raise TypinoxNotImplementedError(
            "AbstractClassVar is not yet supported by Typinox."
        )
    return sanitize_annotation(annotation, cls)


def method_transform_annotations(
    fn: FunctionType, cls: type, policy: TypedPolicy
) -> FunctionType:
    """Sanitize all annotations of a method.

    Also wraps every annotation with ValidatedT[]."""
    annotations = fn.__annotations__
    for key, value in annotations.items():
        if isinstance(value, str):
            warnings.warn(
                f"Typinox: string annotations are not supported: `{value}` in {fn} of {cls}"
            )
            continue
        new_annotation = sanitize_annotation(value, cls)
        if policy.always_validated:
            new_annotation = ValidatedT[new_annotation]  # type: ignore
        if new_annotation is not value:
            annotations[key] = new_annotation
    return fn


def is_magic(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


CallableDescriptor = staticmethod | classmethod | property | EqxWrapMethod

SKIP_MAGIC_MODULES = frozenset(
    [
        "builtins",
        "typing",
        "dataclasses",
        "typinox",
        "typinox.module",
        "typinox._module",
        "equinox",
        "equinox._module",
    ]
)
SKIP_MAGIC_NAMES = frozenset(
    [
        "__validate__",
        "__validate_str__",
        "__validate_self_str__",
    ]
)


def skip_magic(
    name: str, fn: Callable | CallableDescriptor, cls: type, policy: TypedPolicy
) -> bool:
    if name in SKIP_MAGIC_NAMES:
        return True
    if fn.__module__ in SKIP_MAGIC_MODULES:
        return True
    return False


def decorate_method[T: Callable | CallableDescriptor](
    name: str, fn: T, cls: type, policy: TypedPolicy
) -> T:
    """Decorate a method with the typechecker (jaxtyped and beartype).
    Recurses into staticmethods, classmethods and properties."""
    if name in policy.skip_methods:
        return fn
    if getattr(fn, "__no_type_check__", False):
        return fn
    if isinstance(fn, staticmethod):
        actual_method = fn.__func__
        return cast(
            T, staticmethod(decorate_method(name, actual_method, cls, policy))
        )
    if isinstance(fn, classmethod):
        actual_method = fn.__func__
        return cast(
            T, classmethod(decorate_method(name, actual_method, cls, policy))
        )
    if isinstance(fn, property):
        fget = (
            decorate_method(name, fn.fget, cls, policy)
            if fn.fget is not None
            else None
        )
        fset = (
            decorate_method(name, fn.fset, cls, policy)
            if fn.fset is not None
            else None
        )
        fdel = (
            decorate_method(name, fn.fdel, cls, policy)
            if fn.fdel is not None
            else None
        )
        return cast(T, property(fget, fset, fdel))
    # # not needed after equinox 0.13
    # if isinstance(fn, EqxWrapMethod):
    #     return cast(
    #         T, EqxWrapMethod(decorate_method(name, fn.method, cls, policy))
    #     )
    if not callable(fn):
        return fn
    if not inspect.isfunction(fn):
        # We can only wrap Python-native functions.
        debug_warn(
            f"Typinox: attempting to perform typechecking decoration on unknown object: {fn}",
            TypinoxUnknownFunctionWarning,
        )
        return cast(T, fn)
    if marked_as_typed(fn):
        return cast(T, fn)
    if is_magic(name):
        if skip_magic(name, fn, cls, policy):
            return cast(T, fn)
    # Main case: pure-python function.
    pyfunc = cast(FunctionType, fn)
    pyfunc = method_transform_annotations(pyfunc, cls, policy)
    decorated = decorate_function(pyfunc)
    decorated = mark_as_typed(decorated)
    return cast(T, decorated)


class RealTypedModuleMeta(EqxModuleMeta):
    """Metaclass for TypedModule.

    If you want to create a module with a metaclass other than
    :class:`abc.ABCMeta`, :class:`equinox._module._ModuleMeta` or
    :class:`typinox.module.TypedModuleMeta`, you need to create a
    new metaclass that inherits from this class and your metaclass of
    choice.
    """

    def __new__(
        mcs,
        name,
        bases,
        dict_,
        /,
        strict: bool | None = False,
        typed_policy: TypedPolicy | dict | None = None,
        **kwargs,
    ):
        # [Step 1] Create the Module as normal.
        cls = super().__new__(mcs, name, bases, dict_, strict=strict, **kwargs)
        # Assumption:
        # - Every non-magic normal method is wrapped by Equinox.
        # - A __init__ method is created, either by the user or by Equinox.

        # [Step 2] Wrap all methods with the typechecker.
        # [Step 2.0] Prepare the typechecking policy.
        if isinstance(typed_policy, dict):
            typed_policy = TypedPolicy(**typed_policy)
        if typed_policy is None:
            typed_policy = TypedPolicy()
        policy_for_type[cls] = typed_policy
        # [Step 2.1] Wrap the methods with the typechecker.
        for key, value in cls.__dict__.items():
            if key == "__init__":
                # We skip __init__ method generated by Equinox.
                if _has_dataclass_init[cls]:
                    continue
            decorated_value = decorate_method(key, value, cls, typed_policy)
            if decorated_value is not value:
                setattr(cls, key, decorated_value)

        # [Step 3] Add the validator methods.
        old_validate = cls.__dict__.get("__validate__", None)
        old_validate_str = cls.__dict__.get("__validate_str__", None)

        # [Step 3.1] Recursively validate the fields.
        # [Step 3.1.0] Prepare the annotations to check.
        sanitized_annotations = {
            key: sanitize_member_annotation(value, cls)
            for key, value in cls.__annotations__.items()
        }
        # Exclude the fields that are marked as not typechecking.
        for field in dataclasses.fields(cls):
            if not field.metadata.get("typecheck", True):
                sanitized_annotations.pop(field.name, None)

        # [Step 3.1 cont'd] Actually validate the fields.
        def __validate_self_str__(self):
            __tracebackhide__ = True
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
                except ValidationFailed as e:
                    return str(e)
                if result is False:
                    return "it failed its custom validation"
            return ""

        def __validate_str__(self):
            __tracebackhide__ = True
            with jaxtyped("context"):  # type: ignore
                return __validate_self_str__(self)

        # Add the methods to the class.
        __validate_str__.__qualname__ = "__validate_str__"
        ABCMeta.__setattr__(cls, "__validate_self_str__", __validate_self_str__)
        ABCMeta.__setattr__(cls, "__validate_str__", __validate_str__)
        ABCMeta.__setattr__(cls, "__validate__", None)
        return cls

    # Creating an instance with MyModule(...) will call this method.
    def __call__(cls, *args, **kwargs):
        __tracebackhide__ = True
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


RealTypedModuleMeta.__name__ = "TypedModuleMeta"
