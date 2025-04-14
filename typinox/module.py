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
    # Workaround for static type checkers

    type AbstractVar[T] = T | property

    @dataclass_transform(
        field_specifiers=(dataclasses.field, equinox.field, field),
    )
    class TypedModuleMeta(equinox._module._ModuleMeta):  # type: ignore
        pass

    class TypedModule(equinox.Module, metaclass=TypedModuleMeta):
        def __validate_self_str__(self) -> str:
            return ""

        def _validate(self) -> None: ...

else:
    # The real branch; this is what gets executed at runtime

    from ._module import RealTypedModuleMeta

    AbstractVar = equinox.AbstractVar
    TypedModuleMeta = RealTypedModuleMeta

    class TypedModule(equinox.Module, metaclass=TypedModuleMeta):
        """
        Base class. Inherit this to create a typinox typed module.

        This class is a subclass of :class:`equinox.Module` and provides
        automatic type validation for its fields.

        Thanks to :class:`equinox.Module`, every typed module is
        automatically a `dataclass`_ and a `pytree`_.

        .. _dataclass: https://docs.python.org/3/library/dataclasses.html
        .. _pytree: https://docs.jax.dev/en/latest/pytrees.html

        **Fields**

        Declare typed fields at the class level, using the same syntax as
        for `dataclass`_. The class will automatically become a pytree
        containing all the fields. It will also automatically validate
        the types of the fields at runtime.

        .. code-block:: python

            from jaxtyping import Array, Float, Key

            class MyModule(TypedModule):
                weight: Float[Array, "m n"]
                bias: Float[Array, "m"]
                sublayers: list[TypedModule]

        **Initialization**

        A default constructor is provided similar to a dataclass. It fills
        each field with the arguments in order. You can also pass keyword
        arguments to fill the fields of the same name. For example
        ``MyModule(w, sublayers=[m1, m2], bias=b)``.

        Alternatively, you can provide an ``__init__`` method to customize
        the initialization behavior.

        .. code-block:: python

            from jaxtyping import Array, Float, Key

            class MyModule(TypedModule):
                weight: Float[Array, "m n"]
                bias: Float[Array, "m"]
                sublayers: list[TypedModule]

                def __init__(self, m: int, n: int,
                             key: Key[Array, ""],
                             sublayers: list[TypedModule] = [],
                            ):
                    self.weight = jax.random.normal(key, (m, n))
                    self.bias = jax.ones((m,))
                    self.sublayers = sublayers

        **Methods**

        Define methods at the class level just like any other class.
        Every methods is automatically wrapped with :func:`beartype.beartype`
        and :func:`jaxtyping.jaxtyped` to perform run-time type checking.

        .. code-block:: python

            class MyModule(TypedModule):
                # ... same as above

                def __call__(self, x: Float[Array, "n"]) -> Float[Array, "m"]:
                    y = jnp.dot(self.weight, x) + self.bias
                    for layer in self.sublayers:
                        y = layer(y)
                    # if y is not a Float[Array, "m"] at this point,
                    # an error will be raised by beartype
                    return y

        .. tip::

            The method does not have to be named ``__call__()``; it might
            as well be named ``forward()``. The dunder name ``__call__``
            is not specially treated by typinox.
        """

        pass

    def _validate(self) -> None:
        """A helper method to validate the type of the module.

        This is particularly useful for modules that may be vmapped.
        The argument to and return value of :func:`jax.vmap` are
        pytrees of arrays with added dimensions, and thus may be invalid
        modules. For example:

        .. code-block:: python

            def create_module(key: Key[Array, ""]):
                return MyModule(n=3, m=2, key=key, sublayers=[])

            key = jax.random.key(1)
            some_keys = jax.random.split(key, 5)
            some_modules = jax.vmap(create_module)(some_keys)

        In this case, ``some_modules`` has ``MyModule`` as its ``.__class__``,
        but is not a valid ``MyModule`` because its ``weight`` and ``bias``
        have shapes ``(5, 2, 3)`` and ``(5, 2)`` respectively. Therefore,
        ``some_modules._validate()`` will fail.

        .. hint::

            In this case we can annotate it with :class:`Vmapped`, as it
            passes the type check for ``Vmapped[MyModule, "5"]``.

        Returns
        -------
        None

        Raises
        ------
        TypinoxTypeViolation
            If the type of the module is not valid.
        """
        __tracebackhide__ = True
        cls = type(self)
        for kls in cls.__mro__[-2::-1]:
            if hasattr(kls, "__validate_self_str__"):
                validated = kls.__validate_self_str__(self)
                if validated != "":
                    raise TypinoxTypeViolation(
                        f"the value ({self}) is not a {cls}, as {validated}"
                    )

    # Patch the _validate method to the class
    # This is to avoid TypedModuleMeta to attempt to
    #   type check _validate, leading to an infinite recursion
    type.__setattr__(TypedModule, "_validate", _validate)
