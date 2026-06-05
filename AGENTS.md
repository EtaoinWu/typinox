# Agent Instructions

## Project Overview

Typinox is a Python library that enhances runtime type checking for
`jaxtyping`-annotated JAX arrays and `equinox.Module` classes. The public API
is exposed from `typinox/__init__.py` and centers on:

- `TypedModule`, `TypedPolicy`, `field`, and `AbstractVar` in
  `typinox/module.py`.
- `Vmapped`, `VmappedT`, and `VmappedI` in `typinox/vmapped.py`.
- `ValidatedT`, `ValidationFailed`, and custom validation helpers in
  `typinox/validator.py`.
- `typinox.tree.stack()` and `typinox.tree.unstack()` for PyTrees.
- Shape validation helpers in `typinox/shaped.py`.

The implementation uses JAX, Equinox, jaxtyping, and beartype. Several internals
intentionally rely on private APIs from Equinox and jaxtyping, so dependency
updates and runtime behavior changes need careful testing.

## Repository Layout

- `typinox/`: package source.
- `typinox/_module.py`: runtime metaclass implementation for `TypedModule`.
- `typinox/_vmapped.py`: runtime implementation for `Vmapped*` annotations.
- `typinox/module.py` and `typinox/vmapped.py`: public wrappers with
  `TYPE_CHECKING` branches for static type checkers.
- `test/`: pytest suite, including module, vmapped, validator, tree, and
  end-to-end ML tests.
- `docs/`: Sphinx documentation sources and Makefile.
- `pyproject.toml`: package metadata, dependency groups, lint, pytest, pyright,
  and mypy configuration.
- `uv.lock`: lockfile; prefer `uv` commands for local development.

## Environment And Commands

The project declares Python `>=3.14` in `pyproject.toml`. Use the locked
environment when possible:

```bash
uv sync --group dev
```

Common validation commands:

```bash
uv run pytest ./test
uv run ruff check
uv run ruff format --check
uv run pyright .
uv run mypy .
uv run make -C docs html
```

Useful focused test commands:

```bash
uv run pytest ./test/test_vmapped.py
uv run pytest ./test/test_validator.py
uv run pytest ./test/module
uv run pytest ./test/module/test_module_vmapped.py
```

## Coding Conventions

- Keep code formatted for Ruff's configured `line-length = 80`.
- Preserve the modern Python typing style already used here, including PEP 695
  generic syntax such as `def stack[T](...)`.
- Avoid string annotations in runtime-checked code. `_module.py` warns that
  string annotations are not supported by Typinox's annotation transform.
- Prefer existing imports and helper APIs over adding new abstraction layers.
- Keep public aliases in `typinox/__init__.py` stable unless the change is
  intentionally API-facing.
- Use `typinox.field(typecheck=False)` when a `TypedModule` field must be
  excluded from runtime field validation.
- Raise or preserve Typinox-specific exceptions from `typinox/error.py` for
  library errors.

## Runtime Type-Checking Notes

`TypedModule` behavior is implemented by `RealTypedModuleMeta` in
`typinox/_module.py`.

- Class creation decorates methods with `jaxtyped(..., typechecker=beartype)`.
- `Self`, unions, `Annotated`, `Unpack`, `AbstractVar`, and `Vmapped` annotations
  are sanitized before runtime checking.
- Instance creation validates fields unless disabled with
  `TypedPolicy(typecheck_init_result=False)`.
- Method annotations are wrapped in `ValidatedT[...]` when
  `TypedPolicy(always_validated=True)`.
- `typing.no_type_check` and `TypedPolicy(skip_methods=...)` are supported
  escape hatches.

When editing this area, run at least:

```bash
uv run pytest ./test/module ./test/test_validator.py
uv run pyright .
```

`Vmapped` behavior is implemented by `typinox/_vmapped.py`.

- `Vmapped[T, "dims"]` uses `$` to mark where the original shape appears.
- If `$` is omitted, the vmapped axes are prepended.
- Unsupported jaxtyping modifiers such as `*`, `?`, `#`, and `...` should keep
  producing `TypinoxAnnotationError`.
- Shape memo state from jaxtyping is intentionally restored after failed checks.

When editing this area, run at least:

```bash
uv run pytest ./test/test_vmapped.py ./test/module/test_module_vmapped.py
```

## Documentation

Docs are Sphinx-based and live under `docs/src`. API pages are in
`docs/src/api`, usage pages are in `docs/src/usage`, and the development page is
`docs/src/development.rst`.

Build docs with:

```bash
uv run make -C docs html
```

Serve docs locally with:

```bash
uv run make -C docs serve
```

Keep code examples compatible with the doctest namespace configured in
`test/conftest.py`, which provides `jax`, `jnp`, `jt`, `typinox`, `tpx`,
`equinox`, `eqx`, and `chex`.

## Testing Guidance

The default pytest config includes coverage and doctest modules:

```toml
addopts = "--cov=typinox --cov-report=term-missing:skip-covered --doctest-modules"
```

Use focused tests while iterating, then run the full suite before finishing
behavioral changes. For changes to docs examples or docstrings, run pytest too
because doctests are enabled.

## Change Safety

- Do not relax runtime validation without adding tests for the new behavior.
- Do not change dependency bounds casually; Typinox tracks current JAX,
  jaxtyping, Equinox, and beartype behavior closely.
- Be careful around `TYPE_CHECKING` branches in `module.py` and `vmapped.py`;
  they exist to keep static type checkers usable while runtime classes remain
  dynamic.
- Preserve jaxtyping memo restoration in failed `Vmapped` checks.
- Avoid unrelated refactors in `_module.py` and `_vmapped.py`; small changes can
  affect static typing, dataclass generation, pytree behavior, and runtime
  validation at the same time.
