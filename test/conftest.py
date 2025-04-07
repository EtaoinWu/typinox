import chex
import equinox
import jax
import pytest
from jax import (
    numpy as jnp,
    tree as jt,
)

import typinox


@pytest.fixture(autouse=True)
def add_doctest_np(doctest_namespace):
    doctest_namespace["jax"] = jax
    doctest_namespace["jnp"] = jnp
    doctest_namespace["jt"] = jt

    doctest_namespace["typinox"] = typinox
    doctest_namespace["tpx"] = typinox

    doctest_namespace["equinox"] = equinox
    doctest_namespace["eqx"] = equinox

    doctest_namespace["chex"] = chex
