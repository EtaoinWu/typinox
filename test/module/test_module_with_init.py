import chex
from jax import (
    numpy as jnp,
    random as jr,
)
from jaxtyping import Array, Float, Key, Scalar

from typinox import (
    TypedModule,
    set_debug_mode,
)

set_debug_mode(True)


class GaussianGenerator(TypedModule):
    mean: Float[Array, " d"]
    linear: Float[Array, "d d"]

    def __init__(self, mean: Float[Array, " d"], cov: Float[Array, "d d"]):
        self.mean = mean
        self.linear = jnp.linalg.cholesky(cov, upper=True)

    def sample(
        self, key: Key[Scalar, ""], shape: tuple[int, ...] = ()
    ) -> Float[Array, "... d"]:
        z = jr.normal(key, shape + self.mean.shape)
        return self.mean.reshape(
            (1,) * len(shape) + self.mean.shape
        ) + jnp.einsum("...i,ij->...j", z, self.linear)


def test_gaussian_generator():
    key = jr.key(0)
    mean = jnp.array([1.0, 2.0])
    cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
    gen = GaussianGenerator(mean, cov)

    assert isinstance(gen.mean, jnp.ndarray)
    assert isinstance(gen.linear, jnp.ndarray)
    assert gen.mean.shape == (2,)
    assert gen.linear.shape == (2, 2)

    samples = gen.sample(key, shape=(10000,))
    assert samples.shape == (10000, 2)

    # Check that the samples have approximately the correct mean and covariance
    sample_mean = jnp.mean(samples, axis=0)
    sample_cov = jnp.cov(samples, rowvar=False)

    chex.assert_trees_all_close(sample_mean, mean, atol=0.1)
    chex.assert_trees_all_close(sample_cov, cov, atol=0.1)
