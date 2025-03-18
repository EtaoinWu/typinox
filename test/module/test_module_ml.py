import jax
from beartype.typing import Callable
from jax import (
    numpy as jnp,
    random as jr,
    tree as jt,
)
from jaxtyping import Array, Float, Scalar

import typinox as tpx
from typinox import (
    TypedModule,
    ValidatedT,
    Vmapped,
    ensure_shape,
    set_debug_mode,
)

set_debug_mode(True)


class LinearLayer(TypedModule):
    w: Float[Array, "in out"]
    b: Float[Array, " out"]

    def __call__(self, x: Float[Array, " in"]) -> Float[Array, " out"]:
        return jnp.dot(x, self.w) + self.b


LinearLayerT = ValidatedT[LinearLayer]


class ActivationLayer(TypedModule):
    fn: Callable[[Float[Array, " in"]], Float[Array, " out=in"]] = tpx.field(
        static=True
    )

    def __call__(self, x: Float[Array, " in"]) -> Float[Array, " out=in"]:
        return self.fn(x)


ActivationLayerT = ValidatedT[ActivationLayer]


class Sequential(TypedModule):
    layers: list[LinearLayerT | ActivationLayerT]

    def __call__(self, x: Float[Array, " in"]) -> Float[Array, " out"]:
        for layer in self.layers:
            x = layer(x)
        return x


batch_size = 1000


def loss(net, dk) -> Scalar:
    xy = jr.uniform(dk, (batch_size, 2), minval=-1, maxval=1)
    res = jax.vmap(net)(xy).reshape(-1)
    truth = (xy[:, 1] * xy[:, 0] < 0).astype(jnp.float32)
    return jnp.mean((res - truth) ** 2)


@jax.jit
def train_step(net, dk, lr):
    def iter(net, sk):
        l, grad = jax.value_and_grad(loss)(net, sk)
        new_net = jt.map(lambda n, g: n - lr * g, net, grad)
        return new_net, l

    new_net, l = jax.lax.scan(iter, net, jr.split(dk, 500), unroll=2)
    return new_net, jnp.mean(l)


def test_end2end_train():
    key = jr.key(0)
    key11, key12, key21, key22, key31, key32, key = jr.split(key, 7)
    net = Sequential(
        [
            LinearLayer(jr.normal(key11, (2, 3)), jr.normal(key12, (3,))),
            ActivationLayer(jnp.tanh),
            LinearLayer(jr.normal(key21, (3, 4)), jr.normal(key22, (4,))),
            ActivationLayer(jnp.tanh),
            LinearLayer(jr.normal(key31, (4, 1)), jr.normal(key32, (1,))),
            ActivationLayer(jax.nn.sigmoid),
        ]
    )
    data_key, key = jr.split(key, 2)

    for dk in jr.split(data_key, 1):
        net, l = train_step(net, dk, 1.0)

    l = loss(net, key)

    assert l < 0.25 * 0.2


class MixtureOfExperts(TypedModule):
    n_experts: int = tpx.field(static=True)
    experts: Vmapped[Sequential, " n_experts"]
    gate: Sequential

    def __validate__(self):
        ensure_shape(".n_experts", self.n_experts, "n_experts")

    def __call__(self, x: Float[Array, " in"]) -> Float[Array, " out"]:
        g = self.gate(x)
        choice = jax.vmap(lambda expert, weight: weight * expert(x))(
            self.experts, g
        )
        return jnp.sum(choice, axis=0)


MixtureOfExpertsT = ValidatedT[MixtureOfExperts]


def test_end2end_train_moe():
    def gen_expert(k):
        k1, k2, k3, k4 = jr.split(k, 4)
        return Sequential(
            [
                LinearLayer(jr.normal(k1, (2, 3)), jr.normal(k2, (3,))),
                ActivationLayer(jnp.tanh),
                LinearLayer(jr.normal(k3, (3, 1)), jr.normal(k4, (1,))),
            ]
        )

    def gen_gate(k, m):
        k1, k2, k3, k4 = jr.split(k, 4)
        return Sequential(
            [
                LinearLayer(jr.normal(k1, (2, 2)), jr.normal(k2, (2,))),
                ActivationLayer(jax.nn.sigmoid),
                LinearLayer(jr.normal(k3, (2, m)), jr.normal(k4, (m,))),
                ActivationLayer(jax.nn.softmax),
            ]
        )

    def gen_moe(k, m) -> MixtureOfExpertsT:
        k1, k2 = jr.split(k, 2)
        return MixtureOfExperts(
            m, jax.vmap(gen_expert)(jr.split(k1, m)), gen_gate(k2, m)
        )

    key = jr.key(1)
    gen_key, data_key, key = jr.split(key, 3)
    moe = gen_moe(gen_key, 3)

    for dk in jr.split(data_key, 3):
        moe, l = train_step(moe, dk, 1.0)

    l = loss(moe, key)
    assert l < 0.25 * 0.2
