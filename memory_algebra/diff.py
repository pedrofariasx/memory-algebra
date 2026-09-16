from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.nn import softmax


@dataclass(frozen=True, eq=False)
class SoftMemory:
    V: jnp.ndarray
    A: jnp.ndarray
    t: jnp.ndarray
    w: jnp.ndarray


def compose(a: SoftMemory, b: SoftMemory) -> SoftMemory:
    return SoftMemory(
        V=a.V + b.V,
        A=a.A + b.A,
        t=jnp.maximum(a.t, b.t),
        w=jnp.maximum(a.w, b.w),
    )


def retrieve(
    m: SoftMemory,
    q: jnp.ndarray,
    time: float | None = None,
    time_scale: float = 1.0,
    tau: float = 1.0,
) -> jnp.ndarray:
    Vn = m.V / jnp.maximum(jnp.linalg.norm(m.V, axis=1, keepdims=True), 1e-12)
    qn = q / jnp.maximum(jnp.linalg.norm(q), 1e-12)
    s = softmax(Vn @ qn / tau)
    s = s * (0.5 + 0.5 * m.w)
    if time is not None:
        dt = (m.t - time) / jnp.maximum(time_scale, 1e-9)
        s = s * jnp.exp(-(dt * dt))
    return s / jnp.maximum(s.sum(), 1e-12)


def soft_subgraph(m: SoftMemory, p: jnp.ndarray) -> SoftMemory:
    mask = jnp.outer(p, p)
    return SoftMemory(V=m.V * p[:, None], A=m.A * mask, t=m.t, w=m.w * p)


def retract(m: SoftMemory, p: jnp.ndarray) -> SoftMemory:
    keep = 1.0 - p
    return SoftMemory(
        V=m.V * keep[:, None],
        A=m.A * jnp.outer(keep, keep),
        t=m.t,
        w=m.w * keep,
    )


def transform(m: SoftMemory, matrix: jnp.ndarray) -> SoftMemory:
    return SoftMemory(V=m.V @ matrix.T, A=m.A, t=m.t, w=m.w)


def sample_edges(A: jnp.ndarray, key: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:
    logits = jnp.stack([jnp.zeros_like(A), A], axis=-1)
    g = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-9, maxval=1.0)))
    y = softmax((logits + g) / tau, axis=-1)
    return y[..., 1]


def norm(m: SoftMemory) -> jnp.ndarray:
    return m.w.sum() + jnp.abs(m.A).sum()
