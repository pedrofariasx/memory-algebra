from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from memory_algebra.diff import (
    SoftMemory,
    compose,
    norm,
    retract,
    retrieve,
    sample_edges,
    soft_subgraph,
    transform,
)

DIM = 8


def make_states(n=6, d=DIM, seed=3):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(3):
        V = rng.standard_normal((n, d))
        A = np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(A, 0.0)
        t = rng.uniform(0, 10, size=n)
        w = rng.uniform(0, 1, size=n)
        states.append(
            SoftMemory(V=jnp.asarray(V), A=jnp.asarray(A), t=jnp.asarray(t), w=jnp.asarray(w))
        )
    return states


def test_soft_compose_associative():
    a, b, c = make_states()
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))
    for f in ("V", "A", "t", "w"):
        assert jnp.allclose(getattr(left, f), getattr(right, f), atol=1e-12)


def test_soft_compose_commutative():
    a, b, _ = make_states()
    ab, ba = compose(a, b), compose(b, a)
    for f in ("V", "A", "t", "w"):
        assert jnp.allclose(getattr(ab, f), getattr(ba, f), atol=1e-12)


def test_retrieve_differentiable():
    (a, _, _) = make_states()
    q = jnp.asarray(np.random.default_rng(1).standard_normal(DIM))

    def loss(V):
        m = SoftMemory(V=V, A=a.A, t=a.t, w=a.w)
        return retrieve(m, q, tau=0.5)[0]

    g = jax.grad(loss)(a.V)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.abs(g).sum()) > 0.0


def test_retrieve_temporal_shifts_focus():
    n, d = 2, DIM
    V = jnp.eye(n, d)
    A = jnp.zeros((n, n))
    t = jnp.asarray([0.0, 5.0])
    w = jnp.ones(n)
    m = SoftMemory(V=V, A=A, t=t, w=w)
    q = jnp.ones(d)
    p_now = retrieve(m, q, time=0.0, time_scale=0.5)
    p_late = retrieve(m, q, time=5.0, time_scale=0.5)
    assert float(p_now[0]) > float(p_now[1])
    assert float(p_late[1]) > float(p_late[0])


def test_transform_functorial():
    a, b, _ = make_states()
    W = jnp.linalg.qr(jnp.asarray(np.random.default_rng(5).standard_normal((DIM, DIM))))[0]
    lhs = transform(compose(a, b), W)
    rhs = compose(transform(a, W), transform(b, W))
    assert jnp.allclose(lhs.V, rhs.V, atol=1e-10)
    assert jnp.allclose(lhs.A, rhs.A, atol=1e-12)


def test_retract_removes_mass():
    (a, _, _) = make_states()
    p = jax.nn.one_hot(0, len(a.w))
    r = retract(a, p)
    assert float(jnp.abs(r.A[0]).sum()) < 1e-12
    assert float(jnp.abs(r.A[:, 0]).sum()) < 1e-12
    assert float(jnp.abs(r.V[0]).sum()) < 1e-12
    assert norm(r) < norm(a)


def test_soft_subgraph_preserves_focus():
    (a, _, _) = make_states()
    p = retrieve(a, a.V[0], tau=0.1)
    sub = soft_subgraph(a, p)
    assert float(sub.w.sum()) <= float(a.w.sum()) + 1e-9
    assert sub.A.shape == a.A.shape


def test_sample_edges_bounds():
    (_, a, _) = make_states()
    key = jax.random.PRNGKey(0)
    s = sample_edges(a.A, key, tau=0.5)
    assert s.shape == a.A.shape
    assert bool(jnp.all(s >= 0.0)) and bool(jnp.all(s <= 1.0))


def test_gradient_flows_through_compose():
    a, b, _ = make_states()
    q = jnp.asarray(np.random.default_rng(2).standard_normal(DIM))

    def loss(V):
        m = compose(SoftMemory(V=V, A=a.A, t=a.t, w=a.w), b)
        return retrieve(m, q, tau=0.5).sum()

    g = jax.grad(loss)(a.V)
    assert jnp.all(jnp.isfinite(g))
