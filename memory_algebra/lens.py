from __future__ import annotations

from memory_algebra.core import MemoryObject, Query, Retrieval, compose, retract, retrieve


def lens_get(state: MemoryObject, query: Query, theta: float = 0.9) -> Retrieval:
    return retrieve(state, query, theta)


def lens_put(
    state: MemoryObject,
    query: Query,
    delta: MemoryObject,
    theta: float = 0.9,
) -> MemoryObject:
    view = retrieve(state, query, theta).memory
    base = retract(state, view, project=False)
    return compose(base, delta)
