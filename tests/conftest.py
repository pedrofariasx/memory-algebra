from __future__ import annotations

import numpy as np
import pytest

from memory_algebra import orthogonal_concepts

from helpers import DIM, SEED


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def concepts(rng) -> list[np.ndarray]:
    return orthogonal_concepts(DIM, DIM, rng)
