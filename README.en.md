[Versão em português](README.md)

# memory-algebra

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/pedrofariasx/memory-algebra/actions/workflows/tests.yml/badge.svg)](https://github.com/pedrofariasx/memory-algebra/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)](tests/)
[![Preprint](https://img.shields.io/badge/preprint-CC--BY--4.0-orange)](https://zenodo.org/records/22815162)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22815162.svg)](https://doi.org/10.5281/zenodo.22815162)

A **memory algebra** for AI systems that solves **catastrophic interference** (catastrophic forgetting): successive memory compositions without destroying previous representations. Python implementation with formal proofs of exact associativity, empirical validation against baselines (LSTM, vector summation, kNN), and preprint available in PT-BR and EN.

## Central Hypothesis

Current AI memory systems (RNNs, FIFO buffers, vector summation) suffer from **catastrophic interference**: successive compositions dilute prior representations. This project proposes a **memory algebra** structured in four components $(V, G, T, P)$ that simultaneously satisfies:

| Property | Description |
|:-----------:|:----------|
| $\boxed{A}$ Associativity | $(m_1 \oplus m_2) \oplus m_3 \cong m_1 \oplus (m_2 \oplus m_3)$ |
| $\boxed{R}$ Recoverability | $d_s(m, \rho(M, q)) \to 0$ for any inserted memory $m$ |
| $\boxed{S}$ Stability | $\|M_t\| < C$ under pruning $\Pi_C$ |
| $\boxed{P}$ Preservation | $\otimes$ does not destroy recoverable structure |
| $\boxed{E}$ Efficiency | Tractable operations over discrete representations |

---

## Component Structure

A memory is the tuple $m = (N, v, E, \tau, w)$:

- **$V$** (`vectors`): dense embeddings in $\mathbb{R}^d$ — semantic content.
- **$G$** (`edges`): labeled graph between nodes — structural relations.
- **$T$** (`times`): timestamps — temporal conflict resolution.
- **$P$** (`weights`): utility in $[0,1]$ — retention/forgetting control.

---

## Algebraic Operations

| Operation | Symbol | Implementation | Property |
|:---------|:-------:|:--------------|:------------|
| Compose | $\oplus$ | Atomic disjoint union + quotient by $\theta$-merge (`compose`, `quotient`) | Exact associativity |
| Retract | $\ominus$ | Node removal + orthogonal projection (`retract`) | Consistency via lenses |
| Transform | $\otimes$ | Matrix $W$ over vectors + edge renaming (`transform`) | Functoriality |
| Retrieve | $\rho$ | Cosine score × temporal factor × weight + BFS on quotient (`retrieve`) | Recoverability |
| Prune | $\Pi_C$ | Forgetting by $w < \varepsilon$ + shortcuts + capacity ceiling (`prune`) | Stability |

---

## Empirical Results

### 3.1 — Associativity (N=100 memories)

Three composition paths (left, right, parallel) produce **identical signatures**.

```
max_distance(left, right)  = 4.44e-17
max_distance(left, parallel) = 4.44e-17
```

### 3.2 — Catastrophic Interference (50 compositions)

Ability to recover the first memory ($m_1$) after 50 subsequent compositions:

| Method | Final $d_s$ | |
|:-------|:-----------:|:--|
| **Algebra $(V,G,T,P)$** | **0.0** | perfect recovery |
| Vector summation | 0.929 | total dilution |
| LSTM (hidden=dim) | 0.588 | partial forgetting |
| FIFO buffer (cap=10) | 1.0 | total loss |

### 3.3 — Conflict Resolution

```
M = compose("João is a doctor", t=1, "João became an engineer", t=2)
```

- Query at $t=2$ → returns "engineer" first ✓
- Query at $t=1$ → returns "doctor" first ✓
- History preserved (both facts exist in $M$) ✓
- Entity "João" merged in quotient ✓

### 3.4 — Multi-hop Expressivity

```
m1(A→B), m2(B→C), m3(C→D) ⇒ query "A?" with hops=3 recovers D
```

Transitive path inferred without direct edge A→D ✓

### Lens Laws

- **Put-Get:** $d_s(\Delta, get(put(M,q,\Delta), q_\Delta)) = 0.0$ ✓
- **Get-Put:** $put(M, q, get(M,q)) = M$ (isolated views) ✓

---

## Formal Foundation

### Theory (Pillar 1)

- Composition modeled as **pushout** in the category of typed graphs.
- Associativity proof: disjoint union in $\mathbf{Set}$ + unique transitive closure of $R_\theta$.
- Asymmetric delta lenses with Put-Get and Get-Put laws.
- Hybrid metric $d_s = \alpha d_V + \beta d_G + \gamma d_T + \delta d_P$.

Full details in [`docs/formalizacao.md`](docs/formalizacao.md).

### Categorical Ground Truth (Catlab.jl)

`formal/catlab_ground_truth.jl` validates exact pushouts with `is_isomorphic`:

```
[PASS] associativity: ((m1⊕m2)⊕m3) ≅ (m1⊕(m2⊕m3))
[PASS] commutativity: (m2⊕m1) ≅ (m1⊕m2)
[PASS] identity: m1 ⊕ ∅ ≅ m1
ALL CHECKS PASSED
```

### Differentiable Layer (JAX)

`memory_algebra/diff.py` implements the algebra as pure functions over JAX arrays:
- `compose`: adjacency matrix summation — exact matrix associativity.
- `retrieve`: cosine softmax × temporal factor × weight.
- `sample_edges`: Gumbel-Softmax for differentiable edge sampling.
- Gradients flow through `compose` and `retrieve` (verified with `jax.grad`).

---

## Repository Structure

```
memory-algebra/
├── memory_algebra/                  # Python package (algebra)
│   ├── core.py             # MemoryObject, compose, quotient, retrieve, retract, transform
│   ├── diff.py             # Differentiable algebra (JAX)
│   ├── metric.py           # d_s with contextual weights
│   ├── lens.py             # Get/put lenses
│   ├── stability.py        # Pruning Π_C
│   ├── baselines.py        # LSTM baseline
│   └── builder.py          # Constructors and utilities
├── tests/                  # 30 tests (pytest + hypothesis)
├── experiments/
│   ├── run_validation.py   # Protocols 3.1–3.4 + lenses
│   └── results.json        # Numerical results
├── formal/
│   └── catlab_ground_truth.jl  # Categorical validation (Catlab.jl)
├── docs/
│   └── formalizacao.md     # Proofs and theory↔code correspondence
├── instructions.md         # Original execution plan
└── pyproject.toml
```

---

## Running

```bash
# Tests (30 tests)
.venv/bin/python -m pytest

# Full empirical validation
.venv/bin/python experiments/run_validation.py

# Categorical ground truth (requires Julia + Catlab 0.15)
julia formal/catlab_ground_truth.jl
```

---

## Dependencies

| Package | Use |
|:-------|:----|
| numpy | Vectors, linear algebra |
| networkx | Structural graph analysis |
| jax | Differentiable layer |
| pytest + hypothesis | Property-based tests |
| Catlab.jl 0.15 (Julia) | Categorical ground truth |

---

## Conclusion

The algebra $(V, G, T, P)$ resolves the central trade-off: **associative composition without representation destruction**. Results demonstrate that graph + time + utility structure protects individual vectors from dilution — a property no tested baseline (vector summation, LSTM, FIFO) can maintain. The differentiable layer opens a path toward integration with neural learning while preserving algebraic guarantees.

---

## Citation

Farias, P. F. R. (2026). *An Algebra of Memory: Formal Foundations and Empirical Validation of Associative Composition Without Catastrophic Interference*. Zenodo. [https://doi.org/10.5281/zenodo.22815162](https://doi.org/10.5281/zenodo.22815162)
